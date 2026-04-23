"""Integration tests for the on-policy distillation proxy server.

Tests the full HTTP request/response flow through the FastAPI ASGI stack,
exercising real serialization, session management, and reward handling
without requiring GPU or LLM backends.

Scenarios 1-4: No GPU required (mock LLM interactions)
Scenario 5: Requires GPU + SGLang (skipped if unavailable)
"""

from __future__ import annotations

import asyncio
import socket
import threading
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo as CachePRI
from customized_areal.on_policy_distill.proxy.proxy_rollout_server import (
    _admin_api_key,
    _api_key_to_session,
    _capacity,
    _session_cache,
    _session_to_api_key,
    app,
    deserialize_interactions_with_position_rewards,
)
from customized_areal.on_policy_distill.proxy.server import (
    PositionRewardInfo as ServerPRI,
)
from customized_areal.on_policy_distill.proxy.types import (
    InteractionWithTokenLevelReward,
)


# =============================================================================
# Helpers
# =============================================================================


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_mock_interaction(
    completion_id="comp-test",
    output_tokens=None,
    reward=0.0,
):
    """Create a mock InteractionWithTokenLevelReward for injection."""
    output_tokens = output_tokens or [100, 200, 300]
    mock_resp = Mock()
    mock_resp.output_tokens = output_tokens
    mock_resp.input_tokens = [1, 2, 3, 4, 5]
    mock_resp.input_len = 5
    mock_resp.output_len = len(output_tokens)
    mock_resp.output_logprobs = [-0.5] * len(output_tokens)
    mock_resp.output_versions = [0] * len(output_tokens)
    mock_resp.output_top_logprobs = [None] * len(output_tokens)

    interaction = InteractionWithTokenLevelReward(
        model_response=mock_resp,
        messages=[{"role": "user", "content": "Hello"}],
        completion=Mock(id=completion_id),
        reward=reward,
    )
    interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
    return interaction


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_server_state():
    """Reset module-level globals between tests."""
    import customized_areal.on_policy_distill.proxy.proxy_rollout_server as srv

    srv._session_cache.clear()
    srv._api_key_to_session.clear()
    srv._session_to_api_key.clear()
    srv._capacity = 0
    yield
    srv._session_cache.clear()
    srv._api_key_to_session.clear()
    srv._session_to_api_key.clear()


@pytest.fixture
def admin_headers():
    return {"Authorization": f"Bearer {_admin_api_key}"}


@pytest.fixture
def make_client():
    """Factory fixture that creates an httpx AsyncClient as an async context manager.

    Usage in tests:
        async with make_client() as client:
            r = await client.post(...)
    """

    def _make():
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://testserver")

    return _make


async def _start_session(client, admin_headers, task_id="test-task"):
    """Helper: grant capacity, start session, inject a mock interaction, return session info."""
    await client.post("/grant_capacity", headers=admin_headers)
    r = await client.post(
        "/rl/start_session", json={"task_id": task_id}, headers=admin_headers
    )
    assert r.status_code == 200
    data = r.json()
    session_id = data["session_id"]
    session_key = data["api_key"]
    session_headers = {"Authorization": f"Bearer {session_key}"}

    # Inject a mock interaction so reward endpoints have something to target
    interaction = _make_mock_interaction()
    session_data = _session_cache[session_id]
    session_data.completions["comp-test"] = interaction

    return session_id, session_key, session_headers


async def _export_interactions(client, admin_headers, session_id):
    """Helper: end session, export, deserialize."""
    # End session
    session_key = _session_to_api_key[session_id]
    session_headers = {"Authorization": f"Bearer {session_key}"}
    r = await client.post("/rl/end_session", headers=session_headers)
    assert r.status_code == 200

    # Export
    r = await client.post(
        "/export_trajectories",
        json={"session_id": session_id, "discount": 1.0, "style": "individual"},
        headers=admin_headers,
    )
    assert r.status_code == 200

    return deserialize_interactions_with_position_rewards(r.json()["interactions"])


# =============================================================================
# Scenario 1: Token Rewards Round-Trip (no GPU)
# =============================================================================


class TestTokenRewardsRoundTrip:
    """Set token rewards via HTTP, export, verify they survive serialization."""

    @pytest.mark.asyncio
    async def test_token_rewards_in_tensor_dict(self, client, admin_headers):
        session_id, session_key, session_headers = await _start_session(
            client, admin_headers
        )

        # Set token rewards via HTTP
        r = await client.post(
            "/rl/set_token_rewards",
            json={
                "interaction_id": "comp-test",
                "token_rewards": [0.1, 0.2, 0.3],
            },
            headers=session_headers,
        )
        assert r.status_code == 200

        interactions = await _export_interactions(client, admin_headers, session_id)

        assert "comp-test" in interactions
        interaction = interactions["comp-test"]

        # Token rewards flow through to_tensor_dict()
        td = interaction.to_tensor_dict()
        assert "token_rewards" in td
        # First 5 tokens are input (0.0), then 3 output tokens with rewards
        token_rewards = td["token_rewards"].squeeze(0).tolist()
        assert token_rewards[5:] == pytest.approx([0.1, 0.2, 0.3])

        # Scalar reward is NOT automatically set from token rewards
        # (token rewards and scalar reward are intentionally separate)
        # When only token rewards are set, scalar reward stays at its initial value
        assert interaction.reward == 0.0

    @pytest.mark.asyncio
    async def test_set_token_rewards_last_interaction(
        self, client, admin_headers
    ):
        session_id, session_key, session_headers = await _start_session(
            client, admin_headers
        )

        # Set token rewards without specifying interaction_id (uses last)
        r = await client.post(
            "/rl/set_token_rewards",
            json={"token_rewards": [0.4, 0.5, 0.6]},
            headers=session_headers,
        )
        assert r.status_code == 200

        interactions = await _export_interactions(client, admin_headers, session_id)
        interaction = interactions["comp-test"]
        td = interaction.to_tensor_dict()
        token_rewards = td["token_rewards"].squeeze(0).tolist()
        assert token_rewards[5:] == pytest.approx([0.4, 0.5, 0.6])


# =============================================================================
# Scenario 2: Position Rewards Round-Trip (no GPU)
# =============================================================================


class TestPositionRewardsRoundTrip:
    """Set position rewards via HTTP, export, verify correct data survives."""

    @pytest.mark.asyncio
    async def test_position_rewards_survive_export(self, client, admin_headers):
        session_id, session_key, session_headers = await _start_session(
            client, admin_headers
        )

        position_rewards = [
            {
                "position": 0,
                "candidates": ["a", "b"],
                "candidate_token_ids": [10, 20],
                "logprobs": [-1.0, -0.5],
                "rewards": [0.1, 0.5],
                "chosen_index": 1,
            },
            {
                "position": 1,
                "candidates": ["c", "d"],
                "candidate_token_ids": [30, 40],
                "logprobs": [-0.8, -0.3],
                "rewards": [0.2, 0.6],
                "chosen_index": 0,
            },
            {
                "position": 2,
                "candidates": ["e", "f", "g"],
                "candidate_token_ids": [50, 60, 70],
                "logprobs": [-1.2, -0.6, -0.4],
                "rewards": [0.3, 0.7, 0.9],
                "chosen_index": 2,
            },
        ]

        r = await client.post(
            "/rl/set_position_rewards",
            json={
                "interaction_id": "comp-test",
                "position_rewards": position_rewards,
            },
            headers=session_headers,
        )
        assert r.status_code == 200

        interactions = await _export_interactions(client, admin_headers, session_id)
        interaction = interactions["comp-test"]

        # Position rewards survive serialization round-trip
        pr = getattr(interaction, "position_rewards", None)
        assert pr is not None
        assert len(pr) == 3
        assert pr[0].position == 0
        assert pr[0].candidates == ["a", "b"]
        assert pr[0].rewards == [0.1, 0.5]
        assert pr[0].chosen_index == 1
        assert pr[2].candidates == ["e", "f", "g"]
        assert pr[2].chosen_index == 2

        # Derived token rewards (chosen rewards) appear in tensor dict
        td = interaction.to_tensor_dict()
        assert "token_rewards" in td
        token_rewards = td["token_rewards"].squeeze(0).tolist()
        assert token_rewards[5:] == pytest.approx([0.5, 0.2, 0.9])

    @pytest.mark.asyncio
    async def test_position_rewards_with_logprobs_survive(
        self, client, admin_headers
    ):
        session_id, session_key, session_headers = await _start_session(
            client, admin_headers
        )

        position_rewards = [
            {
                "position": 0,
                "candidates": ["a", "b"],
                "logprobs": [-1.0, -0.5],
                "rewards": [0.1, 0.5],
                "chosen_index": 1,
            },
            {
                "position": 1,
                "candidates": ["c", "d"],
                "logprobs": [-0.8, -0.3],
                "rewards": [0.2, 0.6],
                "chosen_index": 0,
            },
            {
                "position": 2,
                "candidates": ["e", "f"],
                "logprobs": [-1.0, -2.0],
                "rewards": [0.3, 0.4],
                "chosen_index": 0,
            },
        ]

        r = await client.post(
            "/rl/set_position_rewards",
            json={
                "interaction_id": "comp-test",
                "position_rewards": position_rewards,
            },
            headers=session_headers,
        )
        assert r.status_code == 200

        # Compute entropy via HTTP
        r = await client.post(
            "/rl/compute_entropy",
            json={"interaction_id": "comp-test"},
            headers=session_headers,
        )
        assert r.status_code == 200
        data = r.json()
        assert "entropies" in data
        assert len(data["entropies"]) == 3
        assert all(e >= 0 for e in data["entropies"])
        assert "avg_entropy" in data


# =============================================================================
# Scenario 3: Scalar/Position Reward Separation (no GPU)
# =============================================================================


class TestScalarPositionRewardSeparation:
    """Verify scalar reward is preserved when position rewards are set.

    This is critical for tree backup advantage computation: the scalar
    (trajectory-level) reward must NOT be overwritten by sum of position
    rewards (which are for distillation loss only).
    """

    @pytest.mark.asyncio
    async def test_scalar_reward_preserved_with_position_rewards(
        self, client, admin_headers
    ):
        session_id, session_key, session_headers = await _start_session(
            client, admin_headers
        )

        # 1. Set scalar reward (trajectory-level, for tree backup)
        r = await client.post(
            "/rl/set_reward",
            json={"interaction_id": "comp-test", "reward": 5.0},
            headers=session_headers,
        )
        assert r.status_code == 200

        # 2. Set position rewards where sum of chosen rewards != 5.0
        position_rewards = [
            {
                "position": 0,
                "candidates": ["a", "b"],
                "rewards": [0.1, 0.5],
                "chosen_index": 1,
            },
            {
                "position": 1,
                "candidates": ["c", "d"],
                "rewards": [0.2, 0.6],
                "chosen_index": 0,
            },
            {
                "position": 2,
                "candidates": ["e", "f", "g"],
                "rewards": [0.3, 0.7, 0.9],
                "chosen_index": 2,
            },
        ]
        r = await client.post(
            "/rl/set_position_rewards",
            json={
                "interaction_id": "comp-test",
                "position_rewards": position_rewards,
            },
            headers=session_headers,
        )
        assert r.status_code == 200

        interactions = await _export_interactions(client, admin_headers, session_id)
        interaction = interactions["comp-test"]

        # Scalar reward MUST be preserved at 5.0
        assert interaction.reward == 5.0, (
            f"Scalar reward should be 5.0, got {interaction.reward}. "
            "Position rewards must NOT overwrite the trajectory-level scalar reward."
        )

        # Position rewards still attached for distillation
        pr = getattr(interaction, "position_rewards", None)
        assert pr is not None
        assert len(pr) == 3

        # Derived token rewards in tensor dict (chosen rewards)
        td = interaction.to_tensor_dict()
        token_rewards = td["token_rewards"].squeeze(0).tolist()
        assert token_rewards[5:] == pytest.approx([0.5, 0.2, 0.9])

        # Scalar reward in tensor dict
        assert td["rewards"].item() == 5.0

    @pytest.mark.asyncio
    async def test_scalar_reward_preserved_with_token_rewards(
        self, client, admin_headers
    ):
        session_id, session_key, session_headers = await _start_session(
            client, admin_headers
        )

        # Set scalar reward first
        r = await client.post(
            "/rl/set_reward",
            json={"interaction_id": "comp-test", "reward": 3.0},
            headers=session_headers,
        )
        assert r.status_code == 200

        # Then set token rewards (sum = 1.2, different from 3.0)
        r = await client.post(
            "/rl/set_token_rewards",
            json={
                "interaction_id": "comp-test",
                "token_rewards": [0.2, 0.4, 0.6],
            },
            headers=session_headers,
        )
        assert r.status_code == 200

        interactions = await _export_interactions(client, admin_headers, session_id)
        interaction = interactions["comp-test"]

        # TokenRewardSessionData.export_interactions preserves scalar reward
        # only if it was already set; if reward is None, it defaults to sum
        assert interaction.reward == 3.0


# =============================================================================
# Scenario 4: Full Workflow Episode (no GPU)
# =============================================================================


class TestFullWorkflowEpisode:
    """Test OpenAIProxyWorkflow.arun_episode against a live server.

    Starts the FastAPI server on a real port via uvicorn so that
    OpenAIProxyClient (which uses aiohttp) can reach it.
    """

    @pytest.fixture
    def live_server(self):
        """Start the FastAPI app on a real port."""
        import uvicorn

        port = _find_free_port()
        config = uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error"
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        # Give server time to start
        import time

        time.sleep(1.5)
        yield f"http://127.0.0.1:{port}"
        server.should_exit = True
        thread.join(timeout=5)

    @pytest.mark.asyncio
    async def test_workflow_episode_with_position_rewards(self, live_server):
        from customized_areal.on_policy_distill.proxy.workflow import (
            OpenAIProxyWorkflow,
        )

        class MockAgent:
            """Agent that sets rewards via proxy_client."""

            async def run(self, data, **extra_kwargs):
                proxy_client = extra_kwargs.get("proxy_client")
                base_url = extra_kwargs.get("base_url")
                api_key = extra_kwargs.get("api_key")

                if proxy_client is not None:
                    # Set scalar reward
                    await proxy_client.set_reward("comp-test", 3.0)

                    # Set position rewards
                    pos_rewards = [
                        CachePRI(
                            position=0,
                            candidates=["a", "b"],
                            candidate_token_ids=[10, 20],
                            logprobs=[-1.0, -0.5],
                            rewards=[0.1, 0.5],
                            chosen_index=1,
                        ),
                        CachePRI(
                            position=1,
                            candidates=["c", "d"],
                            candidate_token_ids=[30, 40],
                            logprobs=[-0.8, -0.3],
                            rewards=[0.2, 0.6],
                            chosen_index=0,
                        ),
                        CachePRI(
                            position=2,
                            candidates=["e", "f", "g"],
                            candidate_token_ids=[50, 60, 70],
                            logprobs=[-1.2, -0.6, -0.4],
                            rewards=[0.3, 0.7, 0.9],
                            chosen_index=2,
                        ),
                    ]
                    await proxy_client.set_position_rewards(
                        "comp-test", pos_rewards
                    )

                # Return dict with position rewards + scalar reward
                return {
                    "comp-test": {
                        "position_rewards": pos_rewards,
                        "scalar_reward": 3.0,
                    }
                }

        # Create workflow
        workflow = OpenAIProxyWorkflow(
            agent=MockAgent(),
            proxy_addr=live_server,
            admin_api_key=_admin_api_key,
            discount=1.0,
            export_style="individual",
        )

        # Mock engine and workflow context
        mock_engine = Mock()
        mock_engine.version = 0

        # We need to inject a mock interaction into the session after it starts
        # but before the agent runs. Patch _grant_capacity to do the injection.
        original_grant = workflow._grant_capacity

        async def patched_grant(session):
            await original_grant(session)
            # After capacity is granted and session starts, inject mock interaction
            # The session should have been created by now
            for sid, sdata in _session_cache.items():
                if len(sdata.completions) == 0:
                    sdata.completions["comp-test"] = _make_mock_interaction(
                        reward=0.0
                    )
                    break

        workflow._grant_capacity = patched_grant

        with patch(
            "customized_areal.on_policy_distill.proxy.workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.get.return_value = Mock(task_id=123)
            mock_ctx.get_aiohttp_session = AsyncMock(
                return_value=AsyncMock()
            )
            mock_ctx.get_httpx_client = AsyncMock(return_value=AsyncMock())
            mock_ctx.stat_scope = Mock(return_value="test")

            with patch(
                "customized_areal.on_policy_distill.proxy.workflow.stats_tracker"
            ) as mock_stats:
                mock_tracker = Mock()
                mock_tracker.scalar = Mock()
                mock_stats.get.return_value = mock_tracker

                result = await workflow.arun_episode(mock_engine, {"prompt": "test"})

        # Verify result is a tensor dict
        assert result is not None
        assert isinstance(result, dict)

        # Position rewards attached
        assert "position_rewards" in result
        all_pr = result["position_rewards"]
        assert len(all_pr) == 3
        assert all_pr[0].position == 0
        assert all_pr[0].chosen_index == 1
        # sample_index should be 0 (first interaction)
        assert all_pr[0].sample_index == 0

        # Token rewards in tensor dict
        assert "token_rewards" in result
        assert "token_reward_mask" in result

        # Input IDs present
        assert "input_ids" in result

    @pytest.mark.asyncio
    async def test_workflow_episode_scalar_reward_only(self, live_server):
        from customized_areal.on_policy_distill.proxy.workflow import (
            OpenAIProxyWorkflow,
        )

        class ScalarAgent:
            async def run(self, data, **extra_kwargs):
                proxy_client = extra_kwargs.get("proxy_client")
                if proxy_client is not None:
                    await proxy_client.set_reward("comp-test", 2.5)
                return 2.5

        workflow = OpenAIProxyWorkflow(
            agent=ScalarAgent(),
            proxy_addr=live_server,
            admin_api_key=_admin_api_key,
            discount=1.0,
            export_style="individual",
        )

        mock_engine = Mock()
        mock_engine.version = 0

        original_grant = workflow._grant_capacity

        async def patched_grant(session):
            await original_grant(session)
            for sid, sdata in _session_cache.items():
                if len(sdata.completions) == 0:
                    sdata.completions["comp-test"] = _make_mock_interaction(
                        reward=0.0
                    )
                    break

        workflow._grant_capacity = patched_grant

        with patch(
            "customized_areal.on_policy_distill.proxy.workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.get.return_value = Mock(task_id=456)
            mock_ctx.get_aiohttp_session = AsyncMock(
                return_value=AsyncMock()
            )
            mock_ctx.get_httpx_client = AsyncMock(return_value=AsyncMock())
            mock_ctx.stat_scope = Mock(return_value="test")

            with patch(
                "customized_areal.on_policy_distill.proxy.workflow.stats_tracker"
            ) as mock_stats:
                mock_tracker = Mock()
                mock_tracker.scalar = Mock()
                mock_stats.get.return_value = mock_tracker

                result = await workflow.arun_episode(
                    mock_engine, {"prompt": "scalar test"}
                )

        assert result is not None
        assert "rewards" in result
        # No position_rewards for scalar-only workflow
        assert "position_rewards" not in result


# =============================================================================
# Scenario 5: GPU + SGLang Integration Test
# =============================================================================

CUDA_AVAILABLE = False
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUWithSGLang:
    """End-to-end test with real SGLang engine and LLM generation.

    This test:
    1. Starts the base proxy rollout server (with SGLang engine)
    2. Initializes SGLang via /create_engine + /call "initialize"
    3. Sends real chat/completions requests through the proxy
    4. Sets token/position rewards via HTTP
    5. Exports interactions and verifies the full data flow
    """

    @pytest.fixture
    def sglang_server(self):
        """Start the base proxy_rollout_server with SGLang support."""
        import uvicorn
        from areal.experimental.openai.proxy.proxy_rollout_server import (
            app as base_app,
        )

        port = _find_free_port()
        config = uvicorn.Config(
            base_app, host="127.0.0.1", port=port, log_level="error"
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        import time

        time.sleep(2.0)
        yield f"http://127.0.0.1:{port}", port
        server.should_exit = True
        thread.join(timeout=5)

    @pytest.mark.asyncio
    async def test_real_llm_with_token_rewards(self, sglang_server):
        import aiohttp

        base_url, port = sglang_server

        # Get admin key from the base server
        from areal.experimental.openai.proxy.proxy_rollout_server import (
            _admin_api_key as base_admin_key,
        )

        admin_headers = {"Authorization": f"Bearer {base_admin_key}"}

        async with aiohttp.ClientSession() as session:
            # 1. Create SGLang engine
            # Find a small model to use
            model_path = "Qwen/Qwen3-0.6B"  # Small model for testing
            r = await session.post(
                f"{base_url}/create_engine",
                json={
                    "engine": "areal.engine.sglang.RemoteSGLangEngine",
                    "engine_name": "sglang-test",
                    "config": {
                        "model": model_path,
                        "tp_size": 1,
                    },
                },
                headers=admin_headers,
            )
            assert r.status == 200

            # 2. Initialize the engine
            r = await session.post(
                f"{base_url}/call",
                json={
                    "method": "initialize",
                    "engine_name": "sglang-test",
                },
                headers=admin_headers,
            )
            assert r.status == 200
            # Wait for engine to initialize
            await asyncio.sleep(10)

            # 3. Grant capacity + start session
            await session.post(f"{base_url}/grant_capacity", headers=admin_headers)
            r = await session.post(
                f"{base_url}/rl/start_session",
                json={"task_id": "gpu-test"},
                headers=admin_headers,
            )
            data = await r.json()
            session_id = data["session_id"]
            session_key = data["api_key"]
            session_headers = {"Authorization": f"Bearer {session_key}"}

            # 4. Send real chat/completions request
            r = await session.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model_path,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 20,
                    "temperature": 0.0,
                },
                headers=session_headers,
            )
            assert r.status == 200
            completion = await r.json()
            completion_id = completion.get("id")
            assert completion_id is not None

            # 5. Set token rewards
            # Get number of output tokens from the completion
            output_tokens = completion["usage"]["completion_tokens"]
            token_rewards = [0.1] * output_tokens

            r = await session.post(
                f"{base_url}/rl/set_token_rewards",
                json={
                    "interaction_id": completion_id,
                    "token_rewards": token_rewards,
                },
                headers=session_headers,
            )
            assert r.status == 200

            # 6. End session + export
            await session.post(f"{base_url}/rl/end_session", headers=session_headers)

            r = await session.post(
                f"{base_url}/export_trajectories",
                json={
                    "session_id": session_id,
                    "discount": 1.0,
                    "style": "individual",
                },
                headers=admin_headers,
            )
            assert r.status == 200
            export_data = await r.json()
            assert "interactions" in export_data
            interactions = export_data["interactions"]
            assert len(interactions) > 0

    @pytest.mark.asyncio
    async def test_real_llm_with_position_rewards(self, sglang_server):
        import aiohttp

        base_url, port = sglang_server

        from areal.experimental.openai.proxy.proxy_rollout_server import (
            _admin_api_key as base_admin_key,
        )

        admin_headers = {"Authorization": f"Bearer {base_admin_key}"}

        async with aiohttp.ClientSession() as session:
            # Create + initialize SGLang engine
            model_path = "Qwen/Qwen3-0.6B"
            r = await session.post(
                f"{base_url}/create_engine",
                json={
                    "engine": "areal.engine.sglang.RemoteSGLangEngine",
                    "engine_name": "sglang-test",
                    "config": {"model": model_path, "tp_size": 1},
                },
                headers=admin_headers,
            )
            assert r.status == 200

            r = await session.post(
                f"{base_url}/call",
                json={"method": "initialize", "engine_name": "sglang-test"},
                headers=admin_headers,
            )
            assert r.status == 200
            await asyncio.sleep(10)

            # Start session + chat/completions
            await session.post(f"{base_url}/grant_capacity", headers=admin_headers)
            r = await session.post(
                f"{base_url}/rl/start_session",
                json={"task_id": "gpu-pos-test"},
                headers=admin_headers,
            )
            data = await r.json()
            session_id = data["session_id"]
            session_key = data["api_key"]
            session_headers = {"Authorization": f"Bearer {session_key}"}

            r = await session.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model_path,
                    "messages": [{"role": "user", "content": "Say 'test'"}],
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
                headers=session_headers,
            )
            assert r.status == 200
            completion = await r.json()
            completion_id = completion.get("id")

            # Set scalar + position rewards
            r = await session.post(
                f"{base_url}/rl/set_reward",
                json={"interaction_id": completion_id, "reward": 4.0},
                headers=session_headers,
            )
            assert r.status == 200

            output_tokens = completion["usage"]["completion_tokens"]
            # Build position rewards for each output token
            position_rewards = []
            for i in range(output_tokens):
                position_rewards.append(
                    {
                        "position": i,
                        "candidates": ["tok_a", "tok_b"],
                        "rewards": [0.1, 0.5],
                        "chosen_index": 1,
                    }
                )

            r = await session.post(
                f"{base_url}/rl/set_position_rewards",
                json={
                    "interaction_id": completion_id,
                    "position_rewards": position_rewards,
                },
                headers=session_headers,
            )
            assert r.status_code == 200 or r.status == 200

            # Export
            await session.post(f"{base_url}/rl/end_session", headers=session_headers)
            r = await session.post(
                f"{base_url}/export_trajectories",
                json={
                    "session_id": session_id,
                    "discount": 1.0,
                    "style": "individual",
                },
                headers=admin_headers,
            )
            assert r.status == 200
            export_data = await r.json()
            assert len(export_data.get("interactions", {})) > 0
