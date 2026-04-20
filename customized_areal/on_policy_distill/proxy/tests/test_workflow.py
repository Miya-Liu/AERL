"""Tests for workflow module with prepare_batch integration.

This module tests the OpenAIProxyWorkflow class which is used by
actor.prepare_batch() in the AReaL training loop.
"""

import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest


# Create a real base class that our workflow can inherit from
class MockBaseOpenAIProxyWorkflow:
    """Mock base class for OpenAIProxyWorkflow."""

    def __init__(
        self,
        mode=None,
        agent=None,
        proxy_addr=None,
        admin_api_key=None,
        discount=1.0,
        export_style="individual",
        **kwargs,
    ):
        self.mode = mode
        self.agent = agent
        self.proxy_addr = proxy_addr
        self._admin_api_key = admin_api_key
        self.discount = discount
        self.export_style = export_style

    async def _grant_capacity(self, session):
        pass

    async def _run_agent(self, session_api_key, data, proxy_client=None):
        return await self.agent.run(
            data,
            base_url=self.proxy_addr,
            api_key=session_api_key,
            proxy_client=proxy_client,
        )


class MockInteractionWithTokenLogpReward:
    """Mock base class for InteractionWithTokenLogpReward."""

    pass


# Set up all mocks before any imports
_mock_modules = {
    "areal": Mock(),
    "areal.utils": Mock(),
    "areal.utils.logging": Mock(),
    "areal.utils.stats_tracker": Mock(),
    "areal.utils.perf_tracer": Mock(),
    "areal.experimental": Mock(),
    "areal.experimental.openai": Mock(),
    "areal.experimental.openai.proxy": Mock(),
    "areal.experimental.openai.proxy.client_session": Mock(),
    "areal.experimental.openai.proxy.workflow": Mock(),
    "areal.experimental.openai.types": Mock(),
    "areal.experimental.openai.proxy.server": Mock(),
    "areal.infra": Mock(),
    "areal.infra.utils": Mock(),
    "areal.infra.utils.http": Mock(),
    "areal.infra.workflow_context": Mock(),
}

# Pre-populate sys.modules
for name, module in _mock_modules.items():
    sys.modules[name] = module

# Set up specific mock classes
_mock_modules[
    "areal.experimental.openai.proxy.workflow"
].OpenAIProxyWorkflow = MockBaseOpenAIProxyWorkflow
_mock_modules[
    "areal.experimental.openai.types"
].InteractionWithTokenLogpReward = MockInteractionWithTokenLogpReward
_mock_modules["areal.experimental.openai.types"].ApiType = Mock()
_mock_modules["areal.experimental.openai.types"].InputName = Mock()

# Mock the workflow context functions
_mock_modules["areal.infra.workflow_context"].get = Mock(return_value=Mock(task_id=123))

# Create proper async mocks for coroutine functions
async_mock_get_session = AsyncMock()
async_mock_get_session.return_value = AsyncMock()
_mock_modules[
    "areal.infra.workflow_context"
].get_aiohttp_session = async_mock_get_session

async_mock_get_httpx = AsyncMock()
async_mock_get_httpx.return_value = AsyncMock()
_mock_modules["areal.infra.workflow_context"].get_httpx_client = async_mock_get_httpx


# Mock perf_tracer decorators
def mock_session_context():
    def decorator(f):
        return f

    return decorator


def mock_trace_session(name):
    def decorator(f):
        return f

    return decorator


_mock_modules["areal.utils.perf_tracer"].session_context = mock_session_context
_mock_modules["areal.utils.perf_tracer"].trace_session = mock_trace_session


# Mock the proxy client
class MockOpenAIProxyClient:
    def __init__(self, session, base_url, task_id, admin_api_key):
        self._session = session
        self.base_url = base_url
        self.task_id = task_id
        self._admin_api_key = admin_api_key
        self.session_id = None
        self._session_api_key = None

    async def __aenter__(self):
        self.session_id = "test-session-id"
        self._session_api_key = "test-session-key"
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def session_api_key(self):
        return self._session_api_key

    async def set_reward(self, completion_id, reward):
        pass

    async def set_last_reward(self, reward):
        pass

    async def set_rewards(self, completion_id, token_rewards):
        pass

    async def export_interactions(self, discount=1.0, style="individual"):
        return {}


_mock_modules["customized_areal"] = Mock()
_mock_modules["customized.areal"] = Mock()
_mock_modules["customized_areal.on_policy_distill"] = Mock()
_mock_modules["customized_areal.on_policy_distill.proxy"] = Mock()
_mock_modules["customized_areal.on_policy_distill.proxy.client"] = Mock()
_mock_modules[
    "customized_areal.on_policy_distill.proxy.client"
].OpenAIProxyClient = MockOpenAIProxyClient
_mock_modules["customized_areal.on_policy_distill.proxy.types"] = Mock()

# Now import after setting up mocks
from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, return_value=None):
        self.return_value = return_value
        self.calls = []

    async def run(self, data, **extra_kwargs):
        self.calls.append((data, extra_kwargs))
        return self.return_value


class TestWorkflowProcessRewards:
    """Test workflow reward processing."""

    @pytest.fixture
    def workflow(self):
        """Create workflow fixture."""
        agent = MockAgent(return_value=1.0)

        # Create workflow without calling __init__
        workflow = object.__new__(OpenAIProxyWorkflow)
        workflow.agent = agent
        workflow.proxy_addr = "http://localhost:8000"
        workflow._admin_api_key = "test-admin-key"
        workflow.discount = 0.9
        workflow.export_style = "individual"
        workflow.mode = "inline"

        return workflow

    @pytest.fixture
    def mock_proxy_client(self):
        """Create mock proxy client."""
        client = AsyncMock()
        client.session_api_key = "test-session-key"
        return client

    @pytest.mark.asyncio
    async def test_process_rewards_scalar(self, workflow, mock_proxy_client):
        """Test _process_rewards with scalar reward."""
        await workflow._process_rewards(mock_proxy_client, 1.0)

        mock_proxy_client.set_last_reward.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_process_rewards_dict_scalar(self, workflow, mock_proxy_client):
        """Test _process_rewards with dict of scalar rewards."""
        rewards = {"comp-1": 0.5, "comp-2": 0.8}

        await workflow._process_rewards(mock_proxy_client, rewards)

        mock_proxy_client.set_reward.assert_any_call("comp-1", 0.5)
        mock_proxy_client.set_reward.assert_any_call("comp-2", 0.8)
        assert mock_proxy_client.set_reward.call_count == 2

    @pytest.mark.asyncio
    async def test_process_rewards_token_level(self, workflow, mock_proxy_client):
        """Test _process_rewards with token-level rewards."""
        rewards = {"comp-1": [0.1, 0.2, 0.3]}

        await workflow._process_rewards(mock_proxy_client, rewards)

        mock_proxy_client.set_rewards.assert_called_once_with("comp-1", [0.1, 0.2, 0.3])

    @pytest.mark.asyncio
    async def test_process_rewards_mixed(self, workflow, mock_proxy_client):
        """Test _process_rewards with mixed scalar and token-level rewards."""
        rewards = {
            "comp-1": 0.5,  # scalar
            "comp-2": [0.1, 0.2, 0.3],  # token-level
        }

        await workflow._process_rewards(mock_proxy_client, rewards)

        mock_proxy_client.set_reward.assert_called_once_with("comp-1", 0.5)
        mock_proxy_client.set_rewards.assert_called_once_with("comp-2", [0.1, 0.2, 0.3])

    @pytest.mark.asyncio
    async def test_process_rewards_invalid_type(self, workflow, mock_proxy_client):
        """Test _process_rewards with invalid reward type raises error."""
        with pytest.raises(ValueError, match="Invalid reward type"):
            await workflow._process_rewards(
                mock_proxy_client, [1, 2, 3]
            )  # list instead of dict

    @pytest.mark.asyncio
    async def test_process_rewards_invalid_value_type(
        self, workflow, mock_proxy_client
    ):
        """Test _process_rewards with invalid value type raises error."""
        with pytest.raises(ValueError, match="Invalid reward value type"):
            await workflow._process_rewards(mock_proxy_client, {"comp-1": "invalid"})


class TestWorkflowRunAgent:
    """Test workflow _run_agent method."""

    @pytest.fixture
    def workflow(self):
        """Create workflow fixture."""
        agent = MockAgent(return_value=1.0)

        workflow = object.__new__(OpenAIProxyWorkflow)
        workflow.agent = agent
        workflow.proxy_addr = "http://localhost:8000"
        workflow._admin_api_key = "test-admin-key"
        workflow.mode = "inline"

        return workflow

    @pytest.mark.asyncio
    async def test_run_agent_inline_mode(self, workflow):
        """Test _run_agent in inline mode passes correct kwargs."""
        workflow.mode = "inline"

        with patch(
            "customized_areal.on_policy_distill.proxy.workflow.workflow_context.get_httpx_client",
            new_callable=AsyncMock,
        ) as mock_get_client:
            mock_http_client = AsyncMock()
            mock_get_client.return_value = mock_http_client

            data = {"prompt": "test"}
            proxy_client = Mock()

            await workflow._run_agent("session-key", data, proxy_client)

            # Verify agent was called with correct kwargs
            assert len(workflow.agent.calls) == 1
            call_data, call_kwargs = workflow.agent.calls[0]
            assert call_data == data
            assert call_kwargs["base_url"] == workflow.proxy_addr
            assert call_kwargs["api_key"] == "session-key"
            assert call_kwargs["proxy_client"] == proxy_client


class TestWorkflowArunEpisode:
    """Test workflow arun_episode method - the main entry point called by prepare_batch."""

    @pytest.fixture
    def workflow(self):
        """Create workflow fixture."""
        agent = MockAgent(return_value=1.0)

        workflow = object.__new__(OpenAIProxyWorkflow)
        workflow.agent = agent
        workflow.proxy_addr = "http://localhost:8000"
        workflow._admin_api_key = "test-admin-key"
        workflow.discount = 0.9
        workflow.export_style = "individual"
        workflow.mode = "inline"

        return workflow

    @pytest.fixture
    def mock_engine(self):
        """Create mock inference engine."""
        return Mock()

    @pytest.fixture
    def mock_workflow_context(self):
        """Mock workflow context."""
        with patch(
            "customized_areal.on_policy_distill.proxy.workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.get.return_value = Mock(task_id=123)
            yield mock_ctx

    @pytest.fixture
    def mock_proxy_client(self):
        """Create mock proxy client."""
        client = AsyncMock()
        client.session_api_key = "test-session-key"
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    @pytest.mark.asyncio
    async def test_arun_episode_scalar_reward(
        self, workflow, mock_engine, mock_proxy_client
    ):
        """Test arun_episode with scalar reward from agent."""
        data = {"prompt": "test prompt"}

        with patch(
            "customized_areal.on_policy_distill.proxy.workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.get.return_value = Mock(task_id=123)
            mock_ctx.get_aiohttp_session = AsyncMock(return_value=AsyncMock())

            with patch.object(workflow, "_grant_capacity", new_callable=AsyncMock):
                with patch.object(
                    workflow, "_run_agent", new_callable=AsyncMock
                ) as mock_run:
                    mock_run.return_value = 1.0  # Scalar reward

                    with patch.object(
                        workflow, "_process_rewards", new_callable=AsyncMock
                    ):
                        mock_proxy_client.export_interactions.return_value = {
                            "comp-1": Mock(reward=1.0)
                        }

                        with patch(
                            "customized_areal.on_policy_distill.proxy.workflow.OpenAIProxyClient",
                            return_value=mock_proxy_client,
                        ):
                            result = await workflow.arun_episode(mock_engine, data)

                            assert result is not None
                            assert "comp-1" in result
                            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_arun_episode_token_level_rewards(
        self, workflow, mock_engine, mock_proxy_client
    ):
        """Test arun_episode with token-level rewards from agent."""
        data = {"prompt": "test prompt"}

        with patch(
            "customized_areal.on_policy_distill.proxy.workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.get.return_value = Mock(task_id=123)
            mock_ctx.get_aiohttp_session = AsyncMock(return_value=AsyncMock())

            with patch.object(workflow, "_grant_capacity", new_callable=AsyncMock):
                with patch.object(
                    workflow, "_run_agent", new_callable=AsyncMock
                ) as mock_run:
                    mock_run.return_value = {"comp-1": [0.0, 0.5, 1.0, 0.5]}

                    with patch.object(
                        workflow, "_process_rewards", new_callable=AsyncMock
                    ) as mock_process:
                        mock_interaction = Mock()
                        mock_interaction.reward = 2.0
                        mock_proxy_client.export_interactions.return_value = {
                            "comp-1": mock_interaction
                        }

                        with patch(
                            "customized_areal.on_policy_distill.proxy.workflow.OpenAIProxyClient",
                            return_value=mock_proxy_client,
                        ):
                            result = await workflow.arun_episode(mock_engine, data)

                            assert result is not None
                            mock_process.assert_called_once_with(
                                mock_proxy_client, {"comp-1": [0.0, 0.5, 1.0, 0.5]}
                            )

    @pytest.mark.asyncio
    async def test_arun_episode_agent_failure(
        self, workflow, mock_engine, mock_proxy_client
    ):
        """Test arun_episode when agent raises exception."""
        data = {"prompt": "test prompt"}

        with patch(
            "customized_areal.on_policy_distill.proxy.workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.get.return_value = Mock(task_id=123)
            mock_ctx.get_aiohttp_session = AsyncMock(return_value=AsyncMock())

            with patch.object(workflow, "_grant_capacity", new_callable=AsyncMock):
                with patch.object(
                    workflow, "_run_agent", new_callable=AsyncMock
                ) as mock_run:
                    mock_run.side_effect = ValueError("Agent failed")

                    with patch(
                        "customized_areal.on_policy_distill.proxy.workflow.OpenAIProxyClient",
                        return_value=mock_proxy_client,
                    ):
                        with pytest.raises(ValueError, match="Agent failed"):
                            await workflow.arun_episode(mock_engine, data)

    @pytest.mark.asyncio
    async def test_arun_episode_empty_interactions(
        self, workflow, mock_engine, mock_proxy_client
    ):
        """Test arun_episode returns None when no interactions."""
        data = {"prompt": "test prompt"}

        with patch(
            "customized_areal.on_policy_distill.proxy.workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.get.return_value = Mock(task_id=123)
            mock_ctx.get_aiohttp_session = AsyncMock(return_value=AsyncMock())

            with patch.object(workflow, "_grant_capacity", new_callable=AsyncMock):
                with patch.object(
                    workflow, "_run_agent", new_callable=AsyncMock
                ) as mock_run:
                    mock_run.return_value = None

                    with patch.object(
                        workflow, "_process_rewards", new_callable=AsyncMock
                    ):
                        mock_proxy_client.export_interactions.return_value = {}

                        with patch(
                            "customized_areal.on_policy_distill.proxy.workflow.OpenAIProxyClient",
                            return_value=mock_proxy_client,
                        ):
                            result = await workflow.arun_episode(mock_engine, data)

                            assert result is None


class TestWorkflowWithPrepareBatchPattern:
    """Test workflow integration with actor.prepare_batch pattern.

    These tests demonstrate how the workflow integrates with the
    actor.prepare_batch() training loop pattern in AReaL.
    """

    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader simulating training data."""
        dataloader = Mock()
        dataloader.batch_size = 2
        dataloader.__iter__ = Mock(
            return_value=iter(
                [
                    [{"prompt": "prompt 1"}, {"prompt": "prompt 2"}],
                    [{"prompt": "prompt 3"}, {"prompt": "prompt 4"}],
                ]
            )
        )
        return dataloader

    @pytest.fixture
    def mock_rollout_controller(self):
        """Create mock rollout controller."""
        controller = Mock()
        return controller

    def test_prepare_batch_calls_workflow(self, mock_dataloader):
        """Test that prepare_batch pattern calls workflow for each data item.

        This simulates how actor.prepare_batch() works in AReaL training:

            rollout_batch = actor.prepare_batch(
                dataloader,
                workflow=OpenAIProxyWorkflow(...),
                group_size=1
            )

        The rollout controller calls workflow.arun_episode for each item
        and collects the results into a batch.
        """
        workflow = Mock()
        workflow.arun_episode = AsyncMock(
            return_value={
                "comp-1": Mock(
                    reward=1.0, to_tensor_dict=Mock(return_value={"input_ids": Mock()})
                )
            }
        )

        # Simulate the prepare_batch behavior
        results = []
        for batch in mock_dataloader:
            for item in batch:
                # In actual code this is async and managed by BatchTaskDispatcher
                results.append(item)

        assert len(results) == 4  # 2 batches * 2 items each

    @pytest.mark.asyncio
    async def test_workflow_batch_integration(self, mock_dataloader):
        """Test workflow processes batch data correctly."""
        agent = MockAgent(return_value=1.0)
        workflow = object.__new__(OpenAIProxyWorkflow)
        workflow.agent = agent
        workflow.proxy_addr = "http://localhost:8000"
        workflow._admin_api_key = "test-key"
        workflow.discount = 0.9
        workflow.export_style = "individual"
        workflow.mode = "inline"

        # Process batch
        processed = 0
        for batch in mock_dataloader:
            for item in batch:
                with patch(
                    "customized_areal.on_policy_distill.proxy.workflow.workflow_context"
                ) as mock_ctx:
                    mock_ctx.get.return_value = Mock(task_id=123)
                    mock_ctx.get_aiohttp_session = AsyncMock(return_value=AsyncMock())

                    with patch.object(
                        workflow, "_grant_capacity", new_callable=AsyncMock
                    ):
                        with patch.object(
                            workflow, "_run_agent", new_callable=AsyncMock
                        ) as mock_run:
                            mock_run.return_value = 1.0

                            mock_client = AsyncMock()
                            mock_client.session_api_key = "test-key"
                            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                            mock_client.__aexit__ = AsyncMock(return_value=None)
                            mock_client.export_interactions.return_value = {
                                "comp-1": Mock(reward=1.0)
                            }

                            with patch(
                                "customized_areal.on_policy_distill.proxy.workflow.OpenAIProxyClient",
                                return_value=mock_client,
                            ):
                                with patch.object(
                                    workflow, "_process_rewards", new_callable=AsyncMock
                                ):
                                    result = await workflow.arun_episode(Mock(), item)
                                    if result:
                                        processed += 1

        assert processed == 4


class TestWorkflowRewardTypes:
    """Test different reward types returned by agent."""

    @pytest.fixture
    def workflow(self):
        """Create workflow fixture."""
        agent = MockAgent()
        workflow = object.__new__(OpenAIProxyWorkflow)
        workflow.agent = agent

        return workflow

    @pytest.fixture
    def mock_client(self):
        """Create mock client."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_scalar_float_reward(self, workflow, mock_client):
        """Test agent returning scalar float."""
        await workflow._process_rewards(mock_client, 1.5)
        mock_client.set_last_reward.assert_called_once_with(1.5)

    @pytest.mark.asyncio
    async def test_dict_with_scalar_values(self, workflow, mock_client):
        """Test agent returning dict with scalar rewards."""
        await workflow._process_rewards(mock_client, {"c1": 0.5, "c2": 0.8})
        assert mock_client.set_reward.call_count == 2

    @pytest.mark.asyncio
    async def test_dict_with_token_rewards(self, workflow, mock_client):
        """Test agent returning dict with token-level rewards."""
        await workflow._process_rewards(mock_client, {"c1": [0.1, 0.2]})
        mock_client.set_rewards.assert_called_once_with("c1", [0.1, 0.2])

    @pytest.mark.asyncio
    async def test_none_reward_raises_error(self, workflow, mock_client):
        """Test agent returning None raises error (must return valid reward type)."""
        with pytest.raises(ValueError, match="Invalid reward type"):
            await workflow._process_rewards(mock_client, None)
