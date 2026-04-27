"""Tests for on-policy distillation pipeline bug fixes."""

import inspect

import pytest


def test_bug1_completion_id_uses_interaction_id():
    """Bug 1: OnPolicyDistillAgent should use the interaction's actual ID
    from the proxy server, not an MD5 hash of completion_messages."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.core.agent",
            fromlist=["OnPolicyDistillAgent"],
        ).OnPolicyDistillAgent.run
    )
    assert "hashlib.md5" not in source, (
        "OnPolicyDistillAgent.run() should not use hashlib.md5 for "
        "completion_id. Use interaction.interaction_id from the proxy server."
    )
    assert "interaction_id" in source, (
        "OnPolicyDistillAgent.run() should use interaction.interaction_id "
        "from the proxy server as the completion_id."
    )


def test_bug4_stale_session_uses_configured_timeout():
    """Bug 4: _cleanup_stale_sessions should use _session_timeout_seconds
    (from config) instead of hardcoded SESSION_TIMEOUT_SECONDS."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.proxy.proxy_rollout_server",
            fromlist=["_cleanup_stale_sessions"],
        )._cleanup_stale_sessions
    )
    for line in source.split("\n"):
        stripped = line.strip()
        if "is_stale" in stripped and "SESSION_TIMEOUT_SECONDS" in stripped:
            pytest.fail(
                "_cleanup_stale_sessions uses hardcoded SESSION_TIMEOUT_SECONDS "
                "instead of _session_timeout_seconds. Custom timeout configs "
                "are silently ignored, causing OOM from uncleaned sessions."
            )


def test_bug2_export_does_not_remove_session():
    """Bug 2: export_trajectories should not remove the session from cache.
    Session removal should be deferred to _cleanup_stale_sessions to avoid
    race conditions when export is called after end_session."""
    module = __import__(
        "customized_areal.on_policy_distill.proxy.proxy_rollout_server",
        fromlist=["export_trajectories"],
    )
    export_trajectories_source = inspect.getsource(module.export_trajectories)
    for line in export_trajectories_source.split("\n"):
        stripped = line.strip()
        if "_session_cache.pop" in stripped:
            pytest.fail(
                "export_trajectories should not remove session from "
                "_session_cache. Defer removal to _cleanup_stale_sessions."
            )


def test_bug5_set_rewards_preserve_scalar():
    """Bug 5: InteractionCache.set_rewards should support preserving
    the scalar reward to avoid _total_reward drift from save/restore."""
    from customized_areal.on_policy_distill.proxy.cache import (
        InteractionCache,
    )
    from customized_areal.on_policy_distill.proxy.types import (
        InteractionWithTokenLevelReward,
    )

    cache = InteractionCache()

    class MockModelResponse:
        output_tokens = [1, 2, 3]
        input_tokens = [0]
        input_len = 1
        output_len = 3
        output_logprobs = [-1.0, -0.5, -0.3]

    interaction = InteractionWithTokenLevelReward(
        messages=[{"role": "user", "content": "hi"}],
        reward=5.0,
        model_response=MockModelResponse(),
    )
    interaction.interaction_id = "test-1"
    cache["test-1"] = interaction

    cache.set_rewards("test-1", [0.1, 0.2, 0.3], preserve_scalar_reward=True)

    assert cache["test-1"].reward == 5.0, (
        f"Scalar reward should be preserved, got {cache['test-1'].reward}"
    )
    assert cache.total_reward == 5.0, (
        f"total_reward should be 5.0, got {cache.total_reward}"
    )


def test_bug5_server_no_save_restore():
    """Bug 5: TokenRewardSessionData.set_token_rewards should not use
    save/restore pattern for scalar reward."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.proxy.server",
            fromlist=["TokenRewardSessionData"],
        ).TokenRewardSessionData.set_token_rewards
    )
    assert "saved_reward" not in source, (
        "TokenRewardSessionData.set_token_rewards should not use a "
        "save/restore pattern for scalar reward. Use preserve_scalar_reward=True "
        "in InteractionCache.set_rewards() instead."
    )


def test_bug6_distribute_position_rewards_warns_on_unmapped():
    """Bug 6: _distribute_position_rewards should warn when a
    position_reward's sample_index doesn't map to any minibatch."""
    import torch
    from unittest.mock import patch

    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.actor import (
        _distribute_position_rewards,
    )

    mb = {
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    mb_inputs = type("MB", (), {"mbs": [mb], "forward_indices": [0, 1]})()

    bad_pr = PositionRewardInfo(
        position=0,
        candidates=["a", "b"],
        candidate_token_ids=[1, 2],
        rewards=[0.5, -0.3],
        chosen_index=0,
        sample_index=99,
    )
    good_pr = PositionRewardInfo(
        position=1,
        candidates=["c", "d"],
        candidate_token_ids=[3, 4],
        rewards=[0.2, -0.1],
        chosen_index=0,
        sample_index=0,
    )

    with patch("customized_areal.on_policy_distill.training.actor.logger") as mock_logger:
        _distribute_position_rewards(mb_inputs, [bad_pr, good_pr])
        warning_calls = [
            c for c in mock_logger.method_calls if "warning" in str(c)
        ]
        assert len(warning_calls) > 0, (
            "_distribute_position_rewards should log a warning when "
            "a position_reward's sample_index doesn't map to any minibatch"
        )
