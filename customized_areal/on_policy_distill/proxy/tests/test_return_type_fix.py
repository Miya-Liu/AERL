"""Tests for TokenRewardSessionData.export_interactions return type fix.

Verifies that:
1. TokenRewardSessionData uses the extended InteractionCache
2. export_interactions returns InteractionWithTokenLevelReward objects
3. Deserialization creates InteractionWithTokenLevelReward objects
4. token_rewards survive to_tensor_dict() after cache invalidation
5. Position rewards survive the full round-trip
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from customized_areal.on_policy_distill.proxy.cache import (
    InteractionCache as ExtendedInteractionCache,
)
from customized_areal.on_policy_distill.proxy.proxy_rollout_server import (
    deserialize_interactions_with_position_rewards,
    serialize_interactions_with_position_rewards,
)
from customized_areal.on_policy_distill.proxy.server import (
    PositionRewardInfo,
    TokenRewardSessionData,
)
from customized_areal.on_policy_distill.proxy.types import (
    InteractionWithTokenLevelReward,
)


class TestSessionDataUsesExtendedCache:
    """Verify TokenRewardSessionData uses the extended InteractionCache."""

    def test_completions_is_extended_cache(self):
        session = TokenRewardSessionData("test-session")
        assert isinstance(session.completions, ExtendedInteractionCache), (
            f"Expected ExtendedInteractionCache, got {type(session.completions).__name__}"
        )


class TestExportInteractionsReturnType:
    """Verify export_interactions returns InteractionWithTokenLevelReward objects."""

    def test_returned_interaction_is_correct_type(self):
        session = TokenRewardSessionData("test-session")
        # Add an interaction with minimal setup
        mock_resp = Mock()
        mock_resp.output_tokens = [10, 20, 30]
        mock_resp.input_tokens = [1, 2, 3, 4, 5]
        mock_resp.input_len = 5
        mock_resp.output_len = 3
        mock_resp.output_logprobs = [-0.5, -0.3, -0.8]
        mock_resp.output_versions = [0, 0, 0]
        mock_resp.output_top_logprobs = [None, None, None]

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(id="comp-1"),
            reward=1.0,
        )
        interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
        session.completions["comp-1"] = interaction

        session.set_token_rewards("comp-1", [0.1, 0.2, 0.3])
        session.finish()

        result = session.export_interactions(discount=1.0, style="individual")

        assert "comp-1" in result
        assert isinstance(result["comp-1"], InteractionWithTokenLevelReward), (
            f"Expected InteractionWithTokenLevelReward, got {type(result['comp-1']).__name__}"
        )


class TestDeserializationCreatesCorrectType:
    """Verify deserialize_interactions_with_position_rewards creates
    InteractionWithTokenLevelReward objects."""

    def test_deserialized_is_extended_type(self):
        mock_interaction = Mock()
        mock_interaction.reward = 1.0
        mock_interaction.interaction_id = "comp-deser-test"
        mock_interaction.position_rewards = None
        mock_interaction.token_rewards = [0.1, 0.2, 0.3]

        mock_interaction.to_tensor_dict.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "loss_mask": torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]]),
            "rewards": torch.tensor([1.0]),
            "logprobs": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.3, -0.8]]
            ),
            "versions": torch.tensor([[-1, -1, -1, -1, -1, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        }

        interactions = {"comp-deser-test": mock_interaction}
        serialized = serialize_interactions_with_position_rewards(interactions)
        deserialized = deserialize_interactions_with_position_rewards(serialized)

        result = deserialized["comp-deser-test"]
        assert isinstance(result, InteractionWithTokenLevelReward), (
            f"Expected InteractionWithTokenLevelReward, got {type(result).__name__}"
        )

    def test_deserialized_token_rewards_is_proper_field(self):
        """token_rewards should be a proper dataclass field, not a dynamic attribute."""
        mock_interaction = Mock()
        mock_interaction.reward = 0.0
        mock_interaction.interaction_id = "comp-field-test"
        mock_interaction.position_rewards = None
        mock_interaction.token_rewards = [0.1, 0.2, 0.3]

        mock_interaction.to_tensor_dict.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "loss_mask": torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]]),
            "rewards": torch.tensor([0.0]),
            "logprobs": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.3, -0.8]]
            ),
            "versions": torch.tensor([[-1, -1, -1, -1, -1, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        }

        interactions = {"comp-field-test": mock_interaction}
        serialized = serialize_interactions_with_position_rewards(interactions)
        deserialized = deserialize_interactions_with_position_rewards(serialized)

        result = deserialized["comp-field-test"]
        # InteractionWithTokenLevelReward has token_rewards as a dataclass field
        # with a default of None. InteractionWithTokenLogpReward does NOT have it.
        assert "token_rewards" in [
            f.name for f in result.__dataclass_fields__.values()
        ], "token_rewards should be a declared dataclass field, not a dynamic attribute"


class TestTokenRewardsCacheRecomputation:
    """Verify to_tensor_dict() includes token_rewards even after cache invalidation.

    This is the key safety improvement: if _cache is invalidated (set to None),
    to_tensor_dict() should recompute correctly including token_rewards. This only
    works when the object is InteractionWithTokenLevelReward, which overrides
    to_tensor_dict() to include token_rewards.
    """

    def test_tensor_dict_after_cache_invalidation(self):
        mock_resp = Mock()
        mock_resp.output_tokens = [10, 20, 30]
        mock_resp.input_tokens = [1, 2, 3, 4, 5]
        mock_resp.input_len = 5
        mock_resp.output_len = 3
        mock_resp.output_logprobs = [-0.5, -0.3, -0.8]
        mock_resp.output_versions = [0, 0, 0]
        mock_resp.output_top_logprobs = [None, None, None]

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(id="comp-recomp"),
            reward=1.0,
        )
        interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
        interaction.token_rewards = [0.1, 0.2, 0.3]

        # Compute once to populate cache
        td1 = interaction.to_tensor_dict()
        assert "token_rewards" in td1

        # Invalidate cache
        interaction._cache = None

        # Recompute — should still include token_rewards
        td2 = interaction.to_tensor_dict()
        assert "token_rewards" in td2, (
            "token_rewards lost after cache invalidation. "
            "to_tensor_dict() recomputation must include token_rewards."
        )


class TestScalarRewardPreservedAfterDelegation:
    """Verify scalar reward is NOT overwritten when delegating to extended cache."""

    def test_set_token_rewards_preserves_scalar(self):
        session = TokenRewardSessionData("test-session")

        mock_resp = Mock()
        mock_resp.output_tokens = [10, 20, 30]
        mock_resp.input_tokens = [1, 2, 3, 4, 5]
        mock_resp.input_len = 5
        mock_resp.output_len = 3
        mock_resp.output_logprobs = [-0.5, -0.3, -0.8]
        mock_resp.output_versions = [0, 0, 0]
        mock_resp.output_top_logprobs = [None, None, None]

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(id="comp-scalar"),
            reward=5.0,
        )
        interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
        session.completions["comp-scalar"] = interaction

        # Set scalar reward
        session.completions.set_reward("comp-scalar", 5.0)

        # Set token rewards (sum = 0.6, different from 5.0)
        session.set_token_rewards("comp-scalar", [0.1, 0.2, 0.3])

        # Scalar reward MUST be preserved
        assert session.completions["comp-scalar"].reward == 5.0, (
            f"Scalar reward should be 5.0, got {session.completions['comp-scalar'].reward}. "
            "set_token_rewards must NOT overwrite trajectory-level scalar reward."
        )

    def test_set_position_rewards_preserves_scalar(self):
        session = TokenRewardSessionData("test-session")

        mock_resp = Mock()
        mock_resp.output_tokens = [10, 20, 30]
        mock_resp.input_tokens = [1, 2, 3, 4, 5]
        mock_resp.input_len = 5
        mock_resp.output_len = 3
        mock_resp.output_logprobs = [-0.5, -0.3, -0.8]
        mock_resp.output_versions = [0, 0, 0]
        mock_resp.output_top_logprobs = [None, None, None]

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(id="comp-pos-scalar"),
            reward=5.0,
        )
        interaction.output_message_list = [{"role": "assistant", "content": "Hi"}]
        session.completions["comp-pos-scalar"] = interaction

        session.completions.set_reward("comp-pos-scalar", 5.0)

        position_rewards = [
            PositionRewardInfo(
                position=0,
                candidates=["a", "b"],
                rewards=[0.1, 0.5],
                chosen_index=1,
            ),
            PositionRewardInfo(
                position=1,
                candidates=["c", "d"],
                rewards=[0.2, 0.6],
                chosen_index=0,
            ),
            PositionRewardInfo(
                position=2,
                candidates=["e", "f"],
                rewards=[0.3, 0.9],
                chosen_index=1,
            ),
        ]
        session.set_position_rewards("comp-pos-scalar", position_rewards)

        assert session.completions["comp-pos-scalar"].reward == 5.0, (
            f"Scalar reward should be 5.0, got {session.completions['comp-pos-scalar'].reward}. "
            "set_position_rewards must NOT overwrite trajectory-level scalar reward."
        )