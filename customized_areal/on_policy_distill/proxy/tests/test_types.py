"""Tests for types module."""

from dataclasses import dataclass

import pytest

from customized_areal.on_policy_distill.proxy.types import (
    InteractionWithTokenLevelReward,
)


@dataclass
class MockModelResponse:
    """Mock model response for testing."""

    output_tokens: list[int]
    input_len: int
    output_len: int
    output_logprobs: list[float] | None = None


@dataclass
class MockCompletion:
    """Mock completion for testing."""

    id: str = "test-completion-id"


class TestInteractionWithTokenLevelReward:
    """Test InteractionWithTokenLevelReward class."""

    @pytest.fixture
    def mock_model_response(self):
        """Create mock model response."""
        return MockModelResponse(
            output_tokens=[100, 200, 300, 400, 500],
            input_len=10,
            output_len=5,
            output_logprobs=[-0.5, -0.3, -1.2, -0.8, -0.1],
        )

    @pytest.fixture
    def mock_completion(self):
        """Create mock completion."""
        return MockCompletion(id="test-completion-id")

    @pytest.fixture
    def interaction(self, mock_model_response, mock_completion):
        """Create interaction for testing."""
        return InteractionWithTokenLevelReward(
            model_response=mock_model_response,
            messages=[{"role": "user", "content": "Hello"}],
            completion=mock_completion,
            reward=0.0,
        )

    def test_init(self, interaction):
        """Test initialization."""
        assert interaction.token_rewards is None
        assert interaction.token_reward_mask is None

    def test_set_token_rewards_valid(self, interaction):
        """Test setting valid token rewards."""
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
        interaction.set_token_rewards(rewards)
        assert interaction.token_rewards == rewards

    def test_set_token_rewards_invalid_length(self, interaction):
        """Test setting token rewards with wrong length raises error."""
        with pytest.raises(ValueError, match="token_rewards length"):
            interaction.set_token_rewards([0.1, 0.2])  # Only 2 rewards for 5 tokens

    def test_set_sparse_token_rewards(self, interaction):
        """Test setting sparse token rewards."""
        token_indices = [1, 3]
        rewards = [0.5, 0.7]
        interaction.set_sparse_token_rewards(token_indices, rewards, default_reward=0.0)

        assert interaction.token_rewards == [0.0, 0.5, 0.0, 0.7, 0.0]
        assert interaction.token_reward_mask == [0, 1, 0, 1, 0]

    def test_set_sparse_token_rewards_out_of_range(self, interaction):
        """Test setting sparse token rewards with out of range index raises error."""
        with pytest.raises(ValueError, match="out of range"):
            interaction.set_sparse_token_rewards([10], [0.5])

    def test_to_tensor_dict_without_token_rewards(self, interaction):
        """Test to_tensor_dict without token-level rewards."""
        tensor_dict = interaction.to_tensor_dict()

        assert "input_ids" in tensor_dict
        assert "loss_mask" in tensor_dict
        assert "rewards" in tensor_dict

    def test_to_tensor_dict_with_token_rewards(self, interaction):
        """Test to_tensor_dict with token-level rewards."""
        interaction.set_token_rewards([0.1, 0.2, 0.3, 0.4, 0.5])
        tensor_dict = interaction.to_tensor_dict()

        assert "rewards" in tensor_dict
        assert "token_reward_mask" in tensor_dict

        # Check shapes
        seq_len = tensor_dict["input_ids"].shape[1]
        assert tensor_dict["rewards"].shape == (1, seq_len)
        assert tensor_dict["token_reward_mask"].shape == (1, seq_len)

    def test_get_reward_stats_without_token_rewards(self, interaction):
        """Test get_reward_stats without token rewards."""
        stats = interaction.get_reward_stats()

        assert stats["reward_type"] == "scalar"
        assert "value" in stats

    def test_get_reward_stats_with_token_rewards(self, interaction):
        """Test get_reward_stats with token rewards."""
        interaction.set_token_rewards([0.1, 0.2, 0.3, 0.4, 0.5])
        stats = interaction.get_reward_stats()

        assert stats["reward_type"] == "token_level"
        assert "mean" in stats
        assert "max" in stats
        assert "min" in stats
        assert "sparsity" in stats

    def test_get_output_logprobs(self, interaction):
        """Test get_output_logprobs method."""
        logprobs = interaction.get_output_logprobs()
        assert logprobs == [-0.5, -0.3, -1.2, -0.8, -0.1]

    def test_compute_entropy_from_logprobs(self, interaction):
        """Test compute_entropy_from_logprobs method."""
        entropy = interaction.compute_entropy_from_logprobs()
        assert len(entropy) == 5
        # Entropy is approximated as -logprob
        assert entropy[0] == 0.5
        assert entropy[1] == 0.3

    def test_get_token_level_logp_stats(self, interaction):
        """Test get_token_level_logp_stats method."""
        stats = interaction.get_token_level_logp_stats()

        assert "logprob_mean" in stats
        assert "logprob_min" in stats
        assert "logprob_max" in stats
        assert "approx_entropy_mean" in stats

    def test_save_logp_and_entropy(self, interaction):
        """Test save_logp_and_entropy method."""
        result = interaction.save_logp_and_entropy()

        assert "logprobs" in result
        assert "entropy" in result
        assert "stats" in result
        assert result["logprobs"] == [-0.5, -0.3, -1.2, -0.8, -0.1]

    def test_save_logp_and_entropy_no_logprobs(self):
        """Test save_logp_and_entropy when model_response has no logprobs."""
        resp = MockModelResponse(
            output_tokens=[100, 200],
            input_len=10,
            output_len=2,
            output_logprobs=None,
        )
        interaction = InteractionWithTokenLevelReward(
            model_response=resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=MockCompletion(),
        )
        result = interaction.save_logp_and_entropy()
        assert result["logprobs"] is None
        assert "error" in result
