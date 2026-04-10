"""Tests for cache module."""

import pytest
import threading
import time
from unittest.mock import Mock, MagicMock

from customized_areal.on_policy_distill.proxy.cache import (
    PositionRewardInfo,
    InteractionCache,
)
from customized_areal.on_policy_distill.proxy.types import (
    InteractionWithTokenLevelReward,
)


class TestPositionRewardInfo:
    """Test PositionRewardInfo dataclass."""

    def test_init_valid(self):
        """Test valid initialization."""
        pri = PositionRewardInfo(
            position=0,
            candidates=["a", "b", "c"],
            candidate_token_ids=[1, 2, 3],
            logprobs=[-1.0, -0.5, -2.0],
            rewards=[0.1, 0.5, -0.2],
            chosen_index=1,
        )
        assert pri.position == 0
        assert len(pri.candidates) == 3

    def test_init_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="candidates"):
            PositionRewardInfo(
                position=0,
                candidates=["a", "b"],
                logprobs=[-1.0, -0.5, -2.0],  # 3 vs 2
                rewards=[0.1, 0.5],
            )

    def test_chosen_token(self):
        """Test chosen_token property."""
        pri = PositionRewardInfo(
            position=0,
            candidates=["a", "b", "c"],
            rewards=[0.1, 0.5, -0.2],
            chosen_index=1,
        )
        assert pri.chosen_token == "b"

    def test_chosen_reward(self):
        """Test chosen_reward property."""
        pri = PositionRewardInfo(
            position=0,
            candidates=["a", "b", "c"],
            rewards=[0.1, 0.5, -0.2],
            chosen_index=1,
        )
        assert pri.chosen_reward == 0.5

    def test_chosen_logprob(self):
        """Test chosen_logprob property."""
        pri = PositionRewardInfo(
            position=0,
            candidates=["a", "b", "c"],
            logprobs=[-1.0, -0.5, -2.0],
            rewards=[0.1, 0.5, -0.2],
            chosen_index=1,
        )
        assert pri.chosen_logprob == -0.5

    def test_get_reward_for_token(self):
        """Test get_reward_for_token method."""
        pri = PositionRewardInfo(
            position=0,
            candidates=["a", "b", "c"],
            rewards=[0.1, 0.5, -0.2],
            chosen_index=1,
        )
        assert pri.get_reward_for_token("b") == 0.5
        assert pri.get_reward_for_token("z") is None


class TestInteractionCache:
    """Test InteractionCache class."""

    @pytest.fixture
    def mock_interaction(self):
        """Create mock interaction."""
        mock_resp = Mock()
        mock_resp.output_tokens = [100, 200, 300]
        mock_resp.input_len = 5
        mock_resp.output_len = 3

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(id="test-completion"),
            reward=0.0,
        )
        return interaction

    def test_init(self):
        """Test initialization."""
        cache = InteractionCache()
        assert len(cache) == 0
        assert cache.total_reward == 0.0

    def test_setitem_and_getitem(self, mock_interaction):
        """Test setting and getting items."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        assert "comp-1" in cache
        assert cache["comp-1"] == mock_interaction

    def test_last_interaction_id(self, mock_interaction):
        """Test last_interaction_id property."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        assert cache.last_interaction_id == "comp-1"

    def test_last_interaction_id_empty(self):
        """Test last_interaction_id with empty cache raises error."""
        cache = InteractionCache()
        with pytest.raises(KeyError):
            _ = cache.last_interaction_id

    def test_set_rewards_valid(self, mock_interaction):
        """Test setting valid rewards."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        cache.set_rewards("comp-1", [0.1, 0.2, 0.3])

        token_rewards = cache.get_token_rewards("comp-1")
        assert token_rewards == [0.1, 0.2, 0.3]

    def test_set_rewards_invalid_completion_id(self):
        """Test setting rewards for non-existent completion raises error."""
        cache = InteractionCache()

        with pytest.raises(KeyError, match="not found"):
            cache.set_rewards("non-existent", [0.1, 0.2])

    def test_set_rewards_empty(self, mock_interaction):
        """Test setting empty rewards raises error."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        with pytest.raises(ValueError, match="cannot be empty"):
            cache.set_rewards("comp-1", [])

    def test_set_last_rewards(self, mock_interaction):
        """Test setting rewards for last interaction."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        cache.set_last_rewards([0.1, 0.2, 0.3])

        token_rewards = cache.get_token_rewards("comp-1")
        assert token_rewards == [0.1, 0.2, 0.3]

    def test_set_reward_scalar(self, mock_interaction):
        """Test setting scalar reward."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        cache.set_reward("comp-1", 5.0)
        assert cache.total_reward == 5.0

    def test_set_last_reward(self, mock_interaction):
        """Test setting scalar reward for last interaction."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        cache.set_last_reward(5.0)
        assert mock_interaction.reward == 5.0

    def test_get_reward_stats_scalar(self, mock_interaction):
        """Test getting reward stats for scalar reward."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction
        cache.set_reward("comp-1", 5.0)

        stats = cache.get_reward_stats("comp-1")
        assert stats["reward_type"] == "scalar"
        assert stats["scalar_reward"] == 5.0

    def test_get_reward_stats_token_level(self, mock_interaction):
        """Test getting reward stats for token-level rewards."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction
        cache.set_rewards("comp-1", [0.1, 0.2, 0.3])

        stats = cache.get_reward_stats("comp-1")
        assert stats["reward_type"] == "token_level"
        assert stats["num_tokens"] == 3
        assert stats["sum"] == 0.6

    def test_get_reward_stats_not_found(self):
        """Test getting reward stats for non-existent completion raises error."""
        cache = InteractionCache()

        with pytest.raises(KeyError, match="not found"):
            cache.get_reward_stats("non-existent")

    def test_set_position_rewards(self, mock_interaction):
        """Test setting position rewards."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

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
                rewards=[0.3, 0.7],
                chosen_index=0,
            ),
        ]

        cache.set_position_rewards("comp-1", position_rewards)

        # Verify token rewards were extracted from chosen tokens
        token_rewards = cache.get_token_rewards("comp-1")
        assert token_rewards == [0.5, 0.2, 0.3]  # Chosen rewards from each position

        # Verify position rewards stored
        stored_position_rewards = cache.get_position_rewards("comp-1")
        assert stored_position_rewards is not None
        assert len(stored_position_rewards) == 3

    def test_get_position_rewards_not_found(self):
        """Test getting position rewards for non-existent completion."""
        cache = InteractionCache()

        result = cache.get_position_rewards("non-existent")
        assert result is None

    def test_get_entropies_not_found(self):
        """Test getting entropies when not computed."""
        cache = InteractionCache()
        mock_resp = Mock()
        mock_resp.output_tokens = [100, 200]
        mock_resp.input_len = 5
        mock_resp.output_len = 2

        interaction = InteractionWithTokenLevelReward(
            model_response=mock_resp,
            messages=[{"role": "user", "content": "Hello"}],
            completion=Mock(),
            reward=0.0,
        )

        cache["comp-1"] = interaction
        entropies = cache.get_entropies("comp-1")
        assert entropies is None

    def test_compute_and_store_entropy(self, mock_interaction):
        """Test computing and storing entropy."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        # Set position rewards with logprobs for entropy computation
        position_rewards = [
            PositionRewardInfo(
                position=0,
                candidates=["a", "b"],
                logprobs=[-1.0, -0.5],
                rewards=[0.1, 0.5],
                chosen_index=1,
            ),
            PositionRewardInfo(
                position=1,
                candidates=["c", "d"],
                logprobs=[-0.8, -0.3],
                rewards=[0.2, 0.6],
                chosen_index=0,
            ),
            PositionRewardInfo(
                position=2,
                candidates=["e", "f", "g"],
                logprobs=[-1.2, -0.6, -0.4],
                rewards=[0.3, 0.7, 0.9],
                chosen_index=0,
            ),
        ]

        cache.set_position_rewards("comp-1", position_rewards)

        # Now compute entropy
        entropies = cache.compute_and_store_entropy("comp-1")

        # Verify entropies were computed
        assert len(entropies) == 3
        assert all(isinstance(e, float) for e in entropies)
        assert all(e >= 0 for e in entropies)  # Entropy should be non-negative

        # Verify stored in interaction
        stored_entropies = cache.get_entropies("comp-1")
        assert stored_entropies == entropies

    def test_compute_entropy_no_position_rewards(self, mock_interaction):
        """Test computing entropy without position rewards raises error."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        with pytest.raises(KeyError, match="not found"):
            cache.compute_and_store_entropy("comp-1")

    def test_compute_entropy_no_logprobs(self, mock_interaction):
        """Test computing entropy without logprobs raises error."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        position_rewards = [
            PositionRewardInfo(
                position=0,
                candidates=["a", "b"],
                logprobs=None,  # No logprobs
                rewards=[0.1, 0.5],
                chosen_index=1,
            ),
        ]

        cache.set_position_rewards("comp-1", position_rewards)

        with pytest.raises(ValueError, match="no logprobs"):
            cache.compute_and_store_entropy("comp-1")

    def test_thread_safety(self, mock_interaction):
        """Test thread safety of cache operations."""
        cache = InteractionCache()
        cache["comp-1"] = mock_interaction

        errors = []

        def set_rewards_thread():
            try:
                for i in range(10):
                    cache.set_rewards("comp-1", [0.1 * i, 0.2 * i, 0.3 * i])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=set_rewards_thread) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
