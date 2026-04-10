"""Tests for server module."""

import pytest
import math
from unittest.mock import Mock, patch, MagicMock

# Mock areal imports
with patch.dict(
    "sys.modules",
    {
        "areal": Mock(),
        "areal.utils": Mock(),
        "areal.utils.logging": Mock(),
        "areal.experimental": Mock(),
        "areal.experimental.openai": Mock(),
        "areal.experimental.openai.proxy": Mock(),
        "areal.experimental.openai.proxy.server": Mock(),
    },
):
    from customized_areal.on_policy_distill.proxy.server import (
        PositionRewardInfo,
        SetTokenRewardsRequest,
        SetPositionRewardsRequest,
        ComputeEntropyRequest,
        ComputeEntropyResponse,
        TokenRewardSessionData,
    )


class TestPositionRewardInfo:
    """Test PositionRewardInfo dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        pri = PositionRewardInfo(
            position=0,
            candidates=["a", "b", "c"],
            candidate_token_ids=[1, 2, 3],
            logprobs=[-1.0, -0.5, -2.0],
            rewards=[0.1, 0.5, -0.2],
            chosen_index=1,
        )
        assert pri.position == 0
        assert pri.chosen_index == 1

    def test_init_default_values(self):
        """Test initialization with default values."""
        pri = PositionRewardInfo(position=0)
        assert pri.candidates == []
        assert pri.rewards == []
        assert pri.chosen_index == 0

    def test_mismatched_candidates_logprobs(self):
        """Test that mismatched candidates and logprobs raises ValueError."""
        with pytest.raises(ValueError, match="candidates"):
            PositionRewardInfo(
                position=0,
                candidates=["a", "b"],
                logprobs=[-1.0, -0.5, -2.0],  # 3 vs 2
                rewards=[0.1, 0.5],
            )

    def test_mismatched_candidates_rewards(self):
        """Test that mismatched candidates and rewards raises ValueError."""
        with pytest.raises(ValueError, match="candidates"):
            PositionRewardInfo(
                position=0,
                candidates=["a", "b"],
                rewards=[0.1, 0.5, -0.2],  # 3 vs 2
            )

    def test_invalid_chosen_index(self):
        """Test that invalid chosen_index raises ValueError."""
        with pytest.raises(ValueError, match="chosen_index"):
            PositionRewardInfo(
                position=0,
                candidates=["a", "b"],
                rewards=[0.1, 0.5],
                chosen_index=5,  # Out of range
            )


class TestTokenRewardSessionData:
    """Test TokenRewardSessionData class."""

    @pytest.fixture
    def session_data(self):
        """Create session data for testing."""
        return TokenRewardSessionData("test-session-id")

    def test_init(self, session_data):
        """Test initialization."""
        assert session_data.session_id == "test-session-id"

    def test_set_token_rewards(self, session_data):
        """Test setting token rewards."""
        session_data.set_token_rewards("interaction-1", [0.1, 0.2, 0.3])

        # Check internal state
        assert session_data._token_rewards["interaction-1"] == [0.1, 0.2, 0.3]

    def test_set_position_rewards(self, session_data):
        """Test setting position rewards."""
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
        ]

        session_data.set_position_rewards("interaction-1", position_rewards)

        # Check internal state
        assert "interaction-1" in session_data._position_rewards
        assert session_data._position_rewards["interaction-1"] == position_rewards

        # Check chosen rewards extracted
        assert session_data._token_rewards["interaction-1"] == [0.5, 0.2]

    def test_compute_entropy_no_position_rewards(self, session_data):
        """Test computing entropy without position rewards raises error."""
        with pytest.raises(ValueError, match="No position rewards"):
            session_data.compute_entropy("interaction-1")

    def test_compute_entropy_no_logprobs(self, session_data):
        """Test computing entropy without logprobs."""
        position_rewards = [
            PositionRewardInfo(
                position=0,
                candidates=["a", "b"],
                logprobs=None,  # No logprobs
                rewards=[0.1, 0.5],
                chosen_index=1,
            ),
        ]

        session_data.set_position_rewards("interaction-1", position_rewards)

        entropies, avg_entropy = session_data.compute_entropy("interaction-1")
        assert entropies == [0.0]
        assert avg_entropy == 0.0

    def test_compute_entropy_valid(self, session_data):
        """Test computing entropy with valid logprobs."""
        position_rewards = [
            PositionRewardInfo(
                position=0,
                candidates=["a", "b", "c"],
                logprobs=[-1.0, -0.5, -2.0],  # Different logprobs
                rewards=[0.1, 0.5, -0.2],
                chosen_index=1,
            ),
            PositionRewardInfo(
                position=1,
                candidates=["d", "e"],
                logprobs=[-0.8, -0.3],
                rewards=[0.2, 0.6],
                chosen_index=0,
            ),
        ]

        session_data.set_position_rewards("interaction-1", position_rewards)

        entropies, avg_entropy = session_data.compute_entropy("interaction-1")

        # Check entropy values are computed
        assert len(entropies) == 2
        assert all(isinstance(e, float) for e in entropies)
        assert all(e >= 0 for e in entropies)  # Entropy should be non-negative

        # Check average
        assert avg_entropy == sum(entropies) / len(entropies)


class TestRequestResponseModels:
    """Test Pydantic request/response models."""

    def test_set_token_rewards_request(self):
        """Test SetTokenRewardsRequest."""
        request = SetTokenRewardsRequest(
            interaction_id="test-id",
            token_rewards=[0.1, 0.2, 0.3],
        )
        assert request.interaction_id == "test-id"
        assert request.token_rewards == [0.1, 0.2, 0.3]

    def test_set_token_rewards_request_no_id(self):
        """Test SetTokenRewardsRequest without interaction_id."""
        request = SetTokenRewardsRequest(
            token_rewards=[0.1, 0.2, 0.3],
        )
        assert request.interaction_id is None
        assert request.token_rewards == [0.1, 0.2, 0.3]

    def test_set_position_rewards_request(self):
        """Test SetPositionRewardsRequest."""
        position_rewards = [
            {
                "position": 0,
                "candidates": ["a", "b"],
                "rewards": [0.1, 0.5],
                "chosen_index": 1,
            }
        ]
        request = SetPositionRewardsRequest(
            interaction_id="test-id",
            position_rewards=position_rewards,
        )
        assert request.interaction_id == "test-id"
        assert len(request.position_rewards) == 1

    def test_compute_entropy_request(self):
        """Test ComputeEntropyRequest."""
        request = ComputeEntropyRequest(interaction_id="test-id")
        assert request.interaction_id == "test-id"

    def test_compute_entropy_response(self):
        """Test ComputeEntropyResponse."""
        response = ComputeEntropyResponse(
            entropies=[0.5, 0.6, 0.7],
            avg_entropy=0.6,
        )
        assert response.entropies == [0.5, 0.6, 0.7]
        assert response.avg_entropy == 0.6
