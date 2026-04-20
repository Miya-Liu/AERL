"""
Extended proxy server with token-level reward support via HTTP API.

This module extends the base proxy server from areal to support:
- Token-wise rewards: one reward per output token via HTTP API
- Position-wise rewards: candidate-wise rewards at each position via HTTP API

This eliminates the need for a local cache by allowing the agent to set
token-level rewards directly on the proxy server.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from pydantic import BaseModel

from areal.experimental.openai.proxy.server import (
    EXPORT_TRAJECTORIES_PATHNAME,
    GRANT_CAPACITY_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    SESSION_TIMEOUT_SECONDS,
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    SessionData,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
    deserialize_interactions,
    serialize_interactions,
)

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

# =============================================================================
# Extended Request/Response Models for Token-Level Rewards
# =============================================================================


class PositionRewardInfo(BaseModel):
    """
    Reward information for a single generation position.

    Stores candidate tokens, their log probabilities, and computed rewards
    at a specific position. Used for KL-based rewards.
    """

    position: int
    candidates: list[str] = []
    candidate_token_ids: list[int] = []
    logprobs: list[float] | None = None  # logp_model for each candidate
    rewards: list[float] = []
    chosen_index: int = 0


class SetTokenRewardsRequest(BaseModel):
    """Request to set token-wise rewards for an interaction."""

    interaction_id: str | None = None
    token_rewards: list[float]


class SetPositionRewardsRequest(BaseModel):
    """Request to set position-wise rewards for an interaction."""

    interaction_id: str | None = None
    position_rewards: list[PositionRewardInfo]


class ComputeEntropyRequest(BaseModel):
    """Request to compute entropy for an interaction."""

    interaction_id: str


class ComputeEntropyResponse(BaseModel):
    """Response containing computed entropy values."""

    entropies: list[float]
    avg_entropy: float


# =============================================================================
# Extended Session Data with Token-Level Reward Support
# =============================================================================


class TokenRewardSessionData(SessionData):
    """
    Extended session data that supports token-level rewards.

    Inherits from base SessionData and adds support for:
    - Token-wise rewards storage
    - Position-wise rewards storage
    - Entropy computation
    """

    def __init__(self, session_id: str):
        super().__init__(session_id)
        # Store token-level rewards separately
        self._token_rewards: dict[str, list[float]] = {}
        self._position_rewards: dict[str, list[PositionRewardInfo]] = {}
        self._lock = threading.Lock()

    def set_token_rewards(
        self, interaction_id: str, token_rewards: list[float]
    ) -> None:
        """
        Set token-wise rewards for an interaction.

        Parameters
        ----------
        interaction_id : str
            The interaction/completion ID
        token_rewards : list[float]
            Token-wise rewards, one per output token
        """
        with self._lock:
            self._token_rewards[interaction_id] = token_rewards
            # Also update scalar reward as sum of token rewards
            scalar_reward = sum(token_rewards)
            if interaction_id in self.completions:
                self.completions.set_reward(interaction_id, scalar_reward)

    def set_position_rewards(
        self, interaction_id: str, position_rewards: list[PositionRewardInfo]
    ) -> None:
        """
        Set position-wise rewards for an interaction.

        Parameters
        ----------
        interaction_id : str
            The interaction/completion ID
        position_rewards : list[PositionRewardInfo]
            Position-wise candidate rewards
        """
        with self._lock:
            self._position_rewards[interaction_id] = position_rewards
            # Extract chosen token rewards for token-wise storage
            chosen_rewards = [
                pr.rewards[pr.chosen_index] if pr.rewards else 0.0
                for pr in position_rewards
            ]
            self._token_rewards[interaction_id] = chosen_rewards
            # Update scalar reward
            scalar_reward = sum(chosen_rewards)
            if interaction_id in self.completions:
                self.completions.set_reward(interaction_id, scalar_reward)

    def compute_entropy(self, interaction_id: str) -> tuple[list[float], float]:
        """
        Compute entropy for an interaction with position rewards.

        Parameters
        ----------
        interaction_id : str
            The interaction/completion ID

        Returns
        -------
        tuple[list[float], float]
            List of entropy values per position and average entropy
        """
        import math

        with self._lock:
            if interaction_id not in self._position_rewards:
                raise ValueError(f"No position rewards found for {interaction_id}")

            position_rewards = self._position_rewards[interaction_id]
            entropies = []

            for pr in position_rewards:
                logprobs = pr.logprobs
                if logprobs is None or len(logprobs) == 0:
                    entropy = 0.0
                else:
                    # Compute entropy from logprobs
                    max_logp = max(logprobs)
                    exp_shifted = [math.exp(lp - max_logp) for lp in logprobs]
                    sum_exp = sum(exp_shifted)
                    probs = [e / sum_exp for e in exp_shifted]
                    entropy = 0.0
                    for p, lp in zip(probs, logprobs):
                        log_p = lp - max_logp - math.log(sum_exp)
                        entropy -= p * log_p
                entropies.append(entropy)

            avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
            return entropies, avg_entropy

    def export_interactions(
        self, discount: float, style: str
    ) -> dict[str, InteractionWithTokenLogpReward]:
        """
        Export interactions with token-level rewards applied.

        Overrides base method to apply token-level rewards before export.
        """
        # Apply token-level rewards to interactions before export
        with self._lock:
            for interaction_id, token_rewards in self._token_rewards.items():
                if interaction_id in self.completions:
                    interaction = self.completions[interaction_id]
                    # Set token rewards on the interaction object
                    if hasattr(interaction, "token_rewards"):
                        interaction.token_rewards = token_rewards
                    # Ensure scalar reward is set
                    interaction.reward = sum(token_rewards)

        # Call base export
        return super().export_interactions(discount, style)


# =============================================================================
# Path Constants for Token-Level Reward Endpoints
# =============================================================================

RL_SET_TOKEN_REWARDS_PATHNAME = "rl/set_token_rewards"
RL_SET_POSITION_REWARDS_PATHNAME = "rl/set_position_rewards"
RL_COMPUTE_ENTROPY_PATHNAME = "rl/compute_entropy"


# =============================================================================
# Re-exports from base server
# =============================================================================

__all__ = [
    # Base classes and models
    "SessionData",
    "TokenRewardSessionData",
    "StartSessionRequest",
    "StartSessionResponse",
    "SetRewardRequest",
    "ExportTrajectoriesRequest",
    "ExportTrajectoriesResponse",
    "serialize_interactions",
    "deserialize_interactions",
    "SESSION_TIMEOUT_SECONDS",
    # Token-level reward models
    "PositionRewardInfo",
    "SetTokenRewardsRequest",
    "SetPositionRewardsRequest",
    "ComputeEntropyRequest",
    "ComputeEntropyResponse",
    # Path constants
    "RL_START_SESSION_PATHNAME",
    "RL_END_SESSION_PATHNAME",
    "RL_SET_REWARD_PATHNAME",
    "RL_SET_TOKEN_REWARDS_PATHNAME",
    "RL_SET_POSITION_REWARDS_PATHNAME",
    "RL_COMPUTE_ENTROPY_PATHNAME",
    "EXPORT_TRAJECTORIES_PATHNAME",
    "GRANT_CAPACITY_PATHNAME",
]
