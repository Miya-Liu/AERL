"""
Extended types for token-level reward support.

This demonstrates how to extend InteractionWithTokenLogpReward to support
token-level rewards instead of just scalar rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Import from AReaL for inheritance
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
)


@dataclass
class InteractionWithTokenLevelReward(InteractionWithTokenLogpReward):
    """
    Extended interaction class that supports token-level rewards.

    Inherits from AReaL's InteractionWithTokenLogpReward and adds token-level
    reward support. The key difference is:

    - Base class: reward is a scalar float applied to all tokens
    - This class: token_rewards is a list of floats, one per output token

    The ``rewards`` key in the tensor dict is always the trajectory-level
    scalar reward (used by tree backup and GAE). Per-token position-level
    rewards are stored in the separate ``token_rewards`` key (used for
    distillation).

    Attributes
    ----------
    token_rewards : list[float] | None
        Per-token reward values. Should have same length as output_tokens.
        If None, falls back to scalar reward behavior.
    token_reward_mask : list[int] | None
        Binary mask indicating which tokens have rewards (1 = has reward, 0 = no reward).
        Useful for sparse token-level rewards.
    """

    # Token-level rewards - one reward per output token
    token_rewards: list[float] | None = None

    # Mask for sparse token-level rewards (1 = token has reward, 0 = no reward)
    token_reward_mask: list[int] | None = None

    def __post_init__(self):
        """Validate token-level reward dimensions."""
        if self.model_response is not None and self.token_rewards is not None:
            expected_len = len(self.model_response.output_tokens)
            if len(self.token_rewards) != expected_len:
                raise ValueError(
                    f"token_rewards length ({len(self.token_rewards)}) must match "
                    f"output_tokens length ({expected_len})"
                )

    def set_token_rewards(self, rewards: list[float]) -> None:
        """
        Set per-token rewards for this interaction.

        Parameters
        ----------
        rewards : list[float]
            List of reward values, one per output token.
        """
        if self.model_response is None:
            raise ValueError("Cannot set token rewards without model_response")

        expected_len = len(self.model_response.output_tokens)
        if len(rewards) != expected_len:
            raise ValueError(
                f"token_rewards length ({len(rewards)}) must match "
                f"output_tokens length ({expected_len})"
            )
        self.token_rewards = rewards
        # Invalidate cache since rewards changed
        self._cache = None

    def set_sparse_token_rewards(
        self,
        token_indices: list[int],
        rewards: list[float],
        default_reward: float = 0.0,
    ) -> None:
        """
        Set rewards for specific tokens only (sparse rewards).

        Parameters
        ----------
        token_indices : list[int]
            Indices of tokens to set rewards for.
        rewards : list[float]
            Reward values for each token index.
        default_reward : float
            Default reward for tokens not in token_indices.
        """
        if self.model_response is None:
            raise ValueError("Cannot set token rewards without model_response")

        output_len = len(self.model_response.output_tokens)
        full_rewards = [default_reward] * output_len
        full_mask = [0] * output_len

        for idx, reward in zip(token_indices, rewards):
            if 0 <= idx < output_len:
                full_rewards[idx] = reward
                full_mask[idx] = 1
            else:
                raise ValueError(f"Token index {idx} out of range [0, {output_len})")

        self.token_rewards = full_rewards
        self.token_reward_mask = full_mask
        self._cache = None

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        """
        Convert to tensor dictionary with token-level reward support.

        Overrides the parent method to support token-level rewards.
        When token_rewards is set, token-level rewards are stored in a
        separate ``token_rewards`` key. The ``rewards`` key is always the
        trajectory-level scalar reward so that tree backup advantage
        computation uses only trajectory-level rewards.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - input_ids: Token IDs (input + output)
            - loss_mask: 0 for input tokens, 1 for output tokens
            - logprobs: Log probabilities (0 for input, actual for output)
            - versions: Weight versions (-1 for input, actual for output)
            - attention_mask: All ones
            - rewards: Trajectory-level scalar reward (for tree backup / GAE)
            - token_rewards: Per-token position-level rewards (for distillation)
            - token_reward_mask: Mask indicating which positions have token-level rewards
        """
        # Use parent implementation for base tensors
        base_result = super().to_tensor_dict()

        # If no token-level rewards, return base result as-is
        if self.token_rewards is None:
            return base_result

        # Get sequence length from input_ids
        seq_len = base_result["input_ids"].shape[1]
        resp = self.model_response
        if resp is not None:
            input_len = resp.input_len
            output_len = resp.output_len
        else:
            # Reconstruct input/output lengths from loss_mask when _cache is pre-populated
            # and model_response is not available (e.g., deserialized from server)
            loss_mask = base_result["loss_mask"].squeeze(0).tolist()
            input_len = loss_mask.index(1) if 1 in loss_mask else seq_len
            output_len = seq_len - input_len

        # Pad or truncate token_rewards to match output_len
        token_rewards = self.token_rewards
        if len(token_rewards) < output_len:
            # Pad with last reward value
            token_rewards = token_rewards + [
                token_rewards[-1] if token_rewards else 0.0
            ] * (output_len - len(token_rewards))
        elif len(token_rewards) > output_len:
            # Truncate
            token_rewards = token_rewards[:output_len]

        # Full sequence: 0 for input, token_rewards for output
        full_token_rewards = [0.0] * input_len + token_rewards

        # Build token reward mask
        if self.token_reward_mask is not None:
            token_mask = self.token_reward_mask
            if len(token_mask) < output_len:
                token_mask = token_mask + [0] * (output_len - len(token_mask))
            elif len(token_mask) > output_len:
                token_mask = token_mask[:output_len]
            full_mask = [0] * input_len + token_mask
        else:
            # All output tokens have rewards
            full_mask = [0] * input_len + [1] * output_len

        # Store token-level rewards separately — do NOT overwrite the
        # trajectory-level scalar ``rewards`` which is used by tree backup
        # advantage computation and GAE.
        base_result["token_rewards"] = torch.tensor(
            full_token_rewards, dtype=torch.float32
        ).unsqueeze(0)
        base_result["token_reward_mask"] = torch.tensor(
            full_mask, dtype=torch.int32
        ).unsqueeze(0)

        return base_result

    def get_reward_stats(self) -> dict[str, Any]:
        """
        Get statistics about rewards for this interaction.

        Returns
        -------
        dict
            Statistics including mean, max, min rewards and sparsity.
        """
        if self.token_rewards is not None:
            rewards = self.token_rewards
            return {
                "reward_type": "token_level",
                "mean": sum(rewards) / len(rewards) if rewards else 0.0,
                "max": max(rewards) if rewards else 0.0,
                "min": min(rewards) if rewards else 0.0,
                "sum": sum(rewards),
                "sparsity": (
                    sum(1 for r in rewards if r == 0.0) / len(rewards)
                    if rewards
                    else 0.0
                ),
                "token_count": len(rewards),
            }
        else:
            return {
                "reward_type": "scalar",
                "value": self.reward,
            }

    def get_output_logprobs(self) -> list[float] | None:
        """
        Get output token log probabilities from model_response.

        Returns
        -------
        list[float] | None
            Log probabilities for each output token, or None if model_response
            is not set or has no logprobs.
        """
        if self.model_response is None:
            return None
        return self.model_response.output_logprobs

    def compute_entropy_from_logprobs(self) -> list[float] | None:
        """
        Compute approximate entropy from output logprobs.

        Note: This computes an approximation since we only have the logprob
        of the selected token, not the full distribution. The entropy is
        estimated as -logprob (higher logprob = lower entropy).

        Returns
        -------
        list[float] | None
            Approximate entropy values for each output token, or None if
            model_response is not set.
        """
        logprobs = self.get_output_logprobs()
        if logprobs is None:
            return None
        # Approximate entropy: -logprob (negative logprob)
        # Higher probability (less negative logprob) = lower entropy
        return [-lp for lp in logprobs]

    def get_token_level_logp_stats(self) -> dict[str, Any] | None:
        """
        Get statistics about token-level log probabilities.

        Returns
        -------
        dict[str, Any] | None
            Statistics including mean, min, max logprobs and approximate entropy,
            or None if model_response is not set.
        """
        logprobs = self.get_output_logprobs()
        if logprobs is None or len(logprobs) == 0:
            return None

        # Compute approximate entropy from logprobs
        entropy = self.compute_entropy_from_logprobs()

        return {
            "logprob_mean": sum(logprobs) / len(logprobs),
            "logprob_min": min(logprobs),
            "logprob_max": max(logprobs),
            "logprob_sum": sum(logprobs),
            "approx_entropy_mean": sum(entropy) / len(entropy) if entropy else 0.0,
            "token_count": len(logprobs),
        }

    def save_logp_and_entropy(self) -> dict[str, Any]:
        """
        Save logp and compute entropy metrics from model_response.

        This method extracts logprobs from model_response and computes
        various entropy-related metrics useful for analysis.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - logprobs: Raw log probabilities for each output token
            - entropy: Approximate entropy computed from logprobs
            - stats: Statistical summary of logprobs and entropy

        Example
        -------
        >>> interaction = InteractionWithTokenLevelReward(
        ...     model_response=model_resp,
        ...     messages=messages,
        ...     completion=completion,
        ... )
        >>> saved_data = interaction.save_logp_and_entropy()
        >>> print(saved_data["logprobs"])  # [-0.5, -0.3, -1.2, ...]
        >>> print(saved_data["entropy"])   # [0.5, 0.3, 1.2, ...]
        """
        logprobs = self.get_output_logprobs()
        if logprobs is None:
            return {
                "logprobs": None,
                "entropy": None,
                "stats": None,
                "error": "model_response or output_logprobs not available",
            }

        entropy = self.compute_entropy_from_logprobs()
        stats = self.get_token_level_logp_stats()

        return {
            "logprobs": logprobs,
            "entropy": entropy,
            "stats": stats,
        }


# Type alias for the interactions dict returned by arun_episode
TokenRewardInteractions = dict[str, InteractionWithTokenLevelReward]
