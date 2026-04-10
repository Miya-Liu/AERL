"""
Mock implementations for token-level reward testing.

This module provides mock classes for testing set_rewards with token-wise rewards
without requiring a full AReaL deployment.

Token-wise rewards means each reward in the list corresponds to one output token
in the completion. For a completion with N output tokens, the reward list should
have N values.

Additional support for candidate-wise rewards at each position:
- At each position i, the model considers multiple candidate tokens (a1, a2, a3, ...)
- Each candidate has logp_model stored in PositionRewardInfo.logprobs

Example
-------
>>> # Position 0: candidates a1, a2, a3 with model logprobs
>>> pos0 = PositionRewardInfo(
...     position=0,
...     candidates=["a1", "a2", "a3"],
...     logprobs=[-1.2, -0.8, -2.0],  # logp_model for each candidate
...     rewards=[0.1, 0.5, -0.2],     # advantage or other reward
...     chosen_index=1  # a2 was selected
... )
>>> cache.set_position_rewards("comp-1", [pos0, pos1, pos2])
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from areal.utils import logging
from .types import InteractionWithTokenLevelReward

logger = logging.getLogger("MockTokenRewardCache")


@dataclass
class PositionRewardInfo:
    """
    Reward information for a single generation position.

    Stores candidate tokens, their log probabilities, and computed rewards
    at a specific position. Used for KL-based rewards.

    Attributes
    ----------
    position : int
        The position index in the completion (0-indexed).
    candidates : list[str]
        List of candidate token strings considered at this position.
    candidate_token_ids : list[int]
        List of candidate token IDs (vocab indices) for each candidate.
        Used during training to gather logprobs for multi-candidate loss.
    logprobs : list[float] | None
        Log probabilities for each candidate from the model (logp_model).
        If None, rewards are expected to be pre-computed.
    rewards : list[float]
        Reward for each candidate. Can be:
        - Computed from logprobs (if logprobs provided): advantage or other metric
        - Pre-computed: e.g., logp_model - logp_teacher
    chosen_index : int
        Index of the actually chosen token in candidates list.
    """

    position: int
    candidates: list[str] = field(default_factory=list)
    candidate_token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] | None = None  # logp_model for each candidate
    rewards: list[float] = field(default_factory=list)
    chosen_index: int = 0

    def __post_init__(self):
        n = len(self.candidates)
        if self.logprobs is not None and len(self.logprobs) != n:
            raise ValueError(
                f"candidates ({n}) and logprobs ({len(self.logprobs)}) must have same length"
            )
        if len(self.rewards) != n:
            raise ValueError(
                f"candidates ({n}) and rewards ({len(self.rewards)}) must have same length"
            )
        if self.candidate_token_ids and len(self.candidate_token_ids) != n:
            raise ValueError(
                f"candidates ({n}) and candidate_token_ids ({len(self.candidate_token_ids)}) must have same length"
            )
        if self.candidates and not (0 <= self.chosen_index < n):
            raise ValueError(f"chosen_index {self.chosen_index} out of range [0, {n})")

    @property
    def chosen_token(self) -> str | None:
        """Get the chosen token string."""
        if not self.candidates:
            return None
        return self.candidates[self.chosen_index]

    @property
    def chosen_reward(self) -> float | None:
        """Get the reward for the chosen token."""
        if not self.rewards:
            return None
        return self.rewards[self.chosen_index]

    @property
    def chosen_logprob(self) -> float | None:
        """Get the log probability for the chosen token."""
        if self.logprobs is None or not self.candidates:
            return None
        return self.logprobs[self.chosen_index]

    def get_reward_for_token(self, token: str) -> float | None:
        """Get reward for a specific token if it was a candidate."""
        try:
            idx = self.candidates.index(token)
            return self.rewards[idx]
        except ValueError:
            return None


class InteractionCache(OrderedDict[str, InteractionWithTokenLevelReward]):
    """
    Cache that supports storing token-wise rewards per completion.

    This is a lightweight implementation for testing and development that
    adds support for:
    - Storing token-wise rewards (one reward per output token)
    - Storing candidate-wise rewards at each position (logp diff for all candidates)
    - set_rewards() for per-token reward assignment
    - Parent-child relationship building (like original InteractionCache)

    Example: Candidate-wise rewards at each position
    -------
    >>> cache = InteractionCache()
    >>> cache["comp-1"] = interaction
    >>> # Position 0: candidates a1,a2,a3, chosen a2, rewards = logp - logp_teacher
    >>> pos0 = PositionRewardInfo(
    ...     position=0,
    ...     candidates=["a1", "a2", "a3"],
    ...     rewards=[0.1, 0.5, -0.2],  # logp_model - logp_teacher
    ...     chosen_index=1  # a2 was chosen
    ... )
    >>> cache.set_position_rewards("comp-1", [pos0, pos1, pos2, pos3])
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_reward_discount_called = False
        self._total_reward = 0.0
        self._total_list_reward: list[float] | None = None
        self._lock = threading.Lock()

    def _is_prefix(self, a: list[dict], b: list[dict]) -> bool:
        """True if a is a prefix of b."""
        if len(a) > len(b):
            return False
        return b[: len(a)] == a

    def _is_similar_on_last_message(
        self, a: list[dict], b: list[dict]
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
        """Check if two message lists are similar on the last message.

        Returns True if a is a prefix of b up to the last message,
        and the last messages share some common keys.
        """
        if len(a) > len(b):
            return False, None, None
        last_a_message = a[-1]
        last_b_message = b[len(a) - 1]

        same_keys = set(last_a_message.keys()).intersection(set(last_b_message.keys()))
        for key in same_keys:
            if last_a_message[key] != last_b_message[key]:
                return False, None, None
        diff_a_message = {k: v for k, v in last_a_message.items() if k not in same_keys}
        diff_b_message = {k: v for k, v in last_b_message.items() if k not in same_keys}
        return True, diff_a_message, diff_b_message

    def __setitem__(
        self,
        key: str,
        value: InteractionWithTokenLevelReward,
    ) -> None:
        """
        Add a new interaction to the cache, building parent-child relationships.

        Mimics the behavior of InteractionCache.__setitem__ from
        areal/experimental/openai/cache.py
        """
        if value.messages is None:
            raise ValueError(
                "Interaction messages must be set to find parent relationship."
            )

        # Find parent for the new interaction using longest prefix rule
        # Sort potential parents by message length (longest first) to find best match
        with self._lock:
            interactions = sorted(
                self.values(), key=lambda x: len(x.messages), reverse=True
            )

            for parent in interactions:
                # Skip interactions still being processed (no output yet)
                if parent.output_message_list is None or parent.messages is None:
                    continue
                parent_data = parent.messages + parent.output_message_list
                if self._is_prefix(parent_data, value.messages):
                    value.parent = parent
                    break
                elif self._is_prefix(parent.messages, value.messages):
                    is_similar, diff_a, diff_b = self._is_similar_on_last_message(
                        parent_data, value.messages
                    )
                    if is_similar:
                        logger.warning(
                            "Found a parent interaction with similar last message content, "
                            "but not a strict prefix match. If you wish to use concat mode and build a conversation tree:\n"
                            "1. For completion, append `chat_completion.choices[0].message.model_dump()` to your messages.\n"
                            "2. For response, extend `[o.model_dump() for o in response.output]` to your messages.\n"
                            f"Different keys in parent last message: {diff_a}\n"
                            f"Different keys in child last message: {diff_b}\n"
                        )

            super().__setitem__(key, value)

    @property
    def last_interaction_id(self) -> str:
        """Return the most recent interaction ID."""
        if not self:
            raise KeyError("No interactions in cache")
        return next(reversed(self))

    @property
    def total_reward(self) -> float:
        """Return the total scalar reward across all interactions."""
        return self._total_reward

    @property
    def total_list_reward(self) -> list[float] | None:
        """Return the summed list reward across all interactions."""
        return self._total_list_reward

    def set_rewards(
        self,
        completion_id: str,
        token_rewards: list[float],
    ) -> None:
        """
        Set token-wise rewards for a specific completion by its ID.

        Each reward in the list corresponds to one output token in the completion.
        The length of token_rewards must match the number of output tokens.

        Parameters
        ----------
        completion_id : str
            The ID of the completion/interaction to set rewards for.
        token_rewards : list[float]
            Token-wise rewards, one per output token in the completion.
            For a completion with N output tokens, this should be a list of N floats.

        Raises
        ------
        KeyError
            If completion_id is not found in cache.
        ValueError
            If token_rewards is empty or length doesn't match output tokens.

        Example
        -------
        >>> # Completion "comp-1" has 3 output tokens
        >>> cache.set_rewards("comp-1", [0.5, 0.3, 0.2])  # 3 token rewards
        """
        with self._lock:
            self._set_rewards_internal(completion_id, token_rewards)

    def set_reward(
        self,
        completion_id: str,
        reward: float,
    ) -> None:
        """
        Set scalar reward for a specific completion by its ID.

        This is the standard scalar reward setter, kept for backward
        compatibility. For list rewards, use set_rewards().

        Parameters
        ----------
        completion_id : str
            The ID of the completion/interaction to set reward for.
        reward : float
            Scalar reward value.
        """
        with self._lock:
            if completion_id not in self:
                raise KeyError(f"Completion {completion_id} not found in cache")

            interaction = self[completion_id]

            # Update scalar reward tracking
            old_reward = interaction.reward or 0.0
            self._total_reward -= old_reward
            interaction.reward = float(reward)
            self._total_reward += float(reward)

    def set_last_reward(self, reward: float) -> None:
        """
        Set scalar reward for the most recent completion.

        Parameters
        ----------
        reward : float
            Scalar reward value for the last interaction.
        """
        self.set_reward(self.last_interaction_id, reward)

    def set_last_rewards(self, token_rewards: list[float]) -> None:
        """
        Set token-wise rewards for the most recent completion.

        Parameters
        ----------
        token_rewards : list[float]
            Token-wise rewards, one per output token.
        """
        self.set_rewards(self.last_interaction_id, token_rewards)

    def _set_rewards_internal(
        self,
        completion_id: str,
        token_rewards: list[float],
    ) -> None:
        """Internal version of set_rewards without lock acquisition.

        Assumes caller already holds self._lock.
        """
        if completion_id not in self:
            raise KeyError(f"Completion {completion_id} not found in cache")

        if len(token_rewards) == 0:
            raise ValueError("Token rewards list cannot be empty")

        interaction = self[completion_id]

        # Validate length matches output tokens if available
        if (
            hasattr(interaction, "model_response")
            and interaction.model_response is not None
        ):
            expected_len = len(interaction.model_response.output_tokens)
            if len(token_rewards) != expected_len:
                raise ValueError(
                    f"Token rewards length ({len(token_rewards)}) must match "
                    f"output tokens length ({expected_len})"
                )

        # Compute scalar sum for total tracking
        scalar_reward = sum(token_rewards)

        # Store the token-wise rewards
        interaction.token_rewards_list = token_rewards  # type: ignore

        # Update token_rewards field if using InteractionWithTokenLevelReward
        if hasattr(interaction, "token_rewards"):
            try:
                interaction.set_token_rewards(token_rewards)
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not set token_rewards: {e}")

        # Update total list reward (element-wise sum)
        if self._total_list_reward is None:
            self._total_list_reward = token_rewards.copy()
        else:
            # Pad to same length
            max_len = max(len(self._total_list_reward), len(token_rewards))
            self._total_list_reward += [0.0] * (
                max_len - len(self._total_list_reward)
            )
            rewards_padded = token_rewards + [0.0] * (max_len - len(token_rewards))
            self._total_list_reward = [
                a + b for a, b in zip(self._total_list_reward, rewards_padded)
            ]

        # Update scalar reward tracking (subtract old, add new)
        old_reward = interaction.reward or 0.0
        self._total_reward -= old_reward
        interaction.reward = float(scalar_reward)
        self._total_reward += float(scalar_reward)

    def _set_position_rewards_internal(
        self,
        completion_id: str,
        position_rewards: list[PositionRewardInfo],
    ) -> None:
        """Internal version of set_position_rewards without lock acquisition.

        Assumes caller already holds self._lock.
        """
        if completion_id not in self:
            raise KeyError(f"Completion {completion_id} not found in cache")

        if not position_rewards:
            raise ValueError("position_rewards cannot be empty")

        interaction = self[completion_id]

        # Store the full position-wise reward info
        interaction.position_rewards = position_rewards  # type: ignore

        # Extract chosen token rewards for simple token-wise storage
        chosen_rewards = [
            pr.rewards[pr.chosen_index] if pr.rewards else 0.0
            for pr in position_rewards
        ]

        # Also set as token-wise rewards for compatibility
        # Use internal method to avoid double lock acquisition
        self._set_rewards_internal(completion_id, chosen_rewards)

        logger.info(
            f"Set position-wise rewards for {completion_id}: "
            f"{len(position_rewards)} positions, "
            f"avg {sum(len(pr.candidates) for pr in position_rewards) / len(position_rewards):.1f} "
            f"candidates per position"
        )

    def set_position_rewards(
        self,
        completion_id: str,
        position_rewards: list[PositionRewardInfo],
    ) -> None:
        """
        Set candidate-wise rewards for each position in the completion.

        This is useful for KL-divergence or advantage-based rewards where
        each candidate token at each position has a reward computed from
        log probability differences (e.g., logp_model - logp_teacher).

        Parameters
        ----------
        completion_id : str
            The ID of the completion/interaction to set rewards for.
        position_rewards : list[PositionRewardInfo]
            List of PositionRewardInfo, one per generation position.
            Each contains candidate tokens, their rewards, and which was chosen.

        Example
        -------
        >>> # Completion [a, b, c, d] generated in 4 steps
        >>> position_rewards = [
        ...     PositionRewardInfo(  # Position 0: generating 'a'
        ...         position=0,
        ...         candidates=["a1", "a2", "a3"],
        ...         rewards=[0.1, 0.5, -0.2],  # logp_model - logp_teacher
        ...         chosen_index=1  # "a2" was chosen (actual token "a")
        ...     ),
        ...     PositionRewardInfo(  # Position 1: generating 'b'
        ...         position=1,
        ...         candidates=["b1", "b2"],
        ...         rewards=[0.3, 0.7],
        ...         chosen_index=0  # "b1" was chosen
        ...     ),
        ...     # ... more positions
        ... ]
        >>> cache.set_position_rewards("comp-1", position_rewards)
        """
        with self._lock:
            self._set_position_rewards_internal(completion_id, position_rewards)

    def get_token_rewards(self, completion_id: str) -> list[float] | None:
        """
        Get the token-wise rewards for a completion if set.

        Parameters
        ----------
        completion_id : str
            The completion ID to look up.

        Returns
        -------
        list[float] | None
            The token-wise rewards list if set, otherwise None.
        """
        interaction = self.get(completion_id)
        if interaction is None:
            return None
        return getattr(interaction, "token_rewards_list", None)  # type: ignore

    def get_position_rewards(
        self, completion_id: str
    ) -> list[PositionRewardInfo] | None:
        """
        Get the position-wise candidate rewards for a completion if set.

        Parameters
        ----------
        completion_id : str
            The completion ID to look up.

        Returns
        -------
        list[PositionRewardInfo] | None
            The position-wise rewards if set, otherwise None.
        """
        interaction = self.get(completion_id)
        if interaction is None:
            return None
        return getattr(interaction, "position_rewards", None)  # type: ignore

    def compute_and_store_entropy(self, completion_id: str) -> list[float]:
        """
        Compute entropy for each position and store in the interaction.

        Uses the logprobs stored in PositionRewardInfo to compute entropy
        at each generation position.

        Parameters
        ----------
        completion_id : str
            The ID of the completion to compute entropy for.

        Returns
        -------
        list[float]
            List of entropy values, one per position.

        Example
        -------
        >>> cache.set_position_rewards("comp-1", position_rewards_with_logprobs)
        >>> entropies = cache.compute_and_store_entropy("comp-1")
        >>> # entropies = [0.8, 0.6, 0.9, 0.5]  # for 4 positions
        """
        with self._lock:
            if completion_id not in self:
                raise KeyError(f"Completion {completion_id} not found")

            interaction = self[completion_id]
            position_rewards = getattr(interaction, "position_rewards", None)

            if position_rewards is None:
                raise ValueError(
                    f"No position rewards found for {completion_id}. "
                    f"Call set_position_rewards first with logprobs."
                )

            for pos_idx, pr in enumerate(position_rewards):
                if pr.logprobs is None:
                    raise ValueError(
                        f"Position {pos_idx} in {completion_id} has no logprobs. "
                        f"logprobs must be provided for all positions."
                    )
                if len(pr.logprobs) == 0:
                    raise ValueError(
                        f"Position {pos_idx} in {completion_id} has empty logprobs. "
                        f"logprobs must have at least one value for each position."
                    )
                if len(pr.logprobs) != len(pr.candidates):
                    raise ValueError(
                        f"Position {pos_idx} in {completion_id}: logprobs length "
                        f"({len(pr.logprobs)}) must match candidates length "
                        f"({len(pr.candidates)}). logprobs must contain logp for all "
                        f"candidate tokens."
                    )
                for cand_idx, logp in enumerate(pr.logprobs):
                    if not isinstance(logp, (int, float)):
                        raise ValueError(
                            f"Position {pos_idx}, candidate {cand_idx} in {completion_id}: "
                            f"logp {logp} is invalid type {type(logp)}. "
                            f"logprobs must contain valid logp values for all candidate tokens."
                        )
                    if not math.isfinite(logp):
                        raise ValueError(
                            f"Position {pos_idx}, candidate {cand_idx} in {completion_id}: "
                            f"logp {logp} is not finite. "
                            f"logprobs must contain valid logp values for all candidate tokens."
                        )

            # Compute entropy for each position
            entropies = []
            for pr in position_rewards:
                logprobs = pr.logprobs
                if logprobs is None or len(logprobs) == 0:
                    entropy = 0.0
                else:
                    max_logp = max(logprobs)
                    exp_shifted = [math.exp(lp - max_logp) for lp in logprobs]
                    sum_exp = sum(exp_shifted)
                    probs = [e / sum_exp for e in exp_shifted]
                    entropy = 0.0
                    for p, lp in zip(probs, logprobs):
                        log_p = lp - max_logp - math.log(sum_exp)
                        entropy -= p * log_p
                entropies.append(entropy)

            # Store in the interaction
            interaction.position_entropies = entropies  # type: ignore

            # Also compute average entropy as a scalar metric
            avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
            interaction.avg_entropy = avg_entropy  # type: ignore

            logger.info(
                f"Computed entropy for {completion_id}: "
                f"mean={avg_entropy:.4f}, min={min(entropies):.4f}, max={max(entropies):.4f}"
            )

            return entropies

    def get_entropies(self, completion_id: str) -> list[float] | None:
        """
        Get computed entropy values for a completion.

        Parameters
        ----------
        completion_id : str
            The completion ID to look up.

        Returns
        -------
        list[float] | None
            List of entropy values per position, or None if not computed.
        """
        interaction = self.get(completion_id)
        if interaction is None:
            return None
        return getattr(interaction, "position_entropies", None)  # type: ignore

    def get_reward_stats(self, completion_id: str) -> dict[str, Any]:
        """
        Get reward statistics for a completion.

        Parameters
        ----------
        completion_id : str
            The completion ID to get stats for.

        Returns
        -------
        dict
            Statistics including scalar reward, token-wise rewards if present,
            position-wise rewards if present, and derived metrics.
        """
        interaction = self.get(completion_id)
        if interaction is None:
            raise KeyError(f"Completion {completion_id} not found")

        stats = {
            "completion_id": completion_id,
            "scalar_reward": interaction.reward,
        }

        # Add token-wise reward stats if present
        token_rewards = getattr(interaction, "token_rewards_list", None)
        if token_rewards:
            stats["token_rewards"] = token_rewards
            stats["num_tokens"] = len(token_rewards)
            stats["sum"] = sum(token_rewards)
            stats["mean"] = sum(token_rewards) / len(token_rewards)
            stats["max"] = max(token_rewards)
            stats["min"] = min(token_rewards)

        # Add position-wise reward stats if present
        position_rewards = getattr(interaction, "position_rewards", None)
        if position_rewards:
            stats["num_positions"] = len(position_rewards)
            total_candidates = sum(len(pr.candidates) for pr in position_rewards)
            stats["total_candidates"] = total_candidates
            stats["avg_candidates_per_position"] = total_candidates / len(
                position_rewards
            )

            # Chosen token reward distribution
            chosen_rewards = [
                pr.chosen_reward
                for pr in position_rewards
                if pr.chosen_reward is not None
            ]
            if chosen_rewards:
                stats["chosen_rewards"] = chosen_rewards
                stats["chosen_reward_mean"] = sum(chosen_rewards) / len(chosen_rewards)
                stats["chosen_reward_max"] = max(chosen_rewards)
                stats["chosen_reward_min"] = min(chosen_rewards)

        # Add token_reward field if present (from InteractionWithTokenLevelReward)
        if hasattr(interaction, "token_rewards") and interaction.token_rewards:
            stats["per_token_rewards"] = interaction.token_rewards

        # Add entropy stats if computed
        entropies = getattr(interaction, "position_entropies", None)
        if entropies:
            stats["position_entropies"] = entropies
            stats["avg_entropy"] = getattr(
                interaction, "avg_entropy", sum(entropies) / len(entropies)
            )
            stats["entropy_min"] = min(entropies)
            stats["entropy_max"] = max(entropies)

        return stats

    def apply_reward_discount(
        self, turn_discount: float = 1.0
    ) -> dict[str, InteractionWithTokenLevelReward]:
        """
        Apply backward discounted rewards across cached completions/responses.

        This method iterates over the cached completions/responses in reverse
        creation (insertion) order and applies a geometric discount to propagate
        reward signal backward in time. The most recent completion/response is
        treated as the starting point.

        Formula: reward[i] = reward[i] + reward[i+1] * turn_discount

        Parameters
        ----------
        turn_discount : float, optional
            The per-turn discount factor applied when propagating reward
            backward from a later completion/response to an earlier one,
            by default 1.0.

        Returns
        -------
        dict[str, InteractionWithTokenLevelReward]
            A shallow copy of the cache after rewards have been updated in-place.

        Raises
        ------
        RuntimeError
            If called more than once on the same cache instance.
        """
        with self._lock:
            if self._apply_reward_discount_called:
                raise RuntimeError("apply_reward_discount should only be called once.")
            self._apply_reward_discount_called = True
            reversed_interactions = list(reversed(self.values()))

            if reversed_interactions:
                current_reward = 0.0
                for i, interaction in enumerate(reversed_interactions):
                    if interaction.reward is None:
                        # Warn for any interaction without a reward set
                        if i == 0:
                            logger.warning(
                                "The most recent interaction does not have a reward set. "
                                "All interactions will have None reward."
                            )
                        else:
                            logger.warning(
                                f"Interaction {len(reversed_interactions) - i - 1} "
                                "does not have a reward set. Using 0.0."
                            )
                        interaction.reward = 0.0

                    current_reward = current_reward * turn_discount + interaction.reward
                    interaction.reward = current_reward
                    # Invalidate cached tensor dict so to_tensor_dict will recalculate
                    # with the updated reward value.
                    interaction._cache = None

            return dict(**self)

    def export_interactions(
        self,
        style: str = "individual",
        reward_discount: float | None = None,
    ) -> dict[str, InteractionWithTokenLevelReward]:
        """
        Export cached completions/responses in different formats.

        When ``style='concat'``, this method constructs a conversation tree by
        linking completions/responses whose input message lists form a strict-prefix
        relationship. Returns only leaf-node completions (those without children).

        When ``style='individual'``, all cached completions/responses are returned
        as-is without constructing the tree.

        Parameters
        ----------
        style : str, optional
            The export style, either ``'concat'`` (build tree and return leaves)
            or ``'individual'`` (return all), by default 'individual'.
        reward_discount : float | None, optional
            If provided, apply reward discounting before export using this factor.

        Returns
        -------
        dict[str, InteractionWithTokenLevelReward]
            Mapping from completion/response ID to interaction objects.

        Raises
        ------
        ValueError
            If an unsupported ``style`` is provided.
        """
        if reward_discount is not None:
            self.apply_reward_discount(turn_discount=reward_discount)

        if len(self) == 0:
            return {}

        # Filter out incomplete interactions (standardized to match OpenAI cache logic)
        complete_cache = {}
        for id, interaction in self.items():
            if (
                interaction.interaction_id is None
                or interaction.output_message_list is None
            ):
                logger.warning(
                    f"Skipping incomplete interaction during export: cache_key={id}, "
                    f"messages={interaction.messages[:1] if interaction.messages else []}..."
                )
                continue
            if interaction.interaction_id != id:
                raise ValueError(
                    f"Interaction ID mismatch: {interaction.interaction_id} != {id}"
                )
            complete_cache[id] = interaction

        if len(complete_cache) == 0:
            return {}

        if style == "concat":
            for interaction in complete_cache.values():
                if interaction.chat_template_type != "concat":
                    raise ValueError(
                        "Cannot export interactions in 'concat' style when "
                        "interaction.chat_template_type != 'concat' for any interaction. "
                        "This is because when applying chat template using some "
                        "tokenizers, there might be some tokens added or removed "
                        "(e.g. think tokens), making it impossible to construct the conversation tree. "
                        "Please use 'individual' style instead."
                    )

            # Build children mapping to find leaf nodes
            has_children = set()
            for obj in complete_cache.values():
                if obj.parent is not None:
                    has_children.add(obj.parent.interaction_id)

            # Return only leaf nodes (nodes without children)
            return {
                id: interaction
                for id, interaction in complete_cache.items()
                if id not in has_children
            }
        elif style == "individual":
            return dict(**complete_cache)
        else:
            raise ValueError(f"Invalid export interactions style {style}")

    def export_with_token_rewards(self) -> dict[str, dict[str, Any]]:
        """
        Export all interactions with their token-wise rewards.

        Returns
        -------
        dict
            Mapping from completion_id to reward data including
            scalar and token-wise rewards.
        """
        result = {}
        for completion_id in self.keys():
            result[completion_id] = self.get_reward_stats(completion_id)
        return result


# Re-export PositionRewardInfo for backward compatibility
# (now defined here and imported by client_session.py)
__all__ = [
    "PositionRewardInfo",
    "InteractionCache",
]
