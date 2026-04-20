"""Reward computation utilities for on-policy distillation.

This module contains:
- utils.py: Helper functions for computing token-level rewards
"""

from .utils import (
    aggregate_interaction_rewards,
    apply_token_reward_mask,
    compute_sparse_rewards,
    compute_token_level_rewards,
    create_reasoning_rewards,
    discount_token_rewards,
    normalize_token_rewards,
)

__all__ = [
    "compute_token_level_rewards",
    "compute_sparse_rewards",
    "apply_token_reward_mask",
    "discount_token_rewards",
    "normalize_token_rewards",
    "create_reasoning_rewards",
    "aggregate_interaction_rewards",
]
