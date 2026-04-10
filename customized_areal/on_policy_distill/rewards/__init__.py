"""Reward computation utilities for on-policy distillation.

This module contains:
- utils.py: Helper functions for computing token-level rewards
"""

from .utils import (
    compute_token_level_rewards,
    compute_sparse_rewards,
    apply_token_reward_mask,
    discount_token_rewards,
    normalize_token_rewards,
    create_reasoning_rewards,
    aggregate_interaction_rewards,
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
