"""Reward computation utilities for on-policy distillation.

.. deprecated::
    This module has been moved to :mod:`areal.utils.token_reward_utils`.
    Please update your imports accordingly.

This module now re-exports from the new location for backward compatibility.
"""

import warnings

# Deprecation warning
warnings.warn(
    "customized_areal.on_policy_distill.rewards.utils has been moved to "
    "areal.utils.token_reward_utils. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from areal.utils.token_reward_utils import (
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
