"""Clip-cov PPO loss module.

Implements covariance-aware PPO clipping from PRIME-RL (https://github.com/PRIME-RL/Entropy-Mechanism-of-RL).
"""

from customized_areal.clip_cov.config import ClipCovConfig
from customized_areal.clip_cov.loss import (
    clip_cov_grpo_loss_fn,
    clip_cov_ppo_actor_loss_fn,
)
from customized_areal.clip_cov.patch import (
    patch_ppo_actor_to_use_clip_cov_loss,
    unpatch_ppo_actor_to_use_clip_cov_loss,
)

__all__ = [
    "ClipCovConfig",
    "clip_cov_ppo_actor_loss_fn",
    "clip_cov_grpo_loss_fn",
    "patch_ppo_actor_to_use_clip_cov_loss",
    "unpatch_ppo_actor_to_use_clip_cov_loss",
]
