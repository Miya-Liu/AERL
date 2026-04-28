"""Patch PPOActor to use clip-cov loss.

This module provides a monkey-patch to replace PPOActor._ppo_update with
a version that uses clip_cov_grpo_loss_fn instead of standard grpo_loss_fn.
"""

from __future__ import annotations

import functools
from typing import Any

from customized_areal.clip_cov.config import ClipCovConfig

from areal.api.cli_args import MicroBatchSpec
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging, stats_tracker
from areal.utils.data import split_padded_tensor_dict_into_mb_list

logger = logging.getLogger("ClipCov")

_patch_applied = False
_original_ppo_update = None
_clip_cov_config: ClipCovConfig | None = None


def patch_ppo_actor_to_use_clip_cov_loss(config: ClipCovConfig) -> None:
    """Patch PPOActor class to use clip_cov_grpo_loss_fn globally.

    This replaces PPOActor._ppo_update with a version that uses
    clip_cov_grpo_loss_fn instead of standard grpo_loss_fn.

    Only patches once, even if called multiple times.

    Args:
        config: ClipCovConfig containing clip_ratio, clip_cov_lb, clip_cov_ub.
    """
    global _patch_applied, _original_ppo_update, _clip_cov_config
    if _patch_applied:
        return

    # Store original for potential future restoration
    _original_ppo_update = PPOActor._ppo_update
    _clip_cov_config = config

    def _ppo_update_with_clip_cov_loss(self, data: dict[str, Any]) -> None:
        """PPO update using clip_cov_grpo_loss_fn."""
        from customized_areal.clip_cov.loss import clip_cov_grpo_loss_fn

        # Remove any reward keys we don't need
        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        self.engine.train()

        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            current_version = self.engine.get_version()

            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        clip_cov_grpo_loss_fn,
                        clip_ratio=_clip_cov_config.clip_ratio,
                        clip_cov_lb=_clip_cov_config.clip_cov_lb,
                        clip_cov_ub=_clip_cov_config.clip_cov_ub,
                        current_version=current_version,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)

    PPOActor._ppo_update = _ppo_update_with_clip_cov_loss
    _patch_applied = True
    logger.info("PPOActor class patched to use clip_cov_grpo_loss_fn")


def unpatch_ppo_actor_to_use_clip_cov_loss() -> None:
    """Restore the original PPOActor._ppo_update."""
    global _patch_applied, _original_ppo_update
    if _patch_applied and _original_ppo_update is not None:
        PPOActor._ppo_update = _original_ppo_update
        _patch_applied = False
        logger.info("PPOActor class unpatched to use original grpo_loss_fn")
