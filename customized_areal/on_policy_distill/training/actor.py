"""Custom PPO Actor that uses grpo_distill_loss_fn.

This module provides a custom PPO actor that uses combined
GRPO and position-level GRPO loss function for on-policy distillation.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.api import AllocationMode
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging, stats_tracker
from areal.utils.data import split_padded_tensor_dict_into_mb_list

logger = logging.getLogger("OnPolicyDistill")

_patch_applied = False


def patch_ppo_actor_class_to_use_distill_loss() -> None:
    """Patch PPOActor class to use grpo_distill_loss_fn globally.

    This replaces PPOActor._ppo_update with a version that uses
    grpo_distill_loss_fn instead of standard grpo_loss_fn.

    Only patches once, even if called multiple times.

    Note:
    -----
    For multi-candidate gathering during training, the AReaL engine must be
    modified to pass logits to the loss function. See ENGINE_MODIFICATION.md
    in the training directory.
    """
    global _patch_applied
    if _patch_applied:
        return

    original_ppo_update = PPOActor._ppo_update

    def _ppo_update_with_distill_loss(self, data: dict[str, Any]) -> None:
        """PPO update using grpo_distill_loss_fn."""
        from ..training.loss import grpo_distill_loss_fn

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
                        grpo_distill_loss_fn,
                        config=self.config,
                        current_version=current_version,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)

            # Log critical denominator stats
            loss_mask = data.get("loss_mask")
            if loss_mask is not None:
                if isinstance(loss_mask, torch.Tensor):
                    n_valid = loss_mask.count_nonzero().item()
                else:
                    n_valid = sum(mb["loss_mask"].count_nonzero().item() for mb in mb_inputs.mbs if "loss_mask" in mb)
                stats_tracker.denominator(n_valid_tokens=n_valid)

    PPOActor._ppo_update = _ppo_update_with_distill_loss
    _patch_applied = True
    logger.info("PPOActor class patched to use grpo_distill_loss_fn")
