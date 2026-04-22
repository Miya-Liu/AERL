"""Custom PPO Actor that uses grpo_distill_loss_fn.

This module provides a custom PPO actor that uses combined
GRPO and position-level GRPO loss function for on-policy distillation.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from areal.api.cli_args import MicroBatchSpec
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

    # Store original for potential future restoration
    _original_ppo_update = PPOActor._ppo_update  # noqa: F841

    def _ppo_update_with_distill_loss(self, data: dict[str, Any]) -> None:
        """PPO update using grpo_distill_loss_fn."""
        from ..training.loss import grpo_distill_loss_fn

        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        # Extract position_rewards before splitting so we can distribute
        # the correct subset to each minibatch.  position_rewards is a
        # Python list and cannot be split by the generic tensor-based
        # minibatch splitter.
        position_rewards = data.pop("position_rewards", None)

        self.engine.train()

        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        # Distribute position_rewards to minibatches based on sample_index.
        # Each PositionRewardInfo.sample_index indicates which batch item
        # it belongs to.  We use the forward_indices from the minibatch
        # split to determine which samples are in which minibatch.
        if position_rewards is not None:
            _distribute_position_rewards(mb_inputs, position_rewards)

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
                    n_valid = sum(
                        mb["loss_mask"].count_nonzero().item()
                        for mb in mb_inputs.mbs
                        if "loss_mask" in mb
                    )
                stats_tracker.denominator(n_valid_tokens=n_valid)

    PPOActor._ppo_update = _ppo_update_with_distill_loss
    _patch_applied = True
    logger.info("PPOActor class patched to use grpo_distill_loss_fn")


def _distribute_position_rewards(mb_inputs, position_rewards: list) -> None:
    """Distribute position_rewards to minibatches based on sample_index.

    Each PositionRewardInfo has a sample_index indicating which batch item
    it belongs to.  The MicroBatchList.forward_indices maps batch items to
    their position in the reordered minibatch sequence.  We use this to
    determine which position_rewards belong to which minibatch.
    """
    if not position_rewards:
        return

    forward_indices = mb_inputs.forward_indices
    # Build mapping: original batch index -> minibatch index
    batch_size = len(forward_indices)
    mb_assignment: list[int | None] = [None] * batch_size
    offset = 0
    for i, mb in enumerate(mb_inputs.mbs):
        mb_bs = mb["attention_mask"].shape[0]
        for j in range(mb_bs):
            orig_idx = forward_indices[offset + j]
            mb_assignment[orig_idx] = i
        offset += mb_bs

    # Group position_rewards by minibatch
    per_mb_prs: dict[int, list] = {}
    for pr in position_rewards:
        mb_i = mb_assignment[pr.sample_index]
        if mb_i is not None:
            per_mb_prs.setdefault(mb_i, []).append(pr)

    # Attach to minibatches
    for i, mb in enumerate(mb_inputs.mbs):
        if i in per_mb_prs:
            mb["position_rewards"] = per_mb_prs[i]
        else:
            mb["position_rewards"] = []
