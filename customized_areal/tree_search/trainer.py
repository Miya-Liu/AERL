"""MCTS Tree Backup PPOTrainer.

Subclass of PPOTrainer that replaces GAE advantage computation with MCTS
tree backup. Uses the same patching pattern as OnPolicyDistillationTrainer
to avoid modifying the base PPOTrainer.
"""

from __future__ import annotations

from typing import Any

import torch

from areal import PPOTrainer
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.turn_splitter import make_turn_splitter

logger = logging.getLogger("TreeBackupPPOTrainer")


def patch_ppo_actor_for_tree_backup(tree_store: MCTSTreeStore, tree_advantage_computer: TreeAdvantageComputer) -> None:
    """Patch PPOActor._compute_advantages to use MCTS tree backup instead of GAE.

    This replaces the GAE computation with tree Q-values while preserving
    the KL reward computation, reward scaling, and other per-step logic.
    The original method is saved and can be restored.
    """
    original_compute_advantages = PPOActor._compute_advantages

    def _tree_backup_compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length (same as original)
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            from areal.utils.rl_functional import reward_overlong_penalty

            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling (same as original)
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)

        # Apply the mask to log probabilities (same as original)
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards (same as original)
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # === TREE BACKUP replaces GAE ===
        # Insert trajectories into the tree and compute tree-based advantages
        tree_store.insert_batch([data])
        tree_advantage_computer.compute([data])

        # Scale advantages by loss_mask (same shape handling as original)
        advantages = data["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(0)

        # Optionally perform advantage normalization (same as original)
        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)

        # Store data in the dict (same as original)
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = old_logp

        return data

    PPOActor._compute_advantages = _tree_backup_compute_advantages
    # Store original for potential restore
    PPOActor._original_compute_advantages = original_compute_advantages


def unpatch_ppo_actor() -> None:
    """Restore the original PPOActor._compute_advantages method."""
    if hasattr(PPOActor, "_original_compute_advantages"):
        PPOActor._compute_advantages = PPOActor._original_compute_advantages
        del PPOActor._original_compute_advantages


class TreeBackupPPOTrainer(PPOTrainer):
    """PPOTrainer with MCTS tree backup replacing GAE advantage computation.

    When tree_backup_config.mode is OFF, behaves exactly like PPOTrainer.
    When mode is IN_TRAINING or CROSS_TRAINING, inserts rollout trajectories
    into a shared compressed trie, runs MCTS backup to compute Q-values, and
    uses those Q-values as the advantage signal instead of GAE.

    Args:
        config: PPOConfig instance.
        tree_backup_config: TreeBackupConfig instance controlling tree behavior.
        train_dataset: Optional training dataset.
        valid_dataset: Optional validation dataset.
    """

    def __init__(
        self,
        config: Any,
        tree_backup_config: TreeBackupConfig | None = None,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
    ):
        self.tree_backup_config = tree_backup_config or TreeBackupConfig()

        # Initialize base PPOTrainer first (sets self.tokenizer etc.)
        super().__init__(config, train_dataset, valid_dataset)

        # Set up tree backup components after base init
        if self.tree_backup_config.mode != TreeBackupMode.OFF:
            turn_splitter = make_turn_splitter(
                self.tokenizer, self.tree_backup_config.assistant_marker
            )
            self.tree_store = MCTSTreeStore(turn_splitter)
            self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
            self.tree_checkpoint_manager = TreeCheckpointManager(
                self.tree_backup_config.checkpoint_dir
            )

            if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
                if self.tree_checkpoint_manager.exists():
                    self.tree_store = self.tree_checkpoint_manager.load(turn_splitter)
                    logger.info("Loaded MCTS tree checkpoint")

            # Patch PPOActor to use tree backup instead of GAE
            patch_ppo_actor_for_tree_backup(self.tree_store, self.tree_advantage_computer)
            logger.info(
                f"MCTS tree backup enabled (mode={self.tree_backup_config.mode.value})"
            )

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int) -> None:
        """Save recover checkpoint including MCTS tree state."""
        super()._save_recover_checkpoint(epoch, epoch_step, global_step)

        if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.info("Saved MCTS tree checkpoint")

    def close(self) -> None:
        """Clean up: unpatch PPOActor and call base close."""
        if self.tree_backup_config.mode != TreeBackupMode.OFF:
            unpatch_ppo_actor()
        super().close()
