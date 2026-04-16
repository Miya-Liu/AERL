# customized_areal/tree_search/trainer.py
"""MCTS Tree Backup PPOTrainer.

Subclass of PPOTrainer that replaces GAE advantage computation with MCTS
tree backup. Patches the outer PPOActor.compute_advantages method so that:
1. The original GAE runs first (KL rewards, scaling, normalization)
2. Trajectories are inserted into the tree with raw rewards
3. Tree Q-values overwrite advantages/returns
4. KL metadata (kl_rewards, tot_rewards) is preserved for logging
"""
from __future__ import annotations

from typing import Any

from areal import PPOTrainer
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.turn_splitter import make_turn_splitter

logger = logging.getLogger("TreeBackupPPOTrainer")


def patch_ppo_actor_for_tree_backup(
    tree_store: MCTSTreeStore, tree_advantage_computer: TreeAdvantageComputer
) -> None:
    """Patch PPOActor.compute_advantages to add MCTS tree backup after GAE.

    The patched method:
    1. Calls the original compute_advantages (full GAE pipeline)
    2. Inserts trajectories into the tree with raw rewards
    3. Overwrites advantages/returns with tree Q-values

    The original method's kl_rewards, tot_rewards, loss_mask, logprobs
    are preserved for logging and downstream use.
    """
    # Preserve the true original if patching twice (don't stack patches)
    if hasattr(PPOActor, "_original_compute_advantages"):
        original_compute_advantages = PPOActor._original_compute_advantages
    else:
        original_compute_advantages = PPOActor.compute_advantages

    def _tree_backup_compute_advantages(
        self, data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # 1. Run original GAE pipeline (KL rewards, scaling, normalization, etc.)
        result = original_compute_advantages(self, data)

        # 2. Insert trajectories into tree with raw rewards, compute tree Q-values
        tree_store.insert_batch(result)
        tree_advantage_computer.compute(result)

        # 3. advantages/returns already overwritten by compute()
        # kl_rewards, tot_rewards, loss_mask, logprobs preserved from GAE
        return result

    PPOActor.compute_advantages = _tree_backup_compute_advantages
    # Store original for restore
    PPOActor._original_compute_advantages = original_compute_advantages


def unpatch_ppo_actor() -> None:
    """Restore the original PPOActor.compute_advantages method."""
    if hasattr(PPOActor, "_original_compute_advantages"):
        PPOActor.compute_advantages = PPOActor._original_compute_advantages
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

            # Patch PPOActor outer method to add tree backup after GAE
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
