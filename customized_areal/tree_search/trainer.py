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

import torch

from areal import PPOTrainer
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import RolloutCacheConfig, TreeBackupConfig, TreeBackupMode
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


def _merge_cached_and_new(
    cached_trajs: list[dict[str, Any]],
    new_trajs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge cached and newly-generated trajectory dicts.

    Both are lists of trajectory dicts with shape [1, seq_len].
    Returns a list with a single concatenated dict of shape [total, max_seqlen].

    Note: cached_trajs come from tree store with shape [1, seq_len] each.
    new_trajs come from GroupedRolloutWorkflow as a single dict with shape
    [group_size, max_seqlen] or as individual [1, seq_len] dicts.
    """
    from areal.utils.data import concat_padded_tensors

    all_trajs = list(cached_trajs)

    # new_trajs might be a single grouped dict or individual dicts
    if len(new_trajs) == 1:
        new_dict = new_trajs[0]
        batch_size = new_dict["input_ids"].shape[0]
        if batch_size > 1:
            # Already grouped — split into individual, then concat all
            for i in range(batch_size):
                single = {}
                for k, v in new_dict.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1:
                        single[k] = v[i:i+1]
                    else:
                        single[k] = v
                all_trajs.append(single)
        else:
            all_trajs.append(new_dict)
    else:
        all_trajs.extend(new_trajs)

    if not all_trajs:
        return []

    # Concatenate all into one grouped dict
    merged = concat_padded_tensors(all_trajs)
    return [merged]


class _CacheAwareBatchBuilder:
    """Splits prompts into cached/partially-cached/not-cached groups."""

    def __init__(self, tree_store: MCTSTreeStore, n_samples: int):
        self.tree_store = tree_store
        self.n_samples = n_samples

    def split_prompts(
        self, prompts: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split prompts into cached and needs-generation groups.

        Returns:
            cached: list of dicts with keys: prompt, cached_count, need_gen_count
            need_gen: list of dicts with keys: prompt
        """
        cached = []
        need_gen = []

        for prompt in prompts:
            query_id = prompt.get("_mcts_query_id", "")
            untrained_count = self.tree_store.get_untrained_count(query_id) if query_id else 0

            if untrained_count >= self.n_samples:
                cached.append({
                    "prompt": prompt,
                    "cached_count": self.n_samples,
                    "need_gen_count": 0,
                })
            elif untrained_count > 0:
                cached.append({
                    "prompt": prompt,
                    "cached_count": untrained_count,
                    "need_gen_count": self.n_samples - untrained_count,
                })
            else:
                need_gen.append({"prompt": prompt})

        return cached, need_gen

    def load_cached_trajectories(
        self, cached_prompts: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load cached trajectories for each prompt.

        Returns: dict mapping query_id -> list of trajectory dicts
        """
        result = {}
        for item in cached_prompts:
            query_id = item["prompt"].get("_mcts_query_id", "")
            if not query_id:
                continue
            trajs = self.tree_store.load_trajectories(query_id, item["cached_count"])
            result[query_id] = trajs
        return result


class CacheAwarePPOTrainer(PPOTrainer):
    """PPOTrainer with rollout caching and tree backup.

    On each training step:
    1. Check cache for available trajectories per prompt
    2. Load cached trajectories, generate only missing ones
    3. Merge cached + new trajectories
    4. Run tree backup advantages on merged batch
    5. Mark used trajectories as trained
    6. Save tree checkpoint
    """

    def __init__(
        self,
        config: Any,
        cache_config: RolloutCacheConfig | None = None,
        tree_backup_config: TreeBackupConfig | None = None,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
    ):
        self.cache_config = cache_config or RolloutCacheConfig()
        self.tree_backup_config = tree_backup_config or TreeBackupConfig()

        # Initialize base PPOTrainer first
        super().__init__(config, train_dataset, valid_dataset)

        # Set up tree backup and cache after base init
        if self.cache_config.enabled and self.tree_backup_config.mode != TreeBackupMode.OFF:
            turn_splitter = make_turn_splitter(
                self.tokenizer, self.tree_backup_config.assistant_marker
            )
            self.tree_store = MCTSTreeStore(turn_splitter)
            self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
            self.tree_checkpoint_manager = TreeCheckpointManager(
                self.cache_config.cache_dir or self.tree_backup_config.checkpoint_dir
            )

            # Load existing tree checkpoint if available
            if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
                if self.tree_checkpoint_manager.exists():
                    self.tree_store = self.tree_checkpoint_manager.load(turn_splitter)
                    logger.info("Loaded MCTS tree checkpoint with cached rollouts")

            # Reset trained flags for new training run from scratch
            self.tree_store.reset_trained_flags()

            # Set up batch builder
            self._batch_builder = _CacheAwareBatchBuilder(
                self.tree_store, self.cache_config.n_samples
            )

            # Patch PPOActor for tree backup
            patch_ppo_actor_for_tree_backup(self.tree_store, self.tree_advantage_computer)
            logger.info(
                f"Cache-aware training enabled (mode={self.tree_backup_config.mode.value}, "
                f"n_samples={self.cache_config.n_samples})"
            )

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int) -> None:
        """Save recover checkpoint including MCTS tree state."""
        super()._save_recover_checkpoint(epoch, epoch_step, global_step)

        if self.cache_config.enabled and self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.info("Saved MCTS tree checkpoint with rollout cache")

    def _mark_trajectories_trained(self, rollout_batch: list[dict[str, Any]]) -> None:
        """Mark all trajectories in the batch as trained."""
        if not self.cache_config.enabled:
            return
        for traj in rollout_batch:
            query_id = traj.get("_mcts_query_id")
            seq_id = traj.get("_mcts_seq_id")
            if query_id is not None and seq_id is not None:
                self.tree_store.set_trained(query_id, seq_id, True)

    def close(self) -> None:
        if self.cache_config.enabled and self.tree_backup_config.mode != TreeBackupMode.OFF:
            unpatch_ppo_actor()
        super().close()


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
