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

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import (
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search.mcts_tree_store import (
    MCTSTreeStore,
    get_query_id_from_messages,
)
from customized_areal.tree_search.turn_splitter import make_turn_splitter

from areal import PPOTrainer
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging

logger = logging.getLogger("TreeBackupPPOTrainer")


def _mark_batch_trained(
    tree_store: MCTSTreeStore, trajectories: list[dict[str, Any]]
) -> None:
    """Mark all trajectories in a batch as trained after tree backup."""
    for traj in trajectories:
        query_id = traj.get("_mcts_query_id")
        if query_id is None:
            continue
        seq_id = traj.get("_mcts_seq_id")
        if seq_id is not None:
            tree_store.set_trained(query_id, seq_id, True)
        seq_ids = traj.get("_mcts_seq_ids")
        if seq_ids is not None:
            for sid in seq_ids:
                tree_store.set_trained(query_id, sid, True)


def patch_ppo_actor_for_tree_backup(
    tree_store: MCTSTreeStore, tree_advantage_computer: TreeAdvantageComputer
) -> None:
    """Patch PPOActor.compute_advantages to add MCTS tree backup after GAE.

    The patched method:
    1. Calls the original compute_advantages (full GAE pipeline)
    2. Inserts trajectories into the tree with raw rewards
    3. Overwrites advantages/returns with tree Q-values
    4. Marks trajectories as trained

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

        # 3. Mark trajectories as trained so they won't be loaded from cache again
        _mark_batch_trained(tree_store, result)

        # 4. Record training step order for replay (skip during replay to avoid duplicates)
        if not getattr(tree_store, "_replay_mode", False):
            global_step = result[0].get("_global_step") if result else None
            tree_store.record_training_step(global_step, result)

        # advantages/returns already overwritten by compute()
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

    Returns a list of trajectory dicts (not merged into a single dict).
    Cached trajectories have shape [1, seq_len] each. New trajectories
    may be grouped (shape [group_size, seq_len]) or individual.

    We avoid merging everything via concat_padded_tensors because
    non-tensor keys like _mcts_seq_id and _mcts_query_id would be
    lost (concat_padded_tensors keeps only the first dict's value
    for non-tensor, non-list keys). Keeping them as separate items
    in the list preserves per-trajectory metadata.
    """
    all_trajs = list(cached_trajs)

    for traj in new_trajs:
        batch_size = traj["input_ids"].shape[0]
        if batch_size == 1:
            all_trajs.append(traj)
        else:
            # Split grouped trajectory into individual items to
            # preserve per-sample _mcts_seq_ids metadata
            for i in range(batch_size):
                single = {}
                for k, v in traj.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1:
                        single[k] = v[i : i + 1]
                    elif isinstance(v, list) and k == "_mcts_seq_ids":
                        single["_mcts_seq_id"] = v[i]
                        single["_mcts_query_id"] = traj.get("_mcts_query_id")
                    else:
                        single[k] = v
                all_trajs.append(single)

    return all_trajs


class _CacheAwareBatchBuilder:
    """Splits prompts into cached/partially-cached/not-cached groups."""

    def __init__(self, tree_store: MCTSTreeStore, n_samples: int, tokenizer: Any):
        self.tree_store = tree_store
        self.n_samples = n_samples
        self.tokenizer = tokenizer

    def split_prompts(
        self, prompts: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split prompts into cached and needs-generation groups.

        Derives _mcts_query_id from the messages in each prompt using
        the tokenizer, then checks the tree store for cached rollouts.

        Returns:
            cached: list of dicts with keys: prompt, query_id, cached_count, need_gen_count
            need_gen: list of dicts with keys: prompt, query_id
        """
        cached = []
        need_gen = []

        for prompt in prompts:
            # Derive query_id from messages via tokenizer
            messages = prompt.get("messages", [])
            if messages:
                query_id = get_query_id_from_messages(messages, self.tokenizer)
            else:
                query_id = prompt.get("_mcts_query_id", "")

            untrained_count = (
                self.tree_store.get_untrained_count(query_id) if query_id else 0
            )

            if untrained_count >= self.n_samples:
                cached.append(
                    {
                        "prompt": prompt,
                        "query_id": query_id,
                        "cached_count": self.n_samples,
                        "need_gen_count": 0,
                    }
                )
            elif untrained_count > 0:
                cached.append(
                    {
                        "prompt": prompt,
                        "query_id": query_id,
                        "cached_count": untrained_count,
                        "need_gen_count": self.n_samples - untrained_count,
                    }
                )
            else:
                need_gen.append({"prompt": prompt, "query_id": query_id})

        return cached, need_gen

    def load_cached_trajectories(
        self, cached_prompts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Load cached trajectories for prompts with available rollouts.

        Returns: flat list of trajectory dicts (shape [1, seq_len] each)
        """
        all_trajs = []
        for item in cached_prompts:
            query_id = item["query_id"]
            if not query_id or item["cached_count"] == 0:
                continue
            trajs = self.tree_store.load_trajectories(query_id, item["cached_count"])
            all_trajs.extend(trajs)
        return all_trajs


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
        if (
            self.cache_config.enabled
            and self.tree_backup_config.mode != TreeBackupMode.OFF
        ):
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
                self.tree_store, self.cache_config.n_samples, self.tokenizer
            )

            # Patch PPOActor for tree backup
            patch_ppo_actor_for_tree_backup(
                self.tree_store, self.tree_advantage_computer
            )
            if self.cache_config.replay:
                self.tree_store._replay_mode = True
                if not self.tree_store._training_history:
                    self.tree_store.build_training_history()
                if not self.tree_store._training_history:
                    raise ValueError(
                        "Cannot replay: no training history found in tree "
                        "checkpoint. Run a training session first."
                    )
                self._replay_global_step = 0
                logger.info(
                    f"Replay mode enabled: {len(self.tree_store._training_history)} "
                    f"training steps available"
                )
            else:
                logger.info(
                    f"Cache-aware training enabled (mode={self.tree_backup_config.mode.value}, "
                    f"n_samples={self.cache_config.n_samples})"
                )

    def _save_recover_checkpoint(
        self, epoch: int, epoch_step: int, global_step: int
    ) -> None:
        """Save recover checkpoint including MCTS tree state."""
        super()._save_recover_checkpoint(epoch, epoch_step, global_step)

        if (
            self.cache_config.enabled
            and self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING
        ):
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.info("Saved MCTS tree checkpoint with rollout cache")

    def _mark_trajectories_trained(self, rollout_batch: list[dict[str, Any]]) -> None:
        """Mark all trajectories in the batch as trained.

        Handles both single trajectories (with _mcts_seq_id) and grouped
        trajectories (with _mcts_seq_ids list).
        """
        if not self.cache_config.enabled:
            return
        for traj in rollout_batch:
            query_id = traj.get("_mcts_query_id")
            if query_id is None:
                continue
            # Single trajectory
            seq_id = traj.get("_mcts_seq_id")
            if seq_id is not None:
                self.tree_store.set_trained(query_id, seq_id, True)
                continue
            # Grouped trajectory
            seq_ids = traj.get("_mcts_seq_ids")
            if seq_ids is not None:
                for sid in seq_ids:
                    self.tree_store.set_trained(query_id, sid, True)

    def _cache_aware_prepare_batch(
        self,
        dataloader,
        workflow,
        workflow_kwargs=None,
        should_accept_fn=None,
        group_size=1,
        dynamic_bs=False,
    ):
        """Cache-aware replacement for prepare_batch.

        1. Pulls a batch from the dataloader
        2. Derives query IDs from prompt messages
        3. Splits prompts into cached / needs-generation
        4. Loads cached trajectories from tree store
        5. Generates missing trajectories via rollout_batch
        6. Merges and returns combined list
        """
        from areal.utils.data import cycle_dataloader

        # Lazily initialize the dataloader iterator
        if not hasattr(self, "_cache_dataloader_iter"):
            self._cache_dataloader_iter = iter(cycle_dataloader(dataloader))

        # Pull a batch of raw data items from the dataloader
        raw_batch = next(self._cache_dataloader_iter)

        # Split into cached / needs-generation
        cached_items, need_gen_items = self._batch_builder.split_prompts(raw_batch)

        # Load cached trajectories
        cached_trajs = self._batch_builder.load_cached_trajectories(cached_items)

        # Generate missing trajectories
        need_gen_prompts = [item["prompt"] for item in need_gen_items]
        if need_gen_prompts:
            # Use rollout_batch to generate with distributed coordination
            new_trajs = self.actor.rollout_batch(
                need_gen_prompts,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
        else:
            new_trajs = []

        n_cached = len(cached_trajs)
        n_new = sum(t["input_ids"].shape[0] for t in new_trajs) if new_trajs else 0
        logger.info(f"Cache-aware rollout: {n_cached} cached, {n_new} newly generated")

        # Merge cached and new trajectories
        return _merge_cached_and_new(cached_trajs, new_trajs)

    def _load_untrained_from_tree_store(self) -> list[dict[str, Any]]:
        """Load untrained trajectories from all tree store queries."""
        all_trajs: list[dict[str, Any]] = []
        for query_id in list(self.tree_store.trees.keys()):
            count = self.tree_store.get_untrained_count(query_id)
            if count > 0:
                n = min(count, self.cache_config.n_samples)
                trajs = self.tree_store.load_trajectories(query_id, n)
                all_trajs.extend(trajs)
        return all_trajs

    def _generate_from_dataloader(
        self,
        dataloader,
        workflow,
        workflow_kwargs=None,
        group_size=1,
    ) -> list[dict[str, Any]]:
        """Generate new rollouts from dataloader prompts."""
        from areal.utils.data import cycle_dataloader

        if not hasattr(self, "_replay_dataloader_iter"):
            self._replay_dataloader_iter = iter(cycle_dataloader(dataloader))

        raw_batch = next(self._replay_dataloader_iter)
        prompts = [item for item in raw_batch]
        if prompts:
            new_trajs = self.actor.rollout_batch(
                prompts,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
            n_new = sum(t["input_ids"].shape[0] for t in new_trajs) if new_trajs else 0
            logger.info(
                f"Replay fallback: generated {n_new} new trajectories from dataloader"
            )
            return new_trajs
        return []

    def _replay_prepare_batch(
        self,
        dataloader,
        workflow,
        workflow_kwargs=None,
        should_accept_fn=None,
        group_size=1,
        dynamic_bs=False,
    ):
        """Replay mode with 3-level fallback: history -> cached untrained -> fresh generation."""
        global_step = self._replay_global_step

        # Level 1: Replay from training history
        if global_step in self.tree_store._training_history:
            pairs = self.tree_store._training_history[global_step]
            trajs = []
            for query_id, seq_id in pairs:
                traj = self.tree_store.load_trajectory_by_seq_id(query_id, seq_id)
                if traj is not None:
                    trajs.append(traj)
                else:
                    logger.warning(
                        f"Replay: trajectory (query_id={query_id}, seq_id={seq_id}) "
                        f"not found, skipping"
                    )
            if trajs:
                self._replay_global_step += 1
                logger.info(
                    f"Replay step {global_step}: {len(trajs)} trajectories from history"
                )
                return trajs
            logger.warning(
                f"Replay step {global_step}: all trajectories missing, falling back"
            )

        # Level 2: Cached untrained from tree store
        cached_trajs = self._load_untrained_from_tree_store()
        if cached_trajs:
            self._replay_global_step += 1
            logger.info(
                f"Replay step {global_step}: {len(cached_trajs)} cached untrained"
            )
            return cached_trajs

        # Level 3: Fresh generation from dataloader
        self._replay_global_step += 1
        return self._generate_from_dataloader(
            dataloader, workflow, workflow_kwargs, group_size
        )

    def train(
        self,
        workflow=None,
        eval_workflow=None,
        workflow_kwargs=None,
        eval_workflow_kwargs=None,
        dynamic_filter_fn=None,
        total_epochs=None,
    ):
        """Train with cache-aware rollout generation.

        Temporarily monkey-patches ``self.actor.prepare_batch`` with a
        cache-aware version that loads cached trajectories and only
        generates missing ones. After training completes (or on error),
        the original prepare_batch is restored.
        """
        if not self.cache_config.enabled:
            return super().train(
                workflow=workflow,
                eval_workflow=eval_workflow,
                workflow_kwargs=workflow_kwargs,
                eval_workflow_kwargs=eval_workflow_kwargs,
                dynamic_filter_fn=dynamic_filter_fn,
                total_epochs=total_epochs,
            )

        # Monkey-patch prepare_batch with cache-aware or replay version
        original_prepare_batch = self.actor.prepare_batch

        if self.cache_config.replay:
            _prepare_batch_fn = self._replay_prepare_batch
        else:

            def _prepare_batch_fn(
                dataloader,
                workflow,
                workflow_kwargs=None,
                should_accept_fn=None,
                group_size=1,
                dynamic_bs=False,
            ):
                return self._cache_aware_prepare_batch(
                    dataloader=dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    should_accept_fn=should_accept_fn,
                    group_size=group_size,
                    dynamic_bs=dynamic_bs,
                )

        self.actor.prepare_batch = _prepare_batch_fn

        try:
            return super().train(
                workflow=workflow,
                eval_workflow=eval_workflow,
                workflow_kwargs=workflow_kwargs,
                eval_workflow_kwargs=eval_workflow_kwargs,
                dynamic_filter_fn=dynamic_filter_fn,
                total_epochs=total_epochs,
            )
        finally:
            # Always restore original prepare_batch
            self.actor.prepare_batch = original_prepare_batch
            # Clean up the dataloader iterator(s)
            if hasattr(self, "_cache_dataloader_iter"):
                del self._cache_dataloader_iter
            if hasattr(self, "_replay_dataloader_iter"):
                del self._replay_dataloader_iter

    def close(self) -> None:
        if (
            self.cache_config.enabled
            and self.tree_backup_config.mode != TreeBackupMode.OFF
        ):
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
            patch_ppo_actor_for_tree_backup(
                self.tree_store, self.tree_advantage_computer
            )
            logger.info(
                f"MCTS tree backup enabled (mode={self.tree_backup_config.mode.value})"
            )

    def _save_recover_checkpoint(
        self, epoch: int, epoch_step: int, global_step: int
    ) -> None:
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
