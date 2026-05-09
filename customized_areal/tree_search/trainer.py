# customized_areal/tree_search/trainer.py
"""MCTS Tree Backup PPOTrainer.

Subclass of PPOTrainer that adds MCTS tree backup to PPO training.

Tree insert and advantage computation happen in _cache_aware_prepare_batch
(where query_id / node_id are still available), before the
concat_padded_tensors pipeline drops non-tensor metadata.

Flow:
1. _cache_aware_prepare_batch: insert trajectories into tree, compute tree
   advantages/returns (TREE mode), mark trained, save checkpoint
2. compute_advantages: GAE runs and overwrites advantages/returns; the patch
   restores tree-computed values (saved pre-GAE in local scope)
3. ppo_update: uses restored tree advantages/returns
"""

from __future__ import annotations

from typing import Any

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import (
    AdvantageMode,
    CacheMode,
    LossMode,
    RolloutCacheConfig,
    TreeBackupConfig,
)
from customized_areal.tree_search.mcts_tree_store import (
    MCTSTreeStore,
    Node,
    _node_to_tensor_dict,
)
from customized_areal.tree_search.patches import TreeSearchPatches

from areal import PPOTrainer
from areal.utils import logging
from areal.utils.environ import is_single_controller

logger = logging.getLogger("TreeBackupPPOTrainer")


def _mark_batch_trained(tree_store: MCTSTreeStore, trajectories: list[Node]) -> None:
    """Mark all trajectories in a batch as trained after tree backup."""
    count = 0
    for traj in trajectories:
        node_id = getattr(traj, "node_id", None)
        if node_id is not None:
            tree_store.set_trained(node_id, True)
            count += 1
    if count:
        logger.debug(f"Marked {count} trajectories as trained")


class _CacheAwareBatchBuilder:
    """Splits prompts into cached and needs-generation groups."""

    def __init__(self, tree_store: MCTSTreeStore, n_samples: int):
        self.tree_store = tree_store
        self.n_samples = n_samples

    def split_prompts(
        self, prompts: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split prompts into cached and needs-generation groups.

        A query is cached only when it has >= n_samples untrained
        trajectories. Otherwise all n_samples are generated fresh
        (partial cache is ignored).

        Returns:
            cached: list of dicts with keys: prompt, query_id, cached_count
            need_gen: list of dicts with keys: prompt, query_id
        """
        cached = []
        need_gen = []

        for prompt in prompts:
            query_id = prompt.get("query_id") or ""

            untrained_count = (
                self.tree_store.get_untrained_count(query_id) if query_id else 0
            )

            logger.debug(
                f"Prompt query_id={query_id}: {untrained_count} untrained "
                f"(need {self.n_samples})"
            )

            if untrained_count >= self.n_samples:
                cached.append(
                    {
                        "prompt": prompt,
                        "query_id": query_id,
                        "cached_count": self.n_samples,
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
            if not query_id:
                continue
            nodes = self.tree_store.load_trajectories(query_id, self.n_samples)
            for node in nodes:
                traj_dict = _node_to_tensor_dict(
                    node, query_id, getattr(node, "node_id", 0)
                )
                all_trajs.append(traj_dict)
        return all_trajs


class CacheAwarePPOTrainer(PPOTrainer):
    """PPOTrainer with rollout caching and tree backup.

    On each training step:
    1. _cache_aware_prepare_batch:
       a. Check cache / generate trajectories
       b. Insert into MCTS tree (while query_id is available)
       c. Compute tree advantages/returns on Node fields (TREE mode)
       d. Mark trajectories as trained
       e. Save tree checkpoint (CROSS_TRAINING mode)
    2. compute_advantages:
       a. Patch saves tree values from traj dicts
       b. GAE runs (overwrites advantages/returns)
       c. Patch restores saved tree values
    3. ppo_update uses restored tree advantages/returns

    Monkey-patches are managed by TreeSearchPatches, which applies them
    at the start of train() and restores them in the finally block.
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

        super().__init__(config, train_dataset, valid_dataset)

        self._patches: TreeSearchPatches | None = None

        if self.cache_config.enabled and self.tree_backup_config.mode != CacheMode.OFF:
            self._init_tree_components()
            self._patches = TreeSearchPatches(
                rollout_engine=self.rollout,
                advantage_mode=self.tree_backup_config.advantage_mode,
                loss_mode=self.tree_backup_config.loss_mode,
                group_size=self.cache_config.n_samples,
            )
            logger.info(
                f"Cache-aware training enabled "
                f"(mode={self.tree_backup_config.mode.value}, "
                f"advantage={self.tree_backup_config.advantage_mode.value}, "
                f"n_samples={self.cache_config.n_samples}, "
                f"loss_mode={self.tree_backup_config.loss_mode.value})"
            )

    def _create_train_engine(self, actor_config, alloc):
        """Override to use MultiCandidateFSDPPPOActor when distill loss is enabled."""
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            if alloc.backend != "fsdp":
                raise ValueError(
                    f"Distillation loss mode requires FSDP backend, "
                    f"got: {alloc.backend}"
                )
            from customized_areal.tree_search.engine import (
                MultiCandidateFSDPPPOActor,
            )

            actor_cls = MultiCandidateFSDPPPOActor
            if is_single_controller():
                actor = actor_cls.as_controller(actor_config, self.scheduler)
            else:
                actor = actor_cls(config=actor_config)
            actor.create_process_group(parallel_strategy=alloc.parallel)
            logger.info(
                f"Created MultiCandidateFSDPPPOActor "
                f"(loss_mode={self.tree_backup_config.loss_mode.value})"
            )
            return actor
        return super()._create_train_engine(actor_config, alloc)

    def _init_tree_components(self) -> None:
        """Create tree store, advantage computer, and checkpoint manager."""
        self.tree_store = MCTSTreeStore()
        self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
        self.tree_checkpoint_manager = TreeCheckpointManager(
            self.cache_config.cache_dir or self.tree_backup_config.checkpoint_dir
        )

        # Load existing tree checkpoint if available (CROSS_TRAINING mode)
        if self.tree_backup_config.mode == CacheMode.CROSS_TRAINING:
            if self.tree_checkpoint_manager.exists():
                self.tree_store = self.tree_checkpoint_manager.load()
                logger.info("Loaded MCTS tree checkpoint with cached rollouts")

        # Reset trained flags for a fresh training run
        self.tree_store.reset_trained_flags()

        self._batch_builder = _CacheAwareBatchBuilder(
            self.tree_store, self.cache_config.n_samples
        )

    def _save_recover_checkpoint(
        self, epoch: int, epoch_step: int, global_step: int
    ) -> None:
        """Save recover checkpoint including MCTS tree state."""
        super()._save_recover_checkpoint(epoch, epoch_step, global_step)

        if (
            self.cache_config.enabled
            and self.tree_backup_config.mode == CacheMode.CROSS_TRAINING
        ):
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.info("Saved MCTS tree checkpoint with rollout cache")

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

        Strategy: load cached trajectories for prompts that have them, and
        generate only for prompts that lack sufficient cache. Both cached and
        newly-generated trajectories are concatenated into a single batch.

        After assembling trajectories, this method also:
        1. Inserts them into the MCTS tree (where query_id / node_id
           are still available, before concat_padded_tensors drops them)
        2. If advantage_mode is TREE, computes tree advantages/returns
           on Node fields (restored post-GAE by patches.py)
        3. Marks trajectories as trained
        4. Saves tree checkpoint (CROSS_TRAINING mode)

        Returns:
            List of trajectory dicts carrying ``query_id`` and ``node_id``
            metadata.

        Raises:
            RuntimeError: If both cache and generation produce no trajectories.
        """
        from areal.utils.data import cycle_dataloader

        # Lazily initialize the dataloader iterator
        if not hasattr(self, "_cache_dataloader_iter"):
            self._cache_dataloader_iter = iter(cycle_dataloader(dataloader))

        # Pull a batch of raw data items from the dataloader
        raw_batch = next(self._cache_dataloader_iter)

        # Split into cached / needs-generation
        cached_items, need_gen_items = self._batch_builder.split_prompts(raw_batch)

        # Load cached trajectories for prompts that have them
        cached_nodes: list = []
        if cached_items:
            cached_nodes = list(
                self._batch_builder.load_cached_trajectories(cached_items)
            )

        # Generate trajectories for prompts that need them
        generated_nodes: list = []
        if need_gen_items:
            n_samples = self.cache_config.n_samples
            gen_prompts = [item["prompt"] for item in need_gen_items]

            logger.info(
                f"Generating trajectories for {len(gen_prompts)} queries "
                f"(group_size={n_samples})"
            )
            new_trajs = self.actor.rollout_batch(
                gen_prompts,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
            if new_trajs:
                generated_nodes = new_trajs

        nodes = cached_nodes + generated_nodes
        logger.info(
            f"Cache-aware rollout: {len(cached_nodes)} cached + "
            f"{len(generated_nodes)} generated = {len(nodes)} total"
        )

        if not nodes:
            raise RuntimeError(
                "No trajectories available for this training step "
                "(both cache and generation returned empty). "
                "Check rollout engine and dataset."
            )

        # --- Tree operations (while query_id / node_id are available) ---

        # Insert trajectories into the MCTS tree
        self.tree_store.insert_batch(nodes)
        logger.debug(f"Inserted {len(nodes)} trajectories into tree")

        # Compute tree advantages (stashed on Node fields, flow through to tensors)
        if self.tree_backup_config.advantage_mode == AdvantageMode.TREE:
            self.tree_advantage_computer.compute(nodes)
            logger.debug(
                f"Computed tree advantages for {len(nodes)} trajectories (mode=TREE)"
            )

        # Mark trajectories as trained so they won't be loaded from cache again
        _mark_batch_trained(self.tree_store, nodes)
        logger.debug(f"Marked {len(nodes)} trajectories as trained")

        # Save tree checkpoint (CROSS_TRAINING mode)
        if self.tree_backup_config.mode == CacheMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.debug("Saved MCTS tree checkpoint after tree operations")

        # --- End tree operations ---

        # Convert Nodes to tensor dicts for the downstream PPO pipeline.
        converted_trajs: list[dict[str, Any]] = []
        for node in nodes:
            query_id = node.query_id
            node_id = node.node_id
            converted_trajs.append(_node_to_tensor_dict(node, query_id, node_id))

        # Inject distillation loss weights into trajectory dicts
        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            for traj in converted_trajs:
                if self.tree_backup_config.loss_mode == LossMode.DISTILL:
                    traj["rl_loss_weight"] = 0.0
                else:
                    traj["rl_loss_weight"] = self.tree_backup_config.rl_loss_weight
                traj["distill_loss_weight"] = (
                    self.tree_backup_config.distill_loss_weight
                )

        return converted_trajs

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

        Monkey-patches ``self.actor.prepare_batch`` with a cache-aware version
        and applies tree search patches via TreeSearchPatches. Both are
        restored in the ``finally`` block, so patches never leak on error.
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

        original_prepare_batch = self.actor.prepare_batch

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

        assert self._patches is not None
        self._patches.apply()
        self.actor.prepare_batch = _prepare_batch_fn

        # Safety: reset stale iterator from a previous crashed train() call
        if hasattr(self, "_cache_dataloader_iter"):
            del self._cache_dataloader_iter

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
            self.actor.prepare_batch = original_prepare_batch
            logger.info("Restored original prepare_batch")
            self._patches.restore()
            if hasattr(self, "_cache_dataloader_iter"):
                del self._cache_dataloader_iter

    def close(self) -> None:
        # Safety net: restore patches if train() was never called
        # or crashed before finally executed.
        if self._patches is not None:
            self._patches.restore()
        super().close()
