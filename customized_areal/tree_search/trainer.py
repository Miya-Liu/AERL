# customized_areal/tree_search/trainer.py
"""MCTS Tree Backup PPOTrainer.

Subclass of PPOTrainer that adds MCTS tree backup to PPO training.

Tree insert and advantage computation happen in _cache_aware_prepare_batch
(where _mcts_query_id / _mcts_seq_id are still available), before the
concat_padded_tensors pipeline drops non-tensor metadata.

Flow:
1. _cache_aware_prepare_batch: insert trajectories into tree, compute tree
   advantages (TREE mode), stash as _tree_advantages/_tree_returns, mark
   trained, save checkpoint
2. compute_advantages: GAE runs and overwrites advantages/returns; the patch
   restores tree values from _tree_advantages/_tree_returns
3. ppo_update: uses restored tree advantages
"""

from __future__ import annotations

from typing import Any

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import (
    AdvantageMode,
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow

from areal import PPOTrainer
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging

logger = logging.getLogger("TreeBackupPPOTrainer")


def _mark_batch_trained(
    tree_store: MCTSTreeStore, trajectories: list[dict[str, Any]]
) -> None:
    """Mark all trajectories in a batch as trained after tree backup."""
    count = 0
    for traj in trajectories:
        query_id = traj.get("_mcts_query_id")
        if query_id is None:
            continue
        seq_id = traj.get("_mcts_seq_id")
        if seq_id is not None:
            tree_store.set_trained(query_id, seq_id, True)
            count += 1
        seq_ids = traj.get("_mcts_seq_ids")
        if seq_ids is not None:
            for sid in seq_ids:
                tree_store.set_trained(query_id, sid, True)
                count += 1
    if count:
        logger.debug(f"Marked {count} trajectories as trained")


def _patch_wrap_openai_agent_for_query_id(rollout_engine: Any) -> None:
    """Patch the engine's _wrap_openai_agent to return QueryIDProxyWorkflow.

    QueryIDProxyWorkflow subclasses OpenAIProxyWorkflow and overrides
    arun_episode to inject data["query_id"] into the trajectory dict as
    ``_mcts_query_id``. This is needed because the async rollout pipeline
    shuffles results, so we cannot match queries to trajectories by position,
    and concat_padded_tensors drops non-tensor keys.

    Args:
        rollout_engine: The rollout inference engine (e.g. RemoteInfEngine)
            that has the ``_wrap_openai_agent`` method.
    """
    engine = rollout_engine
    if not hasattr(engine, "_wrap_openai_agent"):
        logger.warning(
            "Engine has no _wrap_openai_agent method; "
            "query_id injection will not be available"
        )
        return

    original_wrap = engine._wrap_openai_agent

    def _query_id_wrap(agent: Any, proxy_addr: str):
        from areal.api.cli_args import OpenAIProxyConfig

        openai_cfg = engine.config.openai or OpenAIProxyConfig()
        return QueryIDProxyWorkflow(
            mode=openai_cfg.mode,
            agent=agent,
            proxy_addr=proxy_addr,
            admin_api_key=openai_cfg.admin_api_key,
            discount=openai_cfg.turn_discount,
            export_style=openai_cfg.export_style,
            subproc_max_workers=openai_cfg.subproc_max_workers,
            proxy_gateway_addr=getattr(engine, "_proxy_gateway_addr", None),
        )

    engine._wrap_openai_agent = _query_id_wrap
    engine._original_wrap_openai_agent = original_wrap
    logger.info("Patched _wrap_openai_agent to use QueryIDProxyWorkflow")


def _unpatch_wrap_openai_agent(rollout_engine: Any) -> None:
    """Restore the original _wrap_openai_agent method."""
    engine = rollout_engine
    if hasattr(engine, "_original_wrap_openai_agent"):
        engine._wrap_openai_agent = engine._original_wrap_openai_agent
        del engine._original_wrap_openai_agent
        logger.info("Restored original _wrap_openai_agent")


def patch_ppo_actor_for_tree_backup(
    advantage_mode: AdvantageMode = AdvantageMode.TREE,
) -> None:
    """Patch PPOActor.compute_advantages to restore tree advantages after GAE.

    Tree insert and advantage computation already happened in
    _cache_aware_prepare_batch (where _mcts_query_id / _mcts_seq_id are
    available). Tree advantages are stashed as ``_tree_advantages`` /
    ``_tree_returns`` on each trajectory dict so they survive
    concat_padded_tensors → _compute_advantages → split_batch.

    The patched method:
    1. Calls the original compute_advantages (full GAE pipeline), which
       overwrites advantages/returns
    2. If advantage_mode is TREE, restores advantages/returns from
       _tree_advantages/_tree_returns (computed earlier in prepare_batch)
    3. Removes the temporary _tree_advantages/_tree_returns keys

    When advantage_mode is GAE, the original GAE advantages/returns are
    preserved unchanged (no _tree_advantages keys are present).
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
        logger.debug(f"GAE completed for {len(result)} trajectories")

        # 2. Restore tree advantages if present (TREE mode)
        if advantage_mode == AdvantageMode.TREE:
            restored = 0
            for traj in result:
                tree_adv = traj.pop("_tree_advantages", None)
                tree_ret = traj.pop("_tree_returns", None)
                if tree_adv is not None:
                    traj["advantages"] = tree_adv
                    traj["returns"] = tree_ret
                    restored += 1
            if restored:
                logger.debug(
                    f"Restored tree advantages for {restored} trajectories (mode=TREE)"
                )

        return result

    PPOActor.compute_advantages = _tree_backup_compute_advantages
    # Store original for restore
    PPOActor._original_compute_advantages = original_compute_advantages


def unpatch_ppo_actor() -> None:
    """Restore the original PPOActor.compute_advantages method."""
    if hasattr(PPOActor, "_original_compute_advantages"):
        PPOActor.compute_advantages = PPOActor._original_compute_advantages
        del PPOActor._original_compute_advantages


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

        Query ID derivation fallback chain:
        1. ``prompt["query_id"]`` — dataset-provided string (preferred)
        2. ``prompt["_mcts_query_id"]`` — from prior injection
        3. Empty string (no tree lookup possible)

        Returns:
            cached: list of dicts with keys: prompt, query_id, cached_count,
                need_gen_count
            need_gen: list of dicts with keys: prompt, query_id
        """
        cached = []
        need_gen = []

        for prompt in prompts:
            query_id = prompt.get("query_id") or prompt.get("_mcts_query_id") or ""

            untrained_count = (
                self.tree_store.get_untrained_count(query_id) if query_id else 0
            )

            logger.debug(
                f"Prompt query_id={query_id}: {untrained_count} untrained "
                f"(need {self.n_samples})"
            )

            if untrained_count > 0:
                cached_count = min(untrained_count, self.n_samples)
                need_gen_count = max(0, self.n_samples - untrained_count)
                cached.append(
                    {
                        "prompt": prompt,
                        "query_id": query_id,
                        "cached_count": cached_count,
                        "need_gen_count": need_gen_count,
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
    1. _cache_aware_prepare_batch:
       a. Check cache / generate trajectories
       b. Insert into MCTS tree (while _mcts_query_id is available)
       c. Compute tree advantages (TREE mode) and stash as
          _tree_advantages / _tree_returns
       d. Mark trajectories as trained
       e. Save tree checkpoint (CROSS_TRAINING mode)
    2. compute_advantages:
       a. GAE runs (overwrites advantages/returns)
       b. Patch restores tree advantages from _tree_advantages/_tree_returns
    3. ppo_update uses restored tree advantages

    Monkey-patches ``PPOActor.compute_advantages`` at the class level (not
    instance level) so that all PPOActor instances — including those created
    internally by the base PPOTrainer — use the tree backup version. Patches
    are cleaned up in ``close()``.
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

        if (
            self.cache_config.enabled
            and self.tree_backup_config.mode != TreeBackupMode.OFF
        ):
            self._init_tree_components()
            self._init_patches()
            logger.info(
                f"Cache-aware training enabled "
                f"(mode={self.tree_backup_config.mode.value}, "
                f"advantage={self.tree_backup_config.advantage_mode.value}, "
                f"n_samples={self.cache_config.n_samples})"
            )

    def _init_tree_components(self) -> None:
        """Create tree store, advantage computer, and checkpoint manager."""
        self.tree_store = MCTSTreeStore()
        self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
        self.tree_checkpoint_manager = TreeCheckpointManager(
            self.cache_config.cache_dir or self.tree_backup_config.checkpoint_dir
        )

        # Load existing tree checkpoint if available (CROSS_TRAINING mode)
        if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
            if self.tree_checkpoint_manager.exists():
                self.tree_store = self.tree_checkpoint_manager.load()
                logger.info("Loaded MCTS tree checkpoint with cached rollouts")

        # Reset trained flags for a fresh training run
        self.tree_store.reset_trained_flags()

        self._batch_builder = _CacheAwareBatchBuilder(
            self.tree_store, self.cache_config.n_samples, self.tokenizer
        )

    def _init_patches(self) -> None:
        """Apply monkey-patches for tree backup and query_id injection."""
        patch_ppo_actor_for_tree_backup(
            advantage_mode=self.tree_backup_config.advantage_mode,
        )
        logger.info(
            f"Patched compute_advantages for tree backup "
            f"(advantage_mode={self.tree_backup_config.advantage_mode.value})"
        )

        # Patch _wrap_openai_agent to use QueryIDProxyWorkflow so that
        # dataset query_id strings are injected into trajectories as
        # _mcts_query_id. Without this, the async rollout pipeline would
        # lose the query_id because concat_padded_tensors drops non-tensor keys.
        _patch_wrap_openai_agent_for_query_id(self.rollout)

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

        Strategy: if *all* prompts in the batch have enough cached trajectories,
        use cache only. If *any* prompt lacks sufficient cache, regenerate all
        prompts via rollout_batch (all-or-nothing). This avoids mixing cached
        and freshly-generated trajectories in a single batch.

        After assembling trajectories, this method also:
        1. Inserts them into the MCTS tree (where _mcts_query_id / _mcts_seq_id
           are still available, before concat_padded_tensors drops them)
        2. If advantage_mode is TREE, computes tree Q-values and stashes them
           as _tree_advantages / _tree_returns so they survive the GAE pipeline
        3. Marks trajectories as trained
        4. Saves tree checkpoint (CROSS_TRAINING mode)

        Returns:
            List of trajectory dicts from rollout_batch (may be grouped with
            shape [group_size, seq_len]) or cache (shape [1, seq_len]),
            carrying ``_mcts_query_id`` and ``_mcts_seq_id`` metadata.
        """
        from areal.utils.data import cycle_dataloader

        # Lazily initialize the dataloader iterator
        if not hasattr(self, "_cache_dataloader_iter"):
            self._cache_dataloader_iter = iter(cycle_dataloader(dataloader))

        # Pull a batch of raw data items from the dataloader
        raw_batch = next(self._cache_dataloader_iter)

        # Split into cached / needs-generation
        cached_items, need_gen_items = self._batch_builder.split_prompts(raw_batch)

        # All prompts have enough cache -> use cache only
        if not need_gen_items:
            trajs = list(self._batch_builder.load_cached_trajectories(cached_items))
            logger.info(f"Cache-aware rollout: {len(trajs)} cached (all from cache)")
        else:
            # Any prompt lacks cache -> regenerate all prompts via rollout_batch
            n_samples = self.cache_config.n_samples
            all_prompts = [item["prompt"] for item in cached_items] + [
                item["prompt"] for item in need_gen_items
            ]

            logger.info(
                f"Generating trajectories for {len(all_prompts)} query "
                f"(group_size={n_samples})"
            )
            new_trajs = self.actor.rollout_batch(
                all_prompts,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=n_samples,
            )

            n_new = sum(t["input_ids"].shape[0] for t in new_trajs) if new_trajs else 0
            logger.info(f"Cache-aware rollout: 0 cached, {n_new} newly generated")
            trajs = new_trajs

        # --- Tree operations (while _mcts_query_id / _mcts_seq_id are available) ---

        # Insert trajectories into the MCTS tree
        self.tree_store.insert_batch(trajs)
        logger.debug(f"Inserted {len(trajs)} trajectories into tree")

        # Compute tree advantages and stash for post-GAE restoration
        if self.tree_backup_config.advantage_mode == AdvantageMode.TREE:
            self.tree_advantage_computer.compute(trajs)
            for traj in trajs:
                if "advantages" in traj:
                    traj["_tree_advantages"] = traj["advantages"].clone()
                    traj["_tree_returns"] = traj["returns"].clone()
            logger.debug(
                f"Computed tree advantages for {len(trajs)} trajectories (mode=TREE)"
            )

        # Mark trajectories as trained so they won't be loaded from cache again
        _mark_batch_trained(self.tree_store, trajs)
        logger.debug(f"Marked {len(trajs)} trajectories as trained")

        # Save tree checkpoint (CROSS_TRAINING mode)
        if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)
            logger.debug("Saved MCTS tree checkpoint after tree operations")

        # --- End tree operations ---

        return trajs

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
        that loads cached trajectories and only generates missing ones. The
        original ``prepare_batch`` is always restored in the ``finally`` block,
        so the patch never leaks on error.
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

        # Monkey-patch prepare_batch with cache-aware version
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
            logger.info("Restored original prepare_batch")
            # Clean up the dataloader iterator
            if hasattr(self, "_cache_dataloader_iter"):
                del self._cache_dataloader_iter

    def close(self) -> None:
        if (
            self.cache_config.enabled
            and self.tree_backup_config.mode != TreeBackupMode.OFF
        ):
            unpatch_ppo_actor()
            _unpatch_wrap_openai_agent(self.rollout)
        super().close()
