# customized_areal/tree_search/trainer.py
"""MCTS Tree Backup PPOTrainer.

Subclass of PPOTrainer that adds MCTS tree backup to PPO training.
Patches the outer PPOActor.compute_advantages method so that:
1. The original GAE runs first (KL rewards, scaling, normalization)
2. Trajectories are inserted into the tree with raw rewards
3. Depending on advantage_mode config:
   - TREE: tree Q-values overwrite advantages/returns
   - GAE: original GAE advantages/returns are preserved
4. KL metadata (kl_rewards, tot_rewards) is preserved for logging
"""

from __future__ import annotations

from typing import Any

import torch

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import (
    AdvantageMode,
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search.mcts_tree_store import (
    MCTSTreeStore,
    get_query_id_from_messages,
)
from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow
from customized_areal.tree_search.turn_splitter import make_turn_splitter

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


def _patch_wrap_openai_agent_for_query_id(actor: PPOActor) -> None:
    """Patch the engine's _wrap_openai_agent to return QueryIDProxyWorkflow.

    QueryIDProxyWorkflow subclasses OpenAIProxyWorkflow and overrides
    arun_episode to inject data["query_id"] into the trajectory dict as
    ``_mcts_query_id``. This is needed because the async rollout pipeline
    shuffles results, so we cannot match queries to trajectories by position,
    and concat_padded_tensors drops non-tensor keys.
    """
    engine = actor.engine
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


def _unpatch_wrap_openai_agent(actor: PPOActor) -> None:
    """Restore the original _wrap_openai_agent method."""
    engine = actor.engine
    if hasattr(engine, "_original_wrap_openai_agent"):
        engine._wrap_openai_agent = engine._original_wrap_openai_agent
        del engine._original_wrap_openai_agent
        logger.info("Restored original _wrap_openai_agent")


def patch_ppo_actor_for_tree_backup(
    tree_store: MCTSTreeStore,
    tree_advantage_computer: TreeAdvantageComputer,
    advantage_mode: AdvantageMode = AdvantageMode.TREE,
) -> None:
    """Patch PPOActor.compute_advantages to add MCTS tree backup after GAE.

    Modifies ``PPOActor.compute_advantages`` at the class level so all
    instances (including those created internally by the base PPOTrainer)
    use the tree backup version. A subclass override would only apply if
    we also subclassed the actor.

    The patch is idempotent — if ``PPOActor._original_compute_advantages``
    already exists (from a prior patch), it reuses the true original instead
    of stacking patches. Must be cleaned up via ``unpatch_ppo_actor()``.

    The patched method:
    1. Calls the original compute_advantages (full GAE pipeline)
    2. Inserts trajectories into the tree with raw rewards
    3. If advantage_mode is TREE, overwrites advantages/returns with tree Q-values
    4. Marks trajectories as trained
    5. Records training step order

    When advantage_mode is GAE, trajectories are still inserted into the tree
    (for caching and MCTS statistics), but the original GAE advantages/returns
    are preserved unchanged.
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
        logger.debug(f"Step A: GAE completed for {len(result)} trajectories")

        # 2. Insert trajectories into tree with raw rewards
        tree_store.insert_batch(result)
        logger.debug(f"Step B: Inserted {len(result)} trajectories into tree")

        # 3. Overwrite advantages/returns with tree Q-values if TREE mode
        # In TREE mode, tree Q-values replace GAE advantages. In GAE mode,
        # trajectories are still inserted (for caching and MCTS statistics)
        # but the original GAE advantages are preserved.
        if advantage_mode == AdvantageMode.TREE:
            tree_advantage_computer.compute(result)
            logger.debug(
                f"Step C: Computed tree advantages for {len(result)} "
                f"trajectories (mode=TREE)"
            )

        # 4. Mark trajectories as trained so they won't be loaded from cache again
        _mark_batch_trained(tree_store, result)
        logger.debug(f"Step D: Marked {len(result)} trajectories as trained")

        # 5. Record training step order for replay/debugging
        global_step = result[0].get("_global_step") if result else None
        tree_store.record_training_step(global_step, result)

        # advantages/returns already overwritten by compute() in TREE mode,
        # or preserved from GAE in GAE mode.
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


def _split_grouped_trajectories(
    trajs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split grouped trajectory dicts into individual items.

    Grouped trajectories may have shape [group_size, seq_len]. We avoid
    concat_padded_tensors because it keeps only the first dict's value for
    non-tensor, non-list keys, which would lose per-trajectory ``_mcts_query_id``
    and ``_mcts_seq_id``. Keeping them as separate items preserves
    per-trajectory metadata.
    """
    result: list[dict[str, Any]] = []
    for traj in trajs:
        batch_size = traj["input_ids"].shape[0]
        # batch_size == 1 means the trajectory is already individual;
        # appending as-is avoids unnecessary tensor slicing.
        if batch_size == 1:
            result.append(traj)
            continue
        logger.debug(
            f"Split grouped trajectory (batch_size={batch_size}) "
            f"into {batch_size} individual items"
        )
        for i in range(batch_size):
            single: dict[str, Any] = {}
            for k, v in traj.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    single[k] = v[i : i + 1]
                elif isinstance(v, list) and k == "_mcts_seq_ids":
                    single["_mcts_seq_id"] = v[i]
                    single["_mcts_query_id"] = traj.get("_mcts_query_id")
                else:
                    single[k] = v
            result.append(single)
    return result


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
        3. MD5 hash of tokenized messages via ``get_query_id_from_messages``
        4. Empty string (no tree lookup possible)

        Returns:
            cached: list of dicts with keys: prompt, query_id, cached_count,
                need_gen_count
            need_gen: list of dicts with keys: prompt, query_id
        """
        cached = []
        need_gen = []

        for prompt in prompts:
            query_id = prompt.get("query_id") or prompt.get("_mcts_query_id")
            if not query_id:
                messages = prompt.get("messages", [])
                if messages:
                    query_id = get_query_id_from_messages(messages, self.tokenizer)
                else:
                    query_id = ""

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
                        "need_gen_count": 0,
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
    6. Save tree checkpoint (CROSS_TRAINING mode)

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
        turn_splitter = make_turn_splitter(
            self.tokenizer, self.tree_backup_config.assistant_marker
        )
        self.tree_store = MCTSTreeStore(turn_splitter)
        self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
        self.tree_checkpoint_manager = TreeCheckpointManager(
            self.cache_config.cache_dir or self.tree_backup_config.checkpoint_dir
        )

        # Load existing tree checkpoint if available (CROSS_TRAINING mode)
        if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
            if self.tree_checkpoint_manager.exists():
                self.tree_store = self.tree_checkpoint_manager.load(turn_splitter)
                logger.info("Loaded MCTS tree checkpoint with cached rollouts")

        # Reset trained flags for a fresh training run
        self.tree_store.reset_trained_flags()

        self._batch_builder = _CacheAwareBatchBuilder(
            self.tree_store, self.cache_config.n_samples, self.tokenizer
        )

    def _init_patches(self) -> None:
        """Apply monkey-patches for tree backup and query_id injection."""
        patch_ppo_actor_for_tree_backup(
            self.tree_store,
            self.tree_advantage_computer,
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
        _patch_wrap_openai_agent_for_query_id(self.actor)

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

        Returns:
            Flat list of per-sample trajectory dicts, each with shape [1, seq_len],
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
            cached_trajs = self._batch_builder.load_cached_trajectories(cached_items)
            n_cached = len(cached_trajs)
            logger.info(f"Cache-aware rollout: {n_cached} cached (all from cache)")
            return list(cached_trajs)

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
        return _split_grouped_trajectories(new_trajs)

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
            _unpatch_wrap_openai_agent(self.actor)
        super().close()
