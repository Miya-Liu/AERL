# customized_areal/tree_search/patches.py
"""Consolidated monkey-patch manager for tree search training.

All patches needed by CacheAwarePPOTrainer are managed here, providing
atomic apply/restore, crash safety (patches scoped to try/finally),
and a context manager protocol.
"""

from __future__ import annotations

from typing import Any

from customized_areal.tree_search.config import AdvantageMode, LossMode
from customized_areal.tree_search.grouped_workflow import (
    TreeSearchGroupedRolloutWorkflow,
)
from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow
from customized_areal.tree_search.workflow_executor import TreeSearchWorkflowExecutor

from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.trainer.ppo.actor import PPOActor
from areal.utils import logging

logger = logging.getLogger("TreeSearchPatches")


class TreeSearchPatches:
    """Manages all monkey-patches needed for tree search training.

    Tracks every original value before overwriting, provides atomic
    apply/restore, and can be used as a context manager.

    Usage::

        patches = TreeSearchPatches(
            rollout_engine=engine,
            advantage_mode=AdvantageMode.TREE,
            loss_mode=LossMode.GRPO,
            group_size=4,
        )
        patches.apply()
        try:
            ...
        finally:
            patches.restore()

    Or as a context manager::

        with TreeSearchPatches(...) as patches:
            ...
    """

    def __init__(
        self,
        rollout_engine: Any,
        advantage_mode: AdvantageMode,
        loss_mode: LossMode,
        group_size: int,
    ):
        self._engine = self._unwrap_engine(rollout_engine)
        self._advantage_mode = advantage_mode
        self._loss_mode = loss_mode
        self._group_size = group_size

        # (target_obj, attr_name, original_value) for every setattr patch
        self._saved: list[tuple[Any, str, Any]] = []
        # Separate undo for distill loss (uses its own unpatch function)
        self._distill_undo: Any = None
        self._applied = False

    # ------------------------------------------------------------------
    # Engine unwrapping (single copy)
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_engine(engine: Any) -> Any:
        """Unwrap decorators (e.g. RemoteSGLangEngine) to RemoteInfEngine."""
        if not hasattr(engine, "_wrap_openai_agent") and hasattr(engine, "_engine"):
            return engine._engine
        return engine

    # ------------------------------------------------------------------
    # Low-level patch primitives
    # ------------------------------------------------------------------

    def _save_and_set(self, obj: Any, attr: str, new_value: Any) -> None:
        """Save current value of obj.attr, then replace it."""
        original = getattr(obj, attr)
        self._saved.append((obj, attr, original))
        setattr(obj, attr, new_value)

    def _save_and_set_method(self, obj: Any, attr: str, new_method: Any) -> None:
        """Save and replace a method (binds new_method to obj)."""
        original = getattr(obj, attr)
        self._saved.append((obj, attr, original))
        setattr(obj, attr, new_method.__get__(obj, type(obj)))

    # ------------------------------------------------------------------
    # Individual patch builders
    # ------------------------------------------------------------------

    def _build_tree_backup_compute_advantages(self):
        """Build patched compute_advantages that restores tree advantages."""
        if hasattr(PPOActor, "_original_compute_advantages"):
            original = PPOActor._original_compute_advantages
        else:
            original = PPOActor.compute_advantages
        advantage_mode = self._advantage_mode

        def _patched(self_actor, data):
            if advantage_mode == AdvantageMode.TREE:
                saved_adv = [traj.get("advantages") for traj in data]
                saved_ret = [traj.get("returns") for traj in data]
            result = original(self_actor, data)
            if advantage_mode == AdvantageMode.TREE:
                restored = 0
                for i, traj in enumerate(result):
                    if saved_adv[i] is not None:
                        traj["advantages"] = saved_adv[i]
                        traj["returns"] = saved_ret[i]
                        restored += 1
                if restored < len(result):
                    logger.warning(
                        f"Tree advantages missing for "
                        f"{len(result) - restored}/{len(result)} "
                        f"trajectories in TREE mode — fell back to GAE"
                    )
                elif restored:
                    logger.debug(
                        f"Restored tree advantages for {restored} "
                        f"trajectories (mode=TREE)"
                    )
            return result

        return _patched

    def _build_tree_search_wrap(self):
        """Build patched _wrap_openai_agent returning
        TreeSearchGroupedRolloutWorkflow."""
        engine = self._engine
        group_size = self._group_size

        def _tree_search_wrap(agent, proxy_addr):
            agent_cfg = engine.config.agent
            if agent_cfg is None:
                raise RuntimeError(
                    "config.agent is None; tree search workflow requires "
                    "agent configuration. Set agent.mode in the config."
                )
            inner = QueryIDProxyWorkflow(
                mode=agent_cfg.mode,
                agent=agent,
                proxy_addr=proxy_addr,
                admin_api_key=agent_cfg.admin_api_key,
                discount=agent_cfg.turn_discount,
                export_style=agent_cfg.export_style,
                subproc_max_workers=agent_cfg.subproc_max_workers,
                proxy_gateway_addr=getattr(engine, "_proxy_gateway_addr", None),
            )
            return TreeSearchGroupedRolloutWorkflow(
                workflow=inner,
                group_size=group_size,
                logger=logger,
            )

        return _tree_search_wrap

    def _build_patched_resolve(self):
        """Build patched _resolve_workflow that strips outer
        GroupedRolloutWorkflow when the inner workflow is already
        TreeSearchGroupedRolloutWorkflow.

        The upstream _resolve_workflow unconditionally wraps with
        GroupedRolloutWorkflow when group_size > 1
        (remote_inf_engine.py:560-562), but
        TreeSearchGroupedRolloutWorkflow already handles grouping
        internally.
        """
        engine = self._engine
        original_resolve = engine._resolve_workflow

        def _patched_resolve(self_engine, wf, wf_kwargs=None, gs=1):
            resolved = original_resolve(wf, wf_kwargs, gs)
            if isinstance(resolved, GroupedRolloutWorkflow) and isinstance(
                resolved.workflow, TreeSearchGroupedRolloutWorkflow
            ):
                logger.debug(
                    "Skipping outer GroupedRolloutWorkflow wrapper "
                    "(TreeSearchGroupedRolloutWorkflow already handles grouping)"
                )
                return resolved.workflow
            return resolved

        return _patched_resolve

    def _build_tree_search_executor(self):
        """Build a TreeSearchWorkflowExecutor replacing the original."""
        engine = self._engine
        original = engine.workflow_executor

        new_executor = TreeSearchWorkflowExecutor(
            config=engine.config,
            inference_engine=engine,
        )

        # Copy all internal state from the original executor.
        # Using vars() instead of listing attributes by name so that
        # upstream additions are automatically picked up.
        for attr, value in vars(original).items():
            if attr.startswith("__"):
                continue
            if attr in ("config", "inference_engine"):
                continue
            setattr(new_executor, attr, value)
        new_executor._initialized = True

        return new_executor

    # ------------------------------------------------------------------
    # Apply / Restore
    # ------------------------------------------------------------------

    def apply(self) -> None:
        """Apply all patches. On failure, roll back any already-applied patches."""
        if self._applied:
            logger.warning("TreeSearchPatches.apply() called twice; skipping")
            return

        try:
            # Patch 1: PPOActor.compute_advantages (class-level)
            new_compute_adv = self._build_tree_backup_compute_advantages()
            if not hasattr(PPOActor, "_original_compute_advantages"):
                PPOActor._original_compute_advantages = PPOActor.compute_advantages
            self._saved.append(
                (PPOActor, "compute_advantages",
                 PPOActor._original_compute_advantages)
            )
            PPOActor.compute_advantages = new_compute_adv

            # Patch 2: engine._wrap_openai_agent
            if hasattr(self._engine, "_wrap_openai_agent"):
                self._save_and_set(
                    self._engine,
                    "_wrap_openai_agent",
                    self._build_tree_search_wrap(),
                )
            else:
                logger.warning(
                    "Engine has no _wrap_openai_agent method; "
                    "tree search workflow will not be available"
                )

            # Patch 2b: engine._resolve_workflow (double-wrapping prevention)
            if hasattr(self._engine, "_resolve_workflow"):
                self._save_and_set_method(
                    self._engine,
                    "_resolve_workflow",
                    self._build_patched_resolve(),
                )

            # Patch 3: engine.workflow_executor
            if hasattr(self._engine, "workflow_executor"):
                new_executor = self._build_tree_search_executor()
                self._save_and_set(self._engine, "workflow_executor", new_executor)
            else:
                logger.warning(
                    "Engine has no workflow_executor attribute; "
                    "tree search workflow executor will not be available"
                )

            # Patch 4 (conditional): PPOActor._ppo_update distill loss
            if self._loss_mode != LossMode.GRPO:
                from customized_areal.tree_search.training.actor import (
                    patch_ppo_actor_class_to_use_distill_loss,
                    unpatch_ppo_actor_distill_loss,
                )
                patch_ppo_actor_class_to_use_distill_loss()
                self._distill_undo = unpatch_ppo_actor_distill_loss

            self._applied = True
            logger.info(
                f"Applied tree search patches "
                f"(advantage={self._advantage_mode.value}, "
                f"loss={self._loss_mode.value}, "
                f"group_size={self._group_size})"
            )

        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        """Restore all original values in reverse order."""
        if not self._applied and not self._saved and self._distill_undo is None:
            return

        # Restore distill loss first (if applied)
        if self._distill_undo is not None:
            self._distill_undo()
            self._distill_undo = None

        # Restore in reverse order (LIFO)
        for obj, attr, original in reversed(self._saved):
            setattr(obj, attr, original)

        # Clean up idempotency marker
        if hasattr(PPOActor, "_original_compute_advantages"):
            del PPOActor._original_compute_advantages

        self._saved.clear()
        self._applied = False
        logger.info("Restored all tree search patches")

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()
        return False
