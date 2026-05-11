# customized_areal/tree_search/patches.py
"""Consolidated monkey-patch manager for tree search training.

All patches needed by CacheAwarePPOTrainer are managed here, providing
atomic apply/restore, crash safety (patches scoped to try/finally),
and a context manager protocol.
"""

from __future__ import annotations

from typing import Any

from customized_areal.tree_search.config import AdvantageMode, LossMode
from customized_areal.tree_search.proxy_workflow import QueryIDProxyWorkflow

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
        tree_store: Any | None = None,
        advantage_computer: Any | None = None,
    ):
        self._engine = self._unwrap_engine(rollout_engine)
        self._advantage_mode = advantage_mode
        self._loss_mode = loss_mode
        self._group_size = group_size
        self._tree_store = tree_store
        self._advantage_computer = advantage_computer

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
        """Unwrap decorators (e.g. RemoteSGLangEngine) to RemoteInfEngine.

        In single-controller mode the ``rollout_engine`` passed in is a
        ``RolloutController`` (which manages remote workers).  Its
        ``_wrap_openai_agent`` / ``workflow_executor`` live on the
        *worker-side* ``RemoteInfEngine`` instances and cannot be
        monkey-patched from the main process.  Trainer-side fallback
        paths (``_tensor_dicts_to_nodes``) handle the dict-to-Node
        conversion for this case.
        """
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

    # ------------------------------------------------------------------
    # Individual patch builders
    # ------------------------------------------------------------------

    def _build_tree_search_wrap(self):
        """Build patched _wrap_openai_agent returning QueryIDProxyWorkflow."""
        engine = self._engine
        tree_store = self._tree_store
        advantage_computer = self._advantage_computer
        advantage_mode = self._advantage_mode
        group_size = self._group_size

        def _tree_search_wrap(agent, proxy_addr):
            agent_cfg = engine.config.agent
            if agent_cfg is None:
                raise RuntimeError(
                    "config.agent is None; tree search workflow requires "
                    "agent configuration. Set agent.mode in the config."
                )
            return QueryIDProxyWorkflow(
                mode=agent_cfg.mode,
                agent=agent,
                proxy_addr=proxy_addr,
                admin_api_key=agent_cfg.admin_api_key,
                discount=agent_cfg.turn_discount,
                export_style=agent_cfg.export_style,
                subproc_max_workers=agent_cfg.subproc_max_workers,
                proxy_gateway_addr=getattr(engine, "_proxy_gateway_addr", None),
                group_size=group_size,
                tree_store=tree_store,
                advantage_computer=advantage_computer,
                advantage_mode=advantage_mode,
            )

        return _tree_search_wrap

    # ------------------------------------------------------------------
    # Apply / Restore
    # ------------------------------------------------------------------

    def apply(self) -> None:
        """Apply all patches. On failure, roll back any already-applied patches."""
        if self._applied:
            logger.warning("TreeSearchPatches.apply() called twice; skipping")
            return

        try:
            _is_controller = hasattr(self._engine, "inf_engine")

            logger.warning(
                "PATCH_VERIFICATION: TreeSearchPatches.apply — "
                "engine_type=%s, has_inf_engine=%s, is_controller=%s, "
                "has_wrap_openai_agent=%s",
                type(self._engine).__name__,
                hasattr(self._engine, "inf_engine"),
                _is_controller,
                hasattr(self._engine, "_wrap_openai_agent"),
            )

            if not _is_controller:
                # Patch: engine._wrap_openai_agent
                if hasattr(self._engine, "_wrap_openai_agent"):
                    self._save_and_set(
                        self._engine,
                        "_wrap_openai_agent",
                        self._build_tree_search_wrap(),
                    )
                    logger.warning(
                        "PATCH_VERIFICATION: _wrap_openai_agent patched on %s",
                        type(self._engine).__name__,
                    )
                else:
                    logger.warning(
                        "Engine has no _wrap_openai_agent method; "
                        "tree search workflow will not be available"
                    )
            else:
                logger.info(
                    "Engine is a RolloutController; skipping worker-side "
                    "patches (remote engine). Trainer-side "
                    "_tensor_dicts_to_nodes will convert tensor dicts to Nodes."
                )

            # Patch: distill loss (conditional)
            if self._loss_mode != LossMode.GRPO:
                from customized_areal.tree_search.training.actor import (
                    patch_ppo_actor_class_to_use_distill_loss,
                    unpatch_ppo_actor_distill_loss,
                )

                self._distill_undo = unpatch_ppo_actor_distill_loss
                patch_ppo_actor_class_to_use_distill_loss()

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
