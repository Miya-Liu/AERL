"""Tree Search Distilling module for AReaL.

Combines MCTS tree backup advantages with on-policy distillation loss
and rollout caching in a single training step.
"""

from customized_areal.tree_search_distilling.agent import TreeDistillAgent

try:
    from customized_areal.tree_search_distilling.trainer import TreeDistillPPOTrainer
except ModuleNotFoundError:
    TreeDistillPPOTrainer = None  # type: ignore[assignment,misc]

__all__ = ["TreeDistillAgent", "TreeDistillPPOTrainer"]
