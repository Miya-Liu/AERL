"""Training script for tree search distilling.

Combines MCTS tree backup advantages with on-policy distillation loss
and rollout caching in a single training step.

When a teacher model is configured (teacher_model_name is set), position
rewards are computed as student_logp - teacher_logp for distillation.
When no teacher is configured, student position-level logprobs are still
saved for logging, with zero distillation rewards.

Usage:
uv run customized_areal/tpfc/scripts/train_tpfc_tree_distill.py  --config customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct.yaml workflow=customized_areal.tpfc.tpfc_agent.TPFCAgent +cache_dir=customized_areal/tpfc/data/tree_cache 
"""

# ruff: noqa: E402

import pathlib
import sys

project_root = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.on_policy_distill.core.config import OnPolicyDistillConfig
from customized_areal.tree_search.config import (
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search_distilling.trainer import TreeDistillPPOTrainer

from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("TrainTreeSearchDistilling")


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    logger.info("Starting tree search distilling training")

    # Load configuration
    config, overrides = load_expr_config(args, OnPolicyDistillConfig)

    # Extract cache config from overrides or use defaults
    cache_dir = getattr(config, "cache_dir", "")
    n_samples = config.gconfig.n_samples
    assistant_marker = getattr(config, "assistant_marker", "")

    cache_config = RolloutCacheConfig(
        cache_dir=cache_dir,
        enabled=True,
        n_samples=n_samples,
    )

    tree_backup_config = TreeBackupConfig(
        mode=TreeBackupMode.CROSS_TRAINING,
        assistant_marker=assistant_marker,
        checkpoint_dir=cache_dir,
    )

    logger.info(
        "Cache config: dir=%s, n_samples=%d, tree_mode=%s, teacher=%s",
        cache_dir,
        n_samples,
        tree_backup_config.mode.value,
        getattr(config, "teacher_model_name", "") or "(none)",
    )

    # Create trainer and run
    trainer = TreeDistillPPOTrainer(
        config=config,
        cache_config=cache_config,
        tree_backup_config=tree_backup_config,
    )
    trainer.train(workflow=trainer.workflow)

    logger.info("Tree search distilling training completed")


if __name__ == "__main__":
    main()
