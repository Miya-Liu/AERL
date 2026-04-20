"""Training script for on-policy distillation with rollout caching and tree backup.

This script enables:
1. Caching generated rollouts in MCTS tree structures
2. Reusing cached rollouts in subsequent training runs
3. Generating only the remaining samples needed for a GRPO group
4. MCTS tree backup for advantage computation

Usage:
    uv run customized_areal/on_policy_distill/scripts/train_with_cache.py \\
        --config customized_areal/on_policy_distill/configs/config_on_policy_distill.yaml \\
        cache_dir=/path/to/cache

First run: generates all rollouts, saves to cache, trains normally.
Second run (from scratch): loads cached rollouts, generates only missing samples.
"""

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
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("TrainWithCache")


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    logger.info("Starting cache-aware training with tree backup")

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
        f"Cache config: dir={cache_dir}, n_samples={n_samples}, "
        f"tree_mode={tree_backup_config.mode.value}"
    )

    # Create trainer and run
    trainer = CacheAwarePPOTrainer(
        config=config,
        cache_config=cache_config,
        tree_backup_config=tree_backup_config,
    )
    trainer.train()

    logger.info("Cache-aware training completed")


if __name__ == "__main__":
    main()
