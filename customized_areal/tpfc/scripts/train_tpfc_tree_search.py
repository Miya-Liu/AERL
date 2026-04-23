"""Training script for TPFC Agent with MCTS tree backup and rollout caching.

Uses CacheAwarePPOTrainer from tree_search to combine the TPFC agent and
dataset with MCTS tree backup advantages and rollout caching.  Cached
trajectories are reused across training steps; only missing rollouts are
newly generated.

Usage:
    uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml +cache_dir=customized_areal/tpfc/data/tree_cache
"""

import pathlib
import sys

project_root = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.tpfc.tpfc_config import TPFCConfig
from customized_areal.tpfc.tpfc_dataset import get_tpfc_rl_dataset
from customized_areal.tree_search.config import (
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

from areal.api.cli_args import load_expr_config
from areal.utils import logging
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("TrainTPFCTreeSearch")


def _resolve_agent(workflow_path: str):
    """Resolve workflow config to an agent instance for OpenAIProxyWorkflow."""
    try:
        cls = import_from_string(workflow_path)
    except (ImportError, AttributeError):
        logger.warning("Could not import workflow=%s", workflow_path)
        return None

    if isinstance(cls, type) and hasattr(cls, "run"):
        logger.info("Resolved workflow=%s as agent class, instantiating", workflow_path)
        return cls()
    return None


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    logger.info("Starting TPFC tree search training")

    config, overrides = load_expr_config(args, TPFCConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load TPFC dataset
    train_dataset = get_tpfc_rl_dataset(
        path=config.train_dataset.path,
        split="train",
        tokenizer=tokenizer,
        max_length=config.train_dataset.max_length,
    )

    valid_dataset = get_tpfc_rl_dataset(
        path=config.valid_dataset.path,
        split="test",
        tokenizer=tokenizer,
        max_length=config.valid_dataset.max_length,
    )

    logger.info("Loaded %d training samples", len(train_dataset))
    logger.info("Loaded %d validation samples", len(valid_dataset))

    # Resolve agent from workflow config
    agent = _resolve_agent(config.workflow)

    # Build cache / tree backup configs from overrides
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
        "Cache config: dir=%s, n_samples=%d, tree_mode=%s, agent=%s",
        cache_dir,
        n_samples,
        tree_backup_config.mode.value,
        type(agent).__name__ if agent else "(none)",
    )

    # Build workflow kwargs
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=getattr(config.gconfig, "top_p", 1.0),
        max_completion_tokens=config.gconfig.max_new_tokens,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with CacheAwarePPOTrainer(
        config,
        cache_config=cache_config,
        tree_backup_config=tree_backup_config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )

    logger.info("TPFC tree search training completed")


if __name__ == "__main__":
    main()
