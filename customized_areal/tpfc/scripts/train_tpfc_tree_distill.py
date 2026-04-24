"""Training script for TPFC Agent with Tree Search Distilling.

Combines the TPFC dataset and agent with MCTS tree backup advantages,
on-policy distillation loss, and rollout caching via TreeDistillPPOTrainer.

The ``workflow`` CLI override specifies the agent class to use inside
OpenAIProxyWorkflow.  For example, to use TPFCAgent:

    uv run customized_areal/tpfc/scripts/train_tpfc_tree_distill.py \
        --config customized_areal/tpfc/configs/config_tpfc_tree_distill.yaml \
        cache_dir=customized_areal/tpfc/data/tree_cache \
        workflow=customized_areal.tpfc.tpfc_agent.TPFCAgent

When ``workflow`` points to an agent class (has an ``async run`` method),
it is instantiated and passed as the ``agent`` kwarg to
TreeDistillPPOTrainer so that OpenAIProxyWorkflow wraps it.
"""

import pathlib
import sys

project_root = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.on_policy_distill.core.config import OnPolicyDistillConfig
from customized_areal.tpfc.tpfc_dataset import get_tpfc_rl_dataset
from customized_areal.tree_search.config import (
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search_distilling.trainer import TreeDistillPPOTrainer

from areal.api.cli_args import load_expr_config
from areal.utils import logging
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("TrainTPFCTreeDistill")


def _resolve_agent(workflow_path: str):
    """Resolve ``workflow`` config to an agent instance for OpenAIProxyWorkflow.

    If *workflow_path* points to a class with an ``async run`` method it is
    treated as an agent class and instantiated.  Otherwise ``None`` is
    returned and TreeDistillPPOTrainer falls back to its default TreeDistillAgent.
    """
    try:
        cls = import_from_string(workflow_path)
    except (ImportError, AttributeError):
        logger.warning(
            "Could not import workflow=%s, using default agent", workflow_path
        )
        return None

    if isinstance(cls, type) and hasattr(cls, "run"):
        logger.info("Resolved workflow=%s as agent class, instantiating", workflow_path)
        return cls()
    return None


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    logger.info("Starting TPFC tree search distilling training")

    config, overrides = load_expr_config(args, OnPolicyDistillConfig)
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

    # Resolve agent from workflow config override (e.g. TPFCAgent)
    agent = _resolve_agent(config.workflow)

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
        "Cache config: dir=%s, n_samples=%d, tree_mode=%s, teacher=%s, agent=%s",
        cache_dir,
        n_samples,
        tree_backup_config.mode.value,
        getattr(config, "teacher_model_name", "") or "(none)",
        type(agent).__name__ if agent else "(default)",
    )

    trainer = TreeDistillPPOTrainer(
        config=config,
        cache_config=cache_config,
        tree_backup_config=tree_backup_config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        agent=agent,
    )
    trainer.train(workflow=trainer.workflow)

    logger.info("TPFC tree search distilling training completed")


if __name__ == "__main__":
    main()
