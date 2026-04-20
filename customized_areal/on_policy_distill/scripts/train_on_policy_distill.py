"""
Training script for On-Policy Distillation with AReaL.

This script demonstrates how to train using on-policy distillation with
mock OpenAI proxy workflow components.

Usage:
    uv run customized_areal/on_policy_distill/train_on_policy_distill.py \
        --config customized_areal/on_policy_distill/config_on_policy_distill.yaml

Or use the OnPolicyDistillationTrainer directly:
    from customized_areal.on_policy_distill import OnPolicyDistillationTrainer, OnPolicyDistillConfig
    trainer = OnPolicyDistillationTrainer(config)
    trainer.train()
"""

import pathlib
import sys

# Add project root to path so we can import areal
project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.on_policy_distill.config import OnPolicyDistillConfig
from customized_areal.on_policy_distill.trainer import OnPolicyDistillationTrainer

from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("TrainOnPolicyDistill")


def main(args: list[str] | None = None) -> None:
    """Main entry point for on-policy distillation training.

    Args:
        args: Command line arguments. If None, uses sys.argv[1:].
    """
    if args is None:
        args = sys.argv[1:]

    logger.info("Starting on-policy distillation training script")

    # Load configuration
    config, _ = load_expr_config(args, OnPolicyDistillConfig)

    # Create trainer and run training
    trainer = OnPolicyDistillationTrainer(config)
    trainer.train()

    logger.info("Training script completed")


if __name__ == "__main__":
    main()
