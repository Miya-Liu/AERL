"""
Training script for On-Policy Distillation Agent with AReaL.

This script demonstrates how to train the OnPolicyDistillAgent using AReaL's
PPO/GRPO trainer with token-level reward tracking.

Usage:
uv run customized_areal/on_policy_distill/train_with_agent.py \
    --config customized_areal/on_policy_distill/config_on_policy_distill.yaml \
    workflow=customized_areal.on_policy_distill.agent.OnPolicyDistillAgent
"""

import pathlib
import sys

# Add project root to path so we can import areal
project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from customized_areal.on_policy_distill.config import OnPolicyDistillConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, OnPolicyDistillConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    # Build workflow kwargs from config
    gconfig = getattr(config, "gconfig", None)
    if gconfig is not None:
        workflow_kwargs = dict(
            temperature=gconfig.temperature,
            top_p=gconfig.top_p,
            max_completion_tokens=gconfig.max_new_tokens,
        )
    else:
        workflow_kwargs = {}

    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
