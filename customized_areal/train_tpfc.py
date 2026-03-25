"""
Training script for TPFC Agent with AReaL.

This script demonstrates how to train the TPFC Agent using AReaL's
PPO/GRPO trainer with the OpenAI proxy workflow.

Usage:
    python customized_areal/train_tpfc.py \
        --config customized_areal/config_tpfc.yaml \
        workflow=customized_areal.tpfc_agent.TPFCAgent
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
from tpfc_config import TPFCConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, TPFCConfig)
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
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        # For openai
        max_completion_tokens=config.gconfig.max_new_tokens,
    )

    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

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
