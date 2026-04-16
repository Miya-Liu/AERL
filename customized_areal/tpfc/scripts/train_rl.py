"""Custom training script for TPFC RL training."""

import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer

# Import custom dataset loader directly from customized_areal
from customized_areal.dataset.tpfc import get_tpfc_rl_dataset


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Use custom TPFC dataset loader directly
    train_dataset = get_tpfc_rl_dataset(
        split="train",
        path=config.train_dataset.path,
        tokenizer=tokenizer,
        max_length=config.train_dataset.max_length,
    )
    valid_dataset = get_tpfc_rl_dataset(
        split="test",
        path=config.valid_dataset.path,
        tokenizer=tokenizer,
        max_length=config.valid_dataset.max_length,
    )

    workflow_kwargs = dict(
        reward_fn="customized_areal.reward.tpfc_reward.tpfc_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
