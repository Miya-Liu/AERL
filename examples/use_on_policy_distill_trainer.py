"""Example script for using OnPolicyDistillationTrainer.

This script demonstrates how to use the OnPolicyDistillationTrainer
with the customized grpo_distill_loss_fn that supports position_rewards.
"""

import torch
from areal.api.cli_args import PPOConfig
from customized_areal.on_policy_distill import (
    OnPolicyDistillationTrainer,
    OnPolicyDistillConfig,
)


def main():
    # Create configuration
    config = OnPolicyDistillConfig(
        actor=PPOActorConfig(
            path="meta-llama/Llama-2-7b-hf",
            learning_rate=1e-6,
            ppo_n_minibatches=4,
        ),
        train_dataset=TrainDatasetConfig(
            name="gsm8k",
            split="train",
        ),
        valid_dataset=ValidDatasetConfig(
            name="gsm8k",
            split="test",
        ),
        total_train_steps=1000,
        total_train_epochs=1,
    )

    # Create trainer (patches PPOActor to use grpo_distill_loss_fn automatically)
    trainer = OnPolicyDistillationTrainer(
        config=config,
    )

    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
