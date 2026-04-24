"""Offline training script for on-policy distillation with mock data generation, save/load functionality, and CLI parser.

This script provides tools for generating mock training data batches, saving/loading them to disk,
and parsing command-line arguments for offline training.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo


def generate_mock_batch(
    batch_size: int = 4,
    seq_len: int = 128,
    prompt_len: int = 32,
    num_candidates: int = 3,
    vocab_size: int = 32000,
) -> dict[str, Any]:
    """Generate a mock training batch with position rewards.

    Args:
        batch_size: Number of samples per batch.
        seq_len: Total sequence length per sample.
        prompt_len: Length of prompt tokens per sample (no loss).
        num_candidates: Number of candidate tokens per position.
        vocab_size: Vocabulary size for token ID generation.

    Returns:
        Dictionary containing training batch tensors and position rewards.
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    loss_mask[:, prompt_len:] = 1

    logprobs = torch.randn(batch_size, seq_len) * 0.5 - 2.0
    advantages = torch.randn(batch_size, seq_len)

    position_rewards: list[PositionRewardInfo] = []
    output_len = seq_len - prompt_len

    for sample_idx in range(batch_size):
        for pos_idx in range(0, output_len, 4):
            position = prompt_len + pos_idx
            candidates = [
                f"token_{sample_idx}_{pos_idx}_{i}" for i in range(num_candidates)
            ]
            candidate_token_ids = torch.randint(
                0, vocab_size, (num_candidates,)
            ).tolist()
            logprobs_list = (torch.randn(num_candidates) * 0.5 - 2.0).tolist()
            rewards_list = torch.randn(num_candidates).tolist()
            chosen_index = torch.randint(0, num_candidates, (1,)).item()

            position_rewards.append(
                PositionRewardInfo(
                    position=position,
                    candidates=candidates,
                    candidate_token_ids=candidate_token_ids,
                    logprobs=logprobs_list,
                    rewards=rewards_list,
                    chosen_index=chosen_index,
                    sample_index=sample_idx,
                )
            )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "advantages": advantages,
        "position_rewards": position_rewards,
    }


def save_batch(batch: dict[str, Any], path: Path) -> None:
    """Save a training batch to disk, handling PositionRewardInfo serialization.

    Args:
        batch: Batch dictionary to save.
        path: Path to save the batch file.
    """
    save_data: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "position_rewards":
            save_data["position_rewards"] = [
                {
                    "position": pr.position,
                    "candidates": pr.candidates,
                    "candidate_token_ids": pr.candidate_token_ids,
                    "logprobs": pr.logprobs,
                    "rewards": pr.rewards,
                    "chosen_index": pr.chosen_index,
                    "sample_index": pr.sample_index,
                }
                for pr in value
            ]
        else:
            save_data[key] = value
    torch.save(save_data, path)


def load_batch(path: Path) -> dict[str, Any]:
    """Load a training batch from disk, restoring PositionRewardInfo objects.

    Args:
        path: Path to the saved batch file.

    Returns:
        Loaded batch dictionary with PositionRewardInfo objects.
    """
    data = torch.load(path, weights_only=False)
    if "position_rewards" in data:
        data["position_rewards"] = [
            PositionRewardInfo(**pr_dict) for pr_dict in data["position_rewards"]
        ]
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for offline training.

    Args:
        argv: List of arguments to parse. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Offline on-policy distillation training (no inference)"
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/distill")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--distill_loss_weight", type=float, default=0.005)
    parser.add_argument("--rl_loss_weight", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--mock_data", action="store_true", default=True)
    parser.add_argument("--no_mock_data", dest="mock_data", action="store_false")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    return parser.parse_args(argv)


@dataclass
class TrainingConfig:
    model_path: str = ""
    data_path: str = ""
    output_dir: str = "./checkpoints/distill"
    num_epochs: int = 3
    batch_size: int = 4
    seq_len: int = 512
    prompt_len: int = 128
    num_candidates: int = 3
    lr: float = 1e-6
    eps_clip: float = 0.2
    distill_loss_weight: float = 0.005
    rl_loss_weight: float = 1.0
    save_every: int = 100
    vocab_size: int = 32000
    mock_data: bool = True
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = False


def _save_checkpoint(engine, path: str) -> None:
    from areal.api.io_struct import SaveLoadMeta

    meta = SaveLoadMeta(
        path=path,
        weight_format="hf",
        with_optim=True,
        tokenizer=engine.tokenizer,
        processor=None,
    )
    engine.save(meta)


def run_training(config: TrainingConfig) -> dict[str, float]:
    import functools

    from customized_areal.on_policy_distill.engine.fsdp_engine import (
        MultiCandidateFSDPEngine,
    )
    from customized_areal.on_policy_distill.training.actor import (
        patch_ppo_actor_class_to_use_distill_loss,
    )
    from customized_areal.on_policy_distill.training.loss import grpo_distill_loss_fn

    from areal.api.cli_args import FinetuneSpec, OptimizerConfig, PPOActorConfig
    from areal.utils import logging as areal_logging

    logger = areal_logging.getLogger("OfflineTrain")

    # Patch PPOActor for distill loss
    patch_ppo_actor_class_to_use_distill_loss()

    # Create engine config
    engine_config = PPOActorConfig(
        path=config.model_path or "dummy",
        dtype=config.dtype,
        eps_clip=config.eps_clip,
        ppo_n_minibatches=1,
        gradient_checkpointing=config.gradient_checkpointing,
        init_from_scratch=(config.model_path == ""),
        optimizer=OptimizerConfig(type="adam", lr=config.lr),
    )

    # Create engine
    engine = MultiCandidateFSDPEngine(engine_config)
    engine.create_process_group()
    ft_spec = FinetuneSpec(
        total_train_epochs=config.num_epochs,
        dataset_size=config.batch_size * 10,
        train_batch_size=config.batch_size,
    )
    engine.initialize(addr=None, ft_spec=ft_spec)

    # Prepare output directory
    os.makedirs(config.output_dir, exist_ok=True)

    step = 0
    final_loss = 0.0

    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")

        # Determine number of batches per epoch
        if config.mock_data or not config.data_path:
            steps_per_epoch = 10
        else:
            data_dir = Path(config.data_path)
            pt_files = sorted(data_dir.glob("*.pt"))
            steps_per_epoch = max(len(pt_files), 1)

        for batch_idx in range(steps_per_epoch):
            # Load or generate batch
            if config.mock_data or not config.data_path:
                batch = generate_mock_batch(
                    batch_size=config.batch_size,
                    seq_len=config.seq_len,
                    prompt_len=config.prompt_len,
                    num_candidates=config.num_candidates,
                    vocab_size=config.vocab_size,
                )
            else:
                data_dir = Path(config.data_path)
                pt_files = sorted(data_dir.glob("*.pt"))
                if pt_files:
                    batch = load_batch(pt_files[batch_idx % len(pt_files)])
                else:
                    batch = generate_mock_batch(
                        batch_size=config.batch_size,
                        seq_len=config.seq_len,
                        prompt_len=config.prompt_len,
                        num_candidates=config.num_candidates,
                        vocab_size=config.vocab_size,
                    )

            # Move tensors to device
            device = engine.device
            for key in [
                "input_ids",
                "attention_mask",
                "loss_mask",
                "logprobs",
                "advantages",
            ]:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Run training step
            train_stat = engine.train_batch(
                batch,
                loss_fn=functools.partial(
                    grpo_distill_loss_fn,
                    config=engine_config,
                    current_version=engine.get_version(),
                ),
                loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
            )

            final_loss = train_stat.get("loss", 0.0)
            if step % 10 == 0:
                logger.info(
                    f"Step {step} | Loss: {final_loss:.6f} | "
                    f"Epoch {epoch + 1} Batch {batch_idx + 1}/{steps_per_epoch}"
                )

            # Save checkpoint
            if config.save_every > 0 and step > 0 and step % config.save_every == 0:
                ckpt_path = os.path.join(config.output_dir, f"step_{step}")
                logger.info(f"Saving checkpoint to {ckpt_path}")
                _save_checkpoint(engine, ckpt_path)

            step += 1

    # Final checkpoint
    final_ckpt_path = os.path.join(config.output_dir, "final")
    logger.info(f"Saving final checkpoint to {final_ckpt_path}")
    _save_checkpoint(engine, final_ckpt_path)

    return {"final_loss": final_loss}


def main():
    args = parse_args()
    config = TrainingConfig(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        prompt_len=args.prompt_len,
        num_candidates=args.num_candidates,
        lr=args.lr,
        eps_clip=args.eps_clip,
        distill_loss_weight=args.distill_loss_weight,
        rl_loss_weight=args.rl_loss_weight,
        save_every=args.save_every,
        vocab_size=args.vocab_size,
        mock_data=args.mock_data,
        dtype=args.dtype,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    run_training(config)


if __name__ == "__main__":
    main()
