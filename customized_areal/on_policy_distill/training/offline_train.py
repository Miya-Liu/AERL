"""Offline training script for on-policy distillation with mock data generation, save/load functionality, and CLI parser.

This script provides tools for generating mock training data batches, saving/loading them to disk,
and parsing command-line arguments for offline training.
"""

import argparse
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
    parser.add_argument("--gradient_checkpointting", action="store_true", default=False)
    return parser.parse_args(argv)
