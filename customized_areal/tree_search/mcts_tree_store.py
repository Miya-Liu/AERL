#!/usr/bin/env python3
# customized_areal/tree_search/mcts_tree_store.py
"""Flat trajectory store with MCTS statistics.

Replaces the TrieNode-based trie with a per-query list of TrajectoryRecord
objects. Each record stores the complete, unpadded sequence from the rollout,
with turn boundaries derived from loss_mask transitions.

This correctly preserves full multi-turn context (including system prompts,
user questions, and growing conversation history) that the trie structure
discarded when it only stored assistant marker tokens as prompt_tokens.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrajectoryRecord:
    """Stores a complete multi-turn trajectory for cache storage."""

    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    versions: list[int]
    reward: float
    turn_response_starts: list[int]
    turn_response_ends: list[int]


def _find_turn_boundaries(
    loss_mask: list[int],
) -> tuple[list[int], list[int]]:
    """Scan loss_mask for 0→1 and 1→0 transitions.

    Returns (turn_response_starts, turn_response_ends) where each pair
    defines a half-open range [start, end) of response tokens.
    """
    starts: list[int] = []
    ends: list[int] = []
    in_response = False
    for i, v in enumerate(loss_mask):
        if v == 1 and not in_response:
            starts.append(i)
            in_response = True
        elif v == 0 and in_response:
            ends.append(i)
            in_response = False
    if in_response:
        ends.append(len(loss_mask))
    return starts, ends


def _get_query_id(traj: dict[str, Any]) -> str:
    """Derive a query ID from the prompt tokens in a trajectory."""
    loss_mask = traj["loss_mask"]
    input_ids = traj["input_ids"]
    if input_ids.dim() == 2:
        lm = loss_mask[0]
        ids = input_ids[0]
    else:
        lm = loss_mask
        ids = input_ids
    prompt_tokens = ids[lm == 0].tolist()
    prompt_str = ",".join(str(t) for t in prompt_tokens)
    return hashlib.md5(prompt_str.encode()).hexdigest()


class MCTSTreeStore:
    """Flat trajectory store with MCTS statistics.

    Manages multiple trajectories per query, tracks MCTS statistics
    (visit counts, Q-values) per trajectory, and provides cache-aware
    loading of untrained trajectories.
    """

    def __init__(self) -> None:
        self.trajectories: dict[str, list[TrajectoryRecord]] = {}
        self._seq_id_to_key: dict[int, tuple[str, int]] = {}
        self._query_seq_ids: dict[str, list[int]] = {}
        self._next_seq_id: int = 0

        self._visit_counts: dict[int, int] = {}
        self._total_values: dict[int, float] = {}
        self._q_values: dict[int, float] = {}

        self._trained: dict[int, bool] = {}
        self._rewards: dict[int, float] = {}

    def _backup(self, seq_id: int, reward: float) -> None:
        """Update MCTS stats for a single trajectory."""
        self._visit_counts[seq_id] = self._visit_counts.get(seq_id, 0) + 1
        self._total_values[seq_id] = self._total_values.get(seq_id, 0.0) + reward
        self._q_values[seq_id] = self._total_values[seq_id] / self._visit_counts[seq_id]

    def _make_record(
        self, traj: dict[str, Any], idx: int, seq_len: int
    ) -> TrajectoryRecord:
        """Extract an unpadded sample from traj[idx] and derive turn boundaries."""
        input_ids = traj["input_ids"][idx, :seq_len].tolist()
        loss_mask = traj["loss_mask"][idx, :seq_len].tolist()
        logprobs = (
            traj["logprobs"][idx, :seq_len].tolist()
            if "logprobs" in traj
            else [0.0] * seq_len
        )
        versions = (
            traj["versions"][idx, :seq_len].tolist()
            if "versions" in traj
            else [0] * seq_len
        )
        rewards = traj["rewards"]
        reward = rewards[idx].item() if rewards.dim() >= 1 else rewards.item()

        starts, ends = _find_turn_boundaries(loss_mask)
        return TrajectoryRecord(
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            reward=reward,
            turn_response_starts=starts,
            turn_response_ends=ends,
        )

    def _insert_single(self, query_id: str, record: TrajectoryRecord) -> int:
        """Insert a single TrajectoryRecord and assign a seq_id."""
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        idx = len(self.trajectories.setdefault(query_id, []))
        self.trajectories[query_id].append(record)
        self._seq_id_to_key[seq_id] = (query_id, idx)
        self._query_seq_ids.setdefault(query_id, []).append(seq_id)

        self._backup(seq_id, record.reward)
        self._trained[seq_id] = False
        self._rewards[seq_id] = record.reward

        return seq_id

    def insert_batch(self, trajectories: list[dict[str, Any]]) -> None:
        """Insert trajectories into the store.

        Handles both individual (batch_size=1) and grouped (batch_size>1)
        trajectory dicts. Padding is stripped per sample using attention_mask.
        Turn boundaries are derived from loss_mask.

        Trajectories that already carry _mcts_seq_id or _mcts_seq_ids
        are skipped (loaded from cache).
        """
        for traj in trajectories:
            if "_mcts_seq_id" in traj or "_mcts_seq_ids" in traj:
                continue

            input_ids = traj["input_ids"]
            batch_size = input_ids.shape[0]

            if batch_size == 1:
                query_id = traj.get("_mcts_query_id") or _get_query_id(traj)
                seq_len = int(traj["attention_mask"].sum())
                record = self._make_record(traj, 0, seq_len)
                seq_id = self._insert_single(query_id, record)
                traj["_mcts_seq_id"] = seq_id
                traj["_mcts_query_id"] = query_id
            else:
                seq_ids: list[int] = []
                query_id = traj.get("_mcts_query_id")
                for i in range(batch_size):
                    single = {
                        "input_ids": input_ids[i : i + 1],
                        "loss_mask": traj["loss_mask"][i : i + 1],
                        "rewards": traj["rewards"][i : i + 1],
                    }
                    qid = query_id or _get_query_id(single)
                    if query_id is None:
                        query_id = qid
                    seq_len = int(traj["attention_mask"][i].sum())
                    record = self._make_record(traj, i, seq_len)
                    seq_id = self._insert_single(qid, record)
                    seq_ids.append(seq_id)

                traj["_mcts_seq_ids"] = seq_ids
                traj["_mcts_query_id"] = query_id

    def get_advantages(self, query_id: str, seq_id: int) -> torch.Tensor:
        """Return per-token advantages: Q-value on response tokens, 0 on prompt tokens."""
        qid, idx = self._seq_id_to_key[seq_id]
        record = self.trajectories[qid][idx]
        q_val = self._q_values.get(seq_id, 0.0)
        seq_len = len(record.input_ids)
        advantages = torch.zeros(seq_len, dtype=torch.float32)
        for start, end in zip(record.turn_response_starts, record.turn_response_ends):
            advantages[start:end] = q_val
        return advantages

    def get_prompt_mask(self, query_id: str, seq_id: int) -> torch.Tensor:
        """Return boolean mask: True for response tokens, False for prompt."""
        qid, idx = self._seq_id_to_key[seq_id]
        record = self.trajectories[qid][idx]
        return torch.tensor(record.loss_mask, dtype=torch.bool)

    def set_trained(self, query_id: str, seq_id: int, trained: bool = True) -> None:
        self._trained[seq_id] = trained

    def is_trained(self, query_id: str, seq_id: int) -> bool:
        return self._trained.get(seq_id, False)

    def get_reward(self, query_id: str, seq_id: int) -> float:
        return self._rewards.get(seq_id, 0.0)

    def get_untrained_count(self, query_id: str) -> int:
        if query_id not in self._query_seq_ids:
            return 0
        return sum(
            1
            for seq_id in self._query_seq_ids[query_id]
            if not self._trained.get(seq_id, False)
        )

    def get_untrained_seq_ids(self, query_id: str, n_samples: int) -> list[int]:
        if query_id not in self._query_seq_ids:
            return []
        result: list[int] = []
        for seq_id in self._query_seq_ids[query_id]:
            if not self._trained.get(seq_id, False):
                result.append(seq_id)
                if len(result) >= n_samples:
                    break
        return result

    def load_trajectories(self, query_id: str, n_samples: int) -> list[dict[str, Any]]:
        """Load untrained trajectories as [1, seq_len] dicts.

        Returns stored input_ids/loss_mask directly — no reconstruction.
        """
        if query_id not in self.trajectories:
            return []

        untrained_ids = self.get_untrained_seq_ids(query_id, n_samples)
        result: list[dict[str, Any]] = []
        for seq_id in untrained_ids:
            qid, idx = self._seq_id_to_key[seq_id]
            record = self.trajectories[qid][idx]
            seq_len = len(record.input_ids)
            result.append(
                {
                    "input_ids": torch.tensor(
                        record.input_ids, dtype=torch.int32
                    ).unsqueeze(0),
                    "loss_mask": torch.tensor(
                        record.loss_mask, dtype=torch.int32
                    ).unsqueeze(0),
                    "logprobs": torch.tensor(
                        record.logprobs, dtype=torch.float32
                    ).unsqueeze(0),
                    "versions": torch.tensor(
                        record.versions, dtype=torch.int32
                    ).unsqueeze(0),
                    "attention_mask": torch.ones(seq_len, dtype=torch.bool).unsqueeze(
                        0
                    ),
                    "rewards": torch.tensor(
                        [record.reward], dtype=torch.float32
                    ).unsqueeze(0),
                    "_mcts_query_id": query_id,
                    "_mcts_seq_id": seq_id,
                }
            )
        return result

    def reset_trained_flags(self) -> None:
        for key in self._trained:
            self._trained[key] = False

    def clear(self) -> None:
        """Reset all trajectories, stats, and indices."""
        self.trajectories.clear()
        self._seq_id_to_key.clear()
        self._query_seq_ids.clear()
        self._next_seq_id = 0
        self._visit_counts.clear()
        self._total_values.clear()
        self._q_values.clear()
        self._trained.clear()
        self._rewards.clear()
