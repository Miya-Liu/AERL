#!/usr/bin/env python3
# customized_areal/tree_search/mcts_tree_store.py
"""Flat trajectory store with MCTS statistics.

Replaces the TrieNode-based trie with a per-query list of Node
objects. Each record stores the complete, unpadded sequence from the rollout,
with turn boundaries derived from loss_mask transitions.

This correctly preserves full multi-turn context (including system prompts,
user questions, and growing conversation history) that the trie structure
discarded when it only stored assistant marker tokens as prompt_tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from areal.infra.rpc.rtensor import RTensor


def _materialize_rtensors(traj: dict[str, Any]) -> dict[str, Any]:
    """Convert any RTensor values in a trajectory dict to local torch.Tensors."""
    for key, val in traj.items():
        if isinstance(val, RTensor):
            traj[key] = val.to_local()
    return traj


@dataclass
class Node:
    """A single turn in a multi-turn conversation tree.

    Each Node represents one assistant response turn, including its prompt
    context (all tokens from the beginning of the conversation up through
    this turn's response). Nodes are linked via node_id/parent_node_id and
    grouped into episodes via episode_id.

    Metadata fields (_mcts_query_id, _mcts_seq_id) are set by the tree
    store after insertion. advantages/returns are set by the advantage
    computer.
    """

    # Core sequence (full turn: prompt + response)
    input_ids: list[int]
    loss_mask: list[int]              # 0=prompt, 1=response
    logprobs: list[float]             # full sequence (0.0 on prompt positions)
    versions: list[int]               # policy version (-1 on prompt)

    # Tree structure
    node_id: str                      # interaction ID for this turn
    parent_node_id: str | None        # parent interaction ID (None for root)
    episode_id: str                   # groups turns into a trajectory path

    # Reward
    outcome_reward: float = 0.0

    # Response-only (aligned to loss_mask==1 positions)
    topk_ids: list[list[int]] | None = None
    topk_logp: list[list[float]] | None = None
    distill_reward: list[list[float]] | None = None
    teacher_logp: list[list[float]] | None = None


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


def _is_list_dict(traj: dict[str, Any]) -> bool:
    """Check if a trajectory dict uses Python lists instead of tensors."""
    input_ids = traj.get("input_ids")
    return isinstance(input_ids, list)


class MCTSTreeStore:
    """Flat trajectory store with MCTS statistics.

    Manages multiple trajectories per query, tracks MCTS statistics
    (visit counts, Q-values) per trajectory, and provides cache-aware
    loading of untrained trajectories.
    """

    def __init__(self) -> None:
        self.trajectories: dict[str, list[Node]] = {}
        self._seq_id_to_key: dict[int, tuple[str, int]] = {}
        self._query_seq_ids: dict[str, list[int]] = {}
        self._next_seq_id: int = 0

        self._visit_counts: dict[int, int] = {}
        self._total_values: dict[int, float] = {}
        self._q_values: dict[int, float] = {}

        self._trained: dict[int, bool] = {}
        self._rewards: dict[int, float] = {}

        # Tree-search episode metadata
        self._turn_nodes: dict[str, int] = {}  # turn_id → seq_id
        self._normalized_advantages: dict[int, float] = {}

    def _backup(self, seq_id: int, reward: float) -> None:
        """Update MCTS stats for a single trajectory."""
        self._visit_counts[seq_id] = self._visit_counts.get(seq_id, 0) + 1
        self._total_values[seq_id] = self._total_values.get(seq_id, 0.0) + reward
        self._q_values[seq_id] = self._total_values[seq_id] / self._visit_counts[seq_id]

    def _make_record(
        self, traj: dict[str, Any], idx: int, seq_len: int
    ) -> Node:
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
        outcome_reward = rewards[idx].item() if rewards.dim() >= 1 else rewards.item()

        return Node(
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            outcome_reward=outcome_reward,
            node_id="",
            parent_node_id=None,
            episode_id="",
        )

    def _insert_list_dict(self, traj: dict[str, Any]) -> None:
        """Insert a list-based trajectory dict into the tree store."""
        # Extract fields from list dict
        input_ids = traj["input_ids"]
        loss_mask = traj["loss_mask"]
        reward = traj.get("reward", 0.0)
        logprobs = traj.get("logprobs", [0.0] * len(input_ids))
        versions = traj.get("versions", [0] * len(input_ids))

        # New fields
        logp = traj.get("logp")
        topk_ids = traj.get("topk_ids")
        topk_logp = traj.get("topk_logp")
        distill_reward = traj.get("distill_reward")
        teacher_logp = traj.get("teacher_logp")

        # Get query ID
        query_id = traj.get("_mcts_query_id", "")

        # Get turn boundaries
        if "turn_response_starts" in traj and "turn_response_ends" in traj:
            starts = traj["turn_response_starts"]
            ends = traj["turn_response_ends"]
        else:
            starts, ends = _find_turn_boundaries(loss_mask)

        # Create record
        record = TrajectoryRecord(
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            reward=reward,
            turn_response_starts=starts,
            turn_response_ends=ends,
            logp=logp,
            topk_ids=topk_ids,
            topk_logp=topk_logp,
            distill_reward=distill_reward,
            teacher_logp=teacher_logp,
            turn_ids=traj.get("turn_ids"),
            parent_turn_ids=traj.get("parent_turn_ids"),
            turn_rewards=traj.get("turn_rewards"),
            outcome_reward=traj.get("outcome_reward", 0.0),
        )

        # Insert into store
        seq_id = self._insert_single(query_id, record)
        traj["_mcts_seq_id"] = seq_id
        traj["_mcts_query_id"] = query_id

    def _insert_single(self, query_id: str, node: Node) -> int:
        """Insert a single Node and assign a seq_id."""
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        idx = len(self.trajectories.setdefault(query_id, []))
        self.trajectories[query_id].append(node)
        self._seq_id_to_key[seq_id] = (query_id, idx)
        self._query_seq_ids.setdefault(query_id, []).append(seq_id)

        self._backup(seq_id, node.outcome_reward)
        self._trained[seq_id] = False
        self._rewards[seq_id] = node.outcome_reward

        # Register turn_id → seq_id mappings for shared-node MCTS
        if node.node_id:
            if node.node_id not in self._turn_nodes:
                self._turn_nodes[node.node_id] = seq_id

        return seq_id

    def insert_batch(self, trajectories: list[dict[str, Any]]) -> None:
        """Insert trajectories into the store.

        Supports four input formats:

        1. **Per-turn dicts** (from _split_to_turn_dicts): each dict has
           shape [1, seq_len] with _episode_idx, _turn_idx_in_episode,
           _parent_turn_id, _turn_reward, _outcome_reward metadata.
           Turns are grouped by (_mcts_query_id, _episode_idx) into a
           single episode Node.

        2. **Individual trajectory dicts** (batch_size=1): single trajectory
           without episode metadata. Same as legacy behavior.

        3. **Grouped trajectory dicts** (batch_size>1): multiple trajectories
           stacked together. Each row gets its own seq_id.

        4. **List-based dicts**: single trajectory with Python lists instead
           of tensors for all fields.

        Trajectories that already carry _mcts_seq_id or _mcts_seq_ids
        are skipped (loaded from cache).
        """
        # Materialize any RTensor values to local tensors (from RPC transfer)
        trajectories = [_materialize_rtensors(t) for t in trajectories]

        # Separate per-turn dicts, list dicts, and legacy-style dicts
        per_turn_dicts: list[dict[str, Any]] = []
        list_dicts: list[dict[str, Any]] = []
        legacy_dicts: list[dict[str, Any]] = []

        for traj in trajectories:
            if "_mcts_seq_id" in traj or "_mcts_seq_ids" in traj:
                continue
            if _is_list_dict(traj):
                list_dicts.append(traj)
            elif "_episode_idx" in traj:
                per_turn_dicts.append(traj)
            else:
                legacy_dicts.append(traj)

        # Handle per-turn dicts: group by (query_id, episode_idx) → one record
        if per_turn_dicts:
            self._insert_per_turn_dicts(per_turn_dicts)

        # Handle list-based dicts
        for traj in list_dicts:
            self._insert_list_dict(traj)

        # Handle legacy-style dicts
        for traj in legacy_dicts:
            input_ids = traj["input_ids"]
            batch_size = input_ids.shape[0]

            if batch_size == 1:
                query_id = traj.get("_mcts_query_id", "")
                seq_len = int(traj["attention_mask"].sum())
                record = self._make_record(traj, 0, seq_len)
                seq_id = self._insert_single(query_id, record)
                traj["_mcts_seq_id"] = seq_id
                traj["_mcts_query_id"] = query_id
            else:
                seq_ids: list[int] = []
                query_id = traj.get("_mcts_query_id", "")
                for i in range(batch_size):
                    seq_len = int(traj["attention_mask"][i].sum())
                    record = self._make_record(traj, i, seq_len)
                    seq_id = self._insert_single(query_id, record)
                    seq_ids.append(seq_id)

                traj["_mcts_seq_ids"] = seq_ids
                traj["_mcts_query_id"] = query_id

    def _insert_per_turn_dicts(self, turn_dicts: list[dict[str, Any]]) -> None:
        """Group per-turn dicts by (query_id, episode_idx) and insert as episode records."""
        from itertools import groupby

        # Sort by (query_id, episode_idx) for stable grouping
        sorted_turns = sorted(
            turn_dicts,
            key=lambda d: (d.get("_mcts_query_id", ""), d.get("_episode_idx", 0)),
        )

        for (query_id, ep_idx), group_iter in groupby(
            sorted_turns,
            key=lambda d: (d.get("_mcts_query_id", ""), d.get("_episode_idx", 0)),
        ):
            turns = list(group_iter)
            # Sort turns within the episode by turn_idx_in_episode
            turns.sort(key=lambda d: d.get("_turn_idx_in_episode", 0))

            # Collect per-turn data for episode reconstruction
            all_input_ids: list[int] = []
            all_loss_mask: list[int] = []
            all_logprobs: list[float] = []
            all_versions: list[int] = []
            turn_ids: list[str] = []
            parent_turn_ids: list[str | None] = []
            turn_rewards: list[float] = []
            outcome_reward: float = 0.0

            for turn in turns:
                seq_len = int(turn["attention_mask"].sum())
                input_ids = turn["input_ids"][0, :seq_len].tolist()
                loss_mask = turn["loss_mask"][0, :seq_len].tolist()
                logprobs = (
                    turn["logprobs"][0, :seq_len].tolist()
                    if "logprobs" in turn
                    else [0.0] * seq_len
                )
                versions = (
                    turn["versions"][0, :seq_len].tolist()
                    if "versions" in turn
                    else [0] * seq_len
                )

                all_input_ids.extend(input_ids)
                all_loss_mask.extend(loss_mask)
                all_logprobs.extend(logprobs)
                all_versions.extend(versions)

                turn_ids.append(turn.get("_turn_id", ""))
                parent_turn_ids.append(turn.get("_parent_turn_id"))
                turn_rewards.append(turn.get("_turn_reward", 0.0))
                outcome_reward = turn.get("_outcome_reward", 0.0)

            starts, ends = _find_turn_boundaries(all_loss_mask)

            record = TrajectoryRecord(
                input_ids=all_input_ids,
                loss_mask=all_loss_mask,
                logprobs=all_logprobs,
                versions=all_versions,
                reward=outcome_reward,
                turn_response_starts=starts,
                turn_response_ends=ends,
                turn_ids=turn_ids,
                parent_turn_ids=parent_turn_ids,
                turn_rewards=turn_rewards,
                outcome_reward=outcome_reward,
            )
            seq_id = self._insert_single(query_id, record)

            # Set _mcts_seq_id on all turn dicts in this group
            for turn in turns:
                turn["_mcts_seq_id"] = seq_id
                turn["_mcts_query_id"] = query_id

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

    def load_trajectories(
        self, query_id: str, n_samples: int, as_list: bool = False
    ) -> list[dict[str, Any]]:
        """Load untrained trajectories.

        If `as_list=False` (default), returns per-turn dicts with tensors. Each
        episode record is split back into per-turn dicts using
        turn_response_starts/turn_response_ends to slice the stored sequence.
        Metadata keys (_turn_id, _parent_turn_id, _turn_reward, _outcome_reward)
        are populated from Node.

        If `as_list=True`, returns one dict per episode with Python lists instead
        of tensors. No per-turn splitting is done.
        """
        if query_id not in self.trajectories:
            return []

        untrained_ids = self.get_untrained_seq_ids(query_id, n_samples)
        result: list[dict[str, Any]] = []
        for seq_id in untrained_ids:
            qid, idx = self._seq_id_to_key[seq_id]
            record = self.trajectories[qid][idx]
            seq_len = len(record.input_ids)

            # List-based output: one dict per episode
            if as_list:
                traj = {
                    "input_ids": record.input_ids.copy(),
                    "loss_mask": record.loss_mask.copy(),
                    "logprobs": record.logprobs.copy(),
                    "versions": record.versions.copy(),
                    "attention_mask": [1] * seq_len,
                    "reward": record.reward,
                    "_mcts_query_id": query_id,
                    "_mcts_seq_id": seq_id,
                }
                # Add new fields if present
                if record.logp is not None:
                    traj["logp"] = record.logp.copy()
                if record.topk_ids is not None:
                    traj["topk_ids"] = [row.copy() for row in record.topk_ids]
                if record.topk_logp is not None:
                    traj["topk_logp"] = [row.copy() for row in record.topk_logp]
                if record.distill_reward is not None:
                    traj["distill_reward"] = [
                        row.copy() for row in record.distill_reward
                    ]
                if record.teacher_logp is not None:
                    traj["teacher_logp"] = [row.copy() for row in record.teacher_logp]
                # Add turn metadata if present
                if record.turn_ids is not None:
                    traj["turn_ids"] = record.turn_ids.copy()
                    traj["turn_response_starts"] = record.turn_response_starts.copy()
                    traj["turn_response_ends"] = record.turn_response_ends.copy()
                if record.parent_turn_ids is not None:
                    traj["parent_turn_ids"] = record.parent_turn_ids.copy()
                if record.turn_rewards is not None:
                    traj["turn_rewards"] = record.turn_rewards.copy()
                traj["outcome_reward"] = record.outcome_reward
                result.append(traj)
                continue

            # Tensor-based output: per-turn dicts (legacy behavior)
            full_input_ids = torch.tensor(record.input_ids, dtype=torch.int32)
            full_loss_mask = torch.tensor(record.loss_mask, dtype=torch.int32)
            full_logprobs = torch.tensor(record.logprobs, dtype=torch.float32)
            full_versions = torch.tensor(record.versions, dtype=torch.int32)
            full_attention = torch.ones(seq_len, dtype=torch.bool)

            # New fields - full tensors
            full_logp = (
                torch.tensor(record.logp, dtype=torch.float32)
                if record.logp is not None
                else None
            )
            full_topk_ids = (
                torch.tensor(record.topk_ids, dtype=torch.int32)
                if record.topk_ids is not None
                else None
            )
            full_topk_logp = (
                torch.tensor(record.topk_logp, dtype=torch.float32)
                if record.topk_logp is not None
                else None
            )
            full_distill_reward = (
                torch.tensor(record.distill_reward, dtype=torch.float32)
                if record.distill_reward is not None
                else None
            )
            full_teacher_logp = (
                torch.tensor(record.teacher_logp, dtype=torch.float32)
                if record.teacher_logp is not None
                else None
            )

            # If no turn metadata, return as a single dict (legacy behavior)
            if record.turn_ids is None or not record.turn_response_starts:
                traj = {
                    "input_ids": full_input_ids.unsqueeze(0),
                    "loss_mask": full_loss_mask.unsqueeze(0),
                    "logprobs": full_logprobs.unsqueeze(0),
                    "versions": full_versions.unsqueeze(0),
                    "attention_mask": full_attention.unsqueeze(0),
                    "rewards": torch.tensor(
                        [record.reward], dtype=torch.float32
                    ).unsqueeze(0),
                    "_mcts_query_id": query_id,
                    "_mcts_seq_id": seq_id,
                }
                # Add new fields if present
                if full_logp is not None:
                    traj["logp"] = full_logp.unsqueeze(0)
                if full_topk_ids is not None:
                    traj["topk_ids"] = full_topk_ids.unsqueeze(0)
                if full_topk_logp is not None:
                    traj["topk_logp"] = full_topk_logp.unsqueeze(0)
                if full_distill_reward is not None:
                    traj["distill_reward"] = full_distill_reward.unsqueeze(0)
                if full_teacher_logp is not None:
                    traj["teacher_logp"] = full_teacher_logp.unsqueeze(0)
                result.append(traj)
                continue

            # Split episode into per-turn dicts
            n_turns = len(record.turn_response_starts)

            # Precompute per-turn response lengths for correct slicing of
            # response-only fields (logp, topk_ids, teacher_logp, etc.).
            # These fields are concatenated across turns and have length
            # total_response_len, not full_seq_len.
            turn_resp_lens = [
                record.turn_response_ends[t] - record.turn_response_starts[t]
                for t in range(n_turns)
            ]
            resp_offset = 0  # running offset into response-only tensors

            for t in range(n_turns):
                end = record.turn_response_ends[t]
                resp_len = turn_resp_lens[t]

                # For individual-style, each turn needs its own prompt context.
                # Include all tokens from beginning to end of this turn's response.
                turn_seq_len = end
                turn_input_ids = full_input_ids[:turn_seq_len]
                turn_loss_mask = full_loss_mask[:turn_seq_len]
                turn_logprobs = full_logprobs[:turn_seq_len]
                turn_versions = full_versions[:turn_seq_len]
                turn_attention = full_attention[:turn_seq_len]

                turn_reward = (
                    record.turn_rewards[t]
                    if record.turn_rewards and t < len(record.turn_rewards)
                    else 0.0
                )
                turn_id = record.turn_ids[t] if t < len(record.turn_ids) else ""
                parent_turn_id = (
                    record.parent_turn_ids[t]
                    if record.parent_turn_ids and t < len(record.parent_turn_ids)
                    else None
                )

                traj = {
                    "input_ids": turn_input_ids.unsqueeze(0),
                    "loss_mask": turn_loss_mask.unsqueeze(0),
                    "logprobs": turn_logprobs.unsqueeze(0),
                    "versions": turn_versions.unsqueeze(0),
                    "attention_mask": turn_attention.unsqueeze(0),
                    "rewards": torch.tensor([turn_reward], dtype=torch.float32),
                    "_mcts_query_id": query_id,
                    "_mcts_seq_id": seq_id,
                    "_episode_idx": idx,
                    "_turn_idx_in_episode": t,
                    "_turn_id": turn_id,
                    "_parent_turn_id": parent_turn_id,
                    "_turn_reward": turn_reward,
                    "_outcome_reward": record.outcome_reward,
                    "_num_turns_in_episode": n_turns,
                }
                # Add new fields if present — slice by response offset, not
                # full-sequence position, since these are response-only.
                if full_logp is not None:
                    traj["logp"] = full_logp[
                        resp_offset : resp_offset + resp_len
                    ].unsqueeze(0)
                if full_topk_ids is not None:
                    traj["topk_ids"] = full_topk_ids[
                        resp_offset : resp_offset + resp_len
                    ].unsqueeze(0)
                if full_topk_logp is not None:
                    traj["topk_logp"] = full_topk_logp[
                        resp_offset : resp_offset + resp_len
                    ].unsqueeze(0)
                if full_distill_reward is not None:
                    traj["distill_reward"] = full_distill_reward[
                        resp_offset : resp_offset + resp_len
                    ].unsqueeze(0)
                if full_teacher_logp is not None:
                    traj["teacher_logp"] = full_teacher_logp[
                        resp_offset : resp_offset + resp_len
                    ].unsqueeze(0)
                result.append(traj)

                resp_offset += resp_len

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
        self._turn_nodes.clear()
        self._normalized_advantages.clear()
