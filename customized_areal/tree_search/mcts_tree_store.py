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


def _response_span(loss_mask: list[int]) -> tuple[int, int]:
    """Return (start, end) of the first response region in loss_mask."""
    starts, ends = _find_turn_boundaries(loss_mask)
    if starts:
        return starts[0], ends[0]
    return 0, 0


def _node_to_tensor_dict(node: Node, query_id: str, seq_id: int) -> dict[str, Any]:
    """Convert a single Node to a tensor dict with shape [1, seq_len]."""
    seq_len = len(node.input_ids)
    traj: dict[str, Any] = {
        "input_ids": torch.tensor(node.input_ids, dtype=torch.int32).unsqueeze(0),
        "loss_mask": torch.tensor(node.loss_mask, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(node.logprobs, dtype=torch.float32).unsqueeze(0),
        "versions": torch.tensor(node.versions, dtype=torch.int32).unsqueeze(0),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "rewards": torch.tensor([node.outcome_reward], dtype=torch.float32).unsqueeze(0),
        "_mcts_query_id": query_id,
        "_mcts_seq_id": seq_id,
    }
    # Response-only fields: extract response portion from full sequence
    resp_start, resp_end = _response_span(node.loss_mask)
    if node.topk_ids is not None:
        traj["topk_ids"] = torch.tensor(node.topk_ids, dtype=torch.int32).unsqueeze(0)
    if node.topk_logp is not None:
        traj["topk_logp"] = torch.tensor(node.topk_logp, dtype=torch.float32).unsqueeze(0)
    if node.distill_reward is not None:
        traj["distill_reward"] = torch.tensor(node.distill_reward, dtype=torch.float32).unsqueeze(0)
    if node.teacher_logp is not None:
        traj["teacher_logp"] = torch.tensor(node.teacher_logp, dtype=torch.float32).unsqueeze(0)
    # Derived from logprobs for response tokens
    if resp_end > resp_start:
        traj["logp"] = torch.tensor(
            node.logprobs[resp_start:resp_end], dtype=torch.float32
        ).unsqueeze(0)
    # Carry advantages if set by advantage computer
    if hasattr(node, 'advantages') and node.advantages is not None:
        traj["advantages"] = node.advantages.unsqueeze(0) if node.advantages.dim() == 1 else node.advantages
    if hasattr(node, 'returns') and node.returns is not None:
        traj["returns"] = node.returns.unsqueeze(0) if node.returns.dim() == 1 else node.returns
    # Turn metadata
    if node.node_id:
        traj["_turn_id"] = node.node_id
    if node.parent_node_id is not None:
        traj["_parent_turn_id"] = node.parent_node_id
    traj["_turn_reward"] = node.outcome_reward
    traj["_outcome_reward"] = node.outcome_reward
    traj["_episode_idx"] = 0
    traj["_turn_idx_in_episode"] = 0
    traj["_num_turns_in_episode"] = 1
    return traj


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
        input_ids = traj["input_ids"]
        loss_mask = traj["loss_mask"]
        outcome_reward = traj.get("outcome_reward", traj.get("reward", 0.0))
        logprobs = traj.get("logprobs", [0.0] * len(input_ids))
        versions = traj.get("versions", [0] * len(input_ids))

        # New fields
        topk_ids = traj.get("topk_ids")
        topk_logp = traj.get("topk_logp")
        distill_reward = traj.get("distill_reward")
        teacher_logp = traj.get("teacher_logp")

        query_id = traj.get("_mcts_query_id", "")

        node = Node(
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            outcome_reward=outcome_reward,
            node_id=traj.get("node_id", traj.get("turn_ids", [""])[0] if traj.get("turn_ids") else ""),
            parent_node_id=traj.get("parent_node_id", traj.get("parent_turn_ids", [None])[0] if traj.get("parent_turn_ids") else None),
            episode_id=traj.get("episode_id", ""),
            topk_ids=topk_ids,
            topk_logp=topk_logp,
            distill_reward=distill_reward,
            teacher_logp=teacher_logp,
        )

        seq_id = self._insert_single(query_id, node)
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

        # Set metadata directly on the Node (dataclass permits non-field attrs when not frozen)
        object.__setattr__(node, '_mcts_seq_id', seq_id)
        object.__setattr__(node, '_mcts_query_id', query_id)

        self._backup(seq_id, node.outcome_reward)
        self._trained[seq_id] = False
        self._rewards[seq_id] = node.outcome_reward

        # Register node_id → seq_id mapping for shared-node MCTS
        if node.node_id and node.node_id not in self._turn_nodes:
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
        """Insert per-turn dicts as individual Node objects.

        Each turn dict becomes a separate Node in the tree store.
        Turns from the same episode share the same episode_id (derived
        from _mcts_query_id + _episode_idx) and are linked via
        node_id/parent_node_id.
        """
        from itertools import groupby

        sorted_turns = sorted(
            turn_dicts,
            key=lambda d: (d.get("_mcts_query_id", ""), d.get("_episode_idx", 0)),
        )

        for (query_id, ep_idx), group_iter in groupby(
            sorted_turns,
            key=lambda d: (d.get("_mcts_query_id", ""), d.get("_episode_idx", 0)),
        ):
            turns = list(group_iter)
            turns.sort(key=lambda d: d.get("_turn_idx_in_episode", 0))

            episode_id = f"{query_id}_{ep_idx}"

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

                node = Node(
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    logprobs=logprobs,
                    versions=versions,
                    outcome_reward=turn.get("_outcome_reward", 0.0),
                    node_id=turn.get("_turn_id", ""),
                    parent_node_id=turn.get("_parent_turn_id"),
                    episode_id=episode_id,
                )

                seq_id = self._insert_single(query_id, node)
                # Also set metadata on dict for downstream dict-key readers
                turn["_mcts_seq_id"] = seq_id
                turn["_mcts_query_id"] = query_id

    def get_advantages(self, query_id: str, seq_id: int) -> torch.Tensor:
        """Return per-token advantages: Q-value on response tokens, 0 on prompt."""
        qid, idx = self._seq_id_to_key[seq_id]
        node = self.trajectories[qid][idx]
        q_val = self._q_values.get(seq_id, 0.0)
        seq_len = len(node.input_ids)
        advantages = torch.zeros(seq_len, dtype=torch.float32)
        starts, ends = _find_turn_boundaries(node.loss_mask)
        for start, end in zip(starts, ends):
            advantages[start:end] = q_val
        return advantages

    def get_prompt_mask(self, query_id: str, seq_id: int) -> torch.Tensor:
        """Return boolean mask: True for response tokens, False for prompt."""
        qid, idx = self._seq_id_to_key[seq_id]
        node = self.trajectories[qid][idx]
        return torch.tensor(node.loss_mask, dtype=torch.bool)

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
        self, query_id: str, n_samples: int
    ) -> list[Node]:
        """Load untrained trajectories as Node objects.

        Returns per-turn Node objects. Callers can:
        - Read Node attributes directly for advantage computation
        - Convert to tensor dicts via _node_to_tensor_dict() for training

        Each Node carries _mcts_query_id and _mcts_seq_id set during
        insertion (accessible as regular attributes).
        """
        if query_id not in self.trajectories:
            return []

        untrained_ids = self.get_untrained_seq_ids(query_id, n_samples)
        result: list[Node] = []
        for seq_id in untrained_ids:
            qid, idx = self._seq_id_to_key[seq_id]
            node = self.trajectories[qid][idx]
            result.append(node)
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
