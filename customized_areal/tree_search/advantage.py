# customized_areal/tree_search/advantage.py
from __future__ import annotations

from typing import Any

import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node

GRPO_NORM_EPS = 1e-8


class TreeAdvantageComputer:
    """Replace GAE advantages with tree Q-values from MCTS backup.

    Works with both Node objects and legacy trajectory dicts. For Nodes,
    reads query_id and node_id attributes. Sets advantages
    and returns on the object in-place.

    Supports per-query GRPO normalization: Q-values are normalized within
    each query group (all episodes for the same query), producing
    zero-mean unit-variance advantages.
    """

    def __init__(self, tree_store: MCTSTreeStore, grpo_eps: float = GRPO_NORM_EPS):
        self.tree_store = tree_store
        self.grpo_eps = grpo_eps

    def _compute_single(self, seq_id: int, seq_len: int, query_id: str) -> torch.Tensor:
        """Compute tree Q-value advantages for a single sample."""
        normalized_q = self.tree_store._normalized_advantages.get(seq_id)
        if normalized_q is None:
            normalized_q = self.tree_store._q_values.get(seq_id, 0.0)

        prompt_mask = self.tree_store.get_prompt_mask(query_id, seq_id)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        common_len = min(seq_len, prompt_mask.shape[0])
        mask[:common_len] = prompt_mask[:common_len]

        return mask.float() * normalized_q

    @staticmethod
    def _get_query_id(traj: Any) -> str | None:
        """Extract query_id from Node or dict."""
        if isinstance(traj, Node):
            return getattr(traj, "query_id", None)
        if isinstance(traj, dict):
            return traj.get("query_id")
        return None

    @staticmethod
    def _get_seq_len(traj: Any) -> int:
        """Get sequence length from Node, tensor, or list."""
        if isinstance(traj, Node):
            return len(traj.input_ids)
        input_ids = traj["input_ids"]
        if isinstance(input_ids, list):
            return len(input_ids)
        return input_ids.shape[-1]

    def compute(self, trajectories: list[Any]) -> None:
        """Replace GAE advantages with tree Q-values. Mutates objects in-place.

        Handles Node objects (sets node.advantages/node.returns via
        object.__setattr__) and legacy trajectory dicts (sets
        traj["advantages"]/traj["returns"]).

        After inserting all trajectories, performs per-query GRPO normalization
        of Q-values so that episodes within the same query group have
        zero-mean unit-variance advantages.
        """
        # Collect unique (query_id, seq_id) pairs for GRPO normalization.
        query_seq_sets: dict[str, dict[int, None]] = {}

        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            qset = query_seq_sets.setdefault(query_id, {})

            if isinstance(traj, Node):
                seq_id = getattr(traj, "node_id", None)
                if seq_id is not None:
                    qset[seq_id] = None
            elif isinstance(traj, dict):
                if "node_ids" in traj:
                    for seq_id in traj["node_ids"]:
                        qset[seq_id] = None
                elif "node_id" in traj:
                    qset[traj["node_id"]] = None

        # Per-query GRPO normalization (deduplicated seq_ids)
        for query_id, seq_id_set in query_seq_sets.items():
            seq_ids = list(seq_id_set)
            q_values = [self.tree_store._rewards.get(sid, 0.0) for sid in seq_ids]
            if len(q_values) < 2:
                if seq_ids:
                    self.tree_store._normalized_advantages[seq_ids[0]] = q_values[0]
                continue
            mean_q = sum(q_values) / len(q_values)
            var_q = sum((q - mean_q) ** 2 for q in q_values) / len(q_values)
            std_q = var_q**0.5
            for sid, q in zip(seq_ids, q_values):
                self.tree_store._normalized_advantages[sid] = (q - mean_q) / (
                    std_q + self.grpo_eps
                )

        # Compute per-trajectory advantages using normalized Q-values
        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue

            if isinstance(traj, Node):
                seq_id = getattr(traj, "node_id", None)
                if seq_id is None:
                    continue
                seq_len = len(traj.input_ids)
                advantages = self._compute_single(seq_id, seq_len, query_id)
                object.__setattr__(traj, "advantages", advantages)
                object.__setattr__(traj, "returns", advantages.clone())
            elif isinstance(traj, dict):
                seq_len = self._get_seq_len(traj)
                if "node_ids" in traj:
                    seq_ids = traj["node_ids"]
                    all_advantages = []
                    for seq_id in seq_ids:
                        adv = self._compute_single(seq_id, seq_len, query_id)
                        all_advantages.append(adv)
                    advantages = torch.stack(all_advantages, dim=0)
                elif "node_id" in traj:
                    seq_id = traj["node_id"]
                    advantages = self._compute_single(seq_id, seq_len, query_id)
                    advantages = advantages.unsqueeze(0)
                else:
                    continue
                traj["advantages"] = advantages
                traj["returns"] = advantages.clone()
