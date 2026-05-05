# customized_areal/tree_search/advantage.py
from __future__ import annotations

from typing import Any

import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore

GRPO_NORM_EPS = 1e-8


class TreeAdvantageComputer:
    """Replace GAE advantages with tree Q-values from MCTS backup.

    For each trajectory, looks up the Q-values computed by MCTS backup
    and assigns them as advantages. Prompt tokens are zeroed out so that
    only response tokens carry the advantage signal.

    Supports per-query GRPO normalization: Q-values are normalized within
    each query group (all episodes for the same query), producing
    zero-mean unit-variance advantages.
    """

    def __init__(self, tree_store: MCTSTreeStore, grpo_eps: float = GRPO_NORM_EPS):
        self.tree_store = tree_store
        self.grpo_eps = grpo_eps

    def _compute_single(
        self, traj: dict[str, Any], query_id: str, seq_id: int, seq_len: int
    ) -> torch.Tensor:
        """Compute tree Q-value advantages for a single sample.

        Uses normalized Q-value if available (from GRPO normalization),
        otherwise falls back to raw Q-value.
        """
        # Prefer normalized Q-value (from GRPO normalization)
        normalized_q = self.tree_store._normalized_advantages.get(seq_id)
        if normalized_q is None:
            # Fall back to raw Q-value for legacy trajectories
            normalized_q = self.tree_store._q_values.get(seq_id, 0.0)

        prompt_mask = self.tree_store.get_prompt_mask(query_id, seq_id)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        common_len = min(seq_len, prompt_mask.shape[0])
        mask[:common_len] = prompt_mask[:common_len]

        advantages = mask.float() * normalized_q
        return advantages

    @staticmethod
    def _get_seq_len(input_ids) -> int:
        """Get sequence length from tensor [..., seq_len] or list."""
        if isinstance(input_ids, list):
            return len(input_ids)
        return input_ids.shape[-1]

    def compute(self, trajectories: list[dict[str, Any]]) -> None:
        """Replace GAE advantages with tree Q-values. Mutates trajectories in-place.

        Handles list-based trajectory dicts (from new rollouts), tensor
        trajectory dicts with shape [1, seq_len], and grouped tensor dicts
        with shape [group_size, seq_len]. For grouped dicts,
        ``_mcts_seq_ids`` (list of seq_ids) is used to look up Q-values per
        sample.

        After inserting all trajectories, performs per-query GRPO normalization
        of Q-values so that episodes within the same query group have
        zero-mean unit-variance advantages.
        """
        # Collect unique (query_id, seq_id) pairs for GRPO normalization.
        # Use dict to deduplicate: per-turn cached trajectories may share
        # the same seq_id across turns of the same episode.
        query_seq_sets: dict[str, dict[int, None]] = {}

        for traj in trajectories:
            query_id = traj.get("_mcts_query_id")
            if query_id is None:
                continue
            qset = query_seq_sets.setdefault(query_id, {})
            if "_mcts_seq_ids" in traj:
                for seq_id in traj["_mcts_seq_ids"]:
                    qset[seq_id] = None
            elif "_mcts_seq_id" in traj:
                qset[traj["_mcts_seq_id"]] = None

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
            query_id = traj.get("_mcts_query_id")
            if query_id is None:
                continue
            input_ids = traj["input_ids"]
            seq_len = self._get_seq_len(input_ids)

            if "_mcts_seq_ids" in traj:
                seq_ids = traj["_mcts_seq_ids"]
                all_advantages = []
                for seq_id in seq_ids:
                    adv = self._compute_single(traj, query_id, seq_id, seq_len)
                    all_advantages.append(adv)
                advantages = torch.stack(all_advantages, dim=0)
            elif "_mcts_seq_id" in traj:
                seq_id = traj["_mcts_seq_id"]
                advantages = self._compute_single(traj, query_id, seq_id, seq_len)
                advantages = advantages.unsqueeze(0)  # [1, seq_len]
            else:
                continue

            traj["advantages"] = advantages
            traj["returns"] = advantages.clone()
