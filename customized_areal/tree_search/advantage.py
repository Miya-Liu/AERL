# customized_areal/tree_search/advantage.py
from __future__ import annotations

from typing import Any

import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


class TreeAdvantageComputer:
    """Replace GAE advantages with tree Q-values from MCTS backup.

    For each trajectory, looks up the Q-values computed by MCTS backup
    and assigns them as advantages. Prompt tokens are zeroed out so that
    only response tokens carry the advantage signal.
    """

    def __init__(self, tree_store: MCTSTreeStore):
        self.tree_store = tree_store

    def _compute_single(
        self, traj: dict[str, Any], query_id: str, seq_id: int, seq_len: int
    ) -> torch.Tensor:
        """Compute tree Q-value advantages for a single sample."""
        tree_advantages = self.tree_store.get_advantages(query_id, seq_id)

        advantages = torch.zeros(seq_len, dtype=torch.float32)
        common_len = min(seq_len, tree_advantages.shape[0])
        advantages[:common_len] = tree_advantages[:common_len]

        prompt_mask = self.tree_store.get_prompt_mask(query_id, seq_id)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        common_mask_len = min(seq_len, prompt_mask.shape[0])
        mask[:common_mask_len] = prompt_mask[:common_mask_len]
        advantages = advantages * mask.float()

        return advantages

    def compute(self, trajectories: list[dict[str, Any]]) -> None:
        """Replace GAE advantages with tree Q-values. Mutates trajectories in-place.

        Handles both individual trajectory dicts (shape [1, seq_len]) and
        grouped trajectory dicts (shape [group_size, seq_len]). For grouped
        dicts, ``_mcts_seq_ids`` (list of seq_ids) is used to look up
        Q-values per sample.
        """
        for traj in trajectories:
            query_id = traj["_mcts_query_id"]
            input_ids = traj["input_ids"]

            if "_mcts_seq_ids" in traj:
                # Grouped trajectory — compute per-sample advantages
                seq_ids = traj["_mcts_seq_ids"]
                all_advantages = []
                for seq_id in seq_ids:
                    seq_len = input_ids.shape[1]
                    adv = self._compute_single(traj, query_id, seq_id, seq_len)
                    all_advantages.append(adv)
                advantages = torch.stack(all_advantages, dim=0)
            elif "_mcts_seq_id" in traj:
                # Single trajectory
                seq_id = traj["_mcts_seq_id"]
                seq_len = input_ids.shape[-1]
                advantages = self._compute_single(traj, query_id, seq_id, seq_len)
                if input_ids.dim() > 1:
                    advantages = advantages.unsqueeze(0)
            else:
                # No tree metadata — skip (shouldn't happen if insert_batch ran)
                continue

            traj["advantages"] = advantages
            traj["returns"] = advantages.clone()
