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

    def compute(self, trajectories: list[dict[str, Any]]) -> None:
        """Replace GAE advantages with tree Q-values. Mutates trajectories in-place."""
        for traj in trajectories:
            query_id = traj["_mcts_query_id"]
            seq_id = traj["_mcts_seq_id"]
            tree_advantages = self.tree_store.get_advantages(query_id, seq_id)

            input_ids = traj["input_ids"]
            seq_len = input_ids.shape[0]

            # Pad or trim tree_advantages to match trajectory length
            advantages = torch.zeros(seq_len, dtype=torch.float32)
            common_len = min(seq_len, tree_advantages.shape[0])
            advantages[:common_len] = tree_advantages[:common_len]

            # Mask prompt tokens -- advantages only for response tokens
            prompt_mask = self.tree_store.get_prompt_mask(query_id, seq_id)
            # Trim or pad prompt_mask to match trajectory length
            mask = torch.zeros(seq_len, dtype=torch.bool)
            common_mask_len = min(seq_len, prompt_mask.shape[0])
            mask[:common_mask_len] = prompt_mask[:common_mask_len]
            advantages = advantages * mask.float()

            # Match trajectory shape: [1, seq_len] or [group_size, seq_len]
            if input_ids.dim() > 1:
                advantages = advantages.unsqueeze(0)

            traj["advantages"] = advantages
            traj["returns"] = advantages.clone()
