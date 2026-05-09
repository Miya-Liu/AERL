# customized_areal/tree_search/advantage.py
from __future__ import annotations

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node

from areal.utils import logging

GRPO_NORM_EPS = 1e-8

logger = logging.getLogger("TreeAdvantageComputer")


class TreeAdvantageComputer:
    """Replace GAE advantages with tree Q-values from MCTS backup.

    Reads query_id and node_id from Node objects. Sets advantages
    and returns on the Node in-place.

    Supports per-query GRPO normalization: Q-values are normalized within
    each query group (all episodes for the same query), producing
    zero-mean unit-variance advantages.
    """

    def __init__(self, tree_store: MCTSTreeStore, grpo_eps: float = GRPO_NORM_EPS):
        self.tree_store = tree_store
        self.grpo_eps = grpo_eps

    @staticmethod
    def _get_query_id(traj: Node) -> str | None:
        """Extract query_id from Node."""
        return traj.query_id or None

    def compute(self, trajectories: list[Node]) -> None:
        """Replace GAE advantages with tree Q-values. Mutates Nodes in-place.

        Sets node.advantages (per-query GRPO-normalized Q-values) and
        node.returns (per-query GRPO-normalized outcome_rewards), both
        broadcast across response positions.

        Per-query GRPO normalization produces zero-mean unit-variance values
        within each query group for both advantages and returns.
        """
        # Collect unique (query_id → set of node_ids) and reward_per_node
        query_node_sets: dict[str, set[int]] = {}
        node_rewards: dict[int, float] = {}

        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue
            nset = query_node_sets.setdefault(query_id, set())

            node_id = getattr(traj, "node_id", None)
            if node_id is not None:
                nset.add(node_id)
                node_rewards[node_id] = traj.outcome_reward

        # Per-query GRPO normalization of Q-values for advantages
        for query_id, node_id_set in query_node_sets.items():
            node_ids = list(node_id_set)
            q_values = [self.tree_store.get_q_value(nid) for nid in node_ids]
            if len(q_values) < 2:
                logger.warning(
                    "Only %d sample(s) for query_id=%s — GRPO normalization "
                    "produces zero advantages (model will ignore this trajectory). "
                    "Consider increasing n_samples.",
                    len(q_values),
                    query_id,
                )
                for nid in node_ids:
                    self.tree_store.set_normalized_advantage(nid, 0.0)
                    self.tree_store.set_normalized_return(nid, 0.0)
                continue
            mean_q = sum(q_values) / len(q_values)
            var_q = sum((q - mean_q) ** 2 for q in q_values) / max(len(q_values), 1)
            std_q = var_q**0.5
            for nid, q in zip(node_ids, q_values):
                self.tree_store.set_normalized_advantage(
                    nid, (q - mean_q) / (std_q + self.grpo_eps)
                )

        # Per-query GRPO normalization of outcome_rewards for returns
        for query_id, node_id_set in query_node_sets.items():
            node_ids = list(node_id_set)
            rewards = [node_rewards[nid] for nid in node_ids]
            if len(rewards) < 2:
                for nid in node_ids:
                    self.tree_store.set_normalized_return(nid, 0.0)
                continue
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
            std_r = var_r**0.5
            for nid, r in zip(node_ids, rewards):
                self.tree_store.set_normalized_return(
                    nid, (r - mean_r) / (std_r + self.grpo_eps)
                )

        # Compute per-trajectory advantages and returns
        for traj in trajectories:
            query_id = self._get_query_id(traj)
            if query_id is None:
                continue

            node_id = getattr(traj, "node_id", None)
            if node_id is None:
                continue
            mask = self.tree_store.get_prompt_mask(query_id, node_id)
            norm_adv = self.tree_store.get_normalized_advantage(node_id)
            if not self.tree_store.has_normalized_advantage(node_id):
                q_val = self.tree_store.get_q_value(node_id)
                logger.warning(
                    "Normalized advantage missing for node_id=%d (query_id=%s), "
                    "falling back to raw Q-value=%.4f",
                    node_id,
                    query_id,
                    q_val,
                )
                norm_adv = q_val
            advantages = mask.float() * norm_adv
            traj.advantages = advantages
            norm_return = self.tree_store.get_normalized_return(node_id)
            traj.returns = mask.float() * norm_return
