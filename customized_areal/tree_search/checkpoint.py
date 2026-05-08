"""Checkpoint save/load for the flat Node store.

Unlike the old TrieNode-based format, MCTS stats are keyed by seq_id (int)
and serialize directly — no rebuild_mcts_stats() needed after loading.
Old TrieNode-based checkpoints are incompatible and must be discarded.
"""

from __future__ import annotations

import json
import os

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node


class TreeCheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = os.path.join(save_dir, "mcts_trees")

    def exists(self) -> bool:
        return os.path.isdir(self.save_dir) and os.path.isfile(
            os.path.join(self.save_dir, "metadata.json")
        )

    def save(self, tree_store: MCTSTreeStore) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        # Save per-query trajectory records
        for query_id, records in tree_store.trajectories.items():
            data = {"records": [self._serialize_record(r) for r in records]}
            filepath = os.path.join(self.save_dir, f"query_{query_id}.json")
            with open(filepath, "w") as f:
                json.dump(data, f)

        # Save metadata (indices, stats, tracking)
        metadata = {
            "next_node_id": tree_store._next_node_id,
            "node_id_to_key": {
                str(k): [v[0], v[1]] for k, v in tree_store._node_id_to_key.items()
            },
            "query_node_ids": {k: v for k, v in tree_store._query_node_ids.items()},
            "visit_counts": {str(k): v for k, v in tree_store._visit_counts.items()},
            "total_values": {str(k): v for k, v in tree_store._total_values.items()},
            "q_values": {str(k): v for k, v in tree_store._q_values.items()},
            "trained": {str(k): v for k, v in tree_store._trained.items()},
            "rewards": {str(k): v for k, v in tree_store._rewards.items()},
            "normalized_advantages": {
                str(k): v for k, v in tree_store._normalized_advantages.items()
            },
            "turn_nodes": tree_store._turn_nodes,
        }
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load(self) -> MCTSTreeStore:
        store = MCTSTreeStore()

        with open(os.path.join(self.save_dir, "metadata.json")) as f:
            metadata = json.load(f)

        store._next_node_id = metadata.get(
            "next_node_id", metadata.get("next_seq_id", 0)
        )
        node_id_to_key_raw = metadata.get(
            "node_id_to_key", metadata.get("seq_id_to_key", {})
        )
        store._node_id_to_key = {
            int(k): (v[0], v[1]) for k, v in node_id_to_key_raw.items()
        }
        store._query_node_ids = metadata.get(
            "query_node_ids", metadata.get("query_seq_ids", {})
        )
        store._visit_counts = {
            int(k): v for k, v in metadata.get("visit_counts", {}).items()
        }
        store._total_values = {
            int(k): v for k, v in metadata.get("total_values", {}).items()
        }
        store._q_values = {int(k): v for k, v in metadata.get("q_values", {}).items()}
        store._trained = {int(k): v for k, v in metadata.get("trained", {}).items()}
        store._rewards = {int(k): v for k, v in metadata.get("rewards", {}).items()}
        store._normalized_advantages = {
            int(k): v for k, v in metadata.get("normalized_advantages", {}).items()
        }
        store._turn_nodes = metadata.get("turn_nodes", {})

        # Load per-query trajectory records
        for filename in os.listdir(self.save_dir):
            if not filename.startswith("query_") or not filename.endswith(".json"):
                continue
            query_id = filename[len("query_") : -len(".json")]
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath) as f:
                data = json.load(f)
            store.trajectories[query_id] = [
                self._deserialize_record(r) for r in data["records"]
            ]

        return store

    @staticmethod
    def _serialize_record(node: Node) -> dict:
        data = {
            "input_ids": node.input_ids,
            "loss_mask": node.loss_mask,
            "logprobs": node.logprobs,
            "versions": node.versions,
            "outcome_reward": node.outcome_reward,
            "node_id": node.node_id,
            "parent_node_id": node.parent_node_id,
            "episode_id": node.episode_id,
            "query_id": node.query_id,
        }
        if node.topk_ids is not None:
            data["topk_ids"] = node.topk_ids
        if node.topk_logp is not None:
            data["topk_logp"] = node.topk_logp
        if node.distill_reward is not None:
            data["distill_reward"] = node.distill_reward
        if node.teacher_logp is not None:
            data["teacher_logp"] = node.teacher_logp
        return data

    @staticmethod
    def _deserialize_record(data: dict) -> Node:
        return Node(
            input_ids=data["input_ids"],
            loss_mask=data["loss_mask"],
            logprobs=data["logprobs"],
            versions=data["versions"],
            outcome_reward=data.get("outcome_reward", data.get("reward", 0.0)),
            node_id=data.get("node_id", 0),
            parent_node_id=data.get("parent_node_id"),
            episode_id=data.get("episode_id", ""),
            query_id=data.get("query_id", ""),
            topk_ids=data.get("topk_ids"),
            topk_logp=data.get("topk_logp"),
            distill_reward=data.get("distill_reward"),
            teacher_logp=data.get("teacher_logp"),
        )
