"""Checkpoint save/load for the flat TrajectoryRecord store.

Unlike the old TrieNode-based format, MCTS stats are keyed by seq_id (int)
and serialize directly — no rebuild_mcts_stats() needed after loading.
Old TrieNode-based checkpoints are incompatible and must be discarded.
"""

from __future__ import annotations

import json
import os

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, TrajectoryRecord


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
            "next_seq_id": tree_store._next_seq_id,
            "seq_id_to_key": {
                str(k): [v[0], v[1]] for k, v in tree_store._seq_id_to_key.items()
            },
            "query_seq_ids": {k: v for k, v in tree_store._query_seq_ids.items()},
            "visit_counts": {str(k): v for k, v in tree_store._visit_counts.items()},
            "total_values": {str(k): v for k, v in tree_store._total_values.items()},
            "q_values": {str(k): v for k, v in tree_store._q_values.items()},
            "trained": {str(k): v for k, v in tree_store._trained.items()},
            "rewards": {str(k): v for k, v in tree_store._rewards.items()},
        }
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load(self) -> MCTSTreeStore:
        store = MCTSTreeStore()

        with open(os.path.join(self.save_dir, "metadata.json")) as f:
            metadata = json.load(f)

        store._next_seq_id = metadata["next_seq_id"]
        store._seq_id_to_key = {
            int(k): (v[0], v[1]) for k, v in metadata["seq_id_to_key"].items()
        }
        store._query_seq_ids = metadata["query_seq_ids"]
        store._visit_counts = {int(k): v for k, v in metadata["visit_counts"].items()}
        store._total_values = {int(k): v for k, v in metadata["total_values"].items()}
        store._q_values = {int(k): v for k, v in metadata["q_values"].items()}
        store._trained = {int(k): v for k, v in metadata["trained"].items()}
        store._rewards = {int(k): v for k, v in metadata["rewards"].items()}

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
    def _serialize_record(record: TrajectoryRecord) -> dict:
        return {
            "input_ids": record.input_ids,
            "loss_mask": record.loss_mask,
            "logprobs": record.logprobs,
            "versions": record.versions,
            "reward": record.reward,
            "turn_response_starts": record.turn_response_starts,
            "turn_response_ends": record.turn_response_ends,
        }

    @staticmethod
    def _deserialize_record(data: dict) -> TrajectoryRecord:
        return TrajectoryRecord(
            input_ids=data["input_ids"],
            loss_mask=data["loss_mask"],
            logprobs=data["logprobs"],
            versions=data["versions"],
            reward=data["reward"],
            turn_response_starts=data["turn_response_starts"],
            turn_response_ends=data["turn_response_ends"],
        )
