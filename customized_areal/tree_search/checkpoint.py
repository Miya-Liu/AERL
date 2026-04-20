from __future__ import annotations

import json
import os
from collections.abc import Callable

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.trie_node import TrieNode
from customized_areal.tree_search.turn_splitter import Turn


class TreeCheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = os.path.join(save_dir, "mcts_trees")

    def exists(self) -> bool:
        return os.path.isdir(self.save_dir) and os.path.isfile(
            os.path.join(self.save_dir, "metadata.json")
        )

    def save(self, tree_store: MCTSTreeStore) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        for query_id, root in tree_store.trees.items():
            tree_data = {"root": self._serialize_node(root)}
            filepath = os.path.join(self.save_dir, f"query_{query_id}.json")
            with open(filepath, "w") as f:
                json.dump(tree_data, f)

        # Serialize trained flags and rewards
        trained_data = {
            f"{qid}:{sid}": trained
            for (qid, sid), trained in tree_store._trained.items()
        }
        rewards_data = {
            f"{qid}:{sid}": reward for (qid, sid), reward in tree_store._rewards.items()
        }

        metadata = {
            "next_seq_id": tree_store._next_seq_id,
            "trained": trained_data,
            "rewards": rewards_data,
        }
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load(self, turn_splitter: Callable[[list[int]], list[Turn]]) -> MCTSTreeStore:
        store = MCTSTreeStore(turn_splitter)
        with open(os.path.join(self.save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        store._next_seq_id = metadata["next_seq_id"]
        for filename in os.listdir(self.save_dir):
            if not filename.startswith("query_") or not filename.endswith(".json"):
                continue
            query_id = filename[len("query_") : -len(".json")]
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath) as f:
                tree_data = json.load(f)
            root = self._deserialize_node(
                tree_data["root"], parent=None, tree_id=len(store.trees)
            )
            root.sequence_ids = list(root.sequence_ids)
            store.trees[query_id] = root
        return store

    def _serialize_node(self, node: TrieNode) -> dict:
        result = {
            "tree_id": node.tree_id,
            "start_idx": node.start_idx,
            "end_idx": node.end_idx,
            "tokens": node.tokens,
            "sequence_ids": list(node.sequence_ids),
            "children": {
                str(key): self._serialize_node(child)
                for key, child in node.children.items()
            },
        }
        if node.prompt_len > 0:
            result["prompt_len"] = node.prompt_len
        if node.logprobs:
            result["logprobs"] = node.logprobs
        if node.versions:
            result["versions"] = node.versions
        return result

    def _deserialize_node(
        self, data: dict, parent: TrieNode | None, tree_id: int
    ) -> TrieNode:
        node = TrieNode(
            tree_id=tree_id,
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            tokens=data["tokens"],
            sequence_ids=data["sequence_ids"],
            prompt_len=data.get("prompt_len", 0),
            logprobs=data.get("logprobs", []),
            versions=data.get("versions", []),
        )
        if parent is not None:
            node.ancestors = parent.ancestors + [parent]
        for key_str, child_data in data["children"].items():
            key = int(key_str)
            child = self._deserialize_node(child_data, parent=node, tree_id=tree_id)
            node.children[key] = child
        return node
