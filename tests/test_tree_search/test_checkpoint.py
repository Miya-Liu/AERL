import json
import os
import pytest
from customized_areal.tree_search.trie_node import TrieNode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.turn_splitter import Turn


def _simple_splitter(input_ids: list[int]) -> list[Turn]:
    try:
        split_pos = input_ids.index(10)
        return [Turn(prompt_tokens=input_ids[:split_pos], response_tokens=input_ids[split_pos:])]
    except ValueError:
        return [Turn(prompt_tokens=[], response_tokens=list(input_ids))]


class TestTreeCheckpointManager:
    def test_save_and_load(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=2.0)
        store.insert_trajectory("q1", [1, 2, 10, 5, 6], reward=0.0)
        store.insert_trajectory("q2", [7, 8], reward=1.0)

        manager.save(store)
        assert manager.exists()

        loaded = manager.load(_simple_splitter)
        assert len(loaded.trees) == 2
        assert "q1" in loaded.trees
        assert "q2" in loaded.trees

    def test_exists_false_when_no_dir(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path / "nonexistent"))
        assert not manager.exists()

    def test_save_creates_directory(self, tmp_path):
        save_dir = str(tmp_path / "new_dir")
        manager = TreeCheckpointManager(save_dir)
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2], reward=1.0)
        manager.save(store)
        assert os.path.isdir(os.path.join(save_dir, "mcts_trees"))

    def test_load_preserves_seq_id_counter(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2], reward=1.0)
        store.insert_trajectory("q2", [3, 4], reward=0.5)
        manager.save(store)

        loaded = manager.load(_simple_splitter)
        seq_id = loaded.insert_trajectory("q3", [5, 6], reward=1.0)
        assert seq_id == 2
