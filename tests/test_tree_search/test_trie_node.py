import pytest

from customized_areal.tree_search.trie_node import TrieNode
from customized_areal.tree_search.turn_splitter import Turn


class TestTrieNodeCreation:
    def test_root_node(self):
        root = TrieNode(tree_id=0)
        assert root.tree_id == 0
        assert root.tokens == []
        assert root.start_idx == -1
        assert root.end_idx == -1
        assert root.children == {}
        assert root.sequence_ids == []
        assert root.ancestors == []
        assert root.nodes == []

    def test_child_node(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        child = root.add_turn(turn, seq_id=0)
        assert child.tokens == [1, 2, 3, 4]
        assert child.ancestors == [root]
        assert 3 in root.children
        assert root.children[3] is child


class TestTrieNodeAddTurn:
    def test_add_single_turn(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2, 3], response_tokens=[4, 5])
        child = root.add_turn(turn, seq_id=0)
        assert 4 in root.children
        assert root.children[4] is child
        assert child.tokens == [1, 2, 3, 4, 5]
        assert 0 in child.sequence_ids
        assert 0 in root.sequence_ids

    def test_add_two_turns_sequentially(self):
        root = TrieNode(tree_id=0)
        turn1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        turn2 = Turn(prompt_tokens=[5, 6], response_tokens=[7, 8])
        child1 = root.add_turn(turn1, seq_id=0)
        child2 = child1.add_turn(turn2, seq_id=0)
        assert 3 in root.children
        assert 7 in child1.children
        assert child2.tokens == [5, 6, 7, 8]
        assert 0 in child2.sequence_ids

    def test_add_turn_shared_prefix_diverges(self):
        root = TrieNode(tree_id=0)
        turn_a1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        turn_b1 = Turn(prompt_tokens=[1, 2], response_tokens=[5, 6])
        child_a = root.add_turn(turn_a1, seq_id=0)
        child_b = root.add_turn(turn_b1, seq_id=1)
        assert 3 in root.children
        assert 5 in root.children
        assert child_a is root.children[3]
        assert child_b is root.children[5]
        assert 0 in child_a.sequence_ids
        assert 1 in child_b.sequence_ids

    def test_add_turn_empty_response_raises(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[])
        with pytest.raises(ValueError, match="response_tokens must not be empty"):
            root.add_turn(turn, seq_id=0)

    def test_add_turn_existing_child_reuses(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4, 5])
        child1 = root.add_turn(turn, seq_id=0)
        child2 = root.add_turn(turn, seq_id=1)
        assert child1 is child2
        assert 0 in child1.sequence_ids
        assert 1 in child1.sequence_ids


class TestTrieNodeGetPathNodes:
    def test_get_path_single_turn(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        root.add_turn(turn, seq_id=0)
        path = root.get_path_nodes(seq_id=0)
        assert len(path) == 1
        assert path[0].tokens == [1, 2, 3, 4]

    def test_get_path_multiple_turns(self):
        root = TrieNode(tree_id=0)
        turn1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        turn2 = Turn(prompt_tokens=[5, 6], response_tokens=[7, 8])
        child1 = root.add_turn(turn1, seq_id=0)
        child1.add_turn(turn2, seq_id=0)
        path = root.get_path_nodes(seq_id=0)
        assert len(path) == 2
        assert path[0].tokens == [1, 2, 3, 4]
        assert path[1].tokens == [5, 6, 7, 8]

    def test_get_path_missing_seq_id_raises(self):
        root = TrieNode(tree_id=0)
        with pytest.raises(KeyError):
            root.get_path_nodes(seq_id=99)


class TestTrieNodeGetTurnBoundaries:
    def test_get_turn_boundaries(self):
        root = TrieNode(tree_id=0)
        turn1 = Turn(prompt_tokens=[1, 2, 3], response_tokens=[4, 5])
        turn2 = Turn(prompt_tokens=[6, 7], response_tokens=[8])
        child1 = root.add_turn(turn1, seq_id=0)
        child1.add_turn(turn2, seq_id=0)
        boundaries = root.get_turn_boundaries(seq_id=0)
        assert boundaries == [0, 5, 8]

    def test_get_turn_boundaries_single_turn(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        root.add_turn(turn, seq_id=0)
        boundaries = root.get_turn_boundaries(seq_id=0)
        assert boundaries == [0, 4]

    def test_get_turn_boundaries_missing_seq_id_raises(self):
        root = TrieNode(tree_id=0)
        with pytest.raises(KeyError):
            root.get_turn_boundaries(seq_id=99)
