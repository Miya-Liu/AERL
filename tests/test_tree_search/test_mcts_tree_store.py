import torch
import pytest
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.turn_splitter import Turn


def _two_turn_splitter(input_ids: list[int]) -> list[Turn]:
    """Simple splitter for testing: splits at token 10, first half is prompt, second is response."""
    try:
        split_pos = input_ids.index(10)
        return [Turn(prompt_tokens=input_ids[:split_pos], response_tokens=input_ids[split_pos:])]
    except ValueError:
        return [Turn(prompt_tokens=[], response_tokens=list(input_ids))]


class TestMCTSTreeStoreStartSequence:
    def test_start_sequence_creates_root(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        assert seq_id == 0
        assert "q1" in store.trees
        root = store.trees["q1"]
        assert root.tokens == []
        assert root.tree_id == 0

    def test_start_sequence_increments_seq_id(self):
        store = MCTSTreeStore(_two_turn_splitter)
        id0 = store.start_sequence("q1")
        id1 = store.start_sequence("q1")
        id2 = store.start_sequence("q2")
        assert id0 == 0
        assert id1 == 1
        assert id2 == 2

    def test_start_sequence_sets_cursor(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        assert ("q1", seq_id) in store._cursors
        assert store._cursors[("q1", seq_id)] is store.trees["q1"]


class TestMCTSTreeStoreAddTurn:
    def test_add_single_turn(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn = Turn(prompt_tokens=[1, 2, 10], response_tokens=[3, 4])
        store.add_turn("q1", seq_id, turn)
        root = store.trees["q1"]
        assert 3 in root.children
        child = root.children[3]
        assert child.tokens == [1, 2, 10, 3, 4]
        assert 0 in child.sequence_ids

    def test_add_two_turns(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        turn2 = Turn(prompt_tokens=[5, 6], response_tokens=[7, 8])
        store.add_turn("q1", seq_id, turn1)
        store.add_turn("q1", seq_id, turn2)
        root = store.trees["q1"]
        child1 = root.children[3]
        child2 = child1.children[7]
        assert child2.tokens == [5, 6, 7, 8]

    def test_add_turn_advances_cursor(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        store.add_turn("q1", seq_id, turn1)
        root = store.trees["q1"]
        cursor = store._cursors[("q1", seq_id)]
        assert cursor is root.children[3]


class TestMCTSTreeStoreFinishSequence:
    def test_finish_sequence_runs_backup(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        store.add_turn("q1", seq_id, turn)
        store.finish_sequence("q1", seq_id, reward=1.0)
        root = store.trees["q1"]
        child = root.children[3]
        assert store._visit_counts[("q1", id(root))] == 1
        assert store._q_values[("q1", id(root))] == 1.0
        assert store._visit_counts[("q1", id(child))] == 1
        assert store._q_values[("q1", id(child))] == 1.0

    def test_finish_sequence_clears_cursor(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        store.add_turn("q1", seq_id, turn)
        store.finish_sequence("q1", seq_id, reward=1.0)
        assert ("q1", seq_id) not in store._cursors


class TestMCTSTreeStoreInsertTrajectory:
    def test_insert_trajectory_convenience(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        root = store.trees["q1"]
        # _two_turn_splitter splits at token 10: prompt=[1,2], response=[10,3,4]
        # first response token is 10
        assert 10 in root.children
        assert store._visit_counts[("q1", id(root))] == 1

    def test_insert_two_trajectories_shared_prefix(self):
        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        root = store.trees["q1"]
        assert 0 in root.sequence_ids
        assert 1 in root.sequence_ids


class TestMCTSTreeStoreInsertBatch:
    def test_insert_batch(self):
        store = MCTSTreeStore(_two_turn_splitter)
        trajectories = [
            {
                "input_ids": torch.tensor([1, 2, 10, 3, 4]),
                "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
                "rewards": torch.tensor([1.0]),
            },
            {
                "input_ids": torch.tensor([1, 2, 10, 3, 5]),
                "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
                "rewards": torch.tensor([0.5]),
            },
        ]
        store.insert_batch(trajectories)
        assert "_mcts_seq_id" in trajectories[0]
        assert "_mcts_query_id" in trajectories[0]


class TestMCTSTreeStoreGetAdvantages:
    def test_get_advantages_single_trajectory(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=2.0)
        advantages = store.get_advantages("q1", seq_id)
        # _two_turn_splitter: prompt=[1,2], response=[10,3,4] -> combined [1,2,10,3,4] = 5 tokens
        assert advantages.shape == torch.Size([5])
        assert torch.allclose(advantages, torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]))


class TestMCTSTreeStoreClear:
    def test_clear(self):
        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2], reward=1.0)
        store.clear()
        assert len(store.trees) == 0
        assert store._next_seq_id == 0
        assert len(store._cursors) == 0
        assert len(store._visit_counts) == 0
        assert len(store._q_values) == 0


class TestTrieNodeExtendedFields:
    def test_add_turn_stores_logprobs_and_versions(self):
        from customized_areal.tree_search.trie_node import TrieNode
        from customized_areal.tree_search.turn_splitter import Turn

        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        child = root.add_turn(turn, seq_id=0)
        assert child.logprobs == []
        assert child.versions == []

    def test_add_turn_with_logprobs_and_versions(self):
        from customized_areal.tree_search.trie_node import TrieNode
        from customized_areal.tree_search.turn_splitter import Turn

        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        logprobs = [-0.1, -0.2, -0.3, -0.4]
        versions = [0, 0, 0, 0]
        child = root.add_turn(turn, seq_id=0, logprobs=logprobs, versions=versions)
        assert child.logprobs == [-0.1, -0.2, -0.3, -0.4]
        assert child.versions == [0, 0, 0, 0]


class TestMCTSTreeStoreTrainedFlag:
    def test_trained_flag_default_false(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        assert store.is_trained("q1", seq_id) is False

    def test_set_trained(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.set_trained("q1", seq_id, True)
        assert store.is_trained("q1", seq_id) is True

    def test_get_untrained_count(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        s2 = store.insert_trajectory("q1", [1, 2, 10, 3, 6], reward=0.3)
        assert store.get_untrained_count("q1") == 3
        store.set_trained("q1", s0, True)
        assert store.get_untrained_count("q1") == 2

    def test_reset_trained_flags(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.set_trained("q1", s0, True)
        store.reset_trained_flags()
        assert store.is_trained("q1", s0) is False

    def test_reward_stored_per_trajectory(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        assert store.get_reward("q1", s0) == 1.0
        assert store.get_reward("q1", s1) == 0.5
