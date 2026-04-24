import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.turn_splitter import Turn


def _two_turn_splitter(input_ids: list[int]) -> list[Turn]:
    """Simple splitter for testing: splits at token 10, first half is prompt, second is response."""
    try:
        split_pos = input_ids.index(10)
        return [
            Turn(
                prompt_tokens=input_ids[:split_pos],
                response_tokens=input_ids[split_pos:],
            )
        ]
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
                "input_ids": torch.tensor([[1, 2, 10, 3, 4]]),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1]]),
                "rewards": torch.tensor([1.0]),
            },
            {
                "input_ids": torch.tensor([[1, 2, 10, 3, 5]]),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1]]),
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


class TestTrieNodeTrainingSteps:
    def test_training_steps_default_empty(self):
        from customized_areal.tree_search.trie_node import TrieNode

        node = TrieNode(tree_id=0)
        assert node.training_steps == []

    def test_training_steps_append(self):
        from customized_areal.tree_search.trie_node import TrieNode

        node = TrieNode(tree_id=0)
        node.training_steps.append(5)
        node.training_steps.append(10)
        assert node.training_steps == [5, 10]


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


class TestMCTSTreeStoreLoadTrajectories:
    def test_load_trajectories_basic(self):
        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        trajs = store.load_trajectories("q1", n_samples=1)
        assert len(trajs) == 1
        traj = trajs[0]
        assert "input_ids" in traj
        assert "logprobs" in traj
        assert "loss_mask" in traj
        assert "attention_mask" in traj
        assert "rewards" in traj
        assert "versions" in traj
        assert traj["input_ids"].shape[0] == 1
        assert traj["rewards"].item() == 1.0

    def test_load_trajectories_multiple(self):
        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        trajs = store.load_trajectories("q1", n_samples=2)
        assert len(trajs) == 2

    def test_load_trajectories_only_untrained(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        store.set_trained("q1", s0, True)
        trajs = store.load_trajectories("q1", n_samples=2)
        assert len(trajs) == 1
        assert trajs[0]["rewards"].item() == 0.5

    def test_load_trajectories_returns_empty_for_unknown_query(self):
        store = MCTSTreeStore(_two_turn_splitter)
        trajs = store.load_trajectories("nonexistent", n_samples=1)
        assert trajs == []


class TestMCTSTreeStoreInsertBatchWithMetadata:
    def test_insert_batch_stores_logprobs_and_versions(self):
        store = MCTSTreeStore(_two_turn_splitter)
        trajectories = [
            {
                "input_ids": torch.tensor([[1, 2, 10, 3, 4]]),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1]]),
                "rewards": torch.tensor([1.0]),
                "logprobs": torch.tensor([[-0.1, -0.2, -0.3, -0.4, -0.5]]),
                "versions": torch.tensor([[0, 0, 0, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool),
            },
        ]
        store.insert_batch(trajectories)
        query_id = trajectories[0]["_mcts_query_id"]
        trajs = store.load_trajectories(query_id, n_samples=1)
        assert len(trajs) == 1
        torch.testing.assert_close(
            trajs[0]["logprobs"].squeeze(0),
            torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5]),
        )

    def test_insert_batch_stores_reward(self):
        store = MCTSTreeStore(_two_turn_splitter)
        trajectories = [
            {
                "input_ids": torch.tensor([[1, 2, 10, 3, 4]]),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1]]),
                "rewards": torch.tensor([0.75]),
                "logprobs": torch.tensor([[-0.1, -0.2, -0.3, -0.4, -0.5]]),
                "versions": torch.tensor([[0, 0, 0, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool),
            },
        ]
        store.insert_batch(trajectories)
        query_id = trajectories[0]["_mcts_query_id"]
        seq_id = trajectories[0]["_mcts_seq_id"]
        assert store.get_reward(query_id, seq_id) == 0.75


class TestMCTSTreeStoreRecordTrainingStep:
    def test_record_training_step_single_trajectory(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        trajectories = [{"_mcts_query_id": "q1", "_mcts_seq_id": seq_id}]
        store.record_training_step(0, trajectories)
        root = store.trees["q1"]
        leaf = root.get_path_nodes(seq_id)[-1]
        assert leaf.training_steps == [0]
        assert 0 in store._training_history
        assert store._training_history[0] == [("q1", seq_id)]

    def test_record_training_step_multiple_trajectories(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
        trajectories = [
            {"_mcts_query_id": "q1", "_mcts_seq_id": s0},
            {"_mcts_query_id": "q2", "_mcts_seq_id": s1},
        ]
        store.record_training_step(3, trajectories)
        leaf0 = store.trees["q1"].get_path_nodes(s0)[-1]
        assert leaf0.training_steps == [3]
        leaf1 = store.trees["q2"].get_path_nodes(s1)[-1]
        assert leaf1.training_steps == [3]
        assert store._training_history[3] == [("q1", s0), ("q2", s1)]

    def test_record_training_step_grouped_trajectory(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q1", [1, 2, 11, 3, 5], reward=0.5)
        trajectories = [
            {"_mcts_query_id": "q1", "_mcts_seq_ids": [s0, s1]},
        ]
        store.record_training_step(1, trajectories)
        leaf0 = store.trees["q1"].get_path_nodes(s0)[-1]
        leaf1 = store.trees["q1"].get_path_nodes(s1)[-1]
        assert leaf0.training_steps == [1]
        assert leaf1.training_steps == [1]
        assert store._training_history[1] == [("q1", s0), ("q1", s1)]

    def test_record_training_step_skips_missing_global_step(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        trajectories = [{"_mcts_query_id": "q1", "_mcts_seq_id": seq_id}]
        store.record_training_step(None, trajectories)
        leaf = store.trees["q1"].get_path_nodes(seq_id)[-1]
        assert leaf.training_steps == []
        assert len(store._training_history) == 0

    def test_record_training_step_same_trajectory_multiple_steps(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        trajectories = [{"_mcts_query_id": "q1", "_mcts_seq_id": seq_id}]
        store.record_training_step(0, trajectories)
        store.record_training_step(5, trajectories)
        leaf = store.trees["q1"].get_path_nodes(seq_id)[-1]
        assert leaf.training_steps == [0, 5]


class TestMCTSTreeStoreLoadBySeqId:
    def test_load_trajectory_by_seq_id(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        traj = store.load_trajectory_by_seq_id("q1", seq_id)
        assert traj is not None
        assert traj["input_ids"].shape[0] == 1
        assert traj["rewards"].item() == 1.0
        assert traj["_mcts_query_id"] == "q1"
        assert traj["_mcts_seq_id"] == seq_id

    def test_load_trajectory_by_seq_id_unknown(self):
        store = MCTSTreeStore(_two_turn_splitter)
        result = store.load_trajectory_by_seq_id("nonexistent", 0)
        assert result is None

    def test_load_trajectory_by_seq_id_matches_load_trajectories(self):
        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        store.set_trained("q1", s0, True)
        trajs = store.load_trajectories("q1", n_samples=1)
        assert len(trajs) == 1
        assert trajs[0]["_mcts_seq_id"] == s1
        # load_by_seq_id can still load s0 regardless of trained flag
        traj = store.load_trajectory_by_seq_id("q1", s0)
        assert traj is not None
        assert traj["_mcts_seq_id"] == s0


class TestMCTSTreeStoreBuildTrainingHistory:
    def _insert_and_record(self, store, query_id, tokens, reward, step):
        seq_id = store.insert_trajectory(query_id, tokens, reward=reward)
        trajectories = [{"_mcts_query_id": query_id, "_mcts_seq_id": seq_id}]
        store.record_training_step(step, trajectories)
        return seq_id

    def test_build_training_history_from_leaves(self):
        store = MCTSTreeStore(_two_turn_splitter)
        self._insert_and_record(store, "q1", [1, 2, 10, 3, 4], 1.0, 0)
        self._insert_and_record(store, "q2", [5, 6, 10, 7, 8], 0.5, 0)
        self._insert_and_record(store, "q1", [1, 2, 11, 3, 5], 0.3, 1)

        # Clear history and rebuild from leaves
        store._training_history.clear()
        store.build_training_history()

        assert 0 in store._training_history
        assert 1 in store._training_history
        step0_pairs = store._training_history[0]
        assert len(step0_pairs) == 2
        assert len(store._training_history[1]) == 1

    def test_build_training_history_empty(self):
        store = MCTSTreeStore(_two_turn_splitter)
        store.build_training_history()
        assert store._training_history == {}

    def test_build_training_history_preserves_existing(self):
        store = MCTSTreeStore(_two_turn_splitter)
        self._insert_and_record(store, "q1", [1, 2, 10, 3, 4], 1.0, 0)
        original = store._training_history[0]
        store.build_training_history()
        assert store._training_history[0] == original


class TestTreeCheckpointTrainingHistory:
    def test_save_load_training_steps_and_history(self):
        import tempfile

        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
        trajectories = [
            {"_mcts_query_id": "q1", "_mcts_seq_id": s0},
            {"_mcts_query_id": "q2", "_mcts_seq_id": s1},
        ]
        store.record_training_step(0, trajectories)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = TreeCheckpointManager(tmpdir)
            mgr.save(store)

            loaded = mgr.load(_two_turn_splitter)

            # Check _training_history preserved
            assert 0 in loaded._training_history
            assert loaded._training_history[0] == [("q1", s0), ("q2", s1)]

            # Check leaf training_steps preserved
            leaf0 = loaded.trees["q1"].get_path_nodes(s0)[-1]
            assert leaf0.training_steps == [0]
            leaf1 = loaded.trees["q2"].get_path_nodes(s1)[-1]
            assert leaf1.training_steps == [0]

    def test_load_old_checkpoint_without_history(self):
        import json
        import os
        import tempfile

        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = TreeCheckpointManager(tmpdir)
            mgr.save(store)

            # Manually remove training_history from metadata to simulate old checkpoint
            metadata_path = os.path.join(tmpdir, "mcts_trees", "metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)
            del metadata["training_history"]
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            loaded = mgr.load(_two_turn_splitter)
            assert loaded._training_history == {}


class TestRolloutCacheConfig:
    def test_default_replay_is_false(self):
        from customized_areal.tree_search.config import RolloutCacheConfig

        config = RolloutCacheConfig()
        assert config.replay is False

    def test_replay_can_be_set(self):
        from customized_areal.tree_search.config import RolloutCacheConfig

        config = RolloutCacheConfig(replay=True)
        assert config.replay is True


class TestTrainingOrderReplayIntegration:
    def test_record_and_replay_cycle(self):
        """Simulate recording training steps, saving checkpoint, loading, and replaying."""
        import tempfile

        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # === RECORD PHASE ===
            store = MCTSTreeStore(_two_turn_splitter)
            s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
            s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
            s2 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.3)

            # Simulate training step 0: use q1/s0 and q2/s1
            store.record_training_step(
                0,
                [
                    {"_mcts_query_id": "q1", "_mcts_seq_id": s0},
                    {"_mcts_query_id": "q2", "_mcts_seq_id": s1},
                ],
            )
            # Simulate training step 1: use q1/s2
            store.record_training_step(
                1,
                [{"_mcts_query_id": "q1", "_mcts_seq_id": s2}],
            )

            # Save checkpoint
            mgr = TreeCheckpointManager(tmpdir)
            mgr.save(store)

            # === REPLAY PHASE ===
            loaded = mgr.load(_two_turn_splitter)

            # Step 0 replay
            assert 0 in loaded._training_history
            step0_pairs = loaded._training_history[0]
            assert len(step0_pairs) == 2
            assert step0_pairs[0] == ("q1", s0)
            assert step0_pairs[1] == ("q2", s1)

            # Load trajectories in replay order
            replay_trajs = []
            for query_id, seq_id in step0_pairs:
                traj = loaded.load_trajectory_by_seq_id(query_id, seq_id)
                assert traj is not None
                replay_trajs.append(traj)

            assert len(replay_trajs) == 2
            assert replay_trajs[0]["_mcts_query_id"] == "q1"
            assert replay_trajs[1]["_mcts_query_id"] == "q2"

            # Step 1 replay
            step1_pairs = loaded._training_history[1]
            assert len(step1_pairs) == 1
            assert step1_pairs[0] == ("q1", s2)

    def test_build_history_fallback_for_old_checkpoint(self):
        """Test that build_training_history can reconstruct from leaves alone."""
        import json
        import os
        import tempfile

        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            store = MCTSTreeStore(_two_turn_splitter)
            s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
            store.record_training_step(
                0, [{"_mcts_query_id": "q1", "_mcts_seq_id": s0}]
            )

            mgr = TreeCheckpointManager(tmpdir)
            mgr.save(store)

            # Simulate old checkpoint: remove training_history from metadata
            metadata_path = os.path.join(tmpdir, "mcts_trees", "metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)
            del metadata["training_history"]
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Load — _training_history should be empty
            loaded = mgr.load(_two_turn_splitter)
            assert loaded._training_history == {}

            # Fallback: build from leaves
            loaded.build_training_history()
            assert 0 in loaded._training_history
            assert len(loaded._training_history[0]) == 1
            assert loaded._training_history[0][0][0] == "q1"
