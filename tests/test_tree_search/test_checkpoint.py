from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


def _make_store_with_data() -> MCTSTreeStore:
    """Create a store with sample data for checkpoint tests."""
    import torch

    store = MCTSTreeStore()
    t1 = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
        "rewards": torch.tensor([2.0], dtype=torch.float32),
        "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "_mcts_query_id": "q1",
    }
    t2 = {
        "input_ids": torch.tensor([[6, 7, 8]], dtype=torch.int32),
        "loss_mask": torch.tensor([[0, 0, 1]], dtype=torch.int32),
        "rewards": torch.tensor([0.5], dtype=torch.float32),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "_mcts_query_id": "q2",
    }
    store.insert_batch([t1, t2])
    return store


class TestTreeCheckpointManager:
    def test_save_and_load(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)
        assert manager.exists()

        loaded = manager.load()
        assert len(loaded.trajectories) == 2
        assert "q1" in loaded.trajectories
        assert "q2" in loaded.trajectories

    def test_exists_false_when_no_dir(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path / "nonexistent"))
        assert not manager.exists()

    def test_save_creates_directory(self, tmp_path):
        save_dir = str(tmp_path / "new_dir")
        manager = TreeCheckpointManager(save_dir)
        store = _make_store_with_data()
        manager.save(store)
        import os

        assert os.path.isdir(os.path.join(save_dir, "mcts_trees"))

    def test_load_preserves_seq_id_counter(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        loaded = manager.load()
        import torch

        t3 = {
            "input_ids": torch.tensor([[9, 10]], dtype=torch.int32),
            "loss_mask": torch.tensor([[0, 1]], dtype=torch.int32),
            "rewards": torch.tensor([1.0], dtype=torch.float32),
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "_mcts_query_id": "q3",
        }
        loaded.insert_batch([t3])
        assert t3["_mcts_seq_id"] == 2

    def test_load_preserves_trajectory_data(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        loaded = manager.load()
        record_q1 = loaded.trajectories["q1"][0]
        assert record_q1.input_ids == [1, 2, 3, 4, 5]
        assert record_q1.loss_mask == [0, 0, 1, 1, 1]
        assert record_q1.reward == 2.0
        record_q2 = loaded.trajectories["q2"][0]
        assert record_q2.input_ids == [6, 7, 8]
        assert record_q2.reward == 0.5

    def test_load_preserves_mcts_stats(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        manager.save(store)

        loaded = manager.load()
        seq_ids = loaded._query_seq_ids["q1"]
        assert loaded._q_values[seq_ids[0]] == 2.0
        seq_ids = loaded._query_seq_ids["q2"]
        assert loaded._q_values[seq_ids[0]] == 0.5

    def test_load_preserves_trained_flags(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = _make_store_with_data()
        seq_ids = store._query_seq_ids["q1"]
        store.set_trained("q1", seq_ids[0], True)
        manager.save(store)

        loaded = manager.load()
        assert loaded.is_trained("q1", seq_ids[0]) is True

    def test_load_preserves_turn_boundaries(self, tmp_path):
        import torch

        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore()
        traj = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32),
            "loss_mask": torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.int32),
            "rewards": torch.tensor([0.75], dtype=torch.float32),
            "attention_mask": torch.ones(1, 8, dtype=torch.bool),
            "_mcts_query_id": "q1",
        }
        store.insert_batch([traj])
        manager.save(store)

        loaded = manager.load()
        record = loaded.trajectories["q1"][0]
        assert record.turn_response_starts == [2, 6]
        assert record.turn_response_ends == [4, 8]

    def test_save_and_load_new_fields(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore()
        # Insert list-based traj with new fields
        traj = {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [0, 0, 1, 1, 1],
            "reward": 1.0,
            "attention_mask": [1, 1, 1, 1, 1],
            "logp": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "topk_ids": [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]],
            "topk_logp": [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]],
            "distill_reward": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
            "teacher_logp": [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6], [-1.7, -1.8], [-1.9, -2.0]],
            "_mcts_query_id": "q1",
        }
        store.insert_batch([traj])
        manager.save(store)

        loaded = manager.load()
        record = loaded.trajectories["q1"][0]
        assert record.logp == [-0.1, -0.2, -0.3, -0.4, -0.5]
        assert record.topk_ids == [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
        assert record.topk_logp == [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]]
        assert record.distill_reward == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        assert record.teacher_logp == [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6], [-1.7, -1.8], [-1.9, -2.0]]
