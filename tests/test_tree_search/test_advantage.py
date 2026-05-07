import torch

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


def _make_traj(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    reward: float = 1.0,
    query_id: str = "q1",
) -> dict:
    seq_len = len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.int32),
        "loss_mask": torch.tensor([loss_mask], dtype=torch.int32),
        "rewards": torch.tensor([reward], dtype=torch.float32),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "query_id": query_id,
    }


class TestTreeAdvantageComputer:
    def test_compute_single_trajectory(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0)
        store.insert_batch([traj])
        computer.compute([traj])
        assert "advantages" in traj
        assert "returns" in traj
        assert torch.allclose(traj["advantages"][0, :2], torch.zeros(2))
        assert torch.allclose(traj["advantages"][0, 2:], torch.full((3,), 2.0))

    def test_compute_returns_equal_advantages(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        traj = _make_traj([1, 10, 3], [0, 0, 1], reward=1.0)
        store.insert_batch([traj])
        computer.compute([traj])
        assert torch.allclose(traj["returns"], traj["advantages"])

    def test_compute_multi_turn_trajectory(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        traj = _make_traj(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 1, 1, 0, 0, 1, 1],
            reward=0.75,
        )
        store.insert_batch([traj])
        computer.compute([traj])
        assert torch.allclose(traj["advantages"][0, :2], torch.zeros(2))
        assert torch.allclose(traj["advantages"][0, 2:4], torch.full((2,), 0.75))
        assert torch.allclose(traj["advantages"][0, 4:6], torch.zeros(2))
        assert torch.allclose(traj["advantages"][0, 6:8], torch.full((2,), 0.75))

    def test_compute_two_trajectories(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        t1 = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        t2 = _make_traj([5, 6, 7, 8], [0, 0, 1, 1], reward=0.5, query_id="q2")
        store.insert_batch([t1, t2])
        computer.compute([t1, t2])
        assert "advantages" in t1
        assert "advantages" in t2
