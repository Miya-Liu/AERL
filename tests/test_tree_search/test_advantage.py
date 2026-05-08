import torch

from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node


def _make_node(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    reward: float = 1.0,
    query_id: str = "q1",
) -> Node:
    return Node(
        input_ids=input_ids,
        loss_mask=loss_mask,
        logprobs=[0.0] * len(input_ids),
        versions=[-1] * len(input_ids),
        outcome_reward=reward,
        query_id=query_id,
    )


class TestTreeAdvantageComputer:
    def test_compute_single_trajectory(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        node = _make_node([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0)
        store.insert_batch([node])
        computer.compute([node])
        assert node.advantages is not None
        assert node.returns is not None
        # Prompt positions (loss_mask=0) get zero advantage
        assert torch.allclose(node.advantages[:2], torch.zeros(2))
        # Single-sample: zero-mean normalization produces 0.0 advantage
        assert torch.allclose(node.advantages[2:], torch.zeros(3))
        # Returns are still mask * outcome_reward
        assert torch.allclose(node.returns[2:], torch.full((3,), 2.0))

    def test_compute_returns_from_outcome_reward(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        node = _make_node([1, 10, 3], [0, 0, 1], reward=1.0)
        store.insert_batch([node])
        computer.compute([node])
        # Single-sample: advantage = 0 (zero-mean), returns = mask * outcome_reward
        assert torch.allclose(node.advantages, torch.zeros(3))
        assert torch.allclose(node.returns[2], torch.tensor(1.0))
        assert torch.allclose(node.returns[:2], torch.zeros(2))

    def test_compute_multi_turn_trajectory(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        node = _make_node(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 1, 1, 0, 0, 1, 1],
            reward=0.75,
        )
        store.insert_batch([node])
        computer.compute([node])
        # Single-sample: zero-mean normalization → advantage = 0 everywhere
        assert torch.allclose(node.advantages, torch.zeros(8))
        # Returns: mask * outcome_reward on response tokens
        assert torch.allclose(node.returns[:2], torch.zeros(2))
        assert torch.allclose(node.returns[2:4], torch.full((2,), 0.75))
        assert torch.allclose(node.returns[4:6], torch.zeros(2))
        assert torch.allclose(node.returns[6:8], torch.full((2,), 0.75))

    def test_compute_two_trajectories(self):
        store = MCTSTreeStore()
        computer = TreeAdvantageComputer(store)
        t1 = _make_node([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        t2 = _make_node([5, 6, 7, 8], [0, 0, 1, 1], reward=0.5, query_id="q2")
        store.insert_batch([t1, t2])
        computer.compute([t1, t2])
        assert t1.advantages is not None
        assert t2.advantages is not None
