import torch
import pytest
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.turn_splitter import Turn


def _simple_splitter(input_ids: list[int]) -> list[Turn]:
    """Split at token 10 -- everything before is prompt, everything from 10 onward is response."""
    try:
        split_pos = input_ids.index(10)
        return [Turn(prompt_tokens=input_ids[:split_pos], response_tokens=input_ids[split_pos:])]
    except ValueError:
        return [Turn(prompt_tokens=[], response_tokens=list(input_ids))]


class TestTreeAdvantageComputer:
    def test_compute_single_trajectory(self):
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        # _simple_splitter splits at token 10: prompt=[1,2], response=[10,3,4]
        # Trie node stores [1,2,10,3,4] = 5 tokens total
        traj = {
            "input_ids": torch.tensor([1, 2, 10, 3, 4]),
            "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
            "rewards": torch.tensor([2.0]),
        }
        store.insert_batch([traj])
        computer.compute([traj])
        assert "advantages" in traj
        assert "returns" in traj
        # Advantages should be zeroed for prompt tokens (first 2)
        assert torch.allclose(traj["advantages"][:2], torch.zeros(2))
        # Response tokens should have q_value = 2.0
        assert torch.allclose(traj["advantages"][2:], torch.full((3,), 2.0))

    def test_compute_returns_equal_advantages(self):
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        # _simple_splitter: prompt=[1], response=[10,3]
        traj = {
            "input_ids": torch.tensor([1, 10, 3]),
            "loss_mask": torch.tensor([0, 0, 1]),
            "rewards": torch.tensor([1.0]),
        }
        store.insert_batch([traj])
        computer.compute([traj])
        assert torch.allclose(traj["returns"], traj["advantages"])

    def test_compute_two_trajectories(self):
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        # traj1: prompt=[1,2], response=[10,3,4]
        traj1 = {
            "input_ids": torch.tensor([1, 2, 10, 3, 4]),
            "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
            "rewards": torch.tensor([2.0]),
        }
        # traj2: prompt=[5,6], response=[10,7,8]
        traj2 = {
            "input_ids": torch.tensor([5, 6, 10, 7, 8]),
            "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
            "rewards": torch.tensor([0.5]),
        }
        store.insert_batch([traj1, traj2])
        computer.compute([traj1, traj2])
        assert "advantages" in traj1
        assert "advantages" in traj2
