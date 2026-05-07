#!/usr/bin/env python3
from typing import Any

import torch

from customized_areal.tree_search.mcts_tree_store import (
    MCTSTreeStore,
    Node,
    _find_turn_boundaries,
)


class TestFindTurnBoundaries:
    def test_single_turn(self):
        starts, ends = _find_turn_boundaries([0, 0, 0, 1, 1])
        assert starts == [3]
        assert ends == [5]

    def test_multi_turn(self):
        starts, ends = _find_turn_boundaries([0, 0, 1, 1, 0, 0, 1, 1])
        assert starts == [2, 6]
        assert ends == [4, 8]

    def test_all_zeros(self):
        starts, ends = _find_turn_boundaries([0, 0, 0, 0])
        assert starts == []
        assert ends == []

    def test_all_ones(self):
        starts, ends = _find_turn_boundaries([1, 1, 1, 1])
        assert starts == [0]
        assert ends == [4]

    def test_empty(self):
        starts, ends = _find_turn_boundaries([])
        assert starts == []
        assert ends == []

    def test_response_at_end(self):
        starts, ends = _find_turn_boundaries([0, 0, 1, 1, 1])
        assert starts == [2]
        assert ends == [5]

    def test_three_turns(self):
        loss_mask = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        starts, ends = _find_turn_boundaries(loss_mask)
        assert starts == [2, 6, 10]
        assert ends == [4, 8, 12]


class TestNode:
    def test_creation(self):
        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="turn_0",
            parent_node_id=None,
            episode_id="ep_1",
            outcome_reward=1.0,
        )
        assert len(node.input_ids) == 5
        assert node.outcome_reward == 1.0
        assert node.node_id == "turn_0"
        assert node.parent_node_id is None

    def test_new_fields_default_to_none(self):
        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="t1",
            parent_node_id=None,
            episode_id="ep_1",
        )
        assert node.topk_ids is None
        assert node.topk_logp is None
        assert node.distill_reward is None
        assert node.teacher_logp is None

    def test_new_fields_can_be_set(self):
        topk_ids = [[10, 20], [30, 40], [50, 60]]
        topk_logp = [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]]
        distill_reward = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        teacher_logp = [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6]]

        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="t1",
            parent_node_id=None,
            episode_id="ep_1",
            outcome_reward=1.0,
            topk_ids=topk_ids,
            topk_logp=topk_logp,
            distill_reward=distill_reward,
            teacher_logp=teacher_logp,
        )
        assert node.topk_ids == topk_ids
        assert node.topk_logp == topk_logp
        assert node.distill_reward == distill_reward
        assert node.teacher_logp == teacher_logp


def _make_traj(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    reward: float = 1.0,
    logprobs: list[float] | None = None,
    versions: list[int] | None = None,
    query_id: str | None = None,
) -> dict[str, Any]:
    seq_len = len(input_ids)
    traj: dict[str, Any] = {
        "input_ids": torch.tensor([input_ids], dtype=torch.int32),
        "loss_mask": torch.tensor([loss_mask], dtype=torch.int32),
        "rewards": torch.tensor([reward], dtype=torch.float32),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
    }
    if logprobs is not None:
        traj["logprobs"] = torch.tensor([logprobs], dtype=torch.float32)
    if versions is not None:
        traj["versions"] = torch.tensor([versions], dtype=torch.int32)
    if query_id is not None:
        traj["query_id"] = query_id
    return traj


class TestMCTSTreeStoreInsertBatch:
    def test_insert_list_dict_basic(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [0, 0, 1, 1, 1],
            "reward": 1.0,
            "attention_mask": [1, 1, 1, 1, 1],
        }
        store.insert_batch([traj])
        assert "node_id" in traj
        assert "query_id" in traj
        assert len(store.trajectories) == 1
        query_id = traj["query_id"]
        assert len(store.trajectories[query_id]) == 1
        node = store.trajectories[query_id][0]
        assert node.input_ids == [1, 2, 3, 4, 5]
        assert node.loss_mask == [0, 0, 1, 1, 1]

    def test_insert_list_dict_with_new_fields(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": [1, 2, 3],
            "loss_mask": [0, 0, 1],
            "reward": 1.0,
            "attention_mask": [1, 1, 1],
            "topk_ids": [[10, 20], [30, 40], [50, 60]],
            "topk_logp": [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
            "distill_reward": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "teacher_logp": [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6]],
        }
        store.insert_batch([traj])
        query_id = traj["query_id"]
        node = store.trajectories[query_id][0]
        assert node.topk_ids == [[10, 20], [30, 40], [50, 60]]
        assert node.topk_logp == [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]]
        assert node.distill_reward == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        assert node.teacher_logp == [[-1.1, -1.2], [-1.3, -1.4], [-1.5, -1.6]]

    def test_insert_list_dict_with_query_id(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [0, 0, 1, 1, 1],
            "reward": 1.0,
            "attention_mask": [1, 1, 1, 1, 1],
            "query_id": "custom_query_id",
        }
        store.insert_batch([traj])
        assert traj["query_id"] == "custom_query_id"
        assert "custom_query_id" in store.trajectories

    def test_insert_single_trajectory(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        store.insert_batch([traj])
        assert "node_id" in traj
        assert traj["query_id"] == "q1"
        assert len(store.trajectories["q1"]) == 1

    def test_insert_two_trajectories_same_query(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        assert len(store.trajectories["q1"]) == 2
        assert t1["node_id"] != t2["node_id"]

    def test_insert_grouped_trajectory(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.int32),
            "loss_mask": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]], dtype=torch.int32),
            "rewards": torch.tensor([1.0, 0.5], dtype=torch.float32),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool
            ),
            "query_id": "q1",
        }
        store.insert_batch([traj])
        assert "node_ids" in traj
        assert len(traj["node_ids"]) == 2
        node1 = store.trajectories["q1"][1]
        assert len(node1.input_ids) == 3

    def test_insert_skips_already_inserted(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        seq_id_1 = traj["node_id"]
        store.insert_batch([traj])
        assert traj["node_id"] == seq_id_1
        assert len(store.trajectories["q1"]) == 1

    def test_insert_stores_logprobs_and_versions(self):
        store = MCTSTreeStore()
        traj = _make_traj(
            [1, 2, 3, 4, 5],
            [0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 1, 1, 1],
            query_id="q1",
        )
        store.insert_batch([traj])
        node = store.trajectories["q1"][0]
        assert len(node.logprobs) == 5
        for expected, actual in zip([-0.1, -0.2, -0.3, -0.4, -0.5], node.logprobs):
            assert abs(expected - actual) < 1e-6
        assert node.versions == [0, 0, 1, 1, 1]

    def test_insert_node_objects(self):
        """Insert a list of Node objects directly (new workflow pipeline)."""
        store = MCTSTreeStore()
        node = Node(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            node_id="turn_0",
            parent_node_id=None,
            episode_id="q1_0",
            outcome_reward=1.0,
        )
        object.__setattr__(node, "query_id", "q1")
        store.insert_batch([node])
        assert hasattr(node, "node_id")
        assert len(store.trajectories) == 1
        assert len(store.trajectories["q1"]) == 1


class TestMCTSTreeStoreAdvantages:
    def test_get_advantages_single_turn(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        store.insert_batch([traj])
        seq_id = traj["node_id"]
        adv = store.get_advantages("q1", seq_id)
        assert adv.shape == torch.Size([5])
        assert torch.allclose(adv[:2], torch.zeros(2))
        assert torch.allclose(adv[2:], torch.full((3,), 2.0))

    def test_get_advantages_multi_turn(self):
        store = MCTSTreeStore()
        traj = _make_traj(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 1, 1, 0, 0, 1, 1],
            reward=0.75,
            query_id="q1",
        )
        store.insert_batch([traj])
        seq_id = traj["node_id"]
        adv = store.get_advantages("q1", seq_id)
        assert torch.allclose(adv[:2], torch.zeros(2))
        assert torch.allclose(adv[2:4], torch.full((2,), 0.75))
        assert torch.allclose(adv[4:6], torch.zeros(2))
        assert torch.allclose(adv[6:8], torch.full((2,), 0.75))

    def test_get_prompt_mask(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], query_id="q1")
        store.insert_batch([traj])
        mask = store.get_prompt_mask("q1", traj["node_id"])
        assert mask.tolist() == [False, False, True, True, True]


class TestMCTSTreeStoreTrainedFlag:
    def test_trained_flag_default_false(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        assert store.is_trained("q1", traj["node_id"]) is False

    def test_set_trained(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        store.set_trained("q1", traj["node_id"], True)
        assert store.is_trained("q1", traj["node_id"]) is True

    def test_get_untrained_count(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        t3 = _make_traj([7, 8, 9], [0, 0, 1], reward=0.3, query_id="q1")
        store.insert_batch([t1, t2, t3])
        assert store.get_untrained_count("q1") == 3
        store.set_trained("q1", t1["node_id"], True)
        assert store.get_untrained_count("q1") == 2

    def test_reset_trained_flags(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        store.set_trained("q1", traj["node_id"], True)
        store.reset_trained_flags()
        assert store.is_trained("q1", traj["node_id"]) is False

    def test_get_reward(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        assert store.get_reward("q1", t1["node_id"]) == 1.0
        assert store.get_reward("q1", t2["node_id"]) == 0.5


class TestMCTSTreeStoreLoadTrajectories:
    def test_load_trajectories_returns_nodes(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [0, 0, 1, 1, 1],
            "logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "versions": [0, 0, 0, 0, 0],
            "reward": 1.0,
            "attention_mask": [1, 1, 1, 1, 1],
            "topk_ids": [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]],
            "topk_logp": [
                [-0.1, -0.2],
                [-0.3, -0.4],
                [-0.5, -0.6],
                [-0.7, -0.8],
                [-0.9, -1.0],
            ],
            "distill_reward": [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 1.0],
            ],
            "teacher_logp": [
                [-1.1, -1.2],
                [-1.3, -1.4],
                [-1.5, -1.6],
                [-1.7, -1.8],
                [-1.9, -2.0],
            ],
        }
        store.insert_batch([traj])
        loaded = store.load_trajectories(traj["query_id"], n_samples=1)
        assert len(loaded) == 1
        node = loaded[0]
        assert isinstance(node, Node)
        assert node.input_ids == [1, 2, 3, 4, 5]
        assert node.outcome_reward == 1.0
        assert node.topk_ids == [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
        assert node.teacher_logp == [
            [-1.1, -1.2],
            [-1.3, -1.4],
            [-1.5, -1.6],
            [-1.7, -1.8],
            [-1.9, -2.0],
        ]
        assert hasattr(node, "query_id")
        assert hasattr(node, "node_id")

    def test_load_trajectories_basic(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=1.0, query_id="q1")
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        assert len(loaded) == 1
        node = loaded[0]
        assert isinstance(node, Node)
        assert len(node.input_ids) == 5
        assert hasattr(node, "query_id")

    def test_load_trajectories_only_untrained(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        store.set_trained("q1", t1["node_id"], True)
        loaded = store.load_trajectories("q1", n_samples=2)
        assert len(loaded) == 1
        assert loaded[0].outcome_reward == 0.5

    def test_load_trajectories_unknown_query(self):
        store = MCTSTreeStore()
        assert store.load_trajectories("nonexistent", n_samples=1) == []


class TestMCTSTreeStoreClear:
    def test_clear(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        store.clear()
        assert len(store.trajectories) == 0
        assert store._next_seq_id == 0
        assert len(store._visit_counts) == 0
        assert len(store._q_values) == 0


class TestMCTSTreeStoreMCTSStats:
    def test_backup_updates_stats(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], reward=2.0, query_id="q1")
        store.insert_batch([traj])
        seq_id = traj["node_id"]
        assert store._visit_counts[seq_id] == 1
        assert store._q_values[seq_id] == 2.0

    def test_two_trajectories_separate_q_values(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.0, query_id="q1")
        store.insert_batch([t1, t2])
        assert store._q_values[t1["node_id"]] == 1.0
        assert store._q_values[t2["node_id"]] == 0.0
