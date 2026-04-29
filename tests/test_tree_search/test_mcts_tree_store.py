#!/usr/bin/env python3
import torch
from typing import Any

from customized_areal.tree_search.mcts_tree_store import (
    MCTSTreeStore,
    TrajectoryRecord,
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


class TestTrajectoryRecord:
    def test_creation(self):
        record = TrajectoryRecord(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            versions=[0, 0, 0, 0, 0],
            reward=1.0,
            turn_response_starts=[2],
            turn_response_ends=[5],
        )
        assert len(record.input_ids) == 5
        assert record.reward == 1.0


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
        traj["_mcts_query_id"] = query_id
    return traj


class TestMCTSTreeStoreInsertBatch:
    def test_insert_single_trajectory(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        store.insert_batch([traj])
        assert "_mcts_seq_id" in traj
        assert traj["_mcts_query_id"] == "q1"
        assert len(store.trajectories["q1"]) == 1

    def test_insert_two_trajectories_same_query(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        assert len(store.trajectories["q1"]) == 2
        assert t1["_mcts_seq_id"] != t2["_mcts_seq_id"]

    def test_insert_grouped_trajectory(self):
        store = MCTSTreeStore()
        traj = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.int32),
            "loss_mask": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]], dtype=torch.int32),
            "rewards": torch.tensor([1.0, 0.5], dtype=torch.float32),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool
            ),
            "_mcts_query_id": "q1",
        }
        store.insert_batch([traj])
        assert "_mcts_seq_ids" in traj
        assert len(traj["_mcts_seq_ids"]) == 2
        record1 = store.trajectories["q1"][1]
        assert len(record1.input_ids) == 3

    def test_insert_skips_already_inserted(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        seq_id_1 = traj["_mcts_seq_id"]
        store.insert_batch([traj])
        assert traj["_mcts_seq_id"] == seq_id_1
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
        record = store.trajectories["q1"][0]
        # Check logprobs with floating-point tolerance
        assert len(record.logprobs) == 5
        for expected, actual in zip([-0.1, -0.2, -0.3, -0.4, -0.5], record.logprobs):
            assert abs(expected - actual) < 1e-6
        assert record.versions == [0, 0, 1, 1, 1]


class TestMCTSTreeStoreAdvantages:
    def test_get_advantages_single_turn(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=2.0, query_id="q1")
        store.insert_batch([traj])
        seq_id = traj["_mcts_seq_id"]
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
        seq_id = traj["_mcts_seq_id"]
        adv = store.get_advantages("q1", seq_id)
        assert torch.allclose(adv[:2], torch.zeros(2))
        assert torch.allclose(adv[2:4], torch.full((2,), 0.75))
        assert torch.allclose(adv[4:6], torch.zeros(2))
        assert torch.allclose(adv[6:8], torch.full((2,), 0.75))

    def test_get_prompt_mask(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], query_id="q1")
        store.insert_batch([traj])
        mask = store.get_prompt_mask("q1", traj["_mcts_seq_id"])
        assert mask.tolist() == [False, False, True, True, True]


class TestMCTSTreeStoreTrainedFlag:
    def test_trained_flag_default_false(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        assert store.is_trained("q1", traj["_mcts_seq_id"]) is False

    def test_set_trained(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        store.set_trained("q1", traj["_mcts_seq_id"], True)
        assert store.is_trained("q1", traj["_mcts_seq_id"]) is True

    def test_get_untrained_count(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        t3 = _make_traj([7, 8, 9], [0, 0, 1], reward=0.3, query_id="q1")
        store.insert_batch([t1, t2, t3])
        assert store.get_untrained_count("q1") == 3
        store.set_trained("q1", t1["_mcts_seq_id"], True)
        assert store.get_untrained_count("q1") == 2

    def test_reset_trained_flags(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3], [0, 0, 1], query_id="q1")
        store.insert_batch([traj])
        store.set_trained("q1", traj["_mcts_seq_id"], True)
        store.reset_trained_flags()
        assert store.is_trained("q1", traj["_mcts_seq_id"]) is False

    def test_get_reward(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        assert store.get_reward("q1", t1["_mcts_seq_id"]) == 1.0
        assert store.get_reward("q1", t2["_mcts_seq_id"]) == 0.5


class TestMCTSTreeStoreLoadTrajectories:
    def test_load_trajectories_basic(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], reward=1.0, query_id="q1")
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        assert len(loaded) == 1
        t = loaded[0]
        assert t["input_ids"].shape[0] == 1
        assert t["input_ids"].shape[1] == 5
        assert t["rewards"].item() == 1.0
        assert t["_mcts_query_id"] == "q1"
        assert t["_mcts_seq_id"] == traj["_mcts_seq_id"]

    def test_load_trajectories_only_untrained(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])
        store.set_trained("q1", t1["_mcts_seq_id"], True)
        loaded = store.load_trajectories("q1", n_samples=2)
        assert len(loaded) == 1
        assert loaded[0]["rewards"].item() == 0.5

    def test_load_trajectories_preserves_loss_mask(self):
        store = MCTSTreeStore()
        loss_mask = [0, 0, 1, 1, 0, 0, 1, 1]
        traj = _make_traj(
            [1, 2, 3, 4, 5, 6, 7, 8], loss_mask, reward=1.0, query_id="q1"
        )
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        torch.testing.assert_close(
            loaded[0]["loss_mask"].squeeze(0),
            torch.tensor(loss_mask, dtype=torch.int32),
        )

    def test_load_trajectories_unknown_query(self):
        store = MCTSTreeStore()
        assert store.load_trajectories("nonexistent", n_samples=1) == []

    def test_load_trajectories_attention_mask_all_ones(self):
        store = MCTSTreeStore()
        traj = _make_traj([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], query_id="q1")
        store.insert_batch([traj])
        loaded = store.load_trajectories("q1", n_samples=1)
        assert loaded[0]["attention_mask"].all()


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
        seq_id = traj["_mcts_seq_id"]
        assert store._visit_counts[seq_id] == 1
        assert store._q_values[seq_id] == 2.0

    def test_two_trajectories_separate_q_values(self):
        store = MCTSTreeStore()
        t1 = _make_traj([1, 2, 3], [0, 0, 1], reward=1.0, query_id="q1")
        t2 = _make_traj([4, 5, 6], [0, 0, 1], reward=0.0, query_id="q1")
        store.insert_batch([t1, t2])
        assert store._q_values[t1["_mcts_seq_id"]] == 1.0
        assert store._q_values[t2["_mcts_seq_id"]] == 0.0
