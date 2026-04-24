import torch
from unittest.mock import MagicMock, patch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.turn_splitter import Turn


def _two_turn_splitter(input_ids: list[int]) -> list[Turn]:
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


class TestCacheAwareBatchBuilder:
    def test_build_batch_fully_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore(_two_turn_splitter)
        for i in range(4):
            store.insert_trajectory(
                "q1",
                [1, 2, 10, 3, 4 + i],
                reward=1.0 / (i + 1),
                logprobs=[-0.1] * 5,
                versions=[0] * 5,
            )

        builder = _CacheAwareBatchBuilder(store, n_samples=4)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 1
        assert len(need_gen) == 0
        assert cached[0]["cached_count"] == 4
        assert cached[0]["need_gen_count"] == 0

    def test_build_batch_partially_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore(_two_turn_splitter)
        for i in range(2):
            store.insert_trajectory(
                "q1",
                [1, 2, 10, 3, 4 + i],
                reward=1.0 / (i + 1),
                logprobs=[-0.1] * 5,
                versions=[0] * 5,
            )

        builder = _CacheAwareBatchBuilder(store, n_samples=4)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 1
        assert len(need_gen) == 0
        assert cached[0]["cached_count"] == 2
        assert cached[0]["need_gen_count"] == 2

    def test_build_batch_not_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore(_two_turn_splitter)
        builder = _CacheAwareBatchBuilder(store, n_samples=4)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 0
        assert len(need_gen) == 1

    def test_load_cached_trajectories(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory(
            "q1", [1, 2, 10, 3, 4], reward=1.0, logprobs=[-0.1] * 5, versions=[0] * 5
        )
        store.insert_trajectory(
            "q1", [1, 2, 10, 3, 5], reward=0.5, logprobs=[-0.2] * 5, versions=[0] * 5
        )

        builder = _CacheAwareBatchBuilder(store, n_samples=4)
        cached, _ = builder.split_prompts([{"_mcts_query_id": "q1"}])
        loaded = builder.load_cached_trajectories(cached)
        assert "q1" in loaded
        assert len(loaded["q1"]) == 2


class TestMergeCachedAndNew:
    def test_merge_cached_and_new_individual(self):
        from customized_areal.tree_search.trainer import _merge_cached_and_new

        cached = [
            {
                "input_ids": torch.tensor([[1, 2, 10, 3, 4]], dtype=torch.int32),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int32),
                "logprobs": torch.tensor(
                    [[-0.1, -0.2, -0.3, -0.4, -0.5]], dtype=torch.float32
                ),
                "rewards": torch.tensor([[1.0]], dtype=torch.float32),
                "versions": torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.int32),
            },
            {
                "input_ids": torch.tensor([[1, 2, 10, 3, 5]], dtype=torch.int32),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int32),
                "logprobs": torch.tensor(
                    [[-0.2, -0.2, -0.2, -0.2, -0.2]], dtype=torch.float32
                ),
                "rewards": torch.tensor([[0.5]], dtype=torch.float32),
                "versions": torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.int32),
            },
        ]

        new_trajs = [
            {
                "input_ids": torch.tensor([[1, 2, 10, 3, 6, 0]], dtype=torch.int32),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1, 0]], dtype=torch.int32),
                "logprobs": torch.tensor(
                    [[-0.3, -0.3, -0.3, -0.3, -0.3, 0.0]], dtype=torch.float32
                ),
                "rewards": torch.tensor([[0.3]], dtype=torch.float32),
                "versions": torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.int32),
            },
        ]

        merged = _merge_cached_and_new(cached, new_trajs)
        assert len(merged) == 1
        assert merged[0]["input_ids"].shape[0] == 3

    def test_merge_with_grouped_new_trajs(self):
        from customized_areal.tree_search.trainer import _merge_cached_and_new

        cached = [
            {
                "input_ids": torch.tensor([[1, 2, 10, 3, 4]], dtype=torch.int32),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int32),
                "logprobs": torch.tensor(
                    [[-0.1, -0.2, -0.3, -0.4, -0.5]], dtype=torch.float32
                ),
                "rewards": torch.tensor([[1.0]], dtype=torch.float32),
                "versions": torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.int32),
            },
            {
                "input_ids": torch.tensor([[1, 2, 10, 3, 5]], dtype=torch.int32),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int32),
                "logprobs": torch.tensor(
                    [[-0.2, -0.2, -0.2, -0.2, -0.2]], dtype=torch.float32
                ),
                "rewards": torch.tensor([[0.5]], dtype=torch.float32),
                "versions": torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.int32),
            },
        ]

        # Grouped new trajs: shape [2, seq_len]
        new_trajs = [
            {
                "input_ids": torch.tensor(
                    [[1, 2, 10, 3, 6], [1, 2, 10, 3, 7]], dtype=torch.int32
                ),
                "attention_mask": torch.tensor(
                    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool
                ),
                "loss_mask": torch.tensor(
                    [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.int32
                ),
                "logprobs": torch.tensor([[-0.3] * 5, [-0.4] * 5], dtype=torch.float32),
                "rewards": torch.tensor([[0.3], [0.2]], dtype=torch.float32),
                "versions": torch.tensor([[0] * 5, [0] * 5], dtype=torch.int32),
            },
        ]

        merged = _merge_cached_and_new(cached, new_trajs)
        assert len(merged) == 1
        assert merged[0]["input_ids"].shape[0] == 4  # n_samples=4

    def test_merge_empty_cached(self):
        from customized_areal.tree_search.trainer import _merge_cached_and_new

        new_trajs = [
            {
                "input_ids": torch.tensor([[1, 2, 10, 3, 4]], dtype=torch.int32),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool),
                "loss_mask": torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int32),
                "logprobs": torch.tensor([[-0.1] * 5], dtype=torch.float32),
                "rewards": torch.tensor([[1.0]], dtype=torch.float32),
                "versions": torch.tensor([[0] * 5], dtype=torch.int32),
            },
        ]

        merged = _merge_cached_and_new([], new_trajs)
        assert len(merged) == 1
        assert merged[0]["input_ids"].shape[0] == 1

    def test_merge_empty_both(self):
        from customized_areal.tree_search.trainer import _merge_cached_and_new

        merged = _merge_cached_and_new([], [])
        assert merged == []


class TestLoadUntrainedFromTreeStore:
    def test_loads_from_single_query(self):
        """Should load untrained trajectories from a single query_id."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert len(result) == 2
        assert result[0]["_mcts_query_id"] == "q1"
        assert result[1]["_mcts_query_id"] == "q1"

    def test_loads_from_multiple_queries(self):
        """Should load untrained trajectories from all query_ids with untrained paths."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert len(result) == 2
        query_ids = {t["_mcts_query_id"] for t in result}
        assert query_ids == {"q1", "q2"}

    def test_skips_trained_trajectories(self):
        """Should not load trajectories that are already marked trained."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        store.set_trained("q1", s0, True)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert len(result) == 1
        assert result[0]["_mcts_seq_id"] == s1

    def test_respects_n_samples_limit(self):
        """Should not load more than n_samples per query_id."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        for i in range(4):
            store.insert_trajectory("q1", [1, 2, 10, 3, 4 + i], reward=1.0)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=2)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert len(result) == 2

    def test_returns_empty_when_no_untrained(self):
        """Should return empty list when all trajectories are trained."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.set_trained("q1", s0, True)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert result == []

    def test_returns_empty_when_tree_empty(self):
        """Should return empty list when tree store has no trees."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)

        result = CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        assert result == []


class TestGenerateFromDataloader:
    def test_lazy_init_and_generation(self):
        """Should lazily init dataloader iter and call rollout_batch."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4)
        trainer._replay_dataloader_iter = None
        del trainer._replay_dataloader_iter  # Remove the attribute

        # Mock rollout_batch to return a fake trajectory
        fake_traj = {
            "input_ids": torch.tensor([[1, 2, 10, 3, 4]], dtype=torch.int32),
            "rewards": torch.tensor([[1.0]], dtype=torch.float32),
        }
        trainer.actor = MagicMock()
        trainer.actor.rollout_batch = MagicMock(return_value=[fake_traj])

        # Create a mock dataloader that yields batches
        mock_dataloader = MagicMock()
        mock_workflow = MagicMock()

        # Create a simple iterable that yields a batch of prompts
        mock_prompts = [{"messages": [{"role": "user", "content": "hello"}]}]

        with patch(
            "areal.utils.data.cycle_dataloader",
            return_value=iter([mock_prompts]),
        ):
            result = CacheAwarePPOTrainer._generate_from_dataloader(
                trainer,
                dataloader=mock_dataloader,
                workflow=mock_workflow,
                workflow_kwargs=None,
                group_size=1,
            )

        assert len(result) == 1
        trainer.actor.rollout_batch.assert_called_once()

    def test_reuses_existing_iterator(self):
        """Should reuse existing _replay_dataloader_iter if already initialized."""
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)

        # Pre-create a dataloader iterator
        batch = [{"messages": [{"role": "user", "content": "test"}]}]
        existing_iter = iter([batch, batch])
        trainer._replay_dataloader_iter = existing_iter

        fake_traj = {
            "input_ids": torch.tensor([[1, 2, 10, 3, 4]], dtype=torch.int32),
            "rewards": torch.tensor([[1.0]], dtype=torch.float32),
        }
        trainer.actor = MagicMock()
        trainer.actor.rollout_batch = MagicMock(return_value=[fake_traj])

        result = CacheAwarePPOTrainer._generate_from_dataloader(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
            workflow_kwargs=None,
            group_size=1,
        )

        assert len(result) == 1
        # Verify it used the existing iterator (didn't create a new one)
        assert trainer._replay_dataloader_iter is existing_iter

    def test_returns_empty_on_empty_batch(self):
        """Should return empty list when dataloader yields empty batch."""
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer._replay_dataloader_iter = iter([[]])

        result = CacheAwarePPOTrainer._generate_from_dataloader(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
            workflow_kwargs=None,
            group_size=1,
        )

        assert result == []


class TestReplayPrepareBatchFallback:
    def test_level1_replay_returns_when_history_available(self):
        """Level 1: Should return replay trajectories when history exists for the step."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
        store._training_history[0] = [("q1", s0), ("q2", s1)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 0

        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        assert len(result) == 2
        assert trainer._replay_global_step == 1

    def test_level1_partial_load_still_returns(self):
        """Level 1: Should return whatever was loaded even if some are missing."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        # Reference a non-existent seq_id
        store._training_history[0] = [("q1", s0), ("q2", 999)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 0

        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        assert len(result) == 1
        assert result[0]["_mcts_query_id"] == "q1"

    def test_level2_falls_to_cached_untrained(self):
        """Level 2: Should fall back to cached untrained when replay step missing."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        s1 = store.insert_trajectory("q2", [5, 6, 10, 7, 8], reward=0.5)
        # Only history for step 0, not step 1
        store._training_history[0] = [("q1", s0)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 1  # No history for step 1
        # Wire _load_untrained_from_tree_store to call the real method
        trainer._load_untrained_from_tree_store = (
            lambda: CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        )

        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        # s0 and s1 are both untrained (they were never marked trained)
        assert len(result) >= 1
        assert trainer._replay_global_step == 2

    def test_level3_falls_to_dataloader_generation(self):
        """Level 3: Should fall back to dataloader generation when no replay and no untrained."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.set_trained("q1", s0, True)
        # History exists but for a different step
        store._training_history[0] = [("q1", s0)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 1  # No history for step 1

        # Wire _load_untrained_from_tree_store to the real method (returns [])
        trainer._load_untrained_from_tree_store = (
            lambda: CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        )

        # Mock _generate_from_dataloader
        fake_traj = {
            "input_ids": torch.tensor([[5, 6, 10, 7, 8]], dtype=torch.int32),
            "rewards": torch.tensor([[0.5]], dtype=torch.float32),
        }
        trainer._generate_from_dataloader = MagicMock(return_value=[fake_traj])

        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        trainer._generate_from_dataloader.assert_called_once()
        assert result == [fake_traj]
        assert trainer._replay_global_step == 2

    def test_level1_all_missing_falls_to_level2(self):
        """Level 1: When all replay trajectories are missing, fall to Level 2."""
        from customized_areal.tree_search.config import RolloutCacheConfig
        from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

        store = MCTSTreeStore(_two_turn_splitter)
        s0 = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        # History references only non-existent trajectories
        store._training_history[0] = [("q99", 999)]

        trainer = MagicMock(spec=CacheAwarePPOTrainer)
        trainer.tree_store = store
        trainer.cache_config = RolloutCacheConfig(n_samples=4, replay=True)
        trainer._replay_global_step = 0
        # Wire _load_untrained_from_tree_store to call the real method
        trainer._load_untrained_from_tree_store = (
            lambda: CacheAwarePPOTrainer._load_untrained_from_tree_store(trainer)
        )

        # Level 1 fails (all missing), Level 2 finds s0 untrained
        result = CacheAwarePPOTrainer._replay_prepare_batch(
            trainer,
            dataloader=MagicMock(),
            workflow=MagicMock(),
        )

        assert len(result) == 1
        assert result[0]["_mcts_query_id"] == "q1"
