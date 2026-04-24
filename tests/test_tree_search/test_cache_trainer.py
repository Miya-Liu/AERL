import torch
from unittest.mock import MagicMock

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
