import torch

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

        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
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

        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 1
        assert len(need_gen) == 0
        assert cached[0]["cached_count"] == 2
        assert cached[0]["need_gen_count"] == 2

    def test_build_batch_not_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore(_two_turn_splitter)
        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
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

        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        cached, _ = builder.split_prompts([{"_mcts_query_id": "q1"}])
        loaded = builder.load_cached_trajectories(cached)
        assert len(loaded) == 2


class TestSplitGroupedTrajectories:
    def test_individual_kept_as_is(self):
        from customized_areal.tree_search.trainer import _split_grouped_trajectories

        trajs = [
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

        result = _split_grouped_trajectories(trajs)
        assert len(result) == 1
        assert result[0]["input_ids"].shape[0] == 1

    def test_grouped_split_into_individual(self):
        from customized_areal.tree_search.trainer import _split_grouped_trajectories

        # Grouped trajs: shape [2, seq_len]
        trajs = [
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

        result = _split_grouped_trajectories(trajs)
        # 2 items split from grouped = 2 total
        assert len(result) == 2
        assert all(t["input_ids"].shape[0] == 1 for t in result)

    def test_split_empty(self):
        from customized_areal.tree_search.trainer import _split_grouped_trajectories

        assert _split_grouped_trajectories([]) == []
