import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


def _make_traj_for_store(
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
        "_mcts_query_id": query_id,
    }


class TestCacheAwareBatchBuilder:
    def test_build_batch_fully_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        for i in range(4):
            traj = _make_traj_for_store(
                [1, 2, 3, 4, 5 + i],
                [0, 0, 1, 1, 1],
                reward=1.0 / (i + 1),
                query_id="q1",
            )
            store.insert_batch([traj])

        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 1
        assert len(need_gen) == 0
        assert cached[0]["cached_count"] == 4
        assert cached[0]["need_gen_count"] == 0

    def test_build_batch_partially_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        for i in range(2):
            traj = _make_traj_for_store(
                [1, 2, 3, 4, 5 + i],
                [0, 0, 1, 1, 1],
                reward=1.0 / (i + 1),
                query_id="q1",
            )
            store.insert_batch([traj])

        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 1
        assert len(need_gen) == 0
        assert cached[0]["cached_count"] == 2
        assert cached[0]["need_gen_count"] == 2

    def test_build_batch_not_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        prompts = [{"_mcts_query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 0
        assert len(need_gen) == 1

    def test_load_cached_trajectories(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        t1 = _make_traj_for_store([1, 2, 3, 4], [0, 0, 1, 1], reward=1.0, query_id="q1")
        t2 = _make_traj_for_store([5, 6, 7, 8], [0, 0, 1, 1], reward=0.5, query_id="q1")
        store.insert_batch([t1, t2])

        builder = _CacheAwareBatchBuilder(store, n_samples=4, tokenizer=None)
        cached, _ = builder.split_prompts([{"_mcts_query_id": "q1"}])
        loaded = builder.load_cached_trajectories(cached)
        assert len(loaded) == 2
