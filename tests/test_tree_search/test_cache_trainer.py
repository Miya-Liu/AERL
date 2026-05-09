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


class TestCacheAwareBatchBuilder:
    def test_build_batch_fully_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        for i in range(4):
            node = _make_node(
                [1, 2, 3, 4, 5 + i],
                [0, 0, 1, 1, 1],
                reward=1.0 / (i + 1),
                query_id="q1",
            )
            store.insert_batch([node])

        builder = _CacheAwareBatchBuilder(store, n_samples=4)
        prompts = [{"query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 1
        assert len(need_gen) == 0
        assert cached[0]["cached_count"] == 4

    def test_build_batch_partially_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        for i in range(2):
            node = _make_node(
                [1, 2, 3, 4, 5 + i],
                [0, 0, 1, 1, 1],
                reward=1.0 / (i + 1),
                query_id="q1",
            )
            store.insert_batch([node])

        builder = _CacheAwareBatchBuilder(store, n_samples=4)
        prompts = [{"query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        # 2 untrained < 4 n_samples → goes to need_gen, not cached
        assert len(cached) == 0
        assert len(need_gen) == 1

    def test_build_batch_not_cached(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        builder = _CacheAwareBatchBuilder(store, n_samples=4)
        prompts = [{"query_id": "q1"}]
        cached, need_gen = builder.split_prompts(prompts)
        assert len(cached) == 0
        assert len(need_gen) == 1

    def test_load_cached_trajectories(self):
        from customized_areal.tree_search.trainer import _CacheAwareBatchBuilder

        store = MCTSTreeStore()
        for i in range(4):
            node = _make_node(
                [i, i + 1, i + 2, i + 3],
                [0, 0, 1, 1],
                reward=1.0 / (i + 1),
                query_id="q1",
            )
            store.insert_batch([node])

        builder = _CacheAwareBatchBuilder(store, n_samples=4)
        cached, _ = builder.split_prompts([{"query_id": "q1"}])
        loaded = builder.load_cached_trajectories(cached)
        assert len(loaded) == 4
