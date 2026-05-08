"""Tests for tree search bug fixes."""

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, Node


def _make_node(reward: float = 1.0) -> Node:
    """Create a minimal Node for testing."""
    return Node(
        input_ids=[1, 2, 3],
        loss_mask=[0, 1, 1],
        logprobs=[0.0, -0.5, -0.3],
        versions=[-1, 0, 0],
        outcome_reward=reward,
    )


class TestSetTrainedSignature:
    """Bug #13: set_trained should not take unused query_id parameter."""

    def test_set_trained_accepts_node_id_only(self):
        store = MCTSTreeStore()
        node = _make_node()
        store.insert_batch([node])
        node_id = node.node_id

        # Should work with just node_id (no query_id param)
        store.set_trained(node_id, True)
        assert store.is_trained(node_id) is True
        assert store.get_reward(node_id) == 1.0


class TestDictSetInsteadOfDictNone:
    """Bug #14: advantage computer should use set[int] not dict[int, None]."""

    def test_compute_uses_set_for_node_ids(self):
        from customized_areal.tree_search.advantage import TreeAdvantageComputer

        store = MCTSTreeStore()
        for r in [1.0, 2.0, 3.0]:
            node = _make_node(reward=r)
            node.query_id = "q1"
            store.insert_batch([node])

        computer = TreeAdvantageComputer(store)
        nodes = store.load_trajectories("q1", 3)
        computer.compute(nodes)

        for node in nodes:
            assert node.advantages is not None


class TestInsertBatchSkipDuplicates:
    """Bug #1: insert_batch should skip already-inserted nodes."""

    def test_insert_batch_skips_nodes_with_existing_node_id(self):
        store = MCTSTreeStore()
        node = _make_node()
        store.insert_batch([node])
        # First node gets ID 1 (_next_node_id starts at 1)
        first_id = node.node_id
        first_count = len(store.trajectories.get("", []))

        # Re-insert the same node (simulating cache reuse)
        store.insert_batch([node])

        # Should NOT create a duplicate
        assert node.node_id == first_id
        assert len(store.trajectories.get("", [])) == first_count

    def test_insert_batch_allows_new_nodes(self):
        store = MCTSTreeStore()
        node_a = _make_node(reward=1.0)
        node_b = _make_node(reward=2.0)
        store.insert_batch([node_a])
        store.insert_batch([node_b])
        assert node_a.node_id != node_b.node_id
        assert len(store.trajectories.get("", [])) == 2


class TestQueryIdCheckpoint:
    """Bug #3: query_id lost on checkpoint deserialization."""

    def test_query_id_survives_save_load(self, tmp_path):
        from customized_areal.tree_search.checkpoint import TreeCheckpointManager

        store = MCTSTreeStore()
        node = _make_node()
        node.query_id = "test_query_123"
        store.insert_batch([node])
        assert node.query_id == "test_query_123"

        manager = TreeCheckpointManager(str(tmp_path))
        manager.save(store)

        loaded = manager.load()
        loaded_nodes = loaded.trajectories.get("test_query_123", [])
        assert len(loaded_nodes) == 1
        assert getattr(loaded_nodes[0], "query_id", None) == "test_query_123"


class TestBesselVariance:
    """Bug #2: GRPO normalization should use Bessel-corrected variance."""

    def test_uses_sample_variance_not_population(self):
        from customized_areal.tree_search.advantage import TreeAdvantageComputer

        store = MCTSTreeStore()
        # Insert 4 nodes with known rewards
        for r in [1.0, 2.0, 3.0, 4.0]:
            node = _make_node(reward=r)
            node.query_id = "q1"
            store.insert_batch([node])

        computer = TreeAdvantageComputer(store)
        nodes = store.load_trajectories("q1", 4)
        computer.compute(nodes)

        # Population variance of [1,2,3,4] = 1.25, std = 1.118
        # Sample variance of [1,2,3,4] = 5/3 = 1.667, std = 1.291
        # With sample variance: (1 - 2.5) / (1.291 + 1e-8) ≈ -1.161
        # With population variance: (1 - 2.5) / (1.118 + 1e-8) ≈ -2.236
        first_adv = nodes[0].advantages
        response_adv = first_adv[first_adv != 0]
        assert response_adv.numel() > 0
        # Should be close to -1.161 (sample), not -2.236 (population)
        assert abs(response_adv[0].item() - (-1.161)) < 0.05, (
            f"Expected sample variance normalization (~-1.161), got {response_adv[0].item()}"
        )


class TestEpisodeIdUniqueness:
    """Bug #8: episode_id should be unique across queries and epochs."""

    def test_different_groups_get_different_episode_ids(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from customized_areal.tree_search.grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

        # Create two groups of nodes with empty query_id
        node_a = _make_node()
        node_b = _make_node()
        node_c = _make_node()
        node_d = _make_node()

        inner = MagicMock()
        inner.arun_episode = AsyncMock(
            side_effect=[
                [node_a, node_b],
                [node_c, node_d],
            ]
        )

        wf = TreeSearchGroupedRolloutWorkflow(
            workflow=inner, group_size=2, logger=MagicMock()
        )
        result = asyncio.run(wf.arun_episode(MagicMock(), {"query_id": ""}))

        # Nodes in the same group share the same episode_id.
        # Different groups get different episode_ids.
        # Each episode_id contains a UUID hex suffix for uniqueness.
        group_0_ids = {n.episode_id for n in result[:2]}
        group_1_ids = {n.episode_id for n in result[2:]}
        assert len(group_0_ids) == 1, (
            f"Group 0 nodes should share episode_id, got {group_0_ids}"
        )
        assert len(group_1_ids) == 1, (
            f"Group 1 nodes should share episode_id, got {group_1_ids}"
        )
        assert group_0_ids != group_1_ids, (
            f"Different groups should have different episode_ids, got {group_0_ids} vs {group_1_ids}"
        )
        # Verify UUID suffix is present (at least 8 hex chars after the last underscore)
        for eid in [group_0_ids.pop(), group_1_ids.pop()]:
            suffix = eid.rsplit("_", 1)[-1]
            assert len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix), (
                f"episode_id should have 8-char hex UUID suffix, got suffix={suffix} in {eid}"
            )
