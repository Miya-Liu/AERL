from unittest.mock import MagicMock, patch

import pytest

from customized_areal.tree_search.config import AdvantageMode, LossMode
from customized_areal.tree_search.patches import TreeSearchPatches

from areal.trainer.ppo.actor import PPOActor


@pytest.fixture
def saved_ppo_actor_state():
    """Fixture to save and restore PPOActor state around tests"""
    original_compute = PPOActor.compute_advantages
    had_sentinel = hasattr(PPOActor, "_original_compute_advantages")
    original_sentinel = getattr(PPOActor, "_original_compute_advantages", None)
    yield
    PPOActor.compute_advantages = original_compute
    if had_sentinel:
        PPOActor._original_compute_advantages = original_sentinel
    elif hasattr(PPOActor, "_original_compute_advantages"):
        del PPOActor._original_compute_advantages


@pytest.fixture
def mock_engine():
    """Fixture to create a mock engine object for testing"""
    engine = MagicMock()
    engine._wrap_openai_agent = MagicMock(return_value="original_wrap")
    engine._resolve_workflow = MagicMock(return_value="original_resolve")
    engine.workflow_executor = MagicMock()
    engine.config = MagicMock()
    engine.config.agent = MagicMock(
        mode="mode",
        admin_api_key="key",
        turn_discount=1.0,
        export_style="concat",
        subproc_max_workers=1,
    )
    engine._proxy_gateway_addr = None
    return engine


class TestApplyRestore:
    """Test applying and restoring patches"""

    def test_apply_then_restore_restores_originals(
        self, mock_engine, saved_ppo_actor_state
    ):
        patches = TreeSearchPatches(mock_engine, AdvantageMode.TREE, LossMode.GRPO, 4)
        original_compute = PPOActor.compute_advantages
        original_wrap = mock_engine._wrap_openai_agent
        original_resolve = mock_engine._resolve_workflow
        original_executor = mock_engine.workflow_executor

        patches.apply()
        assert PPOActor.compute_advantages != original_compute
        assert mock_engine._wrap_openai_agent != original_wrap
        assert mock_engine._resolve_workflow != original_resolve
        assert mock_engine.workflow_executor != original_executor

        patches.restore()
        assert PPOActor.compute_advantages == original_compute
        assert mock_engine._wrap_openai_agent == original_wrap
        assert mock_engine._resolve_workflow == original_resolve
        assert mock_engine.workflow_executor == original_executor
        assert not hasattr(PPOActor, "_original_compute_advantages")


class TestIdempotency:
    """Test idempotency of apply method"""

    def test_apply_twice_is_noop(self, mock_engine, saved_ppo_actor_state):
        patches = TreeSearchPatches(mock_engine, AdvantageMode.TREE, LossMode.GRPO, 4)
        patches.apply()
        first_compute = PPOActor.compute_advantages
        patches.apply()
        second_compute = PPOActor.compute_advantages
        assert first_compute is second_compute
        patches.restore()


class TestContextManager:
    """Test context manager behavior"""

    def test_context_manager_restores_on_normal_exit(
        self, mock_engine, saved_ppo_actor_state
    ):
        original_compute = PPOActor.compute_advantages
        with TreeSearchPatches(mock_engine, AdvantageMode.TREE, LossMode.GRPO, 4):
            assert PPOActor.compute_advantages != original_compute
        assert PPOActor.compute_advantages == original_compute

    def test_context_manager_restores_on_exception(
        self, mock_engine, saved_ppo_actor_state
    ):
        original_compute = PPOActor.compute_advantages
        with pytest.raises(ValueError):
            with TreeSearchPatches(mock_engine, AdvantageMode.TREE, LossMode.GRPO, 4):
                assert PPOActor.compute_advantages != original_compute
                raise ValueError("Test exception")
        assert PPOActor.compute_advantages == original_compute


class TestTreeSearchWrap:
    """Test tree search wrap functionality"""

    def test_raises_on_missing_agent_config(self, saved_ppo_actor_state):
        engine = MagicMock()
        engine._wrap_openai_agent = MagicMock()
        engine._resolve_workflow = MagicMock()
        engine.workflow_executor = MagicMock()
        engine.config = MagicMock()
        engine.config.agent = None
        engine._proxy_gateway_addr = None

        patches = TreeSearchPatches(engine, AdvantageMode.TREE, LossMode.GRPO, 4)
        wrap_func = patches._build_tree_search_wrap()
        with pytest.raises(RuntimeError, match="config.agent is None"):
            wrap_func(MagicMock(), "test_proxy_addr")

    def test_returns_treesearch_workflow(self, saved_ppo_actor_state):
        from customized_areal.tree_search.grouped_workflow import (
            TreeSearchGroupedRolloutWorkflow,
        )

        engine = MagicMock()
        engine._wrap_openai_agent = MagicMock()
        engine._resolve_workflow = MagicMock()
        engine.workflow_executor = MagicMock()
        engine.config = MagicMock()
        engine.config.agent = MagicMock(
            mode="mode",
            admin_api_key="key",
            turn_discount=1.0,
            export_style="concat",
            subproc_max_workers=1,
        )
        engine._proxy_gateway_addr = None

        patches = TreeSearchPatches(engine, AdvantageMode.TREE, LossMode.GRPO, 4)
        wrap_func = patches._build_tree_search_wrap()

        with patch(
            "customized_areal.tree_search.patches.QueryIDProxyWorkflow",
            return_value=MagicMock(),
        ):
            result = wrap_func(MagicMock(), "test_proxy_addr")
            assert isinstance(result, TreeSearchGroupedRolloutWorkflow)


class TestRestoreSafety:
    """Test restore method safety"""

    def test_restore_without_apply_is_noop(self, mock_engine, saved_ppo_actor_state):
        patches = TreeSearchPatches(mock_engine, AdvantageMode.TREE, LossMode.GRPO, 4)
        patches.restore()
        assert not patches._applied
        assert len(patches._saved) == 0

    def test_restore_twice_is_safe(self, mock_engine, saved_ppo_actor_state):
        patches = TreeSearchPatches(mock_engine, AdvantageMode.TREE, LossMode.GRPO, 4)
        original_compute = PPOActor.compute_advantages
        patches.apply()
        patches.restore()
        patches.restore()
        assert PPOActor.compute_advantages == original_compute
