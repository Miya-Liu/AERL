# tests/test_tree_search/test_trainer.py

from unittest.mock import MagicMock

from customized_areal.tree_search.config import (
    AdvantageMode,
    LossMode,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search.patches import TreeSearchPatches

from areal.trainer.ppo.actor import PPOActor


def _make_mock_engine():
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
    return engine


class TestPatchApplyRestore:
    def setup_method(self):
        # Ensure no leftover sentinel from prior tests
        if hasattr(PPOActor, "_original_compute_advantages"):
            del PPOActor._original_compute_advantages

    def teardown_method(self):
        if hasattr(PPOActor, "_original_compute_advantages"):
            del PPOActor._original_compute_advantages

    def test_apply_replaces_compute_advantages(self):
        original = PPOActor.compute_advantages
        patches = TreeSearchPatches(
            _make_mock_engine(), AdvantageMode.TREE, LossMode.GRPO, 4
        )
        patches.apply()
        assert PPOActor.compute_advantages is not original
        assert hasattr(PPOActor, "_original_compute_advantages")
        patches.restore()

    def test_restore_recovers_original(self):
        original = PPOActor.compute_advantages
        patches = TreeSearchPatches(
            _make_mock_engine(), AdvantageMode.TREE, LossMode.GRPO, 4
        )
        patches.apply()
        patches.restore()
        assert PPOActor.compute_advantages is original
        assert not hasattr(PPOActor, "_original_compute_advantages")

    def test_apply_twice_is_noop(self):
        original = PPOActor.compute_advantages
        patches = TreeSearchPatches(
            _make_mock_engine(), AdvantageMode.TREE, LossMode.GRPO, 4
        )
        patches.apply()
        first_patched = PPOActor.compute_advantages
        patches.apply()  # second call is a no-op
        assert PPOActor.compute_advantages is first_patched
        patches.restore()
        assert PPOActor.compute_advantages is original


class TestRestoreSafety:
    def setup_method(self):
        if hasattr(PPOActor, "_original_compute_advantages"):
            del PPOActor._original_compute_advantages

    def test_restore_without_apply_is_safe(self):
        patches = TreeSearchPatches(
            _make_mock_engine(), AdvantageMode.TREE, LossMode.GRPO, 4
        )
        patches.restore()  # should not raise


class TestTreeBackupConfigDefaults:
    def test_default_mode_is_off(self):
        config = TreeBackupConfig()
        assert config.mode == TreeBackupMode.OFF

    def test_no_assistant_marker_field(self):
        config = TreeBackupConfig()
        assert not hasattr(config, "assistant_marker")
