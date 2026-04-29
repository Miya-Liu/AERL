# tests/test_tree_search/test_trainer.py
from unittest.mock import MagicMock

from customized_areal.tree_search.config import AdvantageMode, TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.trainer import (
    patch_ppo_actor_for_tree_backup,
    unpatch_ppo_actor,
)

from areal.trainer.ppo.actor import PPOActor


class TestPatchOuterMethod:
    def setup_method(self):
        unpatch_ppo_actor()

    def teardown_method(self):
        unpatch_ppo_actor()

    def test_patch_replaces_compute_advantages(self):
        original = PPOActor.compute_advantages
        patch_ppo_actor_for_tree_backup(advantage_mode=AdvantageMode.TREE)
        assert PPOActor.compute_advantages is not original
        assert hasattr(PPOActor, "_original_compute_advantages")

    def test_unpatch_restores_original(self):
        original = PPOActor.compute_advantages
        patch_ppo_actor_for_tree_backup(advantage_mode=AdvantageMode.TREE)
        unpatch_ppo_actor()
        assert PPOActor.compute_advantages is original
        assert not hasattr(PPOActor, "_original_compute_advantages")

    def test_double_patch_replaces_previous(self):
        original = PPOActor.compute_advantages
        patch_ppo_actor_for_tree_backup(advantage_mode=AdvantageMode.TREE)
        first_patched = PPOActor.compute_advantages
        patch_ppo_actor_for_tree_backup(advantage_mode=AdvantageMode.GAE)
        assert PPOActor._original_compute_advantages is original
        unpatch_ppo_actor()
        assert PPOActor.compute_advantages is original


class TestUnpatchSafety:
    def setup_method(self):
        unpatch_ppo_actor()

    def teardown_method(self):
        unpatch_ppo_actor()

    def test_unpatch_without_patch_is_safe(self):
        unpatch_ppo_actor()


class TestTreeBackupConfigDefaults:
    def test_default_mode_is_off(self):
        config = TreeBackupConfig()
        assert config.mode == TreeBackupMode.OFF

    def test_no_assistant_marker_field(self):
        config = TreeBackupConfig()
        assert not hasattr(config, "assistant_marker")
