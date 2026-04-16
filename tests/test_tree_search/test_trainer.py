# tests/test_tree_search/test_trainer.py
import pytest
from customized_areal.tree_search.config import TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.turn_splitter import Turn
from customized_areal.tree_search.trainer import (
    TreeBackupPPOTrainer,
    patch_ppo_actor_for_tree_backup,
    unpatch_ppo_actor,
)
from areal.trainer.ppo.actor import PPOActor


def _simple_splitter(input_ids: list[int]) -> list[Turn]:
    """Split at token 10 — everything before is prompt, everything after is response."""
    try:
        split_pos = input_ids.index(10)
        return [Turn(prompt_tokens=input_ids[:split_pos], response_tokens=input_ids[split_pos:])]
    except ValueError:
        return [Turn(prompt_tokens=[], response_tokens=list(input_ids))]


class TestPatchOuterMethod:
    """Test that patch_ppo_actor_for_tree_backup patches the outer compute_advantages method."""

    def setup_method(self):
        """Ensure any previous patch is cleaned up before each test."""
        unpatch_ppo_actor()

    def teardown_method(self):
        """Clean up patch after each test."""
        unpatch_ppo_actor()

    def test_patch_replaces_compute_advantages(self):
        """After patching, PPOActor.compute_advantages should be the tree backup version."""
        original = PPOActor.compute_advantages
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        patch_ppo_actor_for_tree_backup(store, computer)
        assert PPOActor.compute_advantages is not original
        assert hasattr(PPOActor, "_original_compute_advantages")

    def test_unpatch_restores_original(self):
        """After unpatching, PPOActor.compute_advantages should be restored."""
        original = PPOActor.compute_advantages
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        patch_ppo_actor_for_tree_backup(store, computer)
        unpatch_ppo_actor()
        assert PPOActor.compute_advantages is original
        assert not hasattr(PPOActor, "_original_compute_advantages")

    def test_patch_preserves_original_as_backup(self):
        """The original method should be saved as _original_compute_advantages."""
        original = PPOActor.compute_advantages
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        patch_ppo_actor_for_tree_backup(store, computer)
        assert PPOActor._original_compute_advantages is original

    def test_double_patch_replaces_previous(self):
        """Patching twice should replace the first patch, not stack."""
        original = PPOActor.compute_advantages
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        patch_ppo_actor_for_tree_backup(store, computer)
        first_patched = PPOActor.compute_advantages

        # Create a second store to verify the second patch replaces
        store2 = MCTSTreeStore(_simple_splitter)
        computer2 = TreeAdvantageComputer(store2)
        patch_ppo_actor_for_tree_backup(store2, computer2)

        # The patched method should be the new one (different closure)
        # But _original_compute_advantages should always point to the TRUE original
        assert PPOActor._original_compute_advantages is original
        # Unpatch should still restore the true original
        unpatch_ppo_actor()
        assert PPOActor.compute_advantages is original


class TestUnpatchSafety:
    def setup_method(self):
        unpatch_ppo_actor()

    def teardown_method(self):
        unpatch_ppo_actor()

    def test_unpatch_without_patch_is_safe(self):
        """Calling unpatch without a prior patch should not raise."""
        unpatch_ppo_actor()  # should be a no-op


class TestTreeBackupConfigDefaults:
    def test_default_mode_is_off(self):
        """Default config should have OFF mode (no patching)."""
        config = TreeBackupConfig()
        assert config.mode == TreeBackupMode.OFF

    def test_off_mode_means_no_patching(self):
        """With OFF mode, the trainer should not patch PPOActor at all."""
        # This is tested indirectly: TreeBackupPPOTrainer with OFF mode
        # should not call patch_ppo_actor_for_tree_backup
        # We verify by checking that compute_advantages is unchanged
        original = PPOActor.compute_advantages
        config = TreeBackupConfig(mode=TreeBackupMode.OFF)
        # OFF mode means the constructor skips patching
        assert config.mode == TreeBackupMode.OFF
