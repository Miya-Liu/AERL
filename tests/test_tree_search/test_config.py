from customized_areal.tree_search.config import CacheMode, TreeBackupConfig


class TestCacheMode:
    def test_off_is_default(self):
        config = TreeBackupConfig()
        assert config.mode == CacheMode.OFF

    def test_enum_values(self):
        assert CacheMode.OFF == "off"
        assert CacheMode.IN_TRAINING == "in_training"
        assert CacheMode.CROSS_TRAINING == "cross_training"

    def test_default_checkpoint_dir_empty(self):
        config = TreeBackupConfig()
        assert config.checkpoint_dir == ""

    def test_custom_values(self):
        config = TreeBackupConfig(
            mode=CacheMode.CROSS_TRAINING,
            checkpoint_dir="/tmp/mcts",
        )
        assert config.mode == CacheMode.CROSS_TRAINING
        assert config.checkpoint_dir == "/tmp/mcts"
