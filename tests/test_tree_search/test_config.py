from customized_areal.tree_search.config import TreeBackupConfig, TreeBackupMode


class TestTreeBackupMode:
    def test_off_is_default(self):
        config = TreeBackupConfig()
        assert config.mode == TreeBackupMode.OFF

    def test_enum_values(self):
        assert TreeBackupMode.OFF == "off"
        assert TreeBackupMode.IN_TRAINING == "in_training"
        assert TreeBackupMode.CROSS_TRAINING == "cross_training"

    def test_default_assistant_marker_empty(self):
        config = TreeBackupConfig()
        assert config.assistant_marker == ""

    def test_default_checkpoint_dir_empty(self):
        config = TreeBackupConfig()
        assert config.checkpoint_dir == ""

    def test_custom_values(self):
        config = TreeBackupConfig(
            mode=TreeBackupMode.CROSS_TRAINING,
            assistant_marker="<|im_start|>assistant",
            checkpoint_dir="/tmp/mcts",
        )
        assert config.mode == TreeBackupMode.CROSS_TRAINING
        assert config.assistant_marker == "<|im_start|>assistant"
        assert config.checkpoint_dir == "/tmp/mcts"
