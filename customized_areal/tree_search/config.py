from dataclasses import dataclass, field
from enum import Enum


class TreeBackupMode(str, Enum):
    OFF = "off"
    IN_TRAINING = "in_training"
    CROSS_TRAINING = "cross_training"


@dataclass
class TreeBackupConfig:
    mode: TreeBackupMode = TreeBackupMode.OFF
    assistant_marker: str = ""
    checkpoint_dir: str = ""


@dataclass
class RolloutCacheConfig:
    cache_dir: str = ""
    enabled: bool = True
    n_samples: int = 1
