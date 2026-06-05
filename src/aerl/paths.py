"""Repository-relative paths for a self-contained local checkout."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root() -> Path:
    """Return the repository root (directory containing ``src/aerl``)."""
    env = os.environ.get("AERL_REPO_ROOT")
    if env and env.strip():
        return Path(env.strip()).resolve()
    return Path(__file__).resolve().parents[2]


def default_data_dir(*, repo_root: Path | None = None) -> str:
    """Writable runtime data directory: ``<repo>/.data`` (gitignored)."""
    root = repo_root or find_repo_root()
    return str((root / ".data").resolve())


def resolve_data_dir() -> str:
    """``AERL_DATA_DIR`` if set, otherwise ``<repo>/.data``."""
    raw = os.environ.get("AERL_DATA_DIR")
    if raw and raw.strip():
        return raw.strip()
    return default_data_dir()
