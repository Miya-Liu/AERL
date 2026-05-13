import os
import shutil
import tempfile
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Default env for integration tests; tests that need missing vars use monkeypatch."""
    os.environ.setdefault("UPSTREAM_OPENAI_BASE_URL", "https://upstream.test/v1")
    if "AERL_DATA_DIR" not in os.environ:
        d = tempfile.mkdtemp(prefix="aerl-data-")
        config._aerl_session_data_dir = d  # type: ignore[attr-defined]
        os.environ["AERL_DATA_DIR"] = d


def pytest_unconfigure(config: pytest.Config) -> None:
    d = getattr(config, "_aerl_session_data_dir", None)
    if d and Path(d).exists():
        shutil.rmtree(d, ignore_errors=True)
    # Only remove if we created it (path prefix)
    if d and str(d).startswith(tempfile.gettempdir()) and "aerl-data-" in str(d):
        os.environ.pop("AERL_DATA_DIR", None)
