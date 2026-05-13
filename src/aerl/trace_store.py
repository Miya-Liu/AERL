from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Required keys for proxy-style JSONL records (spec §5); see tests.
PROXY_TRACE_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "request_id",
        "ts_request_received",
        "ts_upstream_sent",
        "ts_response_complete",
        "method",
        "path",
        "upstream_status",
    }
)


class TraceStore:
    """Append-only JSONL trace file under ``data_dir`` (thread-safe)."""

    def __init__(self, data_dir: str, *, filename: str = "traces.jsonl") -> None:
        root = Path(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._path = root / filename
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)

    @property
    def path(self) -> Path:
        return self._path
