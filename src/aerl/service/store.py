from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    run_id: str
    pipeline: str
    status: str
    created_at: str
    updated_at: str
    request: dict[str, Any]
    config_path: str | None = None
    log_path: str | None = None
    exit_code: int | None = None
    error: str | None = None
    pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunStore:
    def __init__(self, data_dir: str) -> None:
        self._runs_root = Path(data_dir) / "runs"
        self._runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def run_dir(self, run_id: str) -> Path:
        return self._runs_root / run_id

    def create_run(self, pipeline: str, request: dict[str, Any]) -> RunRecord:
        run_id = request.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            rid = run_id.strip()
            if len(rid) > 128:
                raise ValueError("run_id must be at most 128 characters")
        else:
            rid = str(uuid.uuid4())

        now = _now_iso()
        record = RunRecord(
            run_id=rid,
            pipeline=pipeline,
            status="accepted",
            created_at=now,
            updated_at=now,
            request=request,
            config_path=None,
            log_path=str(self.run_dir(rid) / "train.log"),
        )
        with self._lock:
            rd = self.run_dir(rid)
            if rd.exists():
                raise FileExistsError(f"Run already exists: {rid}")
            rd.mkdir(parents=True)
            (rd / "request.json").write_text(
                json.dumps(request, indent=2),
                encoding="utf-8",
            )
            self._write_status(record)
        return record

    def get(self, run_id: str) -> RunRecord | None:
        path = self.run_dir(run_id) / "status.json"
        if not path.is_file():
            return None
        with self._lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        return RunRecord(**data)

    def update(self, record: RunRecord) -> None:
        record.updated_at = _now_iso()
        with self._lock:
            self._write_status(record)

    def _write_status(self, record: RunRecord) -> None:
        path = self.run_dir(record.run_id) / "status.json"
        path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
