from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def run_mock(run_dir: Path, request: dict[str, Any]) -> int:
    """Simulate a short GRPO training run (no GPU). Returns exit code."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("mock_grpo: starting\n")
        pipeline = request.get("pipeline", "mock_grpo")
        log.write(f"pipeline={pipeline}\n")
        time.sleep(0.5)
        log.write("mock_grpo: epoch 1/1\n")
        time.sleep(0.5)
        log.write("mock_grpo: completed\n")

    train_id = request.get("run_id") or "mock-train-id"
    (run_dir / "train_id.json").write_text(
        json.dumps({"train_id": str(train_id)}),
        encoding="utf-8",
    )
    return 0
