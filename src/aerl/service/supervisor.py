from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

from aerl.pipelines.registry import PipelineSpec, get_pipeline, load_runner
from aerl.service.store import RunRecord, RunStore


def _merge_overrides(config_path: Path, overrides: dict[str, Any]) -> Path:
    if not overrides:
        return config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raw = {}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    out = config_path.parent / f"{config_path.stem}.merged.yaml"
    out.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    return out


def _run_subprocess(
    store: RunStore,
    record: RunRecord,
    spec: PipelineSpec,
    config_path: Path,
    repo_root: Path,
) -> None:
    entry = spec.resolve_entrypoint()
    if entry is None or not entry.is_file():
        record.status = "failed"
        record.error = f"Entrypoint not found: {spec.entrypoint}"
        store.update(record)
        return

    log_path = Path(record.log_path or store.run_dir(record.run_id) / "train.log")
    record.status = "running"
    store.update(record)

    cmd = [sys.executable, str(entry), "--config", str(config_path)]
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            record.pid = proc.pid
            store.update(record)
            exit_code = proc.wait()
    except OSError as exc:
        record.status = "failed"
        record.error = str(exc)
        record.exit_code = 1
        store.update(record)
        return

    record.exit_code = exit_code
    record.pid = None
    record.status = "completed" if exit_code == 0 else "failed"
    if exit_code != 0:
        record.error = f"Training process exited with code {exit_code}"
    store.update(record)


def _run_inprocess(
    store: RunStore,
    record: RunRecord,
    spec: PipelineSpec,
) -> None:
    run_dir = store.run_dir(record.run_id)
    record.status = "running"
    store.update(record)
    try:
        runner = load_runner(spec)
        request = dict(record.request)
        request.setdefault("run_id", record.run_id)
        exit_code = int(runner(run_dir, request))
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
        record.exit_code = 1
        store.update(record)
        return

    record.exit_code = exit_code
    record.status = "completed" if exit_code == 0 else "failed"
    if exit_code != 0:
        record.error = f"Runner exited with code {exit_code}"
    store.update(record)


def start_run_async(
    store: RunStore,
    record: RunRecord,
    *,
    repo_root: str,
) -> None:
    spec = get_pipeline(record.pipeline)
    overrides = record.request.get("overrides")
    if overrides is not None and not isinstance(overrides, dict):
        overrides = {}

    record.status = "provisioning"
    store.update(record)

    if spec.runner:
        thread = threading.Thread(
            target=_run_inprocess,
            args=(store, record, spec),
            daemon=True,
        )
        thread.start()
        return

    default_cfg = spec.resolve_default_config()
    if default_cfg is None or not default_cfg.is_file():
        record.status = "failed"
        record.error = f"default_config not found: {spec.default_config}"
        store.update(record)
        return

    merged = _merge_overrides(default_cfg, overrides or {})
    dest = store.run_dir(record.run_id) / "config.yaml"
    shutil.copy2(merged, dest)
    record.config_path = str(dest)
    store.update(record)

    thread = threading.Thread(
        target=_run_subprocess,
        args=(store, record, spec, dest, Path(repo_root)),
        daemon=True,
    )
    thread.start()
