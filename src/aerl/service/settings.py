from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    data_dir: str
    listen_host: str
    listen_port: int
    service_token: str | None
    repo_root: str


def load_service_settings() -> ServiceSettings:
    data_dir = os.environ.get("AERL_DATA_DIR")
    if not data_dir or not data_dir.strip():
        raise ValueError("AERL_DATA_DIR is required")

    token = os.environ.get("AERL_SERVICE_TOKEN")
    service_token = token.strip() if token and token.strip() else None

    repo = os.environ.get("AERL_REPO_ROOT")
    if repo and repo.strip():
        repo_root = repo.strip()
    else:
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parents[3])

    return ServiceSettings(
        data_dir=data_dir.strip(),
        listen_host=os.environ.get("AERL_SERVICE_HOST", "0.0.0.0").strip(),
        listen_port=_int("AERL_SERVICE_PORT", 8766),
        service_token=service_token,
        repo_root=repo_root,
    )
