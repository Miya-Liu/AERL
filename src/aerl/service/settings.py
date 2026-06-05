from __future__ import annotations

import os
from dataclasses import dataclass

from aerl.paths import find_repo_root, resolve_data_dir


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
    repo_root = str(find_repo_root())

    token = os.environ.get("AERL_SERVICE_TOKEN")
    service_token = token.strip() if token and token.strip() else None

    return ServiceSettings(
        data_dir=resolve_data_dir(),
        listen_host=os.environ.get("AERL_SERVICE_HOST", "0.0.0.0").strip(),
        listen_port=_int("AERL_SERVICE_PORT", 8766),
        service_token=service_token,
        repo_root=repo_root,
    )
