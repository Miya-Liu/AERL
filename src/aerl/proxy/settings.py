from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from aerl.proxy.llm_trace import PricingTable
from aerl.proxy.pricing_config import load_pricing_table


def _truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def normalize_upstream_base(url: str) -> str:
    u = url.strip()
    if not u:
        raise ValueError("UPSTREAM_OPENAI_BASE_URL is empty")
    parts = urlsplit(u)
    if not parts.scheme or not parts.netloc:
        raise ValueError("UPSTREAM_OPENAI_BASE_URL must include scheme and host")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def join_upstream_subpath(upstream_base: str, subpath: str) -> str:
    """Join normalized upstream base (…/v1) with a path segment like ``chat/completions``."""
    base = upstream_base.rstrip("/")
    tail = subpath.lstrip("/")
    return f"{base}/{tail}"


@dataclass(frozen=True, slots=True)
class Settings:
    upstream_openai_base_url: str
    data_dir: str
    job_webhook_url: str | None
    job_webhook_auth: str | None
    max_body_bytes: int
    max_buffered_request_bytes: int
    ready_check_upstream: bool
    listen_host: str
    listen_port: int
    ready_probe_path: str
    job_webhook_timeout: float
    upstream_timeout: float
    ready_auth: str | None
    max_job_bytes: int
    pricing: PricingTable | None


def load_settings() -> Settings:
    upstream = os.environ.get("UPSTREAM_OPENAI_BASE_URL")
    if not upstream or not upstream.strip():
        raise ValueError("UPSTREAM_OPENAI_BASE_URL is required")

    data_dir = os.environ.get("AERL_DATA_DIR")
    if not data_dir or not data_dir.strip():
        raise ValueError("AERL_DATA_DIR is required")

    webhook = os.environ.get("AERL_JOB_WEBHOOK_URL")
    webhook_url = webhook.strip() if webhook and webhook.strip() else None

    auth = os.environ.get("AERL_JOB_WEBHOOK_AUTH")
    webhook_auth = auth.strip() if auth and auth.strip() else None

    ready_a = os.environ.get("AERL_READY_AUTH")
    ready_auth = ready_a.strip() if ready_a and ready_a.strip() else None

    pricing = load_pricing_table(os.environ.get("AERL_PRICING_JSON"))

    return Settings(
        upstream_openai_base_url=normalize_upstream_base(upstream),
        data_dir=data_dir.strip(),
        job_webhook_url=webhook_url,
        job_webhook_auth=webhook_auth,
        max_body_bytes=_int("AERL_MAX_BODY_BYTES", 4 * 1024 * 1024),
        max_buffered_request_bytes=_int(
            "AERL_MAX_BUFFERED_REQUEST_BYTES", 32 * 1024 * 1024
        ),
        ready_check_upstream=_truthy(os.environ.get("AERL_READY_CHECK_UPSTREAM")),
        listen_host=os.environ.get("AERL_LISTEN_HOST", "0.0.0.0").strip(),
        listen_port=_int("AERL_LISTEN_PORT", 8765),
        ready_probe_path=os.environ.get("AERL_READY_PROBE_PATH", "models").strip(),
        job_webhook_timeout=_float("AERL_JOB_WEBHOOK_TIMEOUT", 30.0),
        upstream_timeout=_float("AERL_UPSTREAM_TIMEOUT", 120.0),
        ready_auth=ready_auth,
        max_job_bytes=_int("AERL_MAX_JOB_BYTES", 1024 * 1024),
        pricing=pricing,
    )
