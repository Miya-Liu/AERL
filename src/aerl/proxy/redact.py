from __future__ import annotations

from collections.abc import Mapping


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a shallow copy of headers with ``Authorization`` redacted (spec §7)."""
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() == "authorization":
            out[key] = _redact_authorization(value)
        else:
            out[key] = value
    return out


def _redact_authorization(value: str) -> str:
    raw = value.strip()
    if raw.lower().startswith("bearer "):
        secret = raw[7:].strip()
        if len(secret) <= 4:
            return "Bearer ****"
        return f"Bearer ****{secret[-4:]}"
    return "[REDACTED]"
