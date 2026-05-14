from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def extract_json_user(request_body: bytes) -> str | None:
    if not request_body:
        return None
    try:
        obj = json.loads(request_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    u = obj.get("user")
    if isinstance(u, str) and u.strip():
        return u.strip()
    return None


def extract_caller_label(headers: Mapping[str, str]) -> str | None:
    """Optional orchestrator identity (not OpenAI auth)."""
    for key in ("X-AERL-User", "X-User-Id", "X-Request-User"):
        v = _header_get_ci(headers, key)
        if v and v.strip():
            return v.strip()
    return None


def _header_get_ci(headers: Mapping[str, str], want: str) -> str | None:
    wl = want.lower()
    for k, v in headers.items():
        if k.lower() == wl:
            return v
    return None


def normalize_usage(usage: Mapping[str, Any]) -> dict[str, int] | None:
    """Return integer token counts suitable for logging and cost math."""
    if not usage:
        return None
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    out: dict[str, int] = {}
    if isinstance(pt, int) and pt >= 0:
        out["prompt_tokens"] = pt
    if isinstance(ct, int) and ct >= 0:
        out["completion_tokens"] = ct
    if isinstance(tt, int) and tt >= 0:
        out["total_tokens"] = tt
    if not out:
        return None
    if "total_tokens" not in out and "prompt_tokens" in out and "completion_tokens" in out:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def build_usage_record(usage_obj: dict[str, Any]) -> dict[str, Any] | None:
    norm = normalize_usage(usage_obj)
    if norm is None:
        return None
    out: dict[str, Any] = dict(norm)
    out["upstream"] = usage_obj
    return out


def extract_usage_from_response_json(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    u = obj.get("usage")
    if not isinstance(u, dict):
        return None
    return build_usage_record(u)


def extract_usage_from_sse_bytes(raw: bytes) -> dict[str, Any] | None:
    """Scan OpenAI-style SSE for the last non-empty ``usage`` object on a ``data:`` line."""
    last: dict[str, Any] | None = None
    buf = raw.decode("utf-8", errors="replace")
    for line in buf.splitlines():
        line = line.rstrip("\r")
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        u = obj.get("usage")
        if isinstance(u, dict) and u:
            merged = extract_usage_from_response_json({"usage": u})
            if merged is not None:
                last = merged
    return last


@dataclass(frozen=True, slots=True)
class PricingTable:
    """USD per 1M tokens (input / output)."""

    default: tuple[float, float] | None
    per_model: dict[str, tuple[float, float]]

    def rates_for_model(self, model: str | None) -> tuple[float, float] | None:
        if model and model in self.per_model:
            return self.per_model[model]
        return self.default


def estimate_cost_usd(
    usage_record: Mapping[str, Any] | None,
    model: str | None,
    pricing: PricingTable | None,
) -> float | None:
    if usage_record is None or pricing is None:
        return None
    if "prompt_tokens" not in usage_record or "completion_tokens" not in usage_record:
        return None
    rates = pricing.rates_for_model(model)
    if rates is None:
        return None
    inp_rate, out_rate = rates
    pt = int(usage_record.get("prompt_tokens", 0))
    ct = int(usage_record.get("completion_tokens", 0))
    return round((pt / 1_000_000.0) * inp_rate + (ct / 1_000_000.0) * out_rate, 8)


class SSEAggregator:
    """Parse OpenAI-style SSE: assistant text deltas + last ``usage`` snapshot."""

    def __init__(self) -> None:
        self._buf = ""
        self._parts: list[str] = []
        self.usage: dict[str, Any] | None = None

    def feed(self, data: bytes) -> None:
        self._buf += data.decode("utf-8", errors="replace")
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                u = obj.get("usage")
                if isinstance(u, dict) and u:
                    merged = extract_usage_from_response_json({"usage": u})
                    if merged is not None:
                        self.usage = merged
            choices = obj.get("choices") if isinstance(obj, dict) else None
            if not isinstance(choices, list) or not choices:
                continue
            c0 = choices[0]
            if not isinstance(c0, dict):
                continue
            delta = c0.get("delta")
            if not isinstance(delta, dict):
                continue
            piece = delta.get("content")
            if isinstance(piece, str):
                self._parts.append(piece)

    @property
    def text(self) -> str:
        return "".join(self._parts)
