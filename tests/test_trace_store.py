from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aerl.redact import redact_headers
from aerl.trace_store import PROXY_TRACE_REQUIRED_KEYS, TraceStore


def test_redact_authorization_bearer_masks_secret():
    h = redact_headers({"Authorization": "Bearer sk-verysecret"})
    assert "verysecret" not in h["Authorization"].lower()
    assert h["Authorization"].startswith("Bearer ****")
    assert h["Authorization"].endswith("cret")


def test_redact_preserves_other_headers():
    h = redact_headers({"Authorization": "Bearer x", "X-Custom": "ok"})
    assert h["X-Custom"] == "ok"


def test_trace_store_append_two_records(tmp_path):
    store = TraceStore(str(tmp_path))
    store.append({"a": 1})
    store.append({"b": 2})
    text = store.path.read_text(encoding="utf-8").splitlines()
    assert len(text) == 2
    assert json.loads(text[0]) == {"a": 1}
    assert json.loads(text[1]) == {"b": 2}


def test_proxy_trace_record_contract(tmp_path):
    """Minimal proxy record shape (plan Task 3 / spec §5)."""
    record = {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "ts_request_received": "2026-05-13T12:00:00+00:00",
        "ts_upstream_sent": "2026-05-13T12:00:00.100000+00:00",
        "ts_response_complete": "2026-05-13T12:00:01+00:00",
        "method": "POST",
        "path": "/v1/chat/completions",
        "upstream_status": 200,
        "model": "gpt-4o-mini",
        "request_body_truncated": False,
        "response_body_truncated": False,
    }
    missing = PROXY_TRACE_REQUIRED_KEYS - record.keys()
    assert not missing
    for key in (
        "ts_request_received",
        "ts_upstream_sent",
        "ts_response_complete",
    ):
        datetime.fromisoformat(record[key])

    store = TraceStore(str(tmp_path))
    store.append(record)
    loaded = json.loads(store.path.read_text(encoding="utf-8").strip())
    assert loaded["request_id"] == record["request_id"]
    assert loaded["upstream_status"] == 200
