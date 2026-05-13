from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from starlette.testclient import TestClient

from aerl.app import create_app


@pytest.fixture
def proxy_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AERL_DATA_DIR", str(tmp_path))
    return TestClient(create_app()), tmp_path


@respx.mock
def test_proxy_post_chat_completion_success(proxy_client):
    client, _data_dir = proxy_client
    upstream = {"id": "upstream-1", "choices": []}
    route = respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=upstream)
    )
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    r = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer sk-secret"},
    )
    assert r.status_code == 200
    assert r.json() == upstream
    assert route.called
    req = route.calls.last.request
    assert req.headers.get("authorization") == "Bearer sk-secret"


@respx.mock
def test_proxy_upstream_401_passthrough(proxy_client):
    client, _ = proxy_client
    err = {"error": {"message": "bad", "type": "auth"}}
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(401, json=err)
    )
    r = client.post("/v1/chat/completions", json={"model": "m"})
    assert r.status_code == 401
    assert r.json() == err


@respx.mock
def test_proxy_log_record_contract(proxy_client):
    client, data_dir = proxy_client
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    rid_hdr = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": []},
        headers={"Authorization": "Bearer sk-secret"},
    ).headers["X-AERL-Request-Id"]
    trace_path = Path(data_dir) / "traces.jsonl"
    line = trace_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    rec = json.loads(line)
    for key in (
        "request_id",
        "ts_request_received",
        "ts_upstream_sent",
        "ts_response_complete",
        "method",
        "path",
        "upstream_status",
    ):
        assert key in rec
    assert rec["request_id"] == rid_hdr
    assert rec["method"] == "POST"
    assert "chat/completions" in rec["path"]
    assert rec["upstream_status"] == 200
    assert rec["model"] == "gpt-test"
    auth = rec["request_headers"].get("Authorization") or rec["request_headers"].get(
        "authorization"
    )
    assert auth
    assert "secret" not in auth


@respx.mock
def test_proxy_upstream_unreachable_502(proxy_client):
    client, _ = proxy_client
    respx.post("https://upstream.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("nope")
    )
    r = client.post("/v1/chat/completions", json={"model": "x"})
    assert r.status_code == 502
    body = r.json()
    assert body["error"]["code"] == "upstream_unreachable"
    assert body["error"]["request_id"] == r.headers["X-AERL-Request-Id"]
