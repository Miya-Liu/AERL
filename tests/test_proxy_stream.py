from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from starlette.testclient import TestClient

from aerl.app import create_app

SSE_FIXTURE = (
    'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    "data: [DONE]\n\n"
).encode()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AERL_DATA_DIR", str(tmp_path))
    return tmp_path


@respx.mock
def test_sse_passthrough_and_aggregated_trace(data_dir):
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=SSE_FIXTURE,
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )
    )
    client = TestClient(create_app())
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [], "stream": True},
    )
    assert r.status_code == 200
    assert r.content == SSE_FIXTURE
    rid_hdr = r.headers["X-AERL-Request-Id"]

    trace_path = Path(data_dir) / "traces.jsonl"
    rec = json.loads(trace_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["request_id"] == rid_hdr
    assert rec["stream"] is True
    assert rec["aggregated_text"] == "Hello"
    assert rec["aggregated_text_truncated"] is False
    assert rec["response_body_omitted"] is True
    assert rec["response_body_truncated"] is False
    assert rec["upstream_status"] == 200
    assert isinstance(rec["latency_ms_total"], (int, float))
    assert isinstance(rec["latency_ms_upstream"], (int, float))


@respx.mock
def test_sse_aggregated_text_truncation(data_dir, monkeypatch):
    monkeypatch.setenv("AERL_MAX_BODY_BYTES", "4")
    big = "abcdefghij"
    sse_line = f'data: {{"choices":[{{"delta":{{"content":"{big}"}}}}]}}\n\n'
    body = sse_line.encode()
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )
    )
    client = TestClient(create_app())
    client.post("/v1/chat/completions", json={"model": "m", "stream": True})
    rec = json.loads((Path(data_dir) / "traces.jsonl").read_text().strip().splitlines()[-1])
    assert rec["aggregated_text_truncated"] is True
    assert "…[truncated]" in rec["aggregated_text"]
