from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from starlette.testclient import TestClient

from aerl.app import create_app


@pytest.fixture
def data_dir_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AERL_DATA_DIR", str(tmp_path))
    return TestClient(create_app()), tmp_path


@respx.mock
def test_trace_includes_usage_and_cost(tmp_path, monkeypatch):
    pricing_path = tmp_path / "p.json"
    pricing_path.write_text(
        json.dumps(
            {
                "default": {
                    "input_per_million_usd": 10.0,
                    "output_per_million_usd": 20.0,
                },
                "per_model": {
                    "gpt-costy": {
                        "input_per_million_usd": 1.0,
                        "output_per_million_usd": 2.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AERL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AERL_PRICING_JSON", str(pricing_path))
    client = TestClient(create_app())

    upstream = {
        "id": "1",
        "choices": [],
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000},
    }
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=upstream)
    )
    client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-costy",
            "messages": [],
            "user": "tenant-42",
        },
        headers={"X-AERL-User": "batch-job-7"},
    )
    rec = json.loads((tmp_path / "traces.jsonl").read_text().strip().splitlines()[-1])
    assert rec["openai_user"] == "tenant-42"
    assert rec["caller_label"] == "batch-job-7"
    assert rec["usage"]["prompt_tokens"] == 1_000_000
    assert rec["usage"]["completion_tokens"] == 500_000
    assert rec["cost_usd_estimated"] == pytest.approx(2.0, rel=1e-6)


@respx.mock
def test_sse_trace_includes_usage_from_final_chunk(data_dir_client):
    client, data_dir = data_dir_client
    sse = (
        'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        'data: {"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )
    )
    client.post("/v1/chat/completions", json={"model": "m", "stream": True})
    rec = json.loads((Path(data_dir) / "traces.jsonl").read_text().strip().splitlines()[-1])
    assert rec["stream"] is True
    assert rec["usage"]["prompt_tokens"] == 2
    assert rec["usage"]["completion_tokens"] == 3
