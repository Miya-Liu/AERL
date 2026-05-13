from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from starlette.testclient import TestClient

from aerl.app import create_app


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AERL_DATA_DIR", str(tmp_path))
    return tmp_path


@respx.mock
def test_request_body_truncation_flag(data_dir, monkeypatch):
    monkeypatch.setenv("AERL_MAX_BODY_BYTES", "128")
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "1", "choices": []})
    )
    client = TestClient(create_app())
    padding = "p" * 400
    client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": padding}]},
    )
    rec = json.loads((Path(data_dir) / "traces.jsonl").read_text().strip().splitlines()[-1])
    assert rec["request_body_truncated"] is True


@respx.mock
def test_response_body_truncation_flag(data_dir, monkeypatch):
    monkeypatch.setenv("AERL_MAX_BODY_BYTES", "96")
    big = {"choices": [{"message": {"content": "z" * 800}}]}
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=big)
    )
    client = TestClient(create_app())
    client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    rec = json.loads((Path(data_dir) / "traces.jsonl").read_text().strip().splitlines()[-1])
    assert rec["response_body_truncated"] is True
