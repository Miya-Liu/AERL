from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest
import respx
from starlette.testclient import TestClient

from aerl.app import create_app


def _last_job_trace(data_dir: Path) -> dict:
    lines = (data_dir / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AERL_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("AERL_JOB_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("AERL_JOB_WEBHOOK_AUTH", raising=False)
    return TestClient(create_app()), tmp_path


def test_job_no_job_id_is_uuid(app_client):
    client, data_dir = app_client
    r = client.post("/aerl/v1/jobs", json={"pipeline": "x"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "accepted"
    uuid.UUID(data["job_id"])
    trace = _last_job_trace(data_dir)
    assert trace["job_id"] == data["job_id"]


def test_job_echo_job_id(app_client):
    client, _data_dir = app_client
    r = client.post("/aerl/v1/jobs", json={"job_id": "  my-job-1  "})
    assert r.status_code == 200
    assert r.json() == {"job_id": "my-job-1", "status": "accepted"}


def test_job_overlong_job_id_gets_uuid(app_client):
    client, data_dir = app_client
    long_id = "a" * 129
    r = client.post("/aerl/v1/jobs", json={"job_id": long_id})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    assert jid != long_id
    uuid.UUID(jid)


def test_job_non_string_job_id_gets_uuid(app_client):
    client, _ = app_client
    r = client.post("/aerl/v1/jobs", json={"job_id": 99})
    jid = r.json()["job_id"]
    uuid.UUID(jid)


def test_job_no_webhook_accepted(app_client):
    client, data_dir = app_client
    r = client.post("/aerl/v1/jobs", json={})
    assert r.json()["status"] == "accepted"
    trace = _last_job_trace(data_dir)
    assert trace["status"] == "accepted"
    assert "webhook_http_status" not in trace


@respx.mock
def test_job_webhook_200_forwarded(app_client, monkeypatch):
    client, data_dir = app_client
    monkeypatch.setenv("AERL_JOB_WEBHOOK_URL", "https://hook.example/notify")
    client = TestClient(create_app())
    route = respx.post("https://hook.example/notify").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    r = client.post("/aerl/v1/jobs", json={"x": 1})
    assert r.status_code == 200
    assert r.json()["status"] == "forwarded"
    assert route.called
    trace = _last_job_trace(data_dir)
    assert trace["webhook_http_status"] == 200


@respx.mock
def test_job_webhook_500_failed(app_client, monkeypatch):
    client, data_dir = app_client
    monkeypatch.setenv("AERL_JOB_WEBHOOK_URL", "https://hook.example/fail")
    client = TestClient(create_app())
    respx.post("https://hook.example/fail").mock(
        return_value=httpx.Response(500, text="no")
    )
    r = client.post("/aerl/v1/jobs", json={})
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    trace = _last_job_trace(data_dir)
    assert trace["webhook_http_status"] == 500


def test_job_invalid_json_error(app_client):
    client, _ = app_client
    r = client.post(
        "/aerl/v1/jobs",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_json"
    assert body["error"]["request_id"] == r.headers["X-AERL-Request-Id"]


@respx.mock
def test_job_webhook_transport_failed(app_client, monkeypatch):
    client, data_dir = app_client
    monkeypatch.setenv("AERL_JOB_WEBHOOK_URL", "https://hook.example/down")
    client = TestClient(create_app())
    respx.post("https://hook.example/down").mock(
        side_effect=httpx.ConnectError("refused")
    )
    r = client.post("/aerl/v1/jobs", json={"a": 1})
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    trace = _last_job_trace(data_dir)
    assert "webhook_error" in trace
    assert trace.get("webhook_http_status") is None


def test_job_body_too_large_413(app_client, monkeypatch):
    client, _ = app_client
    monkeypatch.setenv("AERL_MAX_JOB_BYTES", "50")
    client = TestClient(create_app())
    r = client.post(
        "/aerl/v1/jobs",
        content=b"x" * 51,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


@respx.mock
def test_job_webhook_receives_auth_header(app_client, monkeypatch):
    _client, data_dir = app_client
    monkeypatch.setenv("AERL_JOB_WEBHOOK_URL", "https://hook.example/auth")
    monkeypatch.setenv("AERL_JOB_WEBHOOK_AUTH", "Bearer hook-secret")

    def check(req: httpx.Request) -> httpx.Response:
        assert req.headers.get("authorization") == "Bearer hook-secret"
        return httpx.Response(200)

    respx.post("https://hook.example/auth").mock(side_effect=check)
    client = TestClient(create_app())
    r = client.post("/aerl/v1/jobs", json={})
    assert r.status_code == 200
    assert r.json()["status"] == "forwarded"
    _last_job_trace(data_dir)
