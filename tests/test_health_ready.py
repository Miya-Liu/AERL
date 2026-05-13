# /ready tests (Task 5).

import httpx
import respx
from starlette.testclient import TestClient

from aerl import __version__
from aerl.app import create_app


def test_health_returns_version():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == __version__


def test_ready_without_probe_matches_health(monkeypatch):
    monkeypatch.setenv("AERL_READY_CHECK_UPSTREAM", "false")
    client = TestClient(create_app())
    h = client.get("/health").json()
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == h
    assert "upstream_ok" not in r.json()


@respx.mock
def test_ready_probe_upstream_ok(monkeypatch):
    monkeypatch.setenv("AERL_READY_CHECK_UPSTREAM", "true")
    respx.get("https://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json={})
    )
    client = TestClient(create_app())
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert data["upstream_ok"] is True


@respx.mock
def test_ready_probe_upstream_non2xx_503(monkeypatch):
    monkeypatch.setenv("AERL_READY_CHECK_UPSTREAM", "true")
    respx.get("https://upstream.test/v1/models").mock(
        return_value=httpx.Response(500, text="err")
    )
    client = TestClient(create_app())
    r = client.get("/ready")
    assert r.status_code == 503
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert data["upstream_ok"] is False


@respx.mock
def test_ready_probe_sends_authorization(monkeypatch):
    monkeypatch.setenv("AERL_READY_CHECK_UPSTREAM", "true")
    monkeypatch.setenv("AERL_READY_AUTH", "Bearer probe-token")

    def check_request(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer probe-token"
        return httpx.Response(200, json={})

    respx.get("https://upstream.test/v1/models").mock(side_effect=check_request)
    client = TestClient(create_app())
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["upstream_ok"] is True
