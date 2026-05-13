from starlette.testclient import TestClient

from aerl.app import create_app


def test_health_returns_version():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "version" in r.json()
