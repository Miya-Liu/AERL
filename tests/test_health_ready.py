# /ready tests are added in Task 5 (see implementation plan).

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
