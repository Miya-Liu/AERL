from __future__ import annotations

import json
import time

import pytest
from starlette.testclient import TestClient

from aerl.service.app import create_service_app
from aerl.service.settings import ServiceSettings


@pytest.fixture
def service_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AERL_DATA_DIR", str(tmp_path))
    settings = ServiceSettings(
        data_dir=str(tmp_path),
        listen_host="127.0.0.1",
        listen_port=8766,
        service_token=None,
        repo_root=str(tmp_path),
    )
    with TestClient(create_service_app(settings)) as client:
        yield client


def test_list_pipelines(service_client: TestClient) -> None:
    r = service_client.get("/aerl/v1/training/pipelines")
    assert r.status_code == 200
    body = r.json()
    ids = {p["id"] for p in body["pipelines"]}
    assert "mock_grpo" in ids
    assert "gsm8k_grpo" in ids


def test_mock_grpo_run_e2e(service_client: TestClient, tmp_path) -> None:
    r = service_client.post(
        "/aerl/v1/training/runs",
        json={"pipeline": "mock_grpo", "objective": {"type": "smoke"}},
    )
    assert r.status_code == 202
    body = r.json()
    run_id = body["run_id"]
    assert body["pipeline"] == "mock_grpo"

    final = None
    for _ in range(50):
        g = service_client.get(f"/aerl/v1/training/runs/{run_id}")
        assert g.status_code == 200
        final = g.json()
        if final["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert final is not None
    assert final["status"] == "completed"
    assert final["exit_code"] == 0

    run_dir = tmp_path / "runs" / run_id
    assert (run_dir / "train.log").is_file()
    assert (run_dir / "train_id.json").is_file()
    train_id = json.loads((run_dir / "train_id.json").read_text(encoding="utf-8"))
    assert train_id["train_id"] == run_id


def test_unknown_pipeline(service_client: TestClient) -> None:
    r = service_client.post(
        "/aerl/v1/training/runs",
        json={"pipeline": "does_not_exist"},
    )
    assert r.status_code == 404


def test_service_health(service_client: TestClient) -> None:
    r = service_client.get("/health")
    assert r.status_code == 200
    assert r.json()["component"] == "service"
