from __future__ import annotations

import uuid

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.testclient import TestClient

from aerl.app import create_app
from aerl.errors import aerl_error_response
from aerl.middleware import RequestIdMiddleware


async def _error_endpoint(request):
    return aerl_error_response(
        request, code="bad_request", message="test error", status_code=400
    )


def test_aerl_error_response_shape():
    app = Starlette(
        routes=[Route("/e", _error_endpoint, methods=["GET"])],
        middleware=[Middleware(RequestIdMiddleware)],
    )
    client = TestClient(app)
    r = client.get("/e")
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "bad_request"
    assert body["error"]["message"] == "test error"
    rid = body["error"]["request_id"]
    assert rid == r.headers["X-AERL-Request-Id"]
    uuid.UUID(rid)


def test_health_has_request_id_header():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    rid = r.headers.get("X-AERL-Request-Id")
    assert rid
    uuid.UUID(rid)
