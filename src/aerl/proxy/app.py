from __future__ import annotations

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from aerl import __version__
from aerl.proxy.jobs import post_job
from aerl.proxy.middleware import RequestIdMiddleware
from aerl.proxy.forward import proxy_v1
from aerl.proxy.settings import load_settings
from aerl.proxy.upstream_probe import probe_upstream


async def health(_request):
    return JSONResponse({"status": "ok", "version": __version__})


async def ready(request):
    settings = request.app.state.settings
    body: dict[str, object] = {"status": "ok", "version": __version__}
    if not settings.ready_check_upstream:
        return JSONResponse(body)
    ok, _status = await probe_upstream(settings)
    if ok:
        body["upstream_ok"] = True
        return JSONResponse(body)
    body["upstream_ok"] = False
    return JSONResponse(body, status_code=503)


_V1_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def create_app() -> Starlette:
    settings = load_settings()
    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/ready", ready, methods=["GET"]),
            Route("/aerl/v1/jobs", post_job, methods=["POST"]),
            Route("/v1/{path:path}", proxy_v1, methods=_V1_METHODS),
        ],
        middleware=[Middleware(RequestIdMiddleware)],
    )
    app.state.settings = settings
    return app
