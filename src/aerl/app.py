from __future__ import annotations

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from aerl import __version__
from aerl.jobs import post_job
from aerl.middleware import RequestIdMiddleware
from aerl.settings import load_settings
from aerl.upstream_probe import probe_upstream


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


def create_app() -> Starlette:
    settings = load_settings()
    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/ready", ready, methods=["GET"]),
            Route("/aerl/v1/jobs", post_job, methods=["POST"]),
        ],
        middleware=[Middleware(RequestIdMiddleware)],
    )
    app.state.settings = settings
    return app
