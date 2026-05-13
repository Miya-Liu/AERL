from __future__ import annotations

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from aerl import __version__
from aerl.middleware import RequestIdMiddleware
from aerl.settings import load_settings


async def health(_request):
    return JSONResponse({"status": "ok", "version": __version__})


def create_app() -> Starlette:
    settings = load_settings()
    app = Starlette(
        routes=[Route("/health", health, methods=["GET"])],
        middleware=[Middleware(RequestIdMiddleware)],
    )
    app.state.settings = settings
    return app
