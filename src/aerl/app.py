from __future__ import annotations

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from aerl import __version__


async def health(request):
    return JSONResponse({"status": "ok", "version": __version__})


def create_app() -> Starlette:
    return Starlette(routes=[Route("/health", health, methods=["GET"])])
