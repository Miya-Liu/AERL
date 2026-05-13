"""Tiny OpenAI-shaped stub for local smoke tests (not used by default compose)."""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


async def chat_completions(_request):
    return JSONResponse({"id": "mock-1", "object": "chat.completion", "choices": []})


app = Starlette(
    routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])]
)
