from __future__ import annotations

import httpx

from aerl.proxy.settings import Settings, join_upstream_subpath

_READY_PROBE_TIMEOUT = 5.0


def ready_probe_url(settings: Settings) -> str:
    return join_upstream_subpath(
        settings.upstream_openai_base_url, settings.ready_probe_path
    )


async def probe_upstream(settings: Settings) -> tuple[bool, int | None]:
    """
    GET the configured ready URL. Returns (ok, status_code).
    ``ok`` is True for HTTP 2xx; ``status_code`` is None on transport failure.
    """
    url = ready_probe_url(settings)
    headers: dict[str, str] = {}
    if settings.ready_auth:
        headers["Authorization"] = settings.ready_auth
    try:
        async with httpx.AsyncClient(timeout=_READY_PROBE_TIMEOUT) as client:
            response = await client.get(url, headers=headers)
    except httpx.RequestError:
        return False, None
    return response.is_success, response.status_code
