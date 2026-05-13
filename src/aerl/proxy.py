from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response

from aerl.errors import aerl_error_response
from aerl.redact import redact_headers
from aerl.settings import Settings, join_upstream_subpath
from aerl.trace_store import TraceStore

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def forward_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        lk = key.lower()
        if lk == "host":
            continue
        if lk in ("authorization", "content-type", "user-agent") or lk.startswith(
            "openai-"
        ):
            out[key] = value
    return out


def filter_response_headers(resp: httpx.Response) -> dict[str, str]:
    h: dict[str, str] = {}
    for key, value in resp.headers.multi_items():
        lk = key.lower()
        if lk in _HOP_BY_HOP:
            continue
        h[key] = value
    return h


def _truncate_bytes(raw: bytes, max_bytes: int) -> tuple[Any, bool]:
    if not raw:
        return None, False
    truncated = len(raw) > max_bytes
    chunk = raw[:max_bytes]
    text = chunk.decode("utf-8", errors="replace")
    if truncated:
        text = text + "…[truncated]"
    try:
        return json.loads(chunk.decode("utf-8")), truncated
    except json.JSONDecodeError:
        return text, truncated


def _extract_model(request_body: bytes) -> str | None:
    if not request_body:
        return None
    try:
        obj = json.loads(request_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(obj, dict) and "model" in obj:
        return str(obj["model"])
    return None


async def _proxy_large(
    request: Request,
    upstream_url: str,
    settings: Settings,
    rid: str,
    ts_received: str,
) -> Response:
    req_headers = forward_request_headers(request.headers)
    ts_send = _now_iso()
    store = TraceStore(settings.data_dir)
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout) as client:
            async with client.stream(
                request.method,
                upstream_url,
                headers=req_headers,
                content=request.stream(),
            ) as resp:
                chunks: list[bytes] = []
                async for part in resp.aiter_bytes():
                    chunks.append(part)
                body = b"".join(chunks)
                status = resp.status_code
                out_headers = filter_response_headers(resp)
    except httpx.RequestError as exc:
        return aerl_error_response(
            request,
            code="upstream_unreachable",
            message=str(exc),
            status_code=502,
        )
    ts_done = _now_iso()
    record: dict[str, Any] = {
        "request_id": rid,
        "ts_request_received": ts_received,
        "ts_upstream_sent": ts_send,
        "ts_response_complete": ts_done,
        "method": request.method,
        "path": request.url.path,
        "upstream_status": status,
        "request_body_omitted": True,
        "response_body_omitted": True,
    }
    store.append(record)
    r = Response(content=body, status_code=status, headers=dict(out_headers))
    r.headers["X-AERL-Request-Id"] = rid
    return r


async def proxy_v1(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    rid = request.state.request_id
    path = request.path_params["path"]
    ts_received = _now_iso()
    upstream_url = join_upstream_subpath(settings.upstream_openai_base_url, path)
    method = request.method

    cl = request.headers.get("content-length")
    if method in ("POST", "PUT", "PATCH", "DELETE") and cl:
        try:
            if int(cl) > settings.max_buffered_request_bytes:
                return await _proxy_large(
                    request, upstream_url, settings, rid, ts_received
                )
        except ValueError:
            pass

    body = await request.body()
    if len(body) > settings.max_buffered_request_bytes:
        return await _proxy_large(request, upstream_url, settings, rid, ts_received)

    req_headers = forward_request_headers(request.headers)
    ts_send = _now_iso()
    store = TraceStore(settings.data_dir)

    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout) as client:
            resp = await client.request(
                method,
                upstream_url,
                headers=req_headers,
                content=body if body else None,
            )
    except httpx.RequestError as exc:
        return aerl_error_response(
            request,
            code="upstream_unreachable",
            message=str(exc),
            status_code=502,
        )

    ts_done = _now_iso()
    res_bytes = resp.content

    req_log, req_trunc = _truncate_bytes(body, settings.max_body_bytes)
    res_log, res_trunc = _truncate_bytes(res_bytes, settings.max_body_bytes)

    record: dict[str, Any] = {
        "request_id": rid,
        "ts_request_received": ts_received,
        "ts_upstream_sent": ts_send,
        "ts_response_complete": ts_done,
        "method": method,
        "path": request.url.path,
        "upstream_status": resp.status_code,
        "request_body_truncated": req_trunc,
        "response_body_truncated": res_trunc,
    }
    model = _extract_model(body)
    if model is not None:
        record["model"] = model
    if req_log is not None:
        record["request_body"] = req_log
    if res_log is not None:
        record["response_body"] = res_log

    record["request_headers"] = redact_headers(
        {k: v for k, v in request.headers.items()}
    )
    store.append(record)

    out_h = filter_response_headers(resp)
    response = Response(
        content=res_bytes,
        status_code=resp.status_code,
        headers=dict(out_h),
    )
    response.headers["X-AERL-Request-Id"] = rid
    return response
