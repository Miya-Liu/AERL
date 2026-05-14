from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response

from aerl.errors import aerl_error_response
from aerl.llm_trace import (
    SSEAggregator,
    estimate_cost_usd,
    extract_caller_label,
    extract_json_user,
    extract_usage_from_response_json,
    extract_usage_from_sse_bytes,
)
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


def _latency_ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 3)


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


def _truncate_str_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text, False
    cut = raw[:max_bytes].decode("utf-8", errors="ignore")
    return cut + "…[truncated]", True


def _is_sse_response(status: int, content_type: str | None) -> bool:
    if status != 200:
        return False
    ct = content_type or ""
    return "text/event-stream" in ct.lower()


def _complete_proxy_trace_record(
    record: dict[str, Any],
    *,
    settings: Settings,
    request: Request,
    request_body: bytes,
    response_bytes: bytes,
    response_content_type: str | None,
    upstream_status: int,
    perf_request_start: float,
    perf_upstream_start: float,
    perf_complete: float,
) -> None:
    record["latency_ms_total"] = _latency_ms(perf_request_start, perf_complete)
    record["latency_ms_upstream"] = _latency_ms(perf_upstream_start, perf_complete)

    sse = _is_sse_response(upstream_status, response_content_type)
    record["stream"] = sse

    u = extract_json_user(request_body)
    if u is not None:
        record["openai_user"] = u
    label = extract_caller_label(request.headers)
    if label is not None:
        record["caller_label"] = label

    model = record.get("model")
    if not isinstance(model, str):
        model = _extract_model(request_body)

    usage = None
    if sse:
        agg = SSEAggregator()
        agg.feed(response_bytes)
        usage = agg.usage or extract_usage_from_sse_bytes(response_bytes)
        agg_text, agg_trunc = _truncate_str_utf8(agg.text, settings.max_body_bytes)
        record["aggregated_text"] = agg_text
        record["aggregated_text_truncated"] = agg_trunc
        record["response_body_omitted"] = True
        record["response_body_truncated"] = False
    else:
        res_log, res_trunc = _truncate_bytes(response_bytes, settings.max_body_bytes)
        record["response_body_truncated"] = res_trunc
        if res_log is not None:
            record["response_body"] = res_log
        if response_bytes:
            ct = (response_content_type or "").lower()
            if "json" in ct or response_bytes.lstrip()[:1] in (b"{", b"["):
                try:
                    usage = extract_usage_from_response_json(
                        json.loads(response_bytes.decode("utf-8"))
                    )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    usage = None

    if usage:
        record["usage"] = usage

    cost = estimate_cost_usd(
        usage, str(model) if model else None, settings.pricing
    )
    if cost is not None:
        record["cost_usd_estimated"] = cost


async def _proxy_large(
    request: Request,
    upstream_url: str,
    settings: Settings,
    rid: str,
    ts_received: str,
    perf_request_start: float,
) -> Response:
    req_headers = forward_request_headers(request.headers)
    ts_send = _now_iso()
    perf_upstream_start = time.perf_counter()
    store = TraceStore(settings.data_dir)
    resp_ct: str | None = None
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout) as client:
            async with client.stream(
                request.method,
                upstream_url,
                headers=req_headers,
                content=request.stream(),
            ) as resp:
                resp_ct = resp.headers.get("content-type")
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
    perf_complete = time.perf_counter()
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
    record["request_headers"] = redact_headers(
        {k: v for k, v in request.headers.items()}
    )
    _complete_proxy_trace_record(
        record,
        settings=settings,
        request=request,
        request_body=b"",
        response_bytes=body,
        response_content_type=resp_ct,
        upstream_status=status,
        perf_request_start=perf_request_start,
        perf_upstream_start=perf_upstream_start,
        perf_complete=perf_complete,
    )
    store.append(record)
    r = Response(content=body, status_code=status, headers=dict(out_headers))
    r.headers["X-AERL-Request-Id"] = rid
    return r


async def proxy_v1(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    rid = request.state.request_id
    path = request.path_params["path"]
    perf_request_start = time.perf_counter()
    ts_received = _now_iso()
    upstream_url = join_upstream_subpath(settings.upstream_openai_base_url, path)
    method = request.method

    cl = request.headers.get("content-length")
    if method in ("POST", "PUT", "PATCH", "DELETE") and cl:
        try:
            if int(cl) > settings.max_buffered_request_bytes:
                return await _proxy_large(
                    request, upstream_url, settings, rid, ts_received, perf_request_start
                )
        except ValueError:
            pass

    body = await request.body()
    if len(body) > settings.max_buffered_request_bytes:
        return await _proxy_large(
            request, upstream_url, settings, rid, ts_received, perf_request_start
        )

    req_headers = forward_request_headers(request.headers)
    ts_send = _now_iso()
    perf_upstream_start = time.perf_counter()
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

    perf_complete = time.perf_counter()
    ts_done = _now_iso()
    res_bytes = resp.content
    resp_ct = resp.headers.get("content-type")

    req_log, req_trunc = _truncate_bytes(body, settings.max_body_bytes)

    record: dict[str, Any] = {
        "request_id": rid,
        "ts_request_received": ts_received,
        "ts_upstream_sent": ts_send,
        "ts_response_complete": ts_done,
        "method": method,
        "path": request.url.path,
        "upstream_status": resp.status_code,
        "request_body_truncated": req_trunc,
    }
    model = _extract_model(body)
    if model is not None:
        record["model"] = model
    if req_log is not None:
        record["request_body"] = req_log

    _complete_proxy_trace_record(
        record,
        settings=settings,
        request=request,
        request_body=body,
        response_bytes=res_bytes,
        response_content_type=resp_ct,
        upstream_status=resp.status_code,
        perf_request_start=perf_request_start,
        perf_upstream_start=perf_upstream_start,
        perf_complete=perf_complete,
    )

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
