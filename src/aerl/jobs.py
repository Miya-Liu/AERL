from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

from aerl.errors import aerl_error_response
from aerl.settings import Settings
from aerl.trace_store import TraceStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pick_job_id(payload: dict[str, Any]) -> str:
    raw = payload.get("job_id")
    if isinstance(raw, str):
        s = raw.strip()
        if 1 <= len(s) <= 128:
            return s
    return str(uuid.uuid4())


async def _call_webhook(
    settings: Settings, body: dict[str, Any]
) -> tuple[bool, int | None, str | None]:
    headers: dict[str, str] = {}
    if settings.job_webhook_auth:
        headers["Authorization"] = settings.job_webhook_auth
    try:
        async with httpx.AsyncClient(timeout=settings.job_webhook_timeout) as client:
            resp = await client.post(
                settings.job_webhook_url or "",
                json=body,
                headers=headers,
            )
    except httpx.RequestError as exc:
        return False, None, str(exc)
    return resp.is_success, resp.status_code, None


async def post_job(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    rid = request.state.request_id
    ts_received = _now_iso()
    store = TraceStore(settings.data_dir)

    raw = await request.body()
    if len(raw) > settings.max_job_bytes:
        return aerl_error_response(
            request,
            code="payload_too_large",
            message=f"JSON body exceeds {settings.max_job_bytes} bytes",
            status_code=413,
        )

    ct = request.headers.get("content-type", "")
    if raw and "application/json" not in ct.lower():
        return aerl_error_response(
            request,
            code="unsupported_media_type",
            message="Content-Type must be application/json",
            status_code=415,
        )

    try:
        parsed: Any = json.loads(raw.decode("utf-8")) if raw else {}
    except json.JSONDecodeError:
        return aerl_error_response(
            request,
            code="invalid_json",
            message="Body must be a JSON object",
            status_code=400,
        )
    if not isinstance(parsed, dict):
        return aerl_error_response(
            request,
            code="invalid_json",
            message="Body must be a JSON object",
            status_code=400,
        )
    payload = parsed

    job_id = pick_job_id(payload)
    ts_upstream = ts_received
    ts_complete = ts_received
    status = "accepted"
    webhook_http: int | None = None
    webhook_err: str | None = None

    if settings.job_webhook_url:
        ts_upstream = _now_iso()
        ok, code, err = await _call_webhook(settings, payload)
        ts_complete = _now_iso()
        webhook_http = code
        webhook_err = err
        if ok:
            status = "forwarded"
        else:
            status = "failed"

    record: dict[str, Any] = {
        "event_type": "job",
        "request_id": rid,
        "job_id": job_id,
        "ts_request_received": ts_received,
        "ts_upstream_sent": ts_upstream,
        "ts_response_complete": ts_complete,
        "status": status,
    }
    if webhook_http is not None:
        record["webhook_http_status"] = webhook_http
    if webhook_err:
        record["webhook_error"] = webhook_err
    store.append(record)

    return JSONResponse({"job_id": job_id, "status": status})
