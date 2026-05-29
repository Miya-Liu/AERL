from __future__ import annotations

import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from aerl import __version__
from aerl.pipelines.registry import list_pipelines
from aerl.proxy.errors import aerl_error_response
from aerl.service.settings import ServiceSettings
from aerl.service.store import RunStore
from aerl.service.supervisor import start_run_async


def _check_auth(request: Request, settings: ServiceSettings) -> JSONResponse | None:
    if not settings.service_token:
        return None
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return aerl_error_response(
            request,
            code="unauthorized",
            message="Missing Bearer token",
            status_code=401,
        )
    token = auth[7:].strip()
    if token != settings.service_token:
        return aerl_error_response(
            request,
            code="unauthorized",
            message="Invalid token",
            status_code=401,
        )
    return None


async def service_health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__, "component": "service"})


async def list_training_pipelines(request: Request) -> JSONResponse:
    settings: ServiceSettings = request.app.state.service_settings
    err = _check_auth(request, settings)
    if err is not None:
        return err
    pipelines = [
        {
            "id": p.id,
            "description": p.description,
            "requires_gpu": p.requires_gpu,
            "requires_inference": p.requires_inference,
        }
        for p in list_pipelines()
    ]
    return JSONResponse({"pipelines": pipelines})


async def create_training_run(request: Request) -> JSONResponse:
    settings: ServiceSettings = request.app.state.service_settings
    err = _check_auth(request, settings)
    if err is not None:
        return err

    raw = await request.body()
    try:
        body: Any = json.loads(raw.decode("utf-8")) if raw else {}
    except json.JSONDecodeError:
        return aerl_error_response(
            request,
            code="invalid_json",
            message="Body must be JSON",
            status_code=400,
        )
    if not isinstance(body, dict):
        return aerl_error_response(
            request,
            code="invalid_json",
            message="Body must be a JSON object",
            status_code=400,
        )

    pipeline = body.get("pipeline")
    if not isinstance(pipeline, str) or not pipeline.strip():
        return aerl_error_response(
            request,
            code="invalid_request",
            message="Field 'pipeline' is required",
            status_code=400,
        )
    pipeline = pipeline.strip()

    from aerl.pipelines.registry import get_pipeline

    try:
        get_pipeline(pipeline)
    except KeyError:
        return aerl_error_response(
            request,
            code="unknown_pipeline",
            message=f"Unknown pipeline: {pipeline}",
            status_code=404,
        )

    store: RunStore = request.app.state.run_store
    try:
        record = store.create_run(pipeline, body)
    except FileExistsError as exc:
        return aerl_error_response(
            request,
            code="run_exists",
            message=str(exc),
            status_code=409,
        )

    start_run_async(store, record, repo_root=settings.repo_root)

    return JSONResponse(
        {
            "run_id": record.run_id,
            "status": record.status,
            "pipeline": record.pipeline,
        },
        status_code=202,
    )


async def get_training_run(request: Request) -> JSONResponse:
    settings: ServiceSettings = request.app.state.service_settings
    err = _check_auth(request, settings)
    if err is not None:
        return err

    run_id = request.path_params["run_id"]
    store: RunStore = request.app.state.run_store
    record = store.get(run_id)
    if record is None:
        return aerl_error_response(
            request,
            code="not_found",
            message=f"Run not found: {run_id}",
            status_code=404,
        )
    return JSONResponse(record.to_dict())
