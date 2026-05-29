from __future__ import annotations

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route

from aerl.proxy.middleware import RequestIdMiddleware
from aerl.service.runs import (
    create_training_run,
    get_training_run,
    list_training_pipelines,
    service_health,
)
from aerl.service.settings import ServiceSettings, load_service_settings
from aerl.service.store import RunStore


def create_service_app(settings: ServiceSettings | None = None) -> Starlette:
    cfg = settings or load_service_settings()
    app = Starlette(
        routes=[
            Route("/health", service_health, methods=["GET"]),
            Route("/aerl/v1/training/pipelines", list_training_pipelines, methods=["GET"]),
            Route("/aerl/v1/training/runs", create_training_run, methods=["POST"]),
            Route(
                "/aerl/v1/training/runs/{run_id}",
                get_training_run,
                methods=["GET"],
            ),
        ],
        middleware=[Middleware(RequestIdMiddleware)],
    )
    app.state.service_settings = cfg
    app.state.run_store = RunStore(cfg.data_dir)
    return app
