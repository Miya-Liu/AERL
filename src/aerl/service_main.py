from __future__ import annotations

import uvicorn

from aerl.service.app import create_service_app
from aerl.service.settings import load_service_settings


def main() -> None:
    settings = load_service_settings()
    uvicorn.run(
        create_service_app,
        factory=True,
        host=settings.listen_host,
        port=settings.listen_port,
    )
