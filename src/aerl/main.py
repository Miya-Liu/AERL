from __future__ import annotations

import uvicorn

from aerl.app import create_app
from aerl.settings import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        create_app,
        factory=True,
        host=settings.listen_host,
        port=settings.listen_port,
    )
