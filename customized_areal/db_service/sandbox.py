"""Sandbox lifecycle and cleanup operations.

Ported from le-agent-dev/backend/core/infra/sandbox and
le-agent-dev/backend/core/tasks/service.py.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _get_daytona():
    """Return a lazy-initialized async Daytona client from env vars."""
    try:
        from daytona import AsyncDaytona, DaytonaConfig
    except ImportError as exc:
        raise RuntimeError(
            "daytona SDK is required for sandbox operations. "
            "Install it or ensure it is available in the environment."
        ) from exc

    if not hasattr(_get_daytona, "_client"):
        config = DaytonaConfig(
            api_key=os.environ.get("DAYTONA_API_KEY"),
            api_url=os.environ.get("DAYTONA_SERVER_URL"),
            target=os.environ.get("DAYTONA_TARGET"),
        )
        _get_daytona._client = AsyncDaytona(config)
    return _get_daytona._client


async def delete_sandbox(sandbox_id: str) -> None:
    """Delete a Daytona sandbox by its ID."""
    daytona = _get_daytona()
    sandbox = await daytona.get(sandbox_id)
    await daytona.delete(sandbox)
    logger.info("Sandbox deleted: %s", sandbox_id)


async def cleanup_sandbox_for_task(client, task_id: str) -> None:
    """Delete sandbox via Daytona API for a task.

    Only deletes the Daytona sandbox; DB records are left intact.
    """
    try:
        sandbox_result = (
            await client.table("sandboxes")
            .select("sandbox_id")
            .eq("task_id", task_id)
            .maybe_single()
            .execute()
        )
        if sandbox_result and sandbox_result.data:
            sandbox_id = sandbox_result.data.get("sandbox_id")
            if sandbox_id:
                try:
                    await delete_sandbox(sandbox_id)
                except Exception:
                    logger.warning(
                        "Failed to delete sandbox for task_id=%s",
                        task_id,
                        exc_info=True,
                    )
    except Exception:
        logger.warning(
            "Failed to look up sandbox for deletion: task_id=%s",
            task_id,
            exc_info=True,
        )
