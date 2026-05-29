"""Task-related database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


async def create_task(
    *,
    client,
    account_id: str,
    parent_task_id: str | None = None,
    agent_id: str,
    agent_run_id: str | None = None,
    name: str | None = None,
    project_id: str | None = None,
) -> str:
    """Create a new task record.

    Args:
        client: Supabase client instance (from DBConnection.get_client())
        account_id: The account ID to associate with the task
        parent_task_id: Optional parent task ID for subtasks
        agent_id: The agent ID to associate with the task
        agent_run_id: Optional agent run ID
        name: Optional task name
        project_id: Optional project ID

    Returns:
        The task_id of the created task.

    Example:
        >>> db = DBConnection()
        >>> client = await db.get_client()
        >>> task_id = await create_task(
        ...     client=client,
        ...     account_id="user-123",
        ...     agent_id="agent-456",
        ...     name="My Task"
        ... )
    """
    task_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    task_data: dict[str, Any] = {
        "task_id": task_id,
        "account_id": account_id,
        "status": TaskStatus.PENDING,
        "created_at": now,
    }

    if parent_task_id is not None:
        task_data["parent_task_id"] = parent_task_id
    task_data["agent_id"] = agent_id
    if agent_run_id is not None:
        task_data["agent_run_id"] = agent_run_id
    if name is not None:
        task_data["name"] = name
    if project_id is not None:
        task_data["project_id"] = project_id

    await client.table("tasks").insert(task_data).execute()

    return task_id


async def update_task_status(
    client,
    task_id: str,
    status: TaskStatus,
    *,
    error: str | None = None,
) -> None:
    """Update a task's status with appropriate timestamps.

    Args:
        client: Supabase client instance
        task_id: The task ID to update
        status: New task status
        error: Optional error message (for FAILED status)
    """
    now = datetime.now(UTC).isoformat()

    update_data: dict[str, Any] = {"status": status}

    if status == TaskStatus.RUNNING:
        update_data["started_at"] = now
    elif status in (TaskStatus.COMPLETED, TaskStatus.CANCELED, TaskStatus.FAILED):
        update_data["completed_at"] = now

    if error is not None:
        update_data["error"] = error

    await client.table("tasks").update(update_data).eq("task_id", task_id).execute()
