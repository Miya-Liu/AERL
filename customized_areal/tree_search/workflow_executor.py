from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from areal.api import RolloutWorkflow
from areal.infra import workflow_context
from areal.infra.workflow_context import WorkflowContext
from areal.infra.workflow_executor import WorkflowExecutor, _RolloutTaskInput
from areal.utils import perf_tracer, stats_tracker
from areal.utils.perf_tracer import trace_session_event


@dataclass
class _TreeSearchRolloutResult:
    """Internal wrapper for tree search rollout results containing multiple Node objects."""

    task_id: int
    trajectories: list


class TreeSearchWorkflowExecutor(WorkflowExecutor):
    """WorkflowExecutor subclass that handles list[Node] returns from arun_episode."""

    def _create_workflow_task(
        self, pending_task: _RolloutTaskInput
    ) -> Callable[[], Any]:
        """Create an async function that handles list[Node] returns from arun_episode."""

        async def _execute_workflow() -> _TreeSearchRolloutResult | None:
            """Execute workflow.arun_episode and handle list[Node] returns."""
            task_id = pending_task.task_id

            # Set task_id in ContextVar before entering arun_episode
            perf_tracer.set_task_id(task_id)

            # Set workflow execution context
            workflow_context.set(
                WorkflowContext(is_eval=pending_task.is_eval, task_id=task_id)
            )

            manager = self.staleness_manager
            traj_result: list | None = None
            should_accept_fn = pending_task.should_accept_fn
            should_accept: bool | None = None
            reason: str | None = None

            try:
                traj_result = await pending_task.workflow.arun_episode(
                    self.inference_engine, pending_task.data
                )

                if traj_result is None:
                    should_accept_traj = False
                    reason = "returned_none"
                else:
                    if not isinstance(traj_result, list):
                        raise ValueError(
                            f"Expected list or None from arun_episode, got {type(traj_result)}"
                        )

                    # Apply acceptance function if provided
                    if should_accept_fn is None:
                        should_accept = True
                    else:
                        # For list returns, we accept if at least one trajectory is accepted
                        should_accept = any(
                            should_accept_fn(traj) for traj in traj_result
                        )

                    should_accept_traj = bool(should_accept)
                    if not should_accept_traj and should_accept_fn is not None:
                        reason = "rejected"

                # Skip trajectory dumping for tree search (not used)

                if should_accept_traj:
                    manager.on_rollout_accepted()
                    stats_tracker.get("rollout").scalar(accepted=1)
                    trace_session_event(
                        "mark_finalized",
                        task_id=task_id,
                        status="accepted",
                    )
                    if self.config.enable_rollout_tracing:
                        self.logger.info(
                            f"Finish and accept rollout. {self._rollout_stats()}",
                        )
                    assert traj_result is not None
                    return _TreeSearchRolloutResult(
                        task_id=task_id, trajectories=traj_result
                    )

                manager.on_rollout_rejected()
                stats_tracker.get("rollout").scalar(rejected=1)
                trace_session_event(
                    "mark_finalized",
                    task_id=task_id,
                    status="rejected",
                    reason=reason,
                )
                if self.config.enable_rollout_tracing:
                    self.logger.info(
                        f"Finish but reject rollout. {self._rollout_stats()}",
                    )
                return None

            except Exception as exc:
                manager.on_rollout_rejected()
                stats_tracker.get("rollout").scalar(rejected=1)
                trace_session_event(
                    "mark_finalized",
                    task_id=task_id,
                    status="failed",
                    reason="workflow_exception",
                )
                if self.logger is not None:
                    self.logger.error(
                        "Workflow execution failed: %s", exc, exc_info=True
                    )
                return None

        return _execute_workflow

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[Any | None]:
        """Wait for the completion of `count` workflows and extract trajectories.

        Note: A single submission may produce multiple trajectories (e.g.
        _TreeSearchRolloutResult contains a list of Nodes). This method
        flattens them, so the returned list may be longer than `count`.
        """
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)

        # Log and trace
        if self.config.enable_rollout_tracing:
            self.logger.info("Rollout results are ready!")

        extracted = []
        for r in results:
            if r is None:
                extracted.append(None)
            elif isinstance(r, _TreeSearchRolloutResult):
                extracted.extend(r.trajectories)
            else:
                extracted.append(r.trajectory)

        return extracted

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow,
    ) -> list[Any]:
        """Submit a batch of requests and wait for results.

        Note: The returned list may contain more items than the input batch
        because a single workflow can produce multiple trajectories (e.g. via
        group_size > 1 or _TreeSearchRolloutResult). Nones are filtered out.
        """
        perf_tracer.instant(
            "workflow_executor.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
            )
        results = self.wait(count=len(data))
        # Filter out None and flatten
        return [r for r in results if r is not None]
