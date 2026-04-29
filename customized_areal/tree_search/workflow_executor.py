from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from areal.infra.workflow_executor import WorkflowExecutor, _RolloutTaskInput
from areal.api import RolloutWorkflow
from areal.infra import workflow_context
from areal.infra.workflow_context import WorkflowContext
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.perf_tracer import trace_session_event
from areal.utils.data import concat_padded_tensors


@dataclass
class _TreeSearchRolloutResult:
    """Internal wrapper for tree search rollout results containing multiple trajectories."""

    task_id: int
    trajectories: list[dict[str, Any]]


class TreeSearchWorkflowExecutor(WorkflowExecutor):
    """
    WorkflowExecutor subclass that handles list[dict] returns from arun_episode.

    This executor is designed for tree search workflows that return multiple trajectories
    per episode as a list of dictionaries, rather than a single trajectory dict.
    """

    def _create_workflow_task(
        self, pending_task: _RolloutTaskInput
    ) -> Callable[[], Any]:
        """
        Wrapper to create an async function that handles list[dict] returns from arun_episode.

        This overrides the base class method to:
        - Accept list[dict] from arun_episode
        - Skip InteractionWithTokenLogpReward conversion for tree search trajectories
        - Store results as _TreeSearchRolloutResult
        """

        async def _execute_workflow() -> _TreeSearchRolloutResult | None:
            """Execute workflow.arun_episode and handle list[dict] returns."""
            task_id = pending_task.task_id

            # Set task_id in ContextVar before entering arun_episode
            perf_tracer.set_task_id(task_id)

            # Set workflow execution context
            workflow_context.set(
                WorkflowContext(is_eval=pending_task.is_eval, task_id=task_id)
            )

            manager = self.staleness_manager
            traj_result: list[dict[str, Any]] | dict[str, Any] | None = None
            should_accept_fn = pending_task.should_accept_fn
            should_accept: bool | None = None
            reason: str | None = None

            try:
                traj_result = await pending_task.workflow.arun_episode(
                    self.inference_engine, pending_task.data
                )

                # Handle different return types from arun_episode
                if traj_result is None:
                    should_accept_traj = False
                    reason = "returned_none"
                else:
                    # Convert to list[dict] format
                    if isinstance(traj_result, dict):
                        # Check if it's InteractionWithTokenLogpReward format
                        if all(
                            isinstance(v, InteractionWithTokenLogpReward)
                            for v in traj_result.values()
                        ):
                            traj_result = [
                                concat_padded_tensors(
                                    [v.to_tensor_dict() for v in traj_result.values()]
                                )
                            ]
                        else:
                            traj_result = [traj_result]
                    elif isinstance(traj_result, list):
                        # Verify all items are dicts
                        if not all(isinstance(traj, dict) for traj in traj_result):
                            raise ValueError(
                                f"Expected list of dicts from arun_episode, got {type(traj_result)}"
                            )
                    else:
                        raise ValueError(
                            f"Expected list[dict], dict, or None from arun_episode, got {type(traj_result)}"
                        )

                    # Apply acceptance function if provided
                    if should_accept_fn is None:
                        should_accept = True
                    else:
                        # For list returns, we accept if at least one trajectory is accepted
                        should_accept = any(should_accept_fn(traj) for traj in traj_result)

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
    ) -> list[dict[str, Any] | None]:
        """
        Wait for the completion of `count` workflows and extract trajectories.

        Handles both _TreeSearchRolloutResult and legacy _RolloutResult.
        For _TreeSearchRolloutResult, returns the list of trajectories.
        For legacy _RolloutResult, wraps the single trajectory in a list.
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
                # Handle legacy _RolloutResult
                extracted.append(r.trajectory)

        return extracted

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow,
    ) -> list[dict[str, Any]]:
        """
        Submit a batch of requests and wait for results, flattening list[list[dict]] → list[dict].
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
