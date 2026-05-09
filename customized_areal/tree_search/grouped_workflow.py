# customized_areal/tree_search/grouped_workflow.py
"""Tree-search-aware grouped rollout workflow.

Subclasses GroupedRolloutWorkflow and overrides arun_episode to run multiple
rollouts and collect all per-turn Nodes into a flat list[Node].
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from customized_areal.tree_search.mcts_tree_store import Node

from areal.api import InferenceEngine
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("TreeSearchGroupedWorkflow")


class TreeSearchGroupedRolloutWorkflow(GroupedRolloutWorkflow):
    """GroupedRolloutWorkflow that runs multiple rollouts and collects all per-turn Nodes into a flat list[Node]."""

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> list | None:
        results = await asyncio.gather(
            *[self.workflow.arun_episode(engine, data) for _ in range(self.group_size)],
            return_exceptions=True,
        )

        valid_results = [
            r for r in results if not isinstance(r, Exception) and r is not None
        ]

        if not valid_results:
            return None

        if len(valid_results) < len(results):
            self.logger.warning(
                f"TreeSearchGroupedWorkflow: "
                f"{len(results) - len(valid_results)}/{len(results)} "
                "trajectories returned None, using remaining results"
            )

        first = valid_results[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], Node):
            query_id = data.get("query_id") or ""
            all_nodes: list[Node] = []
            for group_idx, result in enumerate(valid_results):
                episode_id = (
                    f"{query_id}_{group_idx}_{uuid.uuid4().hex[:8]}"
                    if query_id
                    else f"{group_idx}_{uuid.uuid4().hex[:8]}"
                )
                for turn_idx, node in enumerate(result, start=1):
                    node.episode_id = episode_id
                    node.query_id = query_id
                    node.turn_idx = turn_idx
                all_nodes.extend(result)
            return all_nodes if all_nodes else None

        return None
