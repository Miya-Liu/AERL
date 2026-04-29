# customized_areal/tree_search/grouped_workflow.py
"""Tree-search-aware grouped rollout workflow.

Replaces both QueryIDProxyWorkflow and GroupedRolloutWorkflow for tree search
training. Subclasses GroupedRolloutWorkflow and overrides arun_episode to:

1. Accept list[dict] returns from the inner workflow (proxy_workflow now returns list[dict])
2. Merge per-turn dicts into per-episode dicts using Python lists (not tensors)
3. Return list[dict] (one dict per episode)
"""

from __future__ import annotations

import asyncio
from typing import Any

from customized_areal.tree_search.mcts_tree_store import _find_turn_boundaries

from areal.api import InferenceEngine
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.utils import logging
from areal.utils.data import concat_padded_tensors

logger = logging.getLogger("TreeSearchGroupedWorkflow")


def _merge_turn_dicts_to_episode(turn_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-turn dictionaries into a single per-episode dictionary.

    Args:
        turn_dicts: List of per-turn trajectory dicts

    Returns:
        Single per-episode trajectory dict with all turns merged
    """
    if not turn_dicts:
        return {}

    # Concatenate sequence fields
    input_ids = []
    loss_mask = []
    logprobs = []
    versions = []

    # Merge turn-specific fields
    turn_ids = []
    parent_turn_ids = []
    turn_rewards = []

    # Merge response-only fields
    response_ids = []
    logp = []
    topk_ids = []
    topk_logp = []
    distill_reward = []
    teacher_logp = []

    for turn_dict in turn_dicts:
        # Sequence fields
        input_ids.extend(turn_dict["input_ids"])
        loss_mask.extend(turn_dict["loss_mask"])
        logprobs.extend(turn_dict["logprobs"])
        versions.extend(turn_dict["versions"])

        # Turn-specific fields
        turn_ids.extend(turn_dict["turn_ids"])
        parent_turn_ids.extend(turn_dict["parent_turn_ids"])
        turn_rewards.extend(turn_dict["turn_rewards"])

        # Response-only fields
        if "response_ids" in turn_dict:
            response_ids.extend(turn_dict["response_ids"])
        if "logp" in turn_dict:
            logp.extend(turn_dict["logp"])
        if "topk_ids" in turn_dict:
            topk_ids.extend(turn_dict["topk_ids"])
        if "topk_logp" in turn_dict:
            topk_logp.extend(turn_dict["topk_logp"])
        if "distill_reward" in turn_dict:
            distill_reward.extend(turn_dict["distill_reward"])
        if "teacher_logp" in turn_dict:
            teacher_logp.extend(turn_dict["teacher_logp"])

    # Recompute turn boundaries on full loss_mask
    turn_response_starts, turn_response_ends = _find_turn_boundaries(loss_mask)

    # Create episode dict
    episode_dict = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "versions": versions,
        "attention_mask": [1] * len(input_ids),
        "turn_response_starts": turn_response_starts,
        "turn_response_ends": turn_response_ends,
        "turn_ids": turn_ids,
        "parent_turn_ids": parent_turn_ids,
        "turn_rewards": turn_rewards,
        "reward": turn_dicts[-1]["reward"],
        "outcome_reward": turn_dicts[-1]["outcome_reward"],
    }

    # Add response-only fields if present
    if response_ids:
        episode_dict["response_ids"] = response_ids
    if logp:
        episode_dict["logp"] = logp
    if topk_ids:
        episode_dict["topk_ids"] = topk_ids
    if topk_logp:
        episode_dict["topk_logp"] = topk_logp
    if distill_reward:
        episode_dict["distill_reward"] = distill_reward
    if teacher_logp:
        episode_dict["teacher_logp"] = teacher_logp

    return episode_dict


class TreeSearchGroupedRolloutWorkflow(GroupedRolloutWorkflow):
    """GroupedRolloutWorkflow that preserves per-turn episode metadata for tree search.

    When used with individual export style, each turn from each episode becomes
    a separate dict. All turns from an episode are merged into a single per-episode dict
    using Python lists (not tensors).
    """

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        results = await asyncio.gather(
            *[self.workflow.arun_episode(engine, data) for _ in range(self.group_size)]
        )

        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return None

        if len(valid_results) < len(results):
            self.logger.warning(
                f"TreeSearchGroupedWorkflow: "
                f"{len(results) - len(valid_results)}/{len(results)} "
                "trajectories returned None, using remaining results"
            )

        # Check if results are list[dict] (new format)
        first = valid_results[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
            # Merge turn dicts per episode into a single per-episode dict
            episode_trajs: list[dict[str, Any]] = []
            query_id = data.get("query_id", "")

            for result in valid_results:
                merged = _merge_turn_dicts_to_episode(result)
                if merged:
                    if query_id:
                        merged["_mcts_query_id"] = query_id
                    episode_trajs.append(merged)

            if not episode_trajs:
                return None

            return episode_trajs

        # Legacy tensor dicts — fall back to base class behavior
        concatenated = concat_padded_tensors(valid_results)
        return [concatenated] if concatenated else None
