# customized_areal/tree_search/grouped_workflow.py
"""Tree-search-aware grouped rollout workflow.

Replaces both QueryIDProxyWorkflow and GroupedRolloutWorkflow for tree search
training. Subclasses GroupedRolloutWorkflow and overrides arun_episode to:

1. Reconstruct episode metadata from InteractionWithTokenLogpReward parent chains
2. Convert each turn to an individual-style tensor dict [1, seq_len]
3. Stack all turns from all group_size episodes into [total_turns, max_seq_len]
4. Preserve per-episode metadata (turn IDs, parent IDs, rewards) as list-valued keys
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch

from areal.api import InferenceEngine
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.utils import logging
from areal.utils.data import concat_padded_tensors

logger = logging.getLogger("TreeSearchGroupedWorkflow")

EPISODE_LEVEL_METADATA_KEYS = frozenset(
    {
        "_episode_num_turns",
        "_episode_turn_offsets",
        "_episode_turn_ids",
        "_episode_parent_turn_ids",
        "_episode_turn_rewards",
        "_episode_outcome_reward",
    }
)


def _sort_interactions_by_creation(
    interactions: dict[str, InteractionWithTokenLogpReward],
) -> list[InteractionWithTokenLogpReward]:
    """Sort interactions by cache insertion order (dict key order in Python 3.7+)."""
    return list(interactions.values())


def _collect_episode_metadata(
    interactions: list[InteractionWithTokenLogpReward],
) -> dict[str, Any]:
    """Extract episode-level metadata from a sorted list of interactions.

    Returns dict with keys:
        turn_ids: list[str]
        parent_turn_ids: list[str | None]
        turn_rewards: list[float]
        outcome_reward: float
    """
    turn_ids: list[str] = []
    parent_turn_ids: list[str | None] = []
    turn_rewards: list[float] = []

    for interaction in interactions:
        iid = interaction.interaction_id
        turn_ids.append(iid if iid is not None else "")
        parent_iid = (
            interaction.parent.interaction_id
            if interaction.parent is not None
            else None
        )
        parent_turn_ids.append(parent_iid)
        turn_rewards.append(
            interaction.reward if interaction.reward is not None else 0.0
        )

    outcome_reward = turn_rewards[-1] if turn_rewards else 0.0

    return {
        "turn_ids": turn_ids,
        "parent_turn_ids": parent_turn_ids,
        "turn_rewards": turn_rewards,
        "outcome_reward": outcome_reward,
    }


class TreeSearchGroupedRolloutWorkflow(GroupedRolloutWorkflow):
    """GroupedRolloutWorkflow that preserves per-turn episode metadata for tree search.

    When used with individual export style, each turn from each episode becomes
    a separate [1, seq_len] tensor dict. All turns are stacked into
    [total_turns, max_seq_len] with episode-level metadata preserved as
    list-valued keys so that downstream _split_to_turn_dicts can reconstruct
    per-turn dicts with full episode context.
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

        # Check if results are InteractionWithTokenLogpReward dicts
        first = valid_results[0]
        if not (
            isinstance(first, dict)
            and first
            and all(
                isinstance(v, InteractionWithTokenLogpReward) for v in first.values()
            )
        ):
            # Tensor dicts — fall back to base class behavior
            concatenated = concat_padded_tensors(valid_results)
            return [concatenated] if concatenated else None

        # Individual export style: reconstruct episode metadata per episode,
        # convert each turn to tensor dict, then concatenate per episode.
        episode_trajs: list[dict[str, Any]] = []

        query_id = data.get("query_id", "")

        for result in valid_results:
            sorted_interactions = _sort_interactions_by_creation(result)
            metadata = _collect_episode_metadata(sorted_interactions)

            turn_dicts = [
                interaction.to_tensor_dict() for interaction in sorted_interactions
            ]

            if not turn_dicts:
                continue

            # Stack turns for this episode
            ep_traj = concat_padded_tensors(turn_dicts)

            # Add episode-level metadata
            if query_id:
                ep_traj["_mcts_query_id"] = query_id
            ep_traj["_episode_turn_ids"] = metadata["turn_ids"]
            ep_traj["_episode_parent_turn_ids"] = metadata["parent_turn_ids"]
            ep_traj["_episode_turn_rewards"] = metadata["turn_rewards"]
            ep_traj["_episode_outcome_reward"] = metadata["outcome_reward"]
            ep_traj["_episode_num_turns"] = len(turn_dicts)

            episode_trajs.append(ep_traj)

        if not episode_trajs:
            return None

        return episode_trajs


def _split_to_turn_dicts(trajs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split stacked trajectory dicts into flat list of per-turn dicts.

    Each output dict has shape [1, seq_len] and carries episode metadata keys:
        _mcts_query_id, _episode_idx, _turn_idx_in_episode,
        _parent_turn_id, _turn_reward, _outcome_reward

    Episode-level metadata keys (prefixed with _episode_) are removed from
    per-turn dicts; their values are distributed into the turn-level keys.
    """
    flat: list[dict[str, Any]] = []

    for traj in trajs:
        offsets = traj["_episode_turn_offsets"]
        num_turns_list = traj["_episode_num_turns"]
        query_id = traj.get("_mcts_query_id", "")

        for ep_idx, num_turns in enumerate(num_turns_list):
            start = offsets[ep_idx]
            for local_turn_idx in range(num_turns):
                t = start + local_turn_idx
                turn_dict = {}
                for k, v in traj.items():
                    if k in EPISODE_LEVEL_METADATA_KEYS:
                        continue
                    if isinstance(v, torch.Tensor):
                        turn_dict[k] = v[t : t + 1]
                    else:
                        # Non-tensor values (e.g. _mcts_query_id string)
                        # are identical across turns — copy as-is
                        turn_dict[k] = v

                turn_dict["_mcts_query_id"] = query_id
                turn_dict["_episode_idx"] = ep_idx
                turn_dict["_turn_idx_in_episode"] = local_turn_idx
                turn_dict["_parent_turn_id"] = traj["_episode_parent_turn_ids"][ep_idx][
                    local_turn_idx
                ]
                turn_dict["_turn_reward"] = traj["_episode_turn_rewards"][ep_idx][
                    local_turn_idx
                ]
                turn_dict["_outcome_reward"] = traj["_episode_outcome_reward"][ep_idx]

                flat.append(turn_dict)

    return flat
