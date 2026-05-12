# customized_areal/tree_search/tree_search_grouped_workflow.py
"""Tree-search-aware grouped rollout workflow with cache reuse.

Consolidates the functionality of QueryIDProxyWorkflow,
TreeSearchGroupedRolloutWorkflow, and TreeSearchWorkflowExecutor into
a single class that:
- Loads/saves tree_store from a checkpoint directory
- Does per-query cache lookup to determine how many fresh episodes are needed
- Generates only the needed fresh episodes (partial cache reuse)
- Converts fresh results to Nodes, loads cached Nodes, combines them
- Inserts fresh Nodes into tree_store, computes advantages, marks trained
- Saves tree checkpoint
- Returns batched tensor dicts that the base WorkflowExecutor handles natively
"""

from __future__ import annotations

import uuid
from typing import Any

from customized_areal.tree_search.config import AdvantageMode, CacheMode, LossMode
from customized_areal.tree_search.mcts_tree_store import Node

from areal.api import RolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("TreeSearchGroupedWorkflow")


def interactions_dict_to_nodes(interactions: dict[str, Any]) -> list[Node]:
    """Convert dict[str, InteractionWithTokenLogpReward] to list[Node].

    Each interaction becomes one Node representing a single turn.
    """
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    nodes: list[Node] = []

    for turn_idx, (interaction_id, interaction) in enumerate(
        interactions.items(), start=1
    ):
        if not isinstance(interaction, InteractionWithTokenLogpReward):
            logger.warning(
                "Skipping interaction %s (type=%s, expected InteractionWithTokenLogpReward)",
                interaction_id,
                type(interaction).__name__,
            )
            continue
        resp = interaction.model_response
        if resp is None:
            logger.warning(
                "Skipping interaction %s: model_response is None",
                interaction_id,
            )
            continue

        seq_tokens = resp.input_tokens + resp.output_tokens

        if (
            interaction.chat_template_type == "concat"
            and interaction.parent is not None
        ):
            parent_res = interaction.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions)

            if resp.input_len > parent_len:
                logprobs = (
                    parent_logprobs
                    + [0.0] * (resp.input_len - parent_len)
                    + resp.output_logprobs
                )
                loss_mask = (
                    parent_loss_mask
                    + [0] * (resp.input_len - parent_len)
                    + [1] * resp.output_len
                )
                versions = (
                    parent_versions
                    + [-1] * (resp.input_len - parent_len)
                    + resp.output_versions
                )
            else:
                logger.error(
                    "concat mode: resp.input_len (%d) <= parent_len (%d) — "
                    "expected monotonic growth. Zero-filling prompt context.",
                    resp.input_len,
                    parent_len,
                )
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

        outcome_reward = interaction.reward if interaction.reward is not None else 0.0

        topk_ids: list[list[int]] = []
        topk_logp: list[list[float]] = []
        if resp.output_top_logprobs is not None:
            for pos_logprobs in resp.output_top_logprobs:
                ids = []
                logps = []
                for token_id, lp in pos_logprobs:
                    ids.append(token_id)
                    logps.append(lp)
                topk_ids.append(ids)
                topk_logp.append(logps)

        pn_id: str | None = None
        if interaction.parent is not None:
            pn_id = interaction.parent.interaction_id

        node = Node(
            input_ids=seq_tokens,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            outcome_reward=outcome_reward,
            turn_idx=turn_idx,
            node_id=interaction_id,
            parent_node_id=pn_id,
            topk_ids=topk_ids if topk_ids else None,
            topk_logp=topk_logp if topk_logp else None,
        )

        nodes.append(node)

    return nodes


def _nodes_to_batched_tensor_dict(nodes: list[Node]) -> dict[str, Any] | None:
    """Convert list[Node] to a batched tensor dict with metadata.

    Each Node is converted to a [1, seq_len] tensor dict via
    _node_to_tensor_dict, then all are concatenated via
    concat_padded_tensors into a single [N, seq_len] batched dict.

    Returns None if nodes is empty.
    """
    if not nodes:
        return None

    from areal.utils.data import concat_padded_tensors
    from customized_areal.tree_search.mcts_tree_store import _node_to_tensor_dict

    tensor_dicts = [
        _node_to_tensor_dict(
            node,
            query_id=node.query_id or "",
            node_id=node.node_id,
        )
        for node in nodes
    ]
    return concat_padded_tensors(tensor_dicts)
