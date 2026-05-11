# customized_areal/tree_search/proxy_workflow.py
"""OpenAIProxyWorkflow subclass that injects dataset query_id into trajectories.

When rollout_batch returns trajectories asynchronously, the ordering between
submitted prompts and returned trajectories may differ. This workflow injects
the dataset ``query_id`` (a string) from the input ``data`` dict into the
output trajectory as ``query_id``, so that MCTSTreeStore can use it
instead of computing an MD5 hash from prompt tokens.

Usage in config:
    workflow: customized_areal.tree_search.proxy_workflow.QueryIDProxyWorkflow
    workflow_kwargs:
      mode: inline
      agent_path: customized_areal.tpfc.tpfc_agent.TPFCAgent
      ... (other OpenAIProxyWorkflow kwargs)

The ``agent_path`` kwarg is resolved at init time: the class is imported
and instantiated (with no args) to produce the agent object. This lets the
config use a single string path instead of passing a pre-built agent instance.

If ``agent`` is provided directly (as a kwarg), ``agent_path`` is ignored.
"""

from __future__ import annotations

import uuid
from typing import Any

from customized_areal.tree_search.mcts_tree_store import Node

from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.utils import logging
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("QueryIDProxyWorkflow")


def interactions_dict_to_nodes(interactions: dict[str, Any]) -> list[Node]:
    """Convert dict[str, InteractionWithTokenLogpReward] to list[Node].

    Each interaction becomes one Node representing a single turn.
    This is a standalone utility so it can be used both by
    QueryIDProxyWorkflow (on the rollout engine) and by
    CacheAwarePPOTrainer (on the trainer side, when tree search
    patches are not applied to the remote engine).
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

        node = Node(
            input_ids=seq_tokens,
            loss_mask=loss_mask,
            logprobs=logprobs,
            versions=versions,
            outcome_reward=outcome_reward,
            turn_idx=turn_idx,
            topk_ids=topk_ids if topk_ids else None,
            topk_logp=topk_logp if topk_logp else None,
        )

        nodes.append(node)

    return nodes


class QueryIDProxyWorkflow(OpenAIProxyWorkflow):
    """OpenAIProxyWorkflow that preserves dataset query_id in trajectories.

    Overrides ``arun_episode`` to inject ``query_id`` (from
    ``data["query_id"]``) into the returned trajectory dict. The base class
    converts ``InteractionWithTokenLogpReward`` objects to tensor dicts via
    ``concat_padded_tensors``, which drops non-tensor keys. This subclass
    performs the conversion early and adds the query_id before returning.

    Parameters
    ----------
    agent_path : str, optional
        Dotted import path to an agent class (e.g.
        ``"customized_areal.tpfc.tpfc_agent.TPFCAgent"``). The class is
        imported and instantiated with no arguments. Ignored if ``agent``
        is also provided.
    **kwargs
        All other kwargs are forwarded to ``OpenAIProxyWorkflow.__init__``.
    """

    def __init__(self, agent_path: str | None = None, **kwargs: Any) -> None:
        # Resolve agent_path to an agent instance if agent not explicitly given
        if "agent" not in kwargs and agent_path is not None:
            agent_cls = import_from_string(agent_path)
            kwargs["agent"] = agent_cls()
        super().__init__(**kwargs)

    def _interactions_to_nodes(self, interactions: dict[str, Any]) -> list[Node]:
        """Convert InteractionWithTokenLogpReward objects to list[Node].

        Each interaction becomes one Node representing a single turn.
        Delegates to the module-level utility for reuse.
        """
        return interactions_dict_to_nodes(interactions)

    async def arun_episode(self, engine, data: dict) -> list[Node] | None:
        query_id = data.get("query_id") or ""

        result = await super().arun_episode(engine, data)

        if result is None:
            return None

        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        if isinstance(result, dict) and all(
            isinstance(v, InteractionWithTokenLogpReward) for v in result.values()
        ):
            nodes = self._interactions_to_nodes(result)
            episode_id = uuid.uuid4().hex
            for node in nodes:
                node.episode_id = episode_id
                node.query_id = query_id
            return nodes

        if isinstance(result, list):
            logger.warning(
                "QueryIDProxyWorkflow.arun_episode received list result "
                "instead of dict; attempting dict conversion"
            )
            if result and isinstance(result[0], InteractionWithTokenLogpReward):
                converted = {str(i): v for i, v in enumerate(result)}
                nodes = self._interactions_to_nodes(converted)
                episode_id = uuid.uuid4().hex
                for node in nodes:
                    node.episode_id = episode_id
                    node.query_id = query_id
                return nodes

        if result is not None:
            logger.warning(
                "QueryIDProxyWorkflow.arun_episode received unexpected result type: %s",
                type(result).__name__,
            )

        return None
