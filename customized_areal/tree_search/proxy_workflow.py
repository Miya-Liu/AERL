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

    # DEBUG: log input dict structure
    logger.warning(
        "DEBUG interactions_dict_to_nodes: dict has %d keys, "
        "key_samples=%s, value_type_samples=%s",
        len(interactions),
        list(interactions.keys())[:3],
        [type(v).__name__ for v in list(interactions.values())[:3]],
    )

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

        # Derive parent node_id from parent interaction when available
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


class QueryIDProxyWorkflow(OpenAIProxyWorkflow):
    """OpenAIProxyWorkflow that preserves dataset query_id in trajectories.

    Overrides ``arun_episode`` to inject ``query_id`` (from
    ``data["query_id"]``) into the returned trajectory dict. The base class
    converts ``InteractionWithTokenLogpReward`` objects to tensor dicts via
    ``concat_padded_tensors``, which drops non-tensor keys. This subclass
    performs the conversion early and adds the query_id before returning.

    When ``group_size > 1``, this workflow runs multiple episodes internally
    and collects all per-turn Nodes into a flat ``list[Node]``, replacing the
    need for the base ``GroupedRolloutWorkflow`` wrapper (which doesn't
    understand ``list[Node]`` returns).

    Parameters
    ----------
    agent_path : str, optional
        Dotted import path to an agent class (e.g.
        ``"customized_areal.tpfc.tpfc_agent.TPFCAgent"``). The class is
        imported and instantiated with no arguments. Ignored if ``agent``
        is also provided.
    group_size : int, optional
        Number of rollout episodes to run per input. When > 1, this workflow
        handles grouping internally, so the caller (e.g. RemoteInfEngine)
        should set its own ``group_size=1`` to avoid double-wrapping.
    **kwargs
        All other kwargs are forwarded to ``OpenAIProxyWorkflow.__init__``.
    """

    def __init__(
        self, agent_path: str | None = None, group_size: int = 1, **kwargs: Any
    ) -> None:
        # Resolve agent_path to an agent instance if agent not explicitly given
        if "agent" not in kwargs and agent_path is not None:
            agent_cls = import_from_string(agent_path)
            kwargs["agent"] = agent_cls()
        self.group_size = group_size
        # Remove group_size from kwargs before passing to super (it's not an OpenAIProxyWorkflow param)
        kwargs.pop("group_size", None)
        logger.warning(
            "PATCH_VERIFICATION: QueryIDProxyWorkflow.__init__ CALLED — "
            "mode=%s, agent_path=%s, group_size=%d, proxy_addr=%s",
            kwargs.get("mode", "?"),
            agent_path,
            group_size,
            kwargs.get("proxy_addr", "?")[:40] if kwargs.get("proxy_addr") else "?",
        )
        super().__init__(**kwargs)

    def _interactions_to_nodes(self, interactions: dict[str, Any]) -> list[Node]:
        """Convert InteractionWithTokenLogpReward objects to list[Node].

        Each interaction becomes one Node representing a single turn.
        Delegates to the module-level utility for reuse.
        """
        return interactions_dict_to_nodes(interactions)

    def _single_episode(self, engine, data: dict, query_id: str) -> list[Node] | None:
        """Run a single episode and convert to list[Node].

        This is the synchronous version used by arun_episode internally.
        """
        # This will be called from arun_episode which is async
        raise NotImplementedError("Use _async_single_episode instead")

    async def _async_single_episode(
        self, engine, data: dict, query_id: str
    ) -> list[Node] | None:
        """Run a single episode via super().arun_episode and convert to Nodes."""
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
                "QueryIDProxyWorkflow: super().arun_episode returned list "
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
                "QueryIDProxyWorkflow: unexpected result type %s",
                type(result).__name__,
            )

        return None

    async def arun_episode(self, engine, data: dict) -> list[Node] | None:
        query_id = data.get("query_id") or ""

        logger.warning(
            "PATCH_VERIFICATION: QueryIDProxyWorkflow.arun_episode CALLED — "
            "class=%s, query_id=%s, engine_type=%s, group_size=%d",
            type(self).__name__,
            query_id,
            type(engine).__name__,
            self.group_size,
        )

        if self.group_size <= 1:
            return await self._async_single_episode(engine, data, query_id)

        # group_size > 1: run multiple episodes and collect all Nodes
        import asyncio

        results = await asyncio.gather(
            *[
                self._async_single_episode(engine, data, query_id)
                for _ in range(self.group_size)
            ],
            return_exceptions=True,
        )

        valid_results = [
            r for r in results if not isinstance(r, Exception) and r is not None
        ]

        if not valid_results:
            return None

        if len(valid_results) < len(results):
            logger.warning(
                "QueryIDProxyWorkflow: %d/%d episodes returned None or failed",
                len(results) - len(valid_results),
                len(results),
            )

        # Assign distinct episode_ids per group and flatten
        all_nodes: list[Node] = []
        for group_idx, episode_nodes in enumerate(valid_results):
            episode_id = (
                f"{query_id}_{group_idx}_{uuid.uuid4().hex[:8]}"
                if query_id
                else f"{group_idx}_{uuid.uuid4().hex[:8]}"
            )
            for turn_idx, node in enumerate(episode_nodes, start=1):
                node.episode_id = episode_id
                node.query_id = query_id
                if not node.turn_idx:
                    node.turn_idx = turn_idx
            all_nodes.extend(episode_nodes)

        return all_nodes if all_nodes else None
