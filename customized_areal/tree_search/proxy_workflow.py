# customized_areal/tree_search/proxy_workflow.py
"""OpenAIProxyWorkflow subclass that injects dataset query_id into trajectories.

When rollout_batch returns trajectories asynchronously, the ordering between
submitted prompts and returned trajectories may differ. This workflow injects
the dataset ``query_id`` (a string) from the input ``data`` dict into the
output trajectory as ``_mcts_query_id``, so that MCTSTreeStore can use it
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

from typing import Any

from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.utils import logging
from areal.utils.data import concat_padded_tensors
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("QueryIDProxyWorkflow")


class QueryIDProxyWorkflow(OpenAIProxyWorkflow):
    """OpenAIProxyWorkflow that preserves dataset query_id in trajectories.

    Overrides ``arun_episode`` to inject ``_mcts_query_id`` (from
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

    async def arun_episode(self, engine, data: dict[str, Any]) -> dict[str, Any] | None:
        # Extract query_id from the input data before it gets lost
        query_id = data.get("query_id", "")

        # Run the base episode logic
        result = await super().arun_episode(engine, data)

        if result is None:
            return None

        # If result is still InteractionWithTokenLogpReward dict, convert now
        # and inject query_id. concat_padded_tensors is the same function
        # that workflow_executor would call, but we do it early to add
        # the string _mcts_query_id key before the executor sees it.
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        if isinstance(result, dict) and all(
            isinstance(v, InteractionWithTokenLogpReward) for v in result.values()
        ):
            traj = concat_padded_tensors([v.to_tensor_dict() for v in result.values()])
            if query_id:
                traj["_mcts_query_id"] = query_id
            return traj

        # If result is already a tensor dict (shouldn't normally happen),
        # just inject query_id directly
        if isinstance(result, dict) and query_id:
            result["_mcts_query_id"] = query_id

        return result
