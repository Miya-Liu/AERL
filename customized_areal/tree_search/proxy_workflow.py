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

from customized_areal.tree_search.mcts_tree_store import _find_turn_boundaries

from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.utils import logging
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

    def _interactions_to_turn_dicts(
        self, interactions: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert InteractionWithTokenLogpReward objects to per-turn list-based dicts.

        Args:
            interactions: Dict of interaction_id to InteractionWithTokenLogpReward objects

        Returns:
            List of trajectory dicts with Python list fields following the schema:
            {
                "input_ids": list[int],              # unpadded token IDs
                "loss_mask": list[int],              # 0=prompt, 1=response
                "logprobs": list[float],             # chosen token log prob per position
                "versions": list[int],               # policy version per token
                "reward": float,                     # outcome reward
                "turn_response_starts": list[int],   # response start indices
                "turn_response_ends": list[int],     # response end indices
                "turn_ids": list[str],               # interaction ID per turn
                "parent_turn_ids": list[str | None], # parent interaction ID per turn
                "turn_rewards": list[float],         # per-turn reward
                "outcome_reward": float,             # outcome reward
                "response_ids": list[int],           # chosen response token IDs (output_tokens)
                "logp": list[float],                 # chosen token log probs (output_logprobs)
                "topk_ids": list[list[int]],         # top-k candidate token IDs per response position
                "topk_logp": list[list[float]],      # top-k candidate log probs per response position
                "distill_reward": list[list[float]], # per-response-position distillation reward (empty placeholder)
                "teacher_logp": list[list[float]],   # teacher log probs per response position (empty placeholder)
            }
        """
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        traj_dicts: list[dict[str, Any]] = []

        for interaction_id, interaction in interactions.items():
            assert isinstance(interaction, InteractionWithTokenLogpReward)
            resp = interaction.model_response
            assert resp is not None, "Model response is not set."

            # Build base sequence data
            seq_tokens = resp.input_tokens + resp.output_tokens

            if (
                interaction.chat_template_type == "concat"
                and interaction.parent is not None
            ):
                # For concat style with parent, include parent's data
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
                    # If child input is shorter than parent, ignore parent (shouldn't happen)
                    logprobs = [0.0] * resp.input_len + resp.output_logprobs
                    loss_mask = [0] * resp.input_len + [1] * resp.output_len
                    versions = [-1] * resp.input_len + resp.output_versions
            else:
                # Standard case: no parent or not concat style
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions

            reward = interaction.reward if interaction.reward is not None else 0.0

            # Compute turn boundaries
            turn_response_starts, turn_response_ends = _find_turn_boundaries(loss_mask)

            # Extract top-k logprobs
            topk_ids: list[list[int]] = []
            topk_logp: list[list[float]] = []
            if resp.output_top_logprobs is not None:
                for pos_logprobs in resp.output_top_logprobs:
                    ids = []
                    logps = []
                    for token_id, logp in pos_logprobs:
                        ids.append(token_id)
                        logps.append(logp)
                    topk_ids.append(ids)
                    topk_logp.append(logps)

            # Build trajectory dict
            traj = {
                "input_ids": seq_tokens,
                "loss_mask": loss_mask,
                "logprobs": logprobs,
                "versions": versions,
                "attention_mask": [1] * len(seq_tokens),
                "reward": reward,
                "turn_response_starts": turn_response_starts,
                "turn_response_ends": turn_response_ends,
                "turn_ids": [interaction_id],
                "parent_turn_ids": [
                    interaction.parent.interaction_id
                    if interaction.parent and interaction.parent._interaction_id
                    else None
                ],
                "turn_rewards": [reward],
                "outcome_reward": reward,
                "response_ids": resp.output_tokens,
                "logp": resp.output_logprobs,
                "topk_ids": topk_ids,
                "topk_logp": topk_logp,
                "distill_reward": [],
                "teacher_logp": [],
            }

            traj_dicts.append(traj)

        return traj_dicts

    async def arun_episode(
        self, engine, data: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        # Extract query_id from the input data before it gets lost
        query_id = data.get("query_id", "")

        # Run the base episode logic
        result = await super().arun_episode(engine, data)

        if result is None:
            return None

        # If result is still InteractionWithTokenLogpReward dict, convert now
        # and inject query_id
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        if isinstance(result, dict) and all(
            isinstance(v, InteractionWithTokenLogpReward) for v in result.values()
        ):
            traj_dicts = self._interactions_to_turn_dicts(result)
            if query_id:
                for traj in traj_dicts:
                    traj["_mcts_query_id"] = query_id
            return traj_dicts

        # If result is already a list (from wrapped workflow), inject query_id
        if isinstance(result, list):
            if query_id:
                for traj in result:
                    if isinstance(traj, dict):
                        traj["_mcts_query_id"] = query_id
            return result

        # If result is a tensor dict, wrap in list and inject query_id
        if isinstance(result, dict):
            if query_id:
                result["_mcts_query_id"] = query_id
            return [result]

        return None
