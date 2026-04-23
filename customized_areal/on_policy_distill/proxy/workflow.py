"""
Custom workflow demonstrating token-level reward support via HTTP API.

This module provides OpenAIProxyWorkflow, which extends the base OpenAIProxyWorkflow
from AReaL to support setting token-level rewards via the proxy server's HTTP API.

Unlike the previous design, this workflow does NOT use a local cache.
Token-level rewards are sent directly to the proxy server via HTTP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from areal.experimental.openai.proxy.workflow import (
    OpenAIProxyWorkflow as BaseOpenAIProxyWorkflow,
)
from areal.infra import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import session_context, trace_session

from .client import OpenAIProxyClient

if TYPE_CHECKING:
    from areal.experimental.openai.client import TRolloutEngine

logger = logging.getLogger("TokenRewardWorkflow")


class OpenAIProxyWorkflow(BaseOpenAIProxyWorkflow):
    """
    Workflow that supports token-level rewards via HTTP API.

    This workflow uses OpenAIProxyClient to communicate with the proxy server.
    Token-level rewards are sent directly to the server via HTTP API,
    eliminating the need for a local cache.

    The flow is:
    1. Create client with HTTP session
    2. Start session (HTTP call to proxy server)
    3. Run agent with session_api_key for LLM calls through proxy server
    4. Set token-level rewards via HTTP during the session
    5. End session (HTTP call to proxy server)
    6. Export interactions from server (with token-level rewards applied)

    The agent's run() method can return:
    - float: Scalar reward (backward compatible)
    - dict[str, float]: Completion ID -> scalar reward
    - dict[str, list[float]]: Completion ID -> token-level rewards
    - dict[str, dict]: Completion ID -> dict with "position_rewards" and
      "scalar_reward" keys (for distillation workflows)

    Example Agent
    -------------
    >>> class MyTokenRewardAgent:
    ...     async def run(self, data, **extra_kwargs):
    ...         client = extra_kwargs.get("proxy_client")  # OpenAIProxyClient
    ...         # ... generate response using proxy_client.session_api_key ...
    ...         # Return token-level rewards
    ...         return {
    ...             "completion_id": [0.0, 0.0, 0.5, 1.0, 1.0]  # per-token rewards
    ...         }

    Example Usage
    -------------
    >>> workflow = OpenAIProxyWorkflow(
    ...     agent=MyTokenRewardAgent(),
    ...     proxy_addr="http://localhost:8000",
    ...     discount=0.9,
    ... )
    >>> interactions = await workflow.arun_episode(engine, data)
    """

    def __init__(
        self,
        agent: Any,
        proxy_addr: str,
        admin_api_key: str = "dummy-admin-key",
        discount: float = 1.0,
        export_style: str = "individual",
    ):
        """
        Initialize the token reward proxy workflow.

        Parameters
        ----------
        agent : Any
            Agent object with async run() method, or a string import path
            (e.g., ``"customized_areal.on_policy_distill.core.agent.OnPolicyDistillAgent"``).
            When a string is provided, the agent class is imported and instantiated
            on the engine worker. This enables the agent to be passed via
            ``workflow_kwargs`` over RPC serialization.
        proxy_addr : str
            Address of the OpenAI proxy server. When using the rollout
            controller's proxy infrastructure, this is injected per-worker
            by ``TokenRewardRolloutController._create_submit_callback``.
        admin_api_key : str
            Admin API key for proxy server.
        discount : float
            Discount factor for reward backpropagation.
        export_style : str
            Export style ("individual" or "concat").
        """
        # Resolve string import path to agent instance
        if isinstance(agent, str):
            from areal.utils.dynamic_import import import_from_string

            agent_cls = import_from_string(agent)
            agent = agent_cls()

        super().__init__(
            mode="inline",
            agent=agent,
            proxy_addr=proxy_addr,
            admin_api_key=admin_api_key,
            discount=discount,
            export_style=export_style,
        )

    @trace_session("run_agent")
    async def _run_agent(
        self,
        session_api_key: str,
        data: dict,
        proxy_client: OpenAIProxyClient | None = None,
    ) -> Any:
        """Run the agent, passing the client as proxy_client for reward APIs."""
        if self.mode == "inline":
            http_client = await workflow_context.get_httpx_client()
            extra_kwargs = {
                "base_url": self.proxy_addr,
                "http_client": http_client,
                "api_key": session_api_key,
                "proxy_client": proxy_client,
            }
            return await self.agent.run(data, **extra_kwargs)
        if self.mode == "subproc":
            extra_envs = {
                "OPENAI_BASE_URL": self.proxy_addr,
                "OPENAI_API_KEY": session_api_key,
                "ANTHROPIC_BASE_URL": self.proxy_addr,
                "ANTHROPIC_API_KEY": session_api_key,
            }
            import asyncio

            loop = asyncio.get_running_loop()
            # Subproc mode does not support proxy client injection
            return await loop.run_in_executor(
                self._get_executor(),
                self._wrap_run_subproc,
                self.agent,
                data,
                extra_envs,
            )
        if self.mode == "online":
            http_client = await workflow_context.get_httpx_client()
            extra_kwargs = {
                "base_url": self.proxy_addr,
                "http_client": http_client,
                "api_key": self._admin_api_key,
                "proxy_client": proxy_client,
            }
            return await self.agent.run(data, **extra_kwargs)
        raise ValueError(f"Unsupported mode: {self.mode}")

    @staticmethod
    def _wrap_run_subproc(
        agent: Any, data: dict[str, Any], extra_envs: dict[str, str]
    ) -> Any:
        import asyncio
        import os

        for key, value in extra_envs.items():
            os.environ[key] = value
        return asyncio.run(agent.run(data))

    @staticmethod
    def _get_executor():
        from areal.experimental.openai.proxy.workflow import _get_executor

        return _get_executor()

    async def _process_rewards(self, client: OpenAIProxyClient, rewards: Any) -> None:
        """
        Process rewards from agent output and send to proxy server via HTTP.

        Supports:
        - float: Scalar reward for last completion
        - dict[str, float]: Map of completion ID to scalar reward
        - dict[str, list[float]]: Map of completion ID to token-level rewards
        - dict[str, dict]: Map of completion ID to dict with "position_rewards"
          and "scalar_reward" keys (for distillation workflows)

        Parameters
        ----------
        client : OpenAIProxyClient
            The proxy client for HTTP communication.
        rewards : Any
            Rewards from agent output.
        """
        if isinstance(rewards, float):
            # Scalar reward for last completion
            await client.set_last_reward(float(rewards))
            return

        if not isinstance(rewards, dict):
            raise ValueError(
                f"Invalid reward type: {type(rewards)}. Expected float or dict"
            )

        for completion_id, reward_value in rewards.items():
            if isinstance(reward_value, list):
                # Token-level rewards
                await client.set_rewards(completion_id, reward_value)
            elif isinstance(reward_value, float):
                # Scalar reward
                await client.set_reward(completion_id, float(reward_value))
            elif isinstance(reward_value, dict):
                # Position-level rewards with scalar reward
                # Used by distillation workflows (OnPolicyDistillAgent, TreeDistillAgent)
                position_rewards = reward_value.get("position_rewards")
                scalar_reward = reward_value.get("scalar_reward")

                if position_rewards is not None:
                    await client.set_position_rewards(completion_id, position_rewards)
                if scalar_reward is not None:
                    await client.set_reward(completion_id, float(scalar_reward))
            else:
                raise ValueError(
                    f"Invalid reward value type for {completion_id}: {type(reward_value)}. "
                    "Expected float, list[float], or dict"
                )

    @session_context()
    async def arun_episode(
        self, engine: TRolloutEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Run a single episode with token-level reward support via HTTP API.

        This is the main entry point called by AReaL's training loop.

        Converts interactions to a concatenated tensor dict and attaches
        position_rewards separately, avoiding concat_padded_tensors key
        consistency issues when some interactions have position_rewards
        and others don't (e.g., multi-turn conversations).

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine (provided by AReaL).
        data : dict[str, Any]
            Input data for this episode.

        Returns
        -------
        dict[str, Any] | None
            Concatenated tensor dict with position_rewards attached, or None.
        """
        from areal.utils.data import concat_padded_tensors

        task_id = workflow_context.get().task_id
        http_session = await workflow_context.get_aiohttp_session()

        await self._grant_capacity(http_session)

        # Use OpenAIProxyClient that supports token-level rewards via HTTP
        client = OpenAIProxyClient(
            session=http_session,
            base_url=self.proxy_addr,
            task_id=str(task_id),
            admin_api_key=self._admin_api_key,
        )

        async with client:
            try:
                # Run the agent - it uses session_api_key for LLM calls
                # The agent can also use proxy_client to set rewards directly
                rewards = await self._run_agent(
                    client.session_api_key, data, proxy_client=client
                )
            except Exception as e:
                logger.warning(
                    f"Agent task failed: {e}. This trajectory will be rejected."
                )
                raise

            # Apply rewards from the agent
            if rewards is not None:
                await self._process_rewards(client, rewards)

        # Export interactions from the server after session ends
        # The server applies token-level rewards during export
        interactions = await client.export_interactions(
            discount=self.discount,
            style=self.export_style,
        )

        if not interactions:
            logger.warning(
                "No interactions returned from proxy server, trajectory rejected."
            )
            return None

        # Record stats
        last_id = list(interactions.keys())[-1]
        last_reward = interactions[last_id].reward
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)

        # Convert interactions to tensor dict here (instead of letting
        # workflow_executor do it) so we can attach position_rewards
        # separately. This avoids concat_padded_tensors key consistency
        # issues when some interactions have position_rewards and others
        # don't (e.g., in multi-turn conversations).
        tensor_dict = concat_padded_tensors(
            [v.to_tensor_dict() for v in interactions.values()]
        )

        # Collect position_rewards from all interactions that have them.
        # position_rewards is a Python attribute set during server-side
        # export or client-side deserialization. It flows as a list
        # (flat-concatenated across interactions) to the distillation loss.
        # Each PositionRewardInfo gets a sample_index indicating which
        # batch item it belongs to, so that minibatch splitting can
        # correctly partition position_rewards across minibatches.
        # NOTE: This assumes export_style="individual" where each
        # interaction becomes a separate batch item. For "concat" style,
        # positions from subsequent interactions would need cumulative
        # offset adjustment based on prior interactions' output lengths.
        all_position_rewards = []
        for sample_idx, interaction in enumerate(interactions.values()):
            pr = getattr(interaction, "position_rewards", None)
            if pr is not None:
                for p in pr:
                    p.sample_index = sample_idx
                all_position_rewards.extend(pr)

        if all_position_rewards:
            tensor_dict["position_rewards"] = all_position_rewards

        return tensor_dict
