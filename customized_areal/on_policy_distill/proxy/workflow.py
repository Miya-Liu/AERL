"""
Custom workflow demonstrating token-level reward support via HTTP API.

This module provides OpenAIProxyWorkflow, which extends the base OpenAIProxyWorkflow
from AReaL to support setting token-level rewards via the proxy server's HTTP API.

Unlike the previous design, this workflow does NOT use a local cache.
Token-level rewards are sent directly to the proxy server via HTTP.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from areal.infra import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import session_context, trace_session
from areal.experimental.openai.proxy.workflow import (
    OpenAIProxyWorkflow as BaseOpenAIProxyWorkflow,
)
from .client import OpenAIProxyClient
from .types import TokenRewardInteractions

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
            Agent object with async run() method.
        proxy_addr : str
            Address of the OpenAI proxy server.
        admin_api_key : str
            Admin API key for proxy server.
        discount : float
            Discount factor for reward backpropagation.
        export_style : str
            Export style ("individual" or "concat").
        """
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
            from concurrent.futures import ProcessPoolExecutor

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
        import os
        import asyncio

        for key, value in extra_envs.items():
            os.environ[key] = value
        return asyncio.run(agent.run(data))

    @staticmethod
    def _get_executor():
        from areal.experimental.openai.proxy.workflow import _get_executor

        return _get_executor()

    async def _process_rewards(
        self, client: OpenAIProxyClient, rewards: Any
    ) -> None:
        """
        Process rewards from agent output and send to proxy server via HTTP.

        Supports:
        - float: Scalar reward for last completion
        - dict[str, float]: Map of completion ID to scalar reward
        - dict[str, list[float]]: Map of completion ID to token-level rewards

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
            else:
                raise ValueError(
                    f"Invalid reward value type for {completion_id}: {type(reward_value)}. "
                    "Expected float or list[float]"
                )

    @session_context()
    async def arun_episode(
        self, engine: TRolloutEngine, data: dict[str, Any]
    ) -> TokenRewardInteractions | None:
        """
        Run a single episode with token-level reward support via HTTP API.

        This is the main entry point called by AReaL's training loop.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine (provided by AReaL).
        data : dict[str, Any]
            Input data for this episode.

        Returns
        -------
        TokenRewardInteractions | None
            Dictionary of interactions with token-level rewards, or None if rejected.
        """
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

        return interactions
