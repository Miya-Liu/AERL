"""
OpenAI-compatible client with token-level reward support via HTTP API.

This module provides OpenAIProxyClient which extends the real
OpenAIProxyClient from areal to support token-level rewards via HTTP API.

This eliminates the need for a local cache by calling the proxy server's
token-level reward endpoints directly.

Example
-------
>>> http_session = aiohttp.ClientSession()
>>> client = OpenAIProxyClient(
...     session=http_session,
...     base_url="http://localhost:8000",
...     task_id="task-1",
...     admin_api_key="admin-key",
... )
>>> async with client:
...     # Agent uses session_api_key for LLM calls
...     rewards = await agent.run(api_key=client.session_api_key)
...     # Set token-level rewards via HTTP
...     await client.set_rewards("completion-id", [0.5, 0.3, 0.2])
... # Export after session ends
>>> interactions = await client.export_interactions()
"""

from __future__ import annotations

from typing import Any

import aiohttp

from areal.experimental.openai.proxy.client_session import (
    OpenAIProxyClient as BaseOpenAIProxyClient,
)
from areal.experimental.openai.proxy.client_session import (
    post_json_with_retry,
)
from areal.utils import logging

from .proxy_rollout_server import (
    deserialize_interactions_with_position_rewards,
)
from .server import (
    EXPORT_TRAJECTORIES_PATHNAME,
    RL_COMPUTE_ENTROPY_PATHNAME,
    RL_SET_POSITION_REWARDS_PATHNAME,
    RL_SET_TOKEN_REWARDS_PATHNAME,
    ComputeEntropyRequest,
    PositionRewardInfo,
    SetPositionRewardsRequest,
    SetTokenRewardsRequest,
)

logger = logging.getLogger("OpenAIProxyClient")


class OpenAIProxyClient(BaseOpenAIProxyClient):
    """
    Extended OpenAIProxyClient with token-level reward support via HTTP API.

    This class inherits from the real OpenAIProxyClient in
    areal.experimental.openai.proxy.client_session and adds methods
    for setting token-level rewards via HTTP API.

    Unlike the previous design, this client does NOT use a local cache.
    Instead, it calls the proxy server's token-level reward endpoints directly.

    Parameters
    ----------
    session : aiohttp.ClientSession
        HTTP session for requests
    base_url : str
        Base URL of the proxy server
    task_id : str
        Unique identifier for this task
    admin_api_key : str
        Admin API key for management operations

    Example
    -------
    >>> async with aiohttp.ClientSession() as http_session:
    ...     client = OpenAIProxyClient(
    ...         session=http_session,
    ...         base_url="http://localhost:8000",
    ...         task_id="task-1",
    ...         admin_api_key="admin-key",
    ...     )
    ...     async with client:
    ...         # Agent uses session_api_key for LLM calls through proxy server
    ...         rewards = await agent.run(api_key=client.session_api_key)
    ...         # Set token-level rewards via HTTP
    ...         await client.set_rewards("comp-1", [0.5, 0.3, 0.2])
    ...     # Export after session ends
    ...     interactions = await client.export_interactions()
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        task_id: str,
        admin_api_key: str,
    ):
        super().__init__(
            session=session,
            base_url=base_url,
            task_id=task_id,
            admin_api_key=admin_api_key,
        )

    async def set_rewards(
        self,
        completion_id: str | None,
        token_rewards: list[float],
    ) -> None:
        """
        Set token-wise rewards for a completion via HTTP API.

        Parameters
        ----------
        completion_id : str | None
            The completion/interaction ID, or None for the last interaction
        token_rewards : list[float]
            Token-wise rewards, one per output token

        Raises
        ------
        RuntimeError
            If session is not started
        aiohttp.ClientResponseError
            If HTTP request fails
        """
        if self.session_id is None:
            raise RuntimeError("Session not started")

        url = f"{self.base_url}{RL_SET_TOKEN_REWARDS_PATHNAME}"
        payload = SetTokenRewardsRequest(
            interaction_id=completion_id,
            token_rewards=token_rewards,
        )
        headers = self._session_auth_headers()

        await post_json_with_retry(
            self._session,
            url=url,
            payload=payload.model_dump(),
            headers=headers,
        )

        logger.info(
            f"Set {len(token_rewards)} token rewards for {completion_id}: "
            f"sum={sum(token_rewards):.3f}"
        )

    async def set_position_rewards(
        self,
        completion_id: str | None,
        position_rewards: list[PositionRewardInfo],
    ) -> None:
        """
        Set position-wise rewards for a completion via HTTP API.

        Parameters
        ----------
        completion_id : str | None
            The completion/interaction ID, or None for the last interaction
        position_rewards : list[PositionRewardInfo]
            Position-wise candidate rewards

        Raises
        ------
        RuntimeError
            If session is not started
        aiohttp.ClientResponseError
            If HTTP request fails
        """
        if self.session_id is None:
            raise RuntimeError("Session not started")

        url = f"{self.base_url}{RL_SET_POSITION_REWARDS_PATHNAME}"

        # Convert dataclasses to dicts for JSON serialization
        pr_dicts = [
            {
                "position": pr.position,
                "candidates": pr.candidates,
                "candidate_token_ids": pr.candidate_token_ids,
                "logprobs": pr.logprobs,
                "rewards": pr.rewards,
                "chosen_index": pr.chosen_index,
            }
            for pr in position_rewards
        ]

        payload = SetPositionRewardsRequest(
            interaction_id=completion_id,
            position_rewards=pr_dicts,  # type: ignore
        )
        headers = self._session_auth_headers()

        await post_json_with_retry(
            self._session,
            url=url,
            payload=payload.model_dump(),
            headers=headers,
        )

        logger.info(
            f"Set position-wise rewards for {completion_id}: "
            f"{len(position_rewards)} positions"
        )

    async def set_last_rewards(self, token_rewards: list[float]) -> None:
        """
        Set token-wise rewards for the most recent completion.

        Parameters
        ----------
        token_rewards : list[float]
            Token-wise rewards, one per output token

        Raises
        ------
        RuntimeError
            If session is not started
        """
        await self.set_rewards(
            completion_id=None,  # None means "last interaction"
            token_rewards=token_rewards,
        )

    async def set_last_position_rewards(
        self,
        position_rewards: list[PositionRewardInfo],
    ) -> None:
        """
        Set position-wise rewards for the most recent completion.

        Parameters
        ----------
        position_rewards : list[PositionRewardInfo]
            Position-wise candidate rewards

        Raises
        ------
        RuntimeError
            If session is not started
        """
        await self.set_position_rewards(
            completion_id=None,  # None means "last interaction"
            position_rewards=position_rewards,
        )

    async def compute_entropy(self, completion_id: str) -> list[float]:
        """
        Compute entropy for a completion via HTTP API.

        Parameters
        ----------
        completion_id : str
            The completion ID

        Returns
        -------
        list[float]
            Entropy values for each position

        Raises
        ------
        RuntimeError
            If session is not started
        aiohttp.ClientResponseError
            If HTTP request fails
        """
        if self.session_id is None:
            raise RuntimeError("Session not started")

        url = f"{self.base_url}{RL_COMPUTE_ENTROPY_PATHNAME}"
        payload = ComputeEntropyRequest(interaction_id=completion_id)
        headers = self._session_auth_headers()

        async with self._session.post(
            url, json=payload.model_dump(), headers=headers
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["entropies"]

    async def get_entropies(self, completion_id: str) -> list[float] | None:
        """
        Get computed entropy values for a completion.

        This is a convenience method that calls compute_entropy.

        Parameters
        ----------
        completion_id : str
            The completion ID

        Returns
        -------
        list[float] | None
            Entropy values per position, or None if not computed
        """
        try:
            return await self.compute_entropy(completion_id)
        except Exception:
            return None

    async def export_interactions(
        self,
        discount: float = 1.0,
        style: str = "individual",
    ) -> dict:
        """Export interactions with position_rewards support.

        Overrides the base class method to use custom deserialization
        that reconstructs position_rewards for the distillation loss.
        """
        if self.session_id is None:
            raise ValueError("session_id must be set before exporting interactions")

        url = f"{self.base_url}{EXPORT_TRAJECTORIES_PATHNAME}"
        payload = {
            "session_id": self.session_id,
            "discount": discount,
            "style": style,
        }
        headers = self._admin_auth_headers()
        async with self._session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return deserialize_interactions_with_position_rewards(data["interactions"])

    async def get_last_interaction(self) -> Any:
        """Get the most recent interaction from the proxy server.

        Fetches all interactions from the server and returns the last one.
        The interaction includes model_response with output_tokens, input_tokens,
        and output_top_logprobs needed for teacher distillation.

        Returns
        -------
        Any
            The last interaction object, or None if no interactions exist.

        Raises
        ------
        RuntimeError
            If session is not started.
        """
        if self.session_id is None:
            raise RuntimeError("Session not started")

        url = f"{self.base_url}{EXPORT_TRAJECTORIES_PATHNAME}"
        params = {"discount": "1.0", "style": "individual"}
        headers = self._session_auth_headers()

        async with self._session.get(url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        interactions = data.get("interactions", {})
        if not interactions:
            return None

        # Return the last interaction by insertion order
        last_id = list(interactions.keys())[-1]
        return interactions[last_id]


__all__ = ["OpenAIProxyClient"]
