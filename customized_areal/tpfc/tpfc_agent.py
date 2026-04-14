"""
TPFC Agent wrapper for AReaL integration.

This module provides a class-based agent interface consistent with AReaL's
agentic RL training pattern, wrapping the existing run_backend functionality.
"""

import asyncio
from typing import Any

from areal.api import AsyncRewardWrapper
from areal.utils import logging

from customized_areal.tpfc.backend_run import run_backend
from customized_areal.tpfc.gaia_final_reward import compute_reward

logger = logging.getLogger("TPFCAgent")


def tpfc_reward_fn(
    completions: list[dict[str, Any]],
    gt: str = "",
    user_query: str = "",
    judge_model_name: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> float:
    """
    Compute reward using GAIA final reward logic with LLM-as-judge.

    Args:
        completions: List of completion messages from the agent.
        gt: Ground truth answer.
        user_query: The original user question/task.
        judge_model_name: Model name for the judge LLM.
        judge_base_url: Base URL for the judge LLM API.
        judge_api_key: API key for the judge LLM.

    Returns:
        Float reward value (0.0 or 1.0).
    """
    if not completions:
        return 0.0

    # Extract response text from the last assistant message
    response_text = ""
    for msg in reversed(completions):
        if msg.get("role") == "assistant":
            response_text = msg.get("content", "")
            break

    if not response_text:
        return 0.0

    # Use default judge model if not specified
    model_name = "z-ai/glm-5.1"
    base_url = "https://openrouter.ai/api/v1"
    api_key = "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"

    try:
        result = compute_reward(
            response_text=response_text,
            ground_truth=gt,
            user_query=user_query,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )
        return float(result.get("answer_reward", 0))
    except Exception as exc:
        logger.warning("GAIA reward computation failed: %s", exc)
        return 0.0 


class TPFCAgent:
    """
    TPFC Agent for AReaL integration.
    
    This class wraps the run_backend function in a class-based structure
    compatible with AReal's RolloutWorkflow and PPOTrainer.
    
    Usage:
        agent = TPFCAgent()
        reward = await agent.run(data={"prompt": "task description", "answer": "ground_truth"}, **extra_kwargs)
    
    Attributes:
        default_agent_id: Default agent ID for TPFC agent runs.
    """
    
    default_agent_id: str = '8bba75cb-0d87-4efe-b566-87de77335b76'
    
    def __init__(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        judge_model_name: str | None = None,
        judge_base_url: str | None = None,
        judge_api_key: str | None = None,
        **kwargs,
    ):
        """
        Initialize TPFC Agent.

        Args:
            agent_id: Optional agent ID to use. Falls back to default_agent_id.
            user_id: Optional user ID for authentication.
            model_name: Optional model name for LLM calls.
            judge_model_name: Model name for the judge LLM (default: gpt-4o).
            judge_base_url: Base URL for the judge LLM API.
            judge_api_key: API key for the judge LLM.
            **kwargs: Additional configuration options (ignored but accepted for compatibility).
        """
        self.agent_id = agent_id or self.default_agent_id
        self.user_id = user_id
        self.model_name = model_name
        self.judge_model_name = judge_model_name
        self.judge_base_url = judge_base_url
        self.judge_api_key = judge_api_key
    
    async def run(
        self,
        data: dict[str, Any],
        **extra_kwargs,
    ) -> float:
        """
        Execute a single agent run and return the reward.
        
        This method is compatible with AReaL's OpenAIProxyWorkflow interface.
        The http_client from extra_kwargs is used to route LLM calls through
        AReaL's proxy server for token-level tracking.
        
        Args:
            data: Input data for the agent. Expected keys:
                - "messages": List of message dicts (last message contains the task)
                - "answer": Ground truth for reward calculation
            **extra_kwargs: Additional keyword arguments passed by OpenAIProxyWorkflow:
                - http_client: httpx.AsyncClient for proxy routing
                - base_url: Proxy server base URL
                - api_key: Session API key
        
        Returns:
            Float reward value (0.0 to 1.0).
        
        Raises:
            RuntimeError: If agent run fails to start.
            TimeoutError: If agent run doesn't complete within timeout.
        """
        # Extract task description from messages
        messages = data.get("messages", [])
        if messages:
            task_description = messages[-1].get("content", "")
        else:
            task_description = data.get("prompt", "")
        
        # Extract ground truth for reward calculation
        gt = data.get("answer", "")

        # Extract image paths from dataset (new files_path column)
        task_file_path = data.get("files_path", [])
        if task_file_path is None:
            task_file_path = []

        # Get OpenAI proxy parameters (passed by OpenAIProxyWorkflow._run_agent)
        base_url = extra_kwargs.get("base_url")
        http_client = extra_kwargs.get("http_client")
        api_key = extra_kwargs.get("api_key")

        logger.info(
            "TPFCAgent starting run: task=%s, agent_id=%s, has_ground_truth=%s, base_url=%s, n_images=%d",
            task_description[:100] if task_description else None,
            self.agent_id,
            bool(gt),
            base_url,
            len(task_file_path),
        )

        # Execute the backend run
        completion_messages, _final_answer, _log_path, _trace = await run_backend(
            task_description=task_description,
            task_file_path=task_file_path,
            log_path="./log.json",
            task_id="",
            gt=gt,
            tags=[],
            user_id=self.user_id,
            model_name=self.model_name,
            agent_id=self.agent_id,
            base_url=base_url,
            api_key=api_key,
        )
        
        # Calculate reward using reward function
        reward_fn = AsyncRewardWrapper(tpfc_reward_fn)
        reward = await reward_fn(
            completions=completion_messages,
            gt=gt,
            user_query=task_description,
            judge_model_name=self.judge_model_name,
            judge_base_url=self.judge_base_url,
            judge_api_key=self.judge_api_key,
        )
        
        logger.info(
            "TPFCAgent run completed: message_count=%d, reward=%.4f",
            len(completion_messages),
            reward,
        )
        
        return reward
