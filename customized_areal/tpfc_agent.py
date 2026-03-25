"""
TPFC Agent wrapper for AReaL integration.

This module provides a class-based agent interface consistent with AReaL's
agentic RL training pattern, wrapping the existing run_backend functionality.
"""

import os
from typing import Any

import httpx

from areal.api import AsyncRewardWrapper
from areal.utils import logging

from customized_areal.backend_run import run_backend

logger = logging.getLogger("TPFCAgent")


def tpfc_reward_fn(completions: list[dict[str, Any]], gt: str = "") -> float:
    """
    Reward function for TPFC tasks.
    
    This is a placeholder reward function that evaluates the agent's output.
    You should customize this based on your specific task requirements.
    
    Args:
        completions: List of message dictionaries from the agent run.
        gt: Ground truth for evaluation (if available).
        
    Returns:
        A float reward value between 0.0 and 1.0.
    """
    # Simple reward: check if completion is not empty
    if not completions:
        return 0.0
    
    # Extract the last message content if available
    last_message = completions[-1] if completions else {}
    content = last_message.get("content", "")
    
    # If ground truth is provided, you can implement custom logic here
    # For now, we return 1.0 if there's content, 0.0 otherwise
    if gt and content:
        # TODO: Implement task-specific reward logic
        # Example: check if gt appears in content
        return 1.0 if gt.lower() in content.lower() else 0.0
    
    return 1.0 if content else 0.0


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
    
    default_agent_id: str = '89395eb4-dd1a-4a13-932d-4f7d3a17bca6'
    
    def __init__(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        """
        Initialize TPFC Agent.
        
        Args:
            agent_id: Optional agent ID to use. Falls back to default_agent_id.
            user_id: Optional user ID for authentication.
            model_name: Optional model name for LLM calls.
            **kwargs: Additional configuration options (ignored but accepted for compatibility).
        """
        self.agent_id = agent_id or self.default_agent_id
        self.user_id = user_id
        self.model_name = model_name
    
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
        
        # Get OpenAI proxy parameters (passed by OpenAIProxyWorkflow._run_agent)
        base_url = extra_kwargs.get("base_url")
        http_client = extra_kwargs.get("http_client")
        api_key = extra_kwargs.get("api_key")
        
        logger.info(
            "TPFCAgent starting run",
            task_description=task_description[:100] if task_description else None,
            agent_id=self.agent_id,
            has_ground_truth=bool(gt),
            base_url=base_url,
        )
        
        # Execute the backend run
        completion_messages = await run_backend(
            task_description=task_description,
            task_file_path=[],
            log_path="./log.json",
            task_id="",
            gt=gt,
            tags=[],
            user_id=self.user_id,
            model_name=self.model_name,
            agent_id=self.agent_id,
            base_url=base_url,
            http_client=http_client,
            api_key=api_key,
            rebuild_llm_client=True,
        )
        
        # Calculate reward using reward function
        reward_fn = AsyncRewardWrapper(tpfc_reward_fn)
        reward = await reward_fn(completions=completion_messages, gt=gt)
        
        logger.info(
            "TPFCAgent run completed",
            message_count=len(completion_messages),
            reward=reward,
        )
        
        return reward
