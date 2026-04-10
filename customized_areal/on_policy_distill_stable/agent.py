"""
On-Policy Distillation Agent for AReaL integration.

This module provides a class-based agent interface consistent with AReaL's
agentic RL training pattern, using the run_backend function from tpfc.
"""

import os
from typing import Any

import httpx

from areal.api import AsyncRewardWrapper
from areal.utils import logging

from customized_areal.tpfc.backend_run import run_backend
from .cache import PositionRewardInfo

logger = logging.getLogger("OnPolicyDistillAgent")


def accuracy_reward(completion: str | list[dict], ground_truth: str) -> float:
    """Simple accuracy reward function.

    Args:
        completion: The model's completion (string or message list).
        ground_truth: The expected ground truth answer.

    Returns:
        Float reward between 0.0 and 1.0.
    """
    if isinstance(completion, list):
        texts = []
        for msg in completion:
            if isinstance(msg, dict) and "content" in msg:
                texts.append(msg["content"])
        completion_text = " ".join(texts)
    else:
        completion_text = str(completion)

    completion_norm = completion_text.strip().lower()
    ground_truth_norm = str(ground_truth).strip().lower()

    if completion_norm == ground_truth_norm:
        return 1.0

    if ground_truth_norm in completion_norm or completion_norm in ground_truth_norm:
        return 0.5

    return 0.0


def on_policy_distill_reward_fn(
    completions: list[dict[str, Any]], gt: str = ""
) -> float:
    """Reward function for on-policy distillation.

    Args:
        completions: List of completion dictionaries.
        gt: Ground truth answer.

    Returns:
        Float reward between 0.0 and 1.0.
    """
    if not completions:
        return 0.0

    last_completion = completions[-1]
    completion_text = last_completion.get("content", "")

    return accuracy_reward(completion_text, gt)


class OnPolicyDistillAgent:
    """On-Policy Distillation Agent for AReaL integration.

    This class wraps the run_backend function in a class-based structure
    compatible with AReaL's RolloutWorkflow and PPOTrainer.

    Attributes:
        default_agent_id: Default agent ID for agent runs.
    """

    default_agent_id: str = "8bba75cb-0d87-4efe-b566-87de77335b76"

    def __init__(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        """Initialize OnPolicyDistillAgent.

        Args:
            agent_id: Optional agent ID to use. Falls back to default_agent_id.
            user_id: Optional user ID for authentication.
            model_name: Optional model name for LLM calls.
            **kwargs: Additional configuration options (ignored but accepted).
        """
        self.agent_id = agent_id or self.default_agent_id
        self.user_id = user_id
        self.model_name = model_name

        logger.info(
            "OnPolicyDistillAgent initialized: agent_id=%s, model_name=%s",
            self.agent_id,
            self.model_name,
        )

    async def run(
        self,
        data: dict[str, Any],
        **extra_kwargs,
    ) -> float:
        """Execute a single agent run and return the reward.

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
        """
        # Extract task description from messages
        messages = data.get("messages", [])
        if messages:
            task_description = messages[-1].get("content", "")
        else:
            task_description = data.get("prompt", "")

        # Extract ground truth for for reward calculation
        gt = data.get("answer", "")

        # Get OpenAI proxy parameters (passed by OpenAIProxyWorkflow._run_agent)
        base_url = extra_kwargs.get("base_url")
        http_client = extra_kwargs.get("http_client")
        api_key = extra_kwargs.get("api_key")

        logger.info(
            "OnPolicyDistillAgent starting run: task=%s, agent_id=%s, has_ground_truth=%s, base_url=%s",
            task_description[:100] if task_description else None,
            self.agent_id,
            bool(gt),
            base_url,
        )

        try:
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
                # http_client=http_client,
                api_key=api_key,
                rebuild_llm_client=True,
            )

            # Extract token_rewards from message metadata
            position_rewards: list[PositionRewardInfo] = []
            completion_id: str | None = None
            if completion_messages:
                for msg in completion_messages:
                    if isinstance(msg, dict) and "metadata" in msg:
                        metadata = msg["metadata"]
                        if "token_rewards" in metadata:
                            position_rewards = self._convert_to_position_rewards(
                                metadata["token_rewards"]
                            )
                            completion_id = msg.get("id") or str(hash(str(metadata)))
                            logger.info(
                                "Extracted position rewards for completion %s: %d positions",
                                completion_id,
                                len(position_rewards),
                            )
                            break

            # Calculate reward using reward function
            reward_fn = AsyncRewardWrapper(on_policy_distill_reward_fn)
            reward = await reward_fn(completions=completion_messages, gt=gt)

            logger.info(
                "OnPolicyDistillAgent run completed: message_count=%d, reward=%.4f",
                len(completion_messages),
                reward,
            )

            if completion_id is not None and position_rewards:
                return {
                    completion_id: {
                        "position_rewards": position_rewards,
                        "scalar_reward": reward,
                    }
                }
            return reward

        except Exception as e:
            logger.error(f"OnPolicyDistillAgent run failed: {e}")
            return 0.0

    def _convert_to_position_rewards(
        self, token_rewards: list[dict[str, Any]]
    ) -> list[PositionRewardInfo]:
        """Convert token_rewards from message metadata to PositionRewardInfo format.

        The token_rewards comes from manager_idm.py:_compute_token_rewards which computes:
        reward = student_top_logp - teacher_logp for each token at each position.

        Args:
            token_rewards: List from metadata with structure:
                [{
                    "step": int,
                    "token_id": int,
                    "token": str,
                    "student_logp": float,
                    "teacher_logp": float,
                    "top_k_rewards": [{
                        "token_id": int,
                        "student_logp": float,
                        "teacher_logp": float,
                        "reward": float,  # student_logp - teacher_logp
                    }, ...]
                }, ...]

        Returns:
            List of PositionRewardInfo objects for set_position_rewards
        """
        position_rewards = []

        for token_reward in token_rewards:
            step = token_reward["step"]
            token = token_reward["token"]
            top_k_rewards = token_reward.get("top_k_rewards", [])

            if not top_k_rewards:
                continue

            candidates = [str(tkr.get("token_id", "")) for tkr in top_k_rewards]
            candidate_token_ids = [tkr.get("token_id", 0) for tkr in top_k_rewards]

            logprobs = [tkr.get("student_logp", 0.0) for tkr in top_k_rewards]

            rewards = [tkr.get("reward", 0.0) for tkr in top_k_rewards]

            actual_token_id = token_reward["token_id"]
            chosen_index = 0
            for idx, tkr in enumerate(top_k_rewards):
                if tkr.get("token_id") == actual_token_id:
                    chosen_index = idx
                    break

            position_rewards.append(
                PositionRewardInfo(
                    position=step,
                    candidates=candidates,
                    candidate_token_ids=candidate_token_ids,
                    logprobs=logprobs,
                    rewards=rewards,
                    chosen_index=chosen_index,
                )
            )

        return position_rewards
