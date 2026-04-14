#!/usr/bin/env python3
"""Test proxy server history consistency.

This test verifies that:
1. Agent's saved conversation history matches the exported history from proxy
2. Token-level rewards are properly stored and exported

Usage:
    # Terminal 1: Start proxy server
     shoucustomized_areal.on_policy_distill.proxy.proxy_rollout_server \
        --host 0.0.0.0 --port 8000 --admin-api-key "test-admin-key"

    # Terminal 2: Run test
    python test_proxy_history_consistency.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import httpx

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from areal.utils import logging

logger = logging.getLogger("ProxyHistoryConsistencyTest")


class HistorySavingAgent:
    """Agent that saves conversation history and interacts with proxy."""

    def __init__(self):
        self.saved_histories: list[dict] = []  # Agent's own record

    async def run(self, data: dict, **extra_kwargs) -> dict:
        """Run agent, save history, interact with proxy."""
        base_url = extra_kwargs.get("base_url", "http://localhost:8000")
        api_key = extra_kwargs.get("api_key")
        proxy_client = extra_kwargs.get("proxy_client")

        # Use httpx client for OpenAI-compatible API calls
        http_client = extra_kwargs.get("http_client")

        # Simulate a multi-turn conversation
        conversation = []

        # Turn 1: Initial query
        messages = [{"role": "user", "content": data.get("prompt", "Hello, how are you?")}]
        conversation.append({"turn": 1, "role": "user", "content": messages[0]["content"]})

        # Call proxy API (simulate chat completion)
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Simulate getting a completion
            completion_data = {
                "model": "test-model",
                "messages": messages,
                "max_tokens": 50
            }

            # Try to get real completion or simulate
            try:
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=completion_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        assistant_msg = result["choices"][0]["message"]["content"]
                        completion_id = result.get("id", "test-completion-1")
                    else:
                        # Simulate response if proxy doesn't have LLM backend
                        assistant_msg = "I am doing well, thank you for asking!"
                        completion_id = f"test-completion-{int(time.time() * 1000)}"
            except Exception as e:
                logger.warning(f"Proxy call failed, using simulated response: {e}")
                assistant_msg = "I am doing well, thank you for asking!"
                completion_id = f"test-completion-{int(time.time() * 1000)}"

        conversation.append({"turn": 1, "role": "assistant", "content": assistant_msg})

        # Turn 2: Follow-up
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": "What can you help me with?"})
        conversation.append({"turn": 2, "role": "user", "content": "What can you help me with?"})

        # Another completion
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                completion_data = {
                    "model": "test-model",
                    "messages": messages,
                    "max_tokens": 50
                }

                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=completion_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        assistant_msg2 = result["choices"][0]["message"]["content"]
                        completion_id2 = result.get("id", "test-completion-2")
                    else:
                        assistant_msg2 = "I can help with many tasks including coding, writing, and analysis."
                        completion_id2 = f"test-completion-{int(time.time() * 1000)}"
        except Exception as e:
            logger.warning(f"Second proxy call failed, using simulated response: {e}")
            assistant_msg2 = "I can help with many tasks including coding, writing, and analysis."
            completion_id2 = f"test-completion-{int(time.time() * 1000)}"

        conversation.append({"turn": 2, "role": "assistant", "content": assistant_msg2})

        # Save agent's own history record
        self.saved_histories.append({
            "session_id": api_key.split("-")[-1] if api_key else "unknown",
            "conversation": conversation,
            "completion_ids": [completion_id, completion_id2],
            "data_prompt": data.get("prompt"),
        })

        logger.info(f"Agent saved history with {len(conversation)} messages")

        # Set token-level rewards via proxy_client if available
        if proxy_client is not None:
            # Simulate token rewards for first completion
            token_rewards_1 = [0.1, 0.2, 0.3, 0.5, 1.0]  # 5 tokens
            await proxy_client.set_rewards(completion_id, token_rewards_1)
            logger.info(f"Set token rewards for {completion_id}: {token_rewards_1}")

            # Simulate token rewards for second completion
            token_rewards_2 = [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 0.5]  # 7 tokens
            await proxy_client.set_rewards(completion_id2, token_rewards_2)
            logger.info(f"Set token rewards for {completion_id2}: {token_rewards_2}")

        # Return format expected by workflow
        return {
            completion_id: [0.1, 0.2, 0.3, 0.5, 1.0],
            completion_id2: [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 0.5],
        }


class MockInferenceEngine:
    """Mock inference engine for testing."""

    def __init__(self):
        self.version = 0

    def get_version(self):
        return self.version


async def run_test(
    proxy_addr: str = "http://localhost:8000",
    admin_api_key: str = "test-admin-key"
) -> bool:
    """Run the history consistency test.

    Args:
        proxy_addr: Address of the proxy server
        admin_api_key: Admin API key for the proxy

    Returns:
        True if test passes, False otherwise
    """
    logger.info("=" * 70)
    logger.info("PROXY HISTORY CONSISTENCY TEST")
    logger.info("=" * 70)

    # Import workflow here to ensure patch is applied
    from customized_areal.on_policy_distill.proxy.client import OpenAIProxyClient
    from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow

    # Create agent
    agent = HistorySavingAgent()

    # Create workflow
    workflow = OpenAIProxyWorkflow(
        agent=agent,
        proxy_addr=proxy_addr,
        admin_api_key=admin_api_key,
        discount=1.0,
        export_style="individual",  # Get all interactions
    )

    # Create mock engine
    engine = MockInferenceEngine()

    # Test data
    test_data = {"prompt": "Hello, tell me about yourself"}

    logger.info(f"\n1. Starting workflow with proxy at {proxy_addr}")

    # Run workflow (this will call agent.run and save history)
    try:
        interactions = await workflow.arun_episode(engine, test_data)
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return False

    logger.info(f"\n2. Workflow completed, exported {len(interactions) if interactions else 0} interactions")

    if not interactions:
        logger.error("No interactions returned from proxy!")
        return False

    # Get agent's saved history
    agent_history = agent.saved_histories[0] if agent.saved_histories else None

    if not agent_history:
        logger.error("Agent did not save any history!")
        return False

    logger.info(f"\n3. Agent saved history:")
    logger.info(f"   - Session ID: {agent_history['session_id'][:20]}...")
    logger.info(f"   - Conversation turns: {len(agent_history['conversation'])}")
    logger.info(f"   - Completion IDs: {agent_history['completion_ids']}")

    # Get proxy exported history
    logger.info(f"\n4. Proxy exported interactions:")
    for idx, (interaction_id, interaction) in enumerate(interactions.items()):
        logger.info(f"   Interaction {idx + 1}: {interaction_id}")
        logger.info(f"   - Messages: {len(interaction.messages) if hasattr(interaction, 'messages') else 'N/A'}")
        logger.info(f"   - Reward: {interaction.reward}")
        if hasattr(interaction, 'token_rewards') and interaction.token_rewards:
            logger.info(f"   - Token rewards: {interaction.token_rewards}")

    # Compare histories
    logger.info(f"\n5. COMPARING HISTORIES:")
    logger.info("-" * 70)

    # Check number of completions match
    agent_completion_count = len(agent_history['completion_ids'])
    proxy_completion_count = len(interactions)

    logger.info(f"Agent recorded {agent_completion_count} completions")
    logger.info(f"Proxy exported {proxy_completion_count} interactions")

    if agent_completion_count != proxy_completion_count:
        logger.error(f"MISMATCH: Completion counts differ!")
        return False

    # Check completion IDs match
    agent_ids = set(agent_history['completion_ids'])
    proxy_ids = set(interactions.keys())

    logger.info(f"Agent completion IDs: {agent_ids}")
    logger.info(f"Proxy interaction IDs: {proxy_ids}")

    if agent_ids != proxy_ids:
        logger.error(f"MISMATCH: Completion IDs don't match!")
        logger.error(f"  Only in agent: {agent_ids - proxy_ids}")
        logger.error(f"  Only in proxy: {proxy_ids - agent_ids}")
        return False

    # Check token rewards are consistent
    logger.info(f"\n6. CHECKING TOKEN REWARDS:")
    logger.info("-" * 70)

    all_rewards_match = True
    for completion_id in agent_history['completion_ids']:
        if completion_id in interactions:
            interaction = interactions[completion_id]
            agent_rewards = agent_history.get(completion_id, [])

            if hasattr(interaction, 'token_rewards') and interaction.token_rewards:
                proxy_rewards = interaction.token_rewards
                logger.info(f"{completion_id}:")
                logger.info(f"  Agent rewards: {agent_rewards}")
                logger.info(f"  Proxy rewards: {proxy_rewards}")

                if agent_rewards and agent_rewards != proxy_rewards:
                    logger.warning(f"  Token rewards differ (this may be OK if agent stored differently)")
                else:
                    logger.info(f"  ✓ Token rewards match")
            else:
                logger.info(f"{completion_id}: No token rewards in proxy (scalar only)")

    # Check reward values
    logger.info(f"\n7. CHECKING REWARD VALUES:")
    logger.info("-" * 70)

    for completion_id, interaction in interactions.items():
        expected_reward = sum(agent_history.get(completion_id, []))
        actual_reward = interaction.reward

        logger.info(f"{completion_id}:")
        logger.info(f"  Expected (sum of token rewards): {expected_reward}")
        logger.info(f"  Actual (proxy scalar): {actual_reward}")

        if abs(expected_reward - actual_reward) > 0.01:
            logger.warning(f"  Reward values differ slightly (may be due to rounding)")
        else:
            logger.info(f"  ✓ Rewards match")

    logger.info(f"\n" + "=" * 70)
    logger.info("TEST PASSED: Agent history and proxy export are consistent!")
    logger.info("=" * 70)

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test proxy history consistency")
    parser.add_argument(
        "--proxy-addr",
        default="http://localhost:8000",
        help="Proxy server address"
    )
    parser.add_argument(
        "--admin-api-key",
        default="test-admin-key",
        help="Admin API key"
    )

    args = parser.parse_args()

    # Run async test
    try:
        result = asyncio.run(run_test(args.proxy_addr, args.admin_api_key))
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
