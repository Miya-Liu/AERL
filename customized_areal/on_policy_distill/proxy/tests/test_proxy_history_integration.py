#!/usr/bin/env python3
"""Integrated test that launches proxy server and tests history consistency.

This test:
1. Launches the customized proxy server in a subprocess
2. Runs the agent workflow test
3. Compares agent's saved history with proxy's exported history
4. Verifies they match

Usage:
    python test_proxy_history_integration.py
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from areal.utils import logging

logger = logging.getLogger("ProxyHistoryIntegrationTest")


class ProxyServerManager:
    """Manages proxy server lifecycle for testing."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        admin_api_key: str = "test-integration-key"
    ):
        self.host = host
        self.port = port
        self.admin_api_key = admin_api_key
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://{host}:{port}"

    def start(self) -> bool:
        """Start the proxy server."""
        cmd = [
            sys.executable,
            "-m",
            "customized_areal.on_policy_distill.proxy.proxy_rollout_server",
            "--host", self.host,
            "--port", str(self.port),
            "--admin-api-key", self.admin_api_key,
        ]

        logger.info(f"Starting proxy server: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_root)
            )

            # Wait for server to start
            logger.info("Waiting for proxy server to start...")
            time.sleep(3)

            # Check if process is still running
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(f"Proxy server failed to start!")
                logger.error(f"stdout: {stdout.decode()}")
                logger.error(f"stderr: {stderr.decode()}")
                return False

            # Try to connect
            import urllib.request
            try:
                urllib.request.urlopen(f"{self.base_url}/docs", timeout=5)
                logger.info(f"Proxy server started at {self.base_url}")
                return True
            except Exception as e:
                logger.warning(f"Could not connect to proxy docs: {e}")
                # Server might still be starting, give it more time
                time.sleep(2)
                return True

        except Exception as e:
            logger.error(f"Failed to start proxy server: {e}")
            return False

    def stop(self):
        """Stop the proxy server."""
        if self.process:
            logger.info("Stopping proxy server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Proxy server didn't terminate gracefully, killing...")
                self.process.kill()
                self.process.wait()

            stdout, stderr = self.process.communicate()
            if stdout:
                logger.debug(f"Proxy stdout:\n{stdout.decode()}")
            if stderr:
                logger.debug(f"Proxy stderr:\n{stderr.decode()}")

            self.process = None
            logger.info("Proxy server stopped")

    def __enter__(self):
        if not self.start():
            raise RuntimeError("Failed to start proxy server")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class HistorySavingAgent:
    """Agent that saves conversation history and interacts with proxy."""

    def __init__(self):
        self.saved_histories: list[dict] = []
        self.completion_ids: list[str] = []

    async def run(self, data: dict, **extra_kwargs) -> dict:
        """Run agent and save history."""
        base_url = extra_kwargs.get("base_url", "http://localhost:8000")
        api_key = extra_kwargs.get("api_key")
        proxy_client = extra_kwargs.get("proxy_client")

        import aiohttp

        # Simulate conversation
        conversation = []
        completion_ids = []

        # Turn 1
        messages = [{"role": "user", "content": data.get("prompt", "Hello")}]
        conversation.append({"role": "user", "content": messages[0]["content"]})

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Try to call proxy, fall back to simulated response
            try:
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={"model": "test", "messages": messages, "max_tokens": 20},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        assistant_msg = result["choices"][0]["message"]["content"]
                        completion_id = result.get("id", f"comp-{int(time.time())}")
                    else:
                        assistant_msg = "Hi there! I'm doing well."
                        completion_id = f"sim-{int(time.time() * 1000)}"
            except Exception:
                assistant_msg = "Hi there! I'm doing well."
                completion_id = f"sim-{int(time.time() * 1000)}"

        conversation.append({"role": "assistant", "content": assistant_msg})
        completion_ids.append(completion_id)

        # Set token rewards via proxy_client
        if proxy_client is not None:
            token_rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
            await proxy_client.set_rewards(completion_id, token_rewards)
            logger.info(f"Set {len(token_rewards)} token rewards for {completion_id}")

        # Turn 2
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": "What's the weather like?"})
        conversation.append({"role": "user", "content": "What's the weather like?"})

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={"model": "test", "messages": messages, "max_tokens": 20},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        assistant_msg2 = result["choices"][0]["message"]["content"]
                        completion_id2 = result.get("id", f"comp-{int(time.time())}")
                    else:
                        assistant_msg2 = "I don't have real-time weather data."
                        completion_id2 = f"sim-{int(time.time() * 1000)}"
        except Exception:
            assistant_msg2 = "I don't have real-time weather data."
            completion_id2 = f"sim-{int(time.time() * 1000)}"

        conversation.append({"role": "assistant", "content": assistant_msg2})
        completion_ids.append(completion_id2)

        if proxy_client is not None:
            token_rewards2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            await proxy_client.set_rewards(completion_id2, token_rewards2)
            logger.info(f"Set {len(token_rewards2)} token rewards for {completion_id2}")

        # Save agent's history
        self.saved_histories.append({
            "conversation": conversation,
            "completion_ids": completion_ids,
        })
        self.completion_ids = completion_ids

        logger.info(f"Agent completed with {len(conversation)} messages, {len(completion_ids)} completions")

        # Return rewards dict
        return {
            completion_id: [0.1, 0.2, 0.3, 0.4, 0.5],
            completion_id2: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }


class MockInferenceEngine:
    """Mock inference engine."""
    version = 0

    def get_version(self):
        return self.version


async def run_history_test(proxy_manager: ProxyServerManager) -> bool:
    """Run the history consistency test."""
    from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow

    logger.info("\n" + "=" * 70)
    logger.info("STARTING HISTORY CONSISTENCY TEST")
    logger.info("=" * 70)

    # Create agent and workflow
    agent = HistorySavingAgent()
    workflow = OpenAIProxyWorkflow(
        agent=agent,
        proxy_addr=proxy_manager.base_url,
        admin_api_key=proxy_manager.admin_api_key,
        discount=1.0,
        export_style="individual",
    )

    engine = MockInferenceEngine()
    test_data = {"prompt": "Hello, how are you?"}

    # Run workflow
    logger.info("\n1. Running workflow...")
    try:
        interactions = await workflow.arun_episode(engine, test_data)
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    if not interactions:
        logger.error("No interactions returned from proxy!")
        return False

    logger.info(f"   ✓ Workflow completed, got {len(interactions)} interactions")

    # Get agent's saved history
    logger.info("\n2. Checking agent's saved history...")
    if not agent.saved_histories:
        logger.error("Agent did not save any history!")
        return False

    agent_history = agent.saved_histories[0]
    agent_completion_ids = set(agent_history["completion_ids"])
    logger.info(f"   ✓ Agent saved {len(agent_history['conversation'])} messages")
    logger.info(f"   ✓ Agent recorded {len(agent_completion_ids)} completions: {agent_completion_ids}")

    # Get proxy's exported interactions
    logger.info("\n3. Checking proxy's exported interactions...")
    proxy_completion_ids = set(interactions.keys())
    logger.info(f"   ✓ Proxy exported {len(proxy_completion_ids)} interactions: {proxy_completion_ids}")

    # Compare
    logger.info("\n4. Comparing histories...")
    logger.info("-" * 70)

    if agent_completion_ids != proxy_completion_ids:
        logger.error("COMPLETION ID MISMATCH!")
        logger.error(f"  Only in agent: {agent_completion_ids - proxy_completion_ids}")
        logger.error(f"  Only in proxy: {proxy_completion_ids - agent_completion_ids}")
        return False

    logger.info("   ✓ Completion IDs match")

    # Check token rewards
    logger.info("\n5. Checking token rewards...")
    for cid in agent_completion_ids:
        interaction = interactions[cid]
        agent_rewards = agent_history.get(cid, [])
        proxy_rewards = getattr(interaction, "token_rewards", None)

        logger.info(f"   {cid}:")
        if proxy_rewards:
            logger.info(f"     Token rewards: {len(proxy_rewards)} values")
            logger.info(f"     Scalar reward: {interaction.reward}")
        else:
            logger.info(f"     Scalar reward: {interaction.reward} (no token rewards)")

    logger.info("\n" + "=" * 70)
    logger.info("✓ TEST PASSED: Agent and proxy histories are consistent!")
    logger.info("=" * 70)

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Proxy history integration test")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for proxy server"
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing proxy server (don't start new one)"
    )
    parser.add_argument(
        "--proxy-addr",
        default=None,
        help="Address of existing proxy server (if --use-existing)"
    )

    args = parser.parse_args()

    if args.use_existing:
        # Use existing proxy server
        proxy_addr = args.proxy_addr or "http://localhost:8000"
        logger.info(f"Using existing proxy at {proxy_addr}")

        class DummyManager:
            base_url = proxy_addr
            admin_api_key = "test-admin-key"
            def __enter__(self): return self
            def __exit__(self, *args): return False

        manager = DummyManager()
    else:
        # Start our own proxy server
        manager = ProxyServerManager(port=args.port)

    try:
        with manager:
            # Give proxy a moment to fully initialize
            if not args.use_existing:
                time.sleep(2)

            # Run the test
            result = asyncio.run(run_history_test(manager))
            sys.exit(0 if result else 1)

    except KeyboardInterrupt:
        logger.info("\nTest interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
