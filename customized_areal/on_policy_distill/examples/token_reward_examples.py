"""
Example usage of token-level reward workflow.

This script demonstrates how to use the OpenAIProxyWorkflow
with a custom agent that computes token-level rewards.

Usage:
    python example_usage.py --config config.yaml
"""

import os


# Example agent that computes token-level rewards
class MathAgentWithTokenRewards:
    """
    Example math agent that returns token-level rewards.

    This agent solves math problems and computes per-token rewards
    based on:
    1. Whether the token contains a number
    2. Whether the token is part of the correct answer
    3. Position in the sequence
    """

    async def run(self, data: dict, **extra_kwargs):
        """
        Run the agent on a math problem.

        Parameters
        ----------
        data : dict
            Contains "messages" (conversation) and "answer" (ground truth).
        extra_kwargs : dict
            Contains base_url, api_key, http_client from AReaL.

        Returns
        -------
        dict[str, list[float]]
            Mapping from completion ID to per-token rewards.
        """
        from openai import AsyncOpenAI

        # Get connection parameters from AReaL
        base_url = extra_kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        http_client = extra_kwargs.get("http_client")

        # Create OpenAI client pointing to AReaL proxy
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        # Get the math problem
        messages = data.get("messages", [])
        problem = messages[-1]["content"] if messages else ""
        correct_answer = data.get("answer", "")

        # Generate solution
        response = await client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0.7,
        )

        completion_id = response.id
        content = response.choices[0].message.content

        # Tokenize the response (simplified - in practice use actual tokenizer)
        # In real usage, you would use the tokenizer to get actual tokens
        tokens = self._simple_tokenize(content)

        # Compute token-level rewards
        token_rewards = self._compute_token_rewards(
            tokens=tokens,
            content=content,
            correct_answer=correct_answer,
        )

        # Return token-level rewards
        return {completion_id: token_rewards}

    def _simple_tokenize(self, text: str) -> list[str]:
        """Simple tokenization by splitting on whitespace and punctuation."""
        import re

        # Split on whitespace and keep punctuation as separate tokens
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def _compute_token_rewards(
        self, tokens: list[str], content: str, correct_answer: str
    ) -> list[float]:
        """
        Compute per-token rewards.

        Reward strategy:
        - 0.5 for tokens containing numbers
        - 1.0 for tokens matching the correct answer
        - 0.1 * position_weight for later tokens (progressive)
        """
        from math_verify import parse, verify

        rewards = []
        num_tokens = len(tokens)

        # Check if final answer is correct
        is_correct = verify(parse(content), parse(correct_answer))
        base_reward = 1.0 if is_correct else 0.0

        for i, token in enumerate(tokens):
            reward = 0.0

            # 1. Reward tokens containing numbers
            if any(c.isdigit() for c in token):
                reward += 0.3

            # 2. Reward tokens that might be part of correct answer
            token_normalized = token.strip().lower()
            answer_normalized = str(correct_answer).strip().lower()
            if (
                token_normalized in answer_normalized
                or answer_normalized in token_normalized
            ):
                reward += 0.5

            # 3. Position-based reward (later tokens more important)
            position_weight = (i + 1) / num_tokens
            reward += position_weight * 0.2

            # 4. Final correctness reward applied to all tokens
            reward += base_reward * 0.5

            rewards.append(reward)

        return rewards


class SparseRewardAgent:
    """
    Example agent that uses sparse token-level rewards.

    Only rewards specific tokens (e.g., the final numerical answer).
    """

    async def run(self, data: dict, **extra_kwargs):
        """Run agent with sparse rewards."""
        import re

        from openai import AsyncOpenAI

        base_url = extra_kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        http_client = extra_kwargs.get("http_client")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        messages = data.get("messages", [])
        response = await client.chat.completions.create(
            model="default",
            messages=messages,
        )

        completion_id = response.id
        content = response.choices[0].message.content
        tokens = content.split()

        # Find the last number in the response (likely the answer)
        token_rewards = [0.0] * len(tokens)
        token_mask = [0] * len(tokens)

        for i, token in enumerate(tokens):
            if re.match(r"^\d+\.?\d*$", token):  # Numeric token
                # Reward this token as the answer
                token_rewards[i] = 1.0
                token_mask[i] = 1

        # Return token-level rewards directly.
        return {completion_id: token_rewards}


def main(args):
    """
    Main entry point for training with token-level rewards.

    This demonstrates how to integrate OpenAIProxyWorkflow with AReaL.
    """
    # Import our custom workflow

    from areal import PPOTrainer
    from areal.api.cli_args import PPOConfig, load_expr_config
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer

    # Load config
    config, _ = load_expr_config(args, PPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load dataset
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    # Create trainer with token-level reward workflow
    with PPOTrainer(config, train_dataset=train_dataset) as trainer:
        # The trainer will automatically wrap our agent with OpenAIProxyWorkflow
        # because the agent has a run() method
        trainer.train(workflow=MathAgentWithTokenRewards())

    # Note: In practice, you would also need to configure the proxy server
    # and ensure the training loop handles token-level rewards properly.
    # This example shows the high-level integration pattern.


def demo_token_rewards():
    """
    Demonstration of token-level reward computation without actual training.

    This shows how the data structures work.
    """
    from customized_areal.token_reward.interaction_types import (
        InteractionWithTokenLevelReward,
    )
    from customized_areal.token_reward.reward_utils import (
        compute_token_level_rewards,
        normalize_token_rewards,
    )

    from areal.api.io_struct import ModelResponse

    print("=" * 60)
    print("Token-Level Reward Demonstration")
    print("=" * 60)

    # Mock data
    tokens = ["The", "answer", "is", "42", "."]
    correct_answer = "42"

    # Compute token-level rewards
    rewards = compute_token_level_rewards(
        tokens=tokens,
        answer="The answer is 42.",
        correct_answer=correct_answer,
    )
    print(f"\nTokens: {tokens}")
    print(f"Token Rewards: {[f'{r:.2f}' for r in rewards]}")

    # Normalize
    normalized = normalize_token_rewards(rewards, method="minmax")
    print(f"Normalized: {[f'{r:.2f}' for r in normalized]}")

    # Create mock ModelResponse
    model_response = ModelResponse(
        input_tokens=[1, 2, 3],  # Mock input tokens
        output_tokens=[10, 11, 12, 13, 14],  # Corresponding to 5 tokens
        output_logprobs=[-0.5, -0.3, -0.2, -0.1, -0.4],
    )

    # Create interaction with token-level rewards
    interaction = InteractionWithTokenLevelReward(
        model_response=model_response,
        reward=sum(rewards),  # Scalar sum for backward compatibility
        messages=[{"role": "user", "content": "What is 20 + 22?"}],
        token_rewards=rewards,
    )

    print("\nInteraction created successfully!")
    print(f"Scalar reward: {interaction.reward}")
    print(f"Token rewards: {interaction.token_rewards}")

    # Convert to tensor dict
    tensor_dict = interaction.to_tensor_dict()
    print(f"\nTensor dict keys: {list(tensor_dict.keys())}")
    print(f"Input IDs shape: {tensor_dict['input_ids'].shape}")
    print(f"Rewards shape: {tensor_dict['rewards'].shape}")
    print(f"Rewards values: {tensor_dict['rewards']}")

    # Show reward stats
    stats = interaction.get_reward_stats()
    print("\nReward Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


class TokenRewardExampleAgent:
    """
    Example agent demonstrating token-level reward computation.

    This is a mock agent showing how to compute and return token-level rewards.
    """

    async def run(self, data: dict, **extra_kwargs) -> dict[str, list[float]]:
        """
        Run the agent and return token-level rewards.

        Returns
        -------
        dict[str, list[float]]
            Mapping from completion ID to per-token reward list.
        """
        from openai import AsyncOpenAI

        base_url = extra_kwargs.get("base_url")
        api_key = extra_kwargs.get("api_key")
        http_client = extra_kwargs.get("http_client")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        # Get the math problem
        messages = data.get("messages", [])

        # Generate response
        response = await client.chat.completions.create(
            model="default",
            messages=messages,
        )

        completion_id = response.id
        content = response.choices[0].message.content or ""

        tokens = content.split()
        num_tokens = len(tokens)

        # Compute token-level rewards
        # Example: Reward tokens that contain numbers more heavily
        token_rewards = []
        for i, token in enumerate(tokens):
            reward = 0.0

            # Reward number tokens
            if any(c.isdigit() for c in token):
                reward += 0.5

            # Reward the final answer (last few tokens)
            if i >= num_tokens - 3:
                reward += 0.5

            token_rewards.append(reward)

        # Return token-level rewards for this completion
        return {completion_id: token_rewards}


if __name__ == "__main__":
    # Run demonstration
    demo_token_rewards()

    # Uncomment to run actual training:
    # main(sys.argv[1:])
