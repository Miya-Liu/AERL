# Token-Level Reward Extension for AReaL

This module provides mock implementations that extend AReaL's agent workflow to support **token-level rewards** instead of just scalar rewards per interaction.

## Overview

### Current AReaL Behavior

In the standard AReaL implementation:

1. `OpenAIProxyWorkflow.arun_episode()` returns `dict[str, InteractionWithTokenLogpReward]`
2. Each `InteractionWithTokenLogpReward` has a scalar `reward: float` field
3. When `to_tensor_dict()` is called, this scalar reward is broadcast to all tokens via `rewards=torch.tensor([float(reward)])`
4. The `loss_mask` determines which tokens contribute to the loss (0 for input, 1 for output)

### Token-Level Reward Extension

This module adds support for per-token rewards:

1. `InteractionWithTokenLevelReward` extends `InteractionWithTokenLogpReward` with:
   - `token_rewards: list[float] | None` - One reward per output token
   - `token_reward_mask: list[int] | None` - Binary mask for sparse rewards

2. When `to_tensor_dict()` is called:
   - If `token_rewards` is set, rewards are broadcast to the full sequence
   - Input tokens get reward 0.0
   - Output tokens get their respective token rewards
   - The result has shape `[1, seq_len]` instead of `[1]` for scalar rewards

3. `OpenAIProxyWorkflow` wraps agents that return:
   - `float` - Scalar reward (backward compatible)
   - `dict[str, float]` - Completion ID → scalar reward
   - `dict[str, list[float]]` - Completion ID → token-level rewards

## Data Structures

### InteractionWithTokenLevelReward

Extended interaction class that supports token-level rewards.

**Fields**:
- `token_rewards: list[float] | None` - Per-token reward values (one per output token)
- `token_reward_mask: list[int] | None` - Binary mask for sparse rewards (1 = has reward, 0 = no reward)

**Methods**:
- `set_token_rewards(rewards: list[float])` - Set per-token rewards for this interaction
- `set_sparse_token_rewards(token_indices: list[int], rewards: list[float], default_reward: float = 0.0)` - Set rewards for specific tokens only
- `to_tensor_dict() -> dict[str, torch.Tensor]` - Convert to tensor dictionary with token-level reward support
- `get_reward_stats() -> dict[str, Any]` - Get statistics about rewards (mean, max, min, sum, sparsity)
- `get_output_logprobs() -> list[float] | None` - Get output token log probabilities
- `compute_entropy_from_logprobs() -> list[float] | None` - Compute approximate entropy from output logprobs
- `get_token_level_logp_stats() -> dict[str, Any] | None` - Get statistics about token-level log probabilities
- `save_logp_and_entropy() -> dict[str, Any]` - Save logp and compute entropy metrics

### ModelResponse (Token Data)

```python
@dataclass
class ModelResponse:
    input_tokens: list[int]        # Input token IDs
    output_tokens: list[int]       # Output token IDs
    output_logprobs: list[float]   # Log probs for output tokens
    output_versions: list[int]     # Weight versions
```

### Tensor Dict Output

```python
{
    "input_ids": torch.tensor([...], dtype=torch.int32),      # [1, seq_len]
    "loss_mask": torch.tensor([...], dtype=torch.int32),      # [1, seq_len]
    "logprobs": torch.tensor([...], dtype=torch.float32),     # [1, seq_len]
    "versions": torch.tensor([...], dtype=torch.int32),       # [1, seq_len]
    "attention_mask": torch.tensor([...], dtype=torch.bool),  # [1, seq_len]
    "rewards": torch.tensor([...], dtype=torch.float32),      # [1, seq_len] - token-level!
    "token_reward_mask": torch.tensor([...], dtype=torch.int32),  # [1, seq_len]
}
```

## Workflow Classes

### OpenAIProxyWorkflow

Workflow that supports token-level rewards for agent training.

**Parameters**:
- `agent: Any` - Agent object with async run() method
- `proxy_addr: str` - Address of the OpenAI proxy server
- `admin_api_key: str` - Admin API key for proxy server
- `discount: float` - Discount factor for reward backpropagation
- `export_style: str` - Export style ("individual" or "concat")

**Methods**:
- `arun_episode(engine, data) -> TokenRewardInteractions | None` - Run a single episode with token-level reward support

### TokenRewardExampleAgent

Example agent demonstrating token-level reward computation.

## Client Classes

### OpenAIProxyClient

Client session for interacting with the OpenAI proxy server.

**Methods**:
- `async set_reward(completion_id: str, reward: float)` - Set scalar reward
- `async set_rewards(completion_id: str, token_rewards: list[float])` - Set token-wise rewards
- `async set_position_rewards(completion_id: str, position_rewards: list[PositionRewardInfo])` - Set position-wise candidate rewards
- `async set_last_reward(reward: float)` - Set scalar reward for most recent completion
- `async set_last_rewards(token_rewards: list[float])` - Set token-wise rewards for most recent completion
- `async set_last_position_rewards(position_rewards: list[PositionRewardInfo])` - Set position-wise rewards for most recent completion
- `async export_interactions(discount: float = 1.0, style: str = "individual") -> dict[str, Any]` - Export interactions
- `async compute_entropy(completion_id: str) -> list[float]` - Compute entropy for each position
- `async get_entropies(completion_id: str) -> list[float] | None` - Get computed entropy values
- `get_cache() -> InteractionCache` - Get the underlying cache

### InteractionCache

Cache that supports storing token-wise rewards per completion.

**Methods**:
- `set_rewards(completion_id: str, token_rewards: list[float])` - Set token-wise rewards
- `set_reward(completion_id: str, reward: float)` - Set scalar reward
- `set_last_reward(reward: float)` - Set scalar reward for most recent completion
- `set_last_rewards(token_rewards: list[float])` - Set token-wise rewards for most recent completion
- `set_position_rewards(completion_id: str, position_rewards: list[PositionRewardInfo])` - Set candidate-wise rewards for each position
- `get_token_rewards(completion_id: str) -> list[float] | None` - Get token-wise rewards
- `get_position_rewards(completion_id: str) -> list[PositionRewardInfo] | None` - Get position-wise candidate rewards
- `compute_and_store_entropy(completion_id: str) -> list[float]` - Compute entropy for each position
- `get_entropies(completion_id: str) -> list[float] | None` - Get computed entropy values
- `get_reward_stats(completion_id: str) -> dict[str, Any]` - Get reward statistics
- `apply_reward_discount(turn_discount: float = 1.0) -> dict[str, InteractionWithTokenLevelReward]` - Apply backward discounted rewards
- `export_interactions(style: str = "individual", reward_discount: float | None = None) -> dict[str, InteractionWithTokenLevelReward]` - Export cached completions
- `export_with_token_rewards() -> dict[str, dict[str, Any]]` - Export all interactions with token-wise rewards

### PositionRewardInfo

Reward information for a single generation position.

**Fields**:
- `position: int` - The position index in the completion (0-indexed)
- `candidates: list[str]` - List of candidate token strings considered at this position
- `logprobs: list[float] | None` - Log probabilities for each candidate from the model
- `rewards: list[float]` - Reward for each candidate
- `chosen_index: int` - Index of the actually chosen token

**Properties**:
- `chosen_token: str | None` - Get the chosen token string
- `chosen_reward: float | None` - Get the reward for the chosen token
- `chosen_logprob: float | None` - Get the log probability for the chosen token

## Utility Functions

### compute_token_level_rewards()

Compute token-level rewards based on content analysis.

**Parameters**:
- `tokens: list[str]` - List of token strings
- `answer: str` - The generated answer
- `tokenizer: Callable | None` - Optional tokenizer function
- `number_reward: float` - Reward value for tokens containing numbers
- `answer_reward: float` - Reward value for tokens matching the correct answer
- `correct_answer: str | None` - The ground truth correct answer

**Returns**: `list[float]` - Per-token reward values

### compute_sparse_rewards()

Compute sparse token-level rewards for specific target tokens.

**Parameters**:
- `tokens: list[str]` - List of token strings
- `target_token: str` - The target token to reward
- `reward_value: float` - Reward value for matching tokens
- `match_mode: str` - How to match: "exact", "contains", or "regex"

**Returns**: `tuple[list[float], list[int]]` - (token_rewards, token_mask)

### apply_token_reward_mask()

Apply a mask to token rewards.

**Parameters**:
- `token_rewards: list[float]` - Original token rewards
- `mask: list[int]` - Binary mask (1 = keep reward, 0 = use fill_value)
- `fill_value: float` - Value to use for masked positions

**Returns**: `list[float]` - Masked token rewards

### discount_token_rewards()

Apply temporal discounting to token rewards.

**Parameters**:
- `token_rewards: list[float]` - Original token rewards
- `discount_factor: float` - Discount factor (gamma) for TD learning
- `direction: str` - "backward" (future rewards affect past) or "forward" (past affects future)

**Returns**: `list[float]` - Discounted token rewards

### normalize_token_rewards()

Normalize token rewards to a standard range.

**Parameters**:
- `token_rewards: list[float]` - Original token rewards
- `method: str` - Normalization method: "minmax", "zscore", or "softmax"

**Returns**: `list[float]` - Normalized token rewards

### create_reasoning_rewards()

Create rewards for chain-of-thought reasoning steps.

**Parameters**:
- `tokens: list[str]` - List of token strings
- `reasoning_steps: list[str]` - List of reasoning step keywords
- `step_reward: float` - Reward for tokens indicating reasoning steps
- `final_answer_reward: float` - Reward for tokens indicating final answer

**Returns**: `list[float]` - Per-token rewards for reasoning

### aggregate_interaction_rewards()

Aggregate token-level rewards across multiple interactions.

**Parameters**:
- `interactions: dict[str, InteractionWithTokenLevelReward]` - Dictionary of interactions
- `aggregation: str` - Aggregation method: "mean", "sum", "max", or "last"

**Returns**: `dict[str, float]` - Aggregated reward per interaction

## Usage Examples

### 1. Basic Token-Level Rewards

```python
from customized_areal.token_reward import (
    OpenAIProxyWorkflow,
    InteractionWithTokenLevelReward,
)

class MyAgent:
    async def run(self, data, **extra_kwargs):
        client = AsyncOpenAI(...)
        response = await client.chat.completions.create(...)
        
        # Compute per-token rewards
        tokens = tokenize(response.choices[0].message.content)
        token_rewards = []
        for token in tokens:
            reward = 0.0
            if has_number(token):
                reward += 0.5
            token_rewards.append(reward)
        
        # Return token-level rewards
        return {response.id: token_rewards}

# Use with trainer
workflow = OpenAIProxyWorkflow(
    agent=MyAgent(),
    proxy_addr="http://localhost:8000",
)
```

### 2. Using Utility Functions

```python
from customized_areal.token_reward import compute_token_level_rewards

tokens = ["The", "answer", "is", "42"]
rewards = compute_token_level_rewards(
    tokens=tokens,
    answer="The answer is 42",
    correct_answer="42",
    number_reward=0.5,
    answer_reward=1.0,
)
# Result: [0.1, 0.2, 0.3, 1.5]
```

### 3. Sparse Token Rewards

```python
from customized_areal.token_reward import compute_sparse_rewards

# Only reward tokens matching "42"
rewards, mask = compute_sparse_rewards(
    tokens=["The", "answer", "is", "42"],
    target_token="42",
    reward_value=1.0,
    match_mode="exact",
)
# Result: rewards=[0, 0, 0, 1], mask=[0, 0, 0, 1]
```

### 4. Temporal Discounting

```python
from customized_areal.token_reward import discount_token_rewards

# Apply backward discounting
rewards = [0.1, 0.2, 0.3, 1.0]
discounted = discount_token_rewards(rewards, discount_factor=0.9)
# Result: future rewards propagate backward
```

### 5. Using Interaction Methods

```python
from customized_areal.token_reward import InteractionWithTokenLevelReward

interaction = InteractionWithTokenLevelReward(
    model_response=model_resp,
    messages=messages,
    completion=completion,
)

# Set token rewards
interaction.set_token_rewards([0.0, 0.0, 0.5, 1.0, 1.0])

# Set sparse token rewards
interaction.set_sparse_token_rewards(
    token_indices=[2, 3, 4],
    rewards=[0.5, 1.0, 1.0],
    default_reward=0.0
)

# Get reward statistics
stats = interaction.get_reward_stats()
print(f"Mean reward: {stats['mean']}")
print(f"Sparsity: {stats['sparsity']}")

# Save logp and entropy
logp_data = interaction.save_logp_and_entropy()
print(f"Logprobs: {logp_data['logprobs']}")
print(f"Entropy: {logp_data['entropy']}")
```

### 6. Using Client Session

```python
from customized_areal.token_reward import OpenAIProxyClient

client = OpenAIProxyClient()
await client.__aenter__()

# Set token-wise rewards
await client.set_rewards("comp-1", [0.5, 0.3, 0.2])

# Set position-wise rewards
from customized_areal.token_reward.cache import PositionRewardInfo
pos_rewards = [
    PositionRewardInfo(0, ["a1", "a2"], [0.1, 0.5], chosen_index=1),
    PositionRewardInfo(1, ["b1", "b2"], [0.2, 0.3], chosen_index=0),
]
await client.set_position_rewards("comp-1", pos_rewards)

# Compute entropy
entropies = await client.compute_entropy("comp-1")

# Export interactions
interactions = await client.export_interactions()
```

### 7. Using InteractionCache

```python
from customized_areal.token_reward.cache import InteractionCache, PositionRewardInfo

cache = InteractionCache()

# Set token-wise rewards
cache.set_rewards("comp-1", [0.5, 0.3, 0.2])

# Set position-wise rewards
pos_rewards = [
    PositionRewardInfo(0, ["a1", "a2"], [0.1, 0.5], chosen_index=1),
    PositionRewardInfo(1, ["b1", "b2"], [0.2, 0.3], chosen_index=0),
]
cache.set_position_rewards("comp-1", pos_rewards)

# Get reward statistics
stats = cache.get_reward_stats("comp-1")
print(f"Token rewards: {stats['token_rewards']}")
print(f"Mean: {stats['mean']}")

# Compute entropy
entropies = cache.compute_and_store_entropy("comp-1")

# Export with token rewards
result = cache.export_with_token_rewards()
```

## Integration with AReaL Training

To use token-level rewards in AReaL training:

```python
from areal import PPOTrainer
from customized_areal.token_reward import OpenAIProxyWorkflow

# Create trainer
trainer = PPOTrainer(config, train_dataset=train_dataset)

# Pass agent - will be automatically wrapped
with trainer:
    trainer.train(
        workflow=MyTokenRewardAgent()
    )
```

The `OpenAIProxyWorkflow` will:
1. Detect that the agent returns token-level rewards
2. Store them in `InteractionWithTokenLevelReward` objects
3. Export them with proper tensor shapes for training

## Files

- `types.py` - Extended `InteractionWithTokenLevelReward` class
- `workflow.py` - `OpenAIProxyWorkflow` and `TokenRewardExampleAgent`
- `reward_utils.py` - Helper functions for reward computation
- `client_session.py` - `OpenAIProxyClient` for proxy server interaction
- `cache.py` - `InteractionCache` and `PositionRewardInfo` for storing rewards
- `example_usage.py` - Example agents and usage patterns

## Key Differences from Standard AReaL

| Aspect | Standard AReaL | Token-Level Extension |
|--------|---------------|----------------------|
| Reward type | `float` (scalar) | `list[float]` (per-token) |
| Tensor shape | `rewards: [1]` | `rewards: [1, seq_len]` |
| Reward mask | N/A | `token_reward_mask: [1, seq_len]` |
| Sparse rewards | Not supported | Supported via mask |
| Position-wise rewards | Not supported | Supported via `PositionRewardInfo` |
| Entropy computation | N/A | Supported via `compute_entropy()` |
| Backward compat | N/A | Yes (falls back to scalar) |

## Notes

- This is a **mock implementation** for demonstration purposes
- Actual integration requires corresponding changes in the training loop
- The `token_reward_mask` can be used to implement sparse rewards
- `PositionRewardInfo` enables candidate-wise reward tracking at each generation position
- All utility functions in `reward_utils.py` are pure functions for easy testing
- Entropy computation can be performed on position-wise rewards using stored logprobs
