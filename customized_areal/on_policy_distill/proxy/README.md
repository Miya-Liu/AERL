# Token-Level Reward Proxy System

This directory contains a customized OpenAI-compatible proxy system for AReaL that
supports **token-level rewards** via HTTP API. Unlike the base AReaL implementation
which only supports scalar rewards, this system allows assigning different reward values
to individual tokens in the generated output.

## Overview

The token-level reward proxy system enables fine-grained RL training where each output
token can have its own reward signal. This is useful for:

- **Sparse reward assignment**: Reward only specific tokens (e.g., final answer tokens)
- **Credit assignment**: Different rewards for different parts of the response
- **KL-regularized rewards**: Position-wise candidate rewards with entropy computation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AReaL Training Loop                                  │
│  actor.prepare_batch(workflow=OpenAIProxyWorkflow)                          │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ OpenAIProxyWorkflow.arun_episode()                                 │    │
│  │  1. Start session via HTTP (POST /rl/start_session)                │    │
│  │  2. Run agent with session_api_key                                  │    │
│  │  3. Agent calls LLM via OpenAI SDK (through proxy server)          │    │
│  │  4. Agent sets token-level rewards via HTTP API                    │    │
│  │  5. End session via HTTP (POST /rl/end_session)                    │    │
│  │  6. Export interactions with token-level rewards                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Proxy Rollout Server (FastAPI)                                      │    │
│  │  - /chat/completions  - OpenAI-compatible generation                │    │
│  │  - /rl/set_token_rewards  - Set per-token rewards                   │    │
│  │  - /rl/set_position_rewards - Set position-wise candidate rewards   │    │
│  │  - /rl/compute_entropy - Compute entropy from position rewards      │    │
│  │  - /export_trajectories - Get trajectory with token-level rewards   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ SGLang/vLLM Inference Engine                                        │    │
│  │  - Generates tokens                                                 │    │
│  │  - Records logprobs                                                 │    │
│  │  - Applies token-level rewards for training                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

| File                      | Purpose                                                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `workflow.py`             | `OpenAIProxyWorkflow` - Main workflow class that orchestrates episodes with token-level reward support            |
| `client.py`               | `OpenAIProxyClient` - HTTP client with methods for setting token-level and position-level rewards                 |
| `server.py`               | Data models and session management for token-level rewards (`TokenRewardSessionData`, `PositionRewardInfo`, etc.) |
| `types.py`                | `InteractionWithTokenLevelReward` - Extended interaction type with token-level reward fields                      |
| `proxy_rollout_server.py` | Extended FastAPI server with token-level reward endpoints                                                         |

## Key Features

### 1. Token-Level Rewards

Assign a different reward value to each output token:

```python
# Agent returns token-level rewards
return {
    "completion_id": [0.0, 0.0, 0.5, 1.0, 1.0]  # Per-token rewards
}
```

### 2. Position-Level Rewards

Assign candidate-wise rewards at each generation position (useful for KL-regularized
training):

```python
position_rewards = [
    PositionRewardInfo(
        position=0,
        candidates=["The", "A", "This"],
        candidate_token_ids=[101, 102, 103],
        logprobs=[-0.5, -1.2, -2.0],  # Model logprobs
        rewards=[1.0, 0.5, 0.3],      # Assigned rewards
        chosen_index=0,               # Which token was selected
    ),
    # ... more positions
]
await client.set_position_rewards("completion_id", position_rewards)
```

### 3. HTTP-Based Communication

Unlike previous designs that used local caches, this system communicates rewards via
HTTP API:

- **Scalable**: Rewards are stored on the proxy server, not locally
- **Clean separation**: Agent and trainer communicate via well-defined HTTP endpoints
- **No cache synchronization issues**: Single source of truth on the server

## Usage

### Basic Example

```python
from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow

class MyTokenRewardAgent:
    async def run(self, data, **extra_kwargs):
        # Get proxy client for setting rewards
        proxy_client = extra_kwargs.get("proxy_client")
        api_key = extra_kwargs.get("api_key")

        # Use OpenAI SDK with the proxy
        import openai
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=extra_kwargs.get("base_url")
        )

        # Generate response
        response = await client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": data["prompt"]}]
        )

        # Extract completion ID from response
        completion_id = response.id

        # Set token-level rewards (e.g., reward last token more)
        num_tokens = len(response.choices[0].logprobs.content)
        token_rewards = [0.0] * (num_tokens - 1) + [1.0]

        await proxy_client.set_rewards(completion_id, token_rewards)

        return {completion_id: token_rewards}

# Create workflow
workflow = OpenAIProxyWorkflow(
    agent=MyTokenRewardAgent(),
    proxy_addr="http://localhost:8000",
    discount=0.9,
)

# Use in training
rollout_batch = actor.prepare_batch(
    dataloader,
    workflow=workflow,
    group_size=1
)
```

### Position-Level Rewards Example

```python
from customized_areal.on_policy_distill.proxy.types import PositionRewardInfo

class PositionRewardAgent:
    async def run(self, data, **extra_kwargs):
        proxy_client = extra_kwargs.get("proxy_client")
        api_key = extra_kwargs.get("api_key")

        # ... generate response and get top-k logprobs ...

        # Set position-wise rewards
        position_rewards = []
        for pos, top_k in enumerate(top_k_logprobs):
            pr = PositionRewardInfo(
                position=pos,
                candidates=[t["token"] for t in top_k],
                candidate_token_ids=[t["token_id"] for t in top_k],
                logprobs=[t["logprob"] for t in top_k],
                rewards=self._compute_rewards(top_k),  # Your reward function
                chosen_index=0,  # Index of selected token
            )
            position_rewards.append(pr)

        await proxy_client.set_position_rewards(completion_id, position_rewards)

        # Return scalar reward for backward compatibility
        return total_reward
```

## HTTP API Endpoints

### Token-Level Rewards

```http
POST /rl/set_token_rewards
Authorization: Bearer {session_api_key}
Content-Type: application/json

{
    "interaction_id": "completion-id",
    "token_rewards": [0.0, 0.5, 1.0, 0.5]
}
```

### Position-Level Rewards

```http
POST /rl/set_position_rewards
Authorization: Bearer {session_api_key}
Content-Type: application/json

{
    "interaction_id": "completion-id",
    "position_rewards": [
        {
            "position": 0,
            "candidates": ["The", "A"],
            "candidate_token_ids": [101, 102],
            "logprobs": [-0.5, -1.2],
            "rewards": [1.0, 0.5],
            "chosen_index": 0
        }
    ]
}
```

### Compute Entropy

```http
POST /rl/compute_entropy
Authorization: Bearer {session_api_key}
Content-Type: application/json

{
    "interaction_id": "completion-id"
}

Response:
{
    "entropies": [0.5, 0.3, 0.8],
    "avg_entropy": 0.53
}
```

## Reward Types Supported

The agent's `run()` method can return rewards in several formats:

| Return Type              | Example                    | Description                           |
| ------------------------ | -------------------------- | ------------------------------------- |
| `float`                  | `1.0`                      | Scalar reward for the last completion |
| `dict[str, float]`       | `{"id1": 1.0, "id2": 0.5}` | Completion ID → scalar reward         |
| `dict[str, list[float]]` | `{"id1": [0.0, 0.5, 1.0]}` | Completion ID → token-level rewards   |

## Comparison with Base AReaL

| Feature             | Base AReaL              | This Customization           |
| ------------------- | ----------------------- | ---------------------------- |
| Reward type         | Scalar only             | Token-level + Position-level |
| Reward storage      | Server-side             | Server-side (HTTP API)       |
| Cache               | None                    | None (HTTP-based)            |
| Modes               | inline, subproc, online | inline only                  |
| Entropy computation | Not supported           | Via `/rl/compute_entropy`    |

See `workflow_comparison.md` for a detailed comparison.

## Configuration

When initializing the training, ensure the proxy server endpoints are enabled:

```python
# In your training config
from areal.api.cli_args import OpenAIProxyConfig

openai_cfg = OpenAIProxyConfig(
    mode="inline",  # Required for token-level rewards
    admin_api_key="your-admin-key",
    # ... other config
)
```

## Testing

Run the tests to verify the system:

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
cd customized_areal/on_policy_distill/proxy/tests
python -m pytest test_token_rewards.py -v
```

## Implementation Notes

### Thread Safety

- `TokenRewardSessionData` uses `threading.Lock` for concurrent access to reward storage
- HTTP endpoints are stateless and session-scoped

### Reward Application

Token-level rewards are applied during `export_interactions()`:

1. Server stores rewards via `set_token_rewards()` or `set_position_rewards()`
1. When session ends, `export_interactions()` applies rewards to interactions
1. Rewards are propagated through the `InteractionWithTokenLevelReward` object
1. `to_tensor_dict()` broadcasts token rewards to the full sequence tensor

### Memory Management

- Rewards are stored on the proxy server, not in the training process
- Session data is cleaned up after `export_interactions()`
- Use `max_concurrent_rollouts` config to limit memory usage
