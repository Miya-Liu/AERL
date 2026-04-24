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
- **On-policy distillation**: Teacher logprobs vs. student logprobs at each position

## Architecture

The system has five components that form a layered architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  AReaL Training Loop                                                        │
│  actor.prepare_batch(workflow=OpenAIProxyWorkflow)                          │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ workflow.py  —  OpenAIProxyWorkflow                                 │    │
│  │  Orchestrates the full episode lifecycle:                           │    │
│  │  1. Creates OpenAIProxyClient (HTTP session to proxy server)        │    │
│  │  2. Starts session via POST /rl/start_session                      │    │
│  │  3. Runs agent; agent uses session_api_key for LLM calls           │    │
│  │  4. Agent sets rewards via proxy_client HTTP methods                │    │
│  │  5. Ends session via POST /rl/end_session                          │    │
│  │  6. Exports interactions with token-level rewards applied           │    │
│  │  7. Converts to tensor dict + attaches position_rewards             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │ uses                                                                  │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ client.py  —  OpenAIProxyClient                                     │    │
│  │  HTTP client that talks to the proxy server:                        │    │
│  │  - set_rewards()          → POST /rl/set_token_rewards              │    │
│  │  - set_position_rewards() → POST /rl/set_position_rewards           │    │
│  │  - compute_entropy()      → POST /rl/compute_entropy                │    │
│  │  - export_interactions()  → POST /export_trajectories               │    │
│  │  Inherits base session management (start/end/grant_capacity)        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │ HTTP                                                                  │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ proxy_rollout_server.py  —  FastAPI Server                          │    │
│  │  The HTTP server that runs on each inference worker:                │    │
│  │  - OpenAI-compatible endpoints (/chat/completions, /responses,      │    │
│  │    /anthropic/messages) for LLM generation                          │    │
│  │  - Token-level reward endpoints (/rl/set_token_rewards,             │    │
│  │    /rl/set_position_rewards, /rl/compute_entropy)                   │    │
│  │  - Admin endpoints (start/end session, export, grant capacity)      │    │
│  │  - Engine management (/create_engine, /call, /health, etc.)         │    │
│  │  - Uses TokenRewardSessionData to store rewards per session         │    │
│  │  - Serializes position_rewards for HTTP transport                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │ uses                                                                  │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ server.py  —  Data Models & Session Logic                           │    │
│  │  - PositionRewardInfo (Pydantic): per-position candidate rewards    │    │
│  │  - TokenRewardSessionData: extends base SessionData with            │    │
│  │    _token_rewards, _position_rewards storage, entropy computation,  │    │
│  │    and export_interactions() that applies rewards to interactions   │    │
│  │  - Request/Response models for token reward endpoints               │    │
│  │  - Path constants for HTTP endpoints                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │ stores data in                                                        │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ cache.py  —  InteractionCache (Testing/Mock)                        │    │
│  │  Local in-memory cache for testing without a running server:        │    │
│  │  - OrderedDict of InteractionWithTokenLevelReward                   │    │
│  │  - Parent-child relationship building via message prefix matching   │    │
│  │  - set_rewards(), set_position_rewards() with validation            │    │
│  │  - compute_and_store_entropy() from logprobs                       │    │
│  │  - apply_reward_discount() for backward reward propagation          │    │
│  │  - export_interactions() with "individual" or "concat" styles       │    │
│  │  NOT used in production — the HTTP path (client + server) is the    │    │
│  │  real data flow. cache.py exists for unit tests and local dev.      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How the Components Work Together

### Production Data Flow (HTTP Path)

This is the path used during actual distributed training:

```
AReaL Training Loop
    │
    ▼
OpenAIProxyWorkflow.arun_episode()
    │
    ├─► Creates OpenAIProxyClient (HTTP session)
    │
    ├─► client.start_session() ──HTTP──► proxy_rollout_server /rl/start_session
    │                                      creates TokenRewardSessionData
    │                                      returns session_api_key
    │
    ├─► agent.run(data, proxy_client=client, api_key=session_api_key)
    │       │
    │       ├─► Agent calls LLM ──HTTP──► proxy_rollout_server /chat/completions
    │       │                                  (proxied to SGLang/vLLM engine)
    │       │                                  stores completion in SessionData
    │       │
    │       └─► Agent sets rewards ──HTTP──► proxy_rollout_server /rl/set_token_rewards
    │           via proxy_client                or /rl/set_position_rewards
    │                                        SessionData stores rewards per interaction
    │
    ├─► client.end_session() ──HTTP──► proxy_rollout_server /rl/end_session
    │                                      marks session completed
    │
    ├─► client.export_interactions() ──HTTP──► proxy_rollout_server /export_trajectories
    │                                           TokenRewardSessionData.export_interactions()
    │                                             applies token_rewards & position_rewards
    │                                           serialize_interactions_with_position_rewards()
    │                                           deserialized by client
    │
    └─► Workflow converts to tensor dict
        attaches position_rewards list for distillation loss
        returns to AReaL training loop
```

### Role of Each File

| File | Role | Key Classes/Functions | Production? |
|------|------|-----------------------|-------------|
| `workflow.py` | Orchestration layer | `OpenAIProxyWorkflow` — episode lifecycle, reward processing, tensor conversion | Yes |
| `client.py` | HTTP client | `OpenAIProxyClient` — sends reward/export requests to server | Yes |
| `proxy_rollout_server.py` | HTTP server | FastAPI app with all endpoints, serialization, engine management | Yes |
| `server.py` | Data models & logic | `TokenRewardSessionData`, `PositionRewardInfo`, request/response models | Yes |
| `cache.py` | Local test cache | `InteractionCache`, `PositionRewardInfo` (dataclass) | No (testing only) |

### Dual Implementation: cache.py vs. server.py

The same reward concepts exist in two parallel implementations:

| Concept | `cache.py` (Local/Testing) | `server.py` + `proxy_rollout_server.py` (Production/HTTP) |
|---------|---------------------------|----------------------------------------------------------|
| Position rewards | `PositionRewardInfo` (dataclass) | `PositionRewardInfo` (Pydantic model) |
| Reward storage | `InteractionCache._lock` + dict | `TokenRewardSessionData._lock` + dict |
| Set token rewards | `InteractionCache.set_rewards()` | `POST /rl/set_token_rewards` → `SessionData.set_token_rewards()` |
| Set position rewards | `InteractionCache.set_position_rewards()` | `POST /rl/set_position_rewards` → `SessionData.set_position_rewards()` |
| Compute entropy | `InteractionCache.compute_and_store_entropy()` | `POST /rl/compute_entropy` → `SessionData.compute_entropy()` |
| Export | `InteractionCache.export_interactions()` | `POST /export_trajectories` → `SessionData.export_interactions()` |
| Parent-child tree | Built in `__setitem__` via prefix matching | Built in base `SessionData.completions` (same `InteractionCache`) |

`cache.py` is used for **unit tests** where you want to test reward logic without
running an HTTP server. The production path goes through `client.py` →
`proxy_rollout_server.py` → `server.py` over HTTP.

### Reward Processing in workflow.py

The workflow's `_process_rewards()` method handles multiple reward formats from agents:

```
Agent returns ──► _process_rewards() ──► HTTP call to server

float              → client.set_last_reward()        → POST /rl/set_reward
dict[str, float]   → client.set_reward(id, val)       → POST /rl/set_reward
dict[str, list]    → client.set_rewards(id, list)      → POST /rl/set_token_rewards
dict[str, dict]    → client.set_position_rewards(...)  → POST /rl/set_position_rewards
                    + client.set_reward(id, scalar)     → POST /rl/set_reward
```

The `dict[str, dict]` format (with `"position_rewards"` and `"scalar_reward"` keys)
is used by distillation agents (OnPolicyDistillAgent, TreeDistillAgent). Scalar and
position rewards are set independently so tree backup advantage computation uses only
the trajectory-level scalar reward, while position rewards feed the distillation loss.

### Export and Tensor Conversion

After `export_interactions()`, the workflow does custom tensor conversion instead of
letting AReaL's default `workflow_executor` handle it. This is because:

1. `position_rewards` is a Python attribute (not in `to_tensor_dict()`) to avoid
   `concat_padded_tensors` key consistency issues when some interactions have
   position_rewards and others don't (e.g., multi-turn conversations).
2. Each `PositionRewardInfo` gets a `sample_index` indicating which batch item it
   belongs to, so minibatch splitting can correctly partition them.

## File Structure

| File | Purpose |
|------|---------|
| `workflow.py` | `OpenAIProxyWorkflow` — orchestrates episodes with token-level reward support |
| `client.py` | `OpenAIProxyClient` — HTTP client with methods for setting token/position rewards |
| `server.py` | Data models and session logic (`TokenRewardSessionData`, `PositionRewardInfo`, etc.) |
| `proxy_rollout_server.py` | FastAPI server with token-level reward endpoints + serialization |
| `cache.py` | `InteractionCache` — local in-memory cache for testing (not used in production) |
| `types.py` | `InteractionWithTokenLevelReward` — extended interaction type with token reward fields |

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
training and distillation):

```python
from customized_areal.on_policy_distill.proxy.server import PositionRewardInfo

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

All reward communication goes through HTTP API:

- **Scalable**: Rewards stored on the proxy server, not locally
- **Clean separation**: Agent and trainer communicate via well-defined HTTP endpoints
- **No cache synchronization**: Single source of truth on the server

## Usage

### Basic Example

```python
from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow

class MyTokenRewardAgent:
    async def run(self, data, **extra_kwargs):
        proxy_client = extra_kwargs.get("proxy_client")
        api_key = extra_kwargs.get("api_key")

        import openai
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=extra_kwargs.get("base_url")
        )

        response = await client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": data["prompt"]}]
        )

        completion_id = response.id
        num_tokens = len(response.choices[0].logprobs.content)
        token_rewards = [0.0] * (num_tokens - 1) + [1.0]

        await proxy_client.set_rewards(completion_id, token_rewards)

        return {completion_id: token_rewards}

workflow = OpenAIProxyWorkflow(
    agent=MyTokenRewardAgent(),
    proxy_addr="http://localhost:8000",
    discount=0.9,
)
```

### Position-Level Rewards (Distillation) Example

```python
from customized_areal.on_policy_distill.proxy.server import PositionRewardInfo

class PositionRewardAgent:
    async def run(self, data, **extra_kwargs):
        proxy_client = extra_kwargs.get("proxy_client")
        api_key = extra_kwargs.get("api_key")

        # ... generate response and get top-k logprobs ...

        position_rewards = []
        for pos, top_k in enumerate(top_k_logprobs):
            pr = PositionRewardInfo(
                position=pos,
                candidates=[t["token"] for t in top_k],
                candidate_token_ids=[t["token_id"] for t in top_k],
                logprobs=[t["logprob"] for t in top_k],
                rewards=self._compute_rewards(top_k),
                chosen_index=0,
            )
            position_rewards.append(pr)

        await proxy_client.set_position_rewards(completion_id, position_rewards)

        # Return dict format for distillation workflows
        return {
            completion_id: {
                "position_rewards": position_rewards,
                "scalar_reward": total_reward,
            }
        }
```

## HTTP API Endpoints

### Session Management

| Endpoint | Auth | Description |
|----------|------|-------------|
| `POST /rl/start_session` | Admin | Start a new session, get session_api_key |
| `POST /rl/end_session` | Session | End session, mark as completed |
| `POST /rl/grant_capacity` | Admin | Increment session capacity |
| `POST /export_trajectories` | Admin | Export interactions with rewards applied |

### LLM Generation (OpenAI-Compatible)

| Endpoint | Auth | Description |
|----------|------|-------------|
| `POST /chat/completions` | Session | OpenAI chat completions (streaming supported) |
| `POST /responses` | Session | OpenAI responses API |
| `POST /anthropic/messages` | Session | Anthropic Messages API (auto-translated) |

### Reward Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `POST /rl/set_reward` | Session | Set scalar reward for an interaction |
| `POST /rl/set_token_rewards` | Session | Set per-token rewards for an interaction |
| `POST /rl/set_position_rewards` | Session | Set position-wise candidate rewards |
| `POST /rl/compute_entropy` | Session | Compute entropy from position rewards |

### Engine Management

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | None | Health check |
| `POST /alloc_ports` | None | Allocate free ports |
| `POST /configure` | None | Set server configuration |
| `POST /set_env` | None | Set environment variables |
| `POST /create_engine` | None | Create inference engine instance |
| `POST /call` | None | Call engine method (e.g., initialize) |

## Reward Types Supported

The agent's `run()` method can return rewards in several formats:

| Return Type | Example | Description |
|-------------|---------|-------------|
| `float` | `1.0` | Scalar reward for the last completion |
| `dict[str, float]` | `{"id1": 1.0, "id2": 0.5}` | Completion ID to scalar reward |
| `dict[str, list[float]]` | `{"id1": [0.0, 0.5, 1.0]}` | Completion ID to token-level rewards |
| `dict[str, dict]` | `{"id1": {"position_rewards": [...], "scalar_reward": 1.0}}` | For distillation workflows |

## Comparison with Base AReaL

| Feature | Base AReaL | This Customization |
|---------|------------|-------------------|
| Reward type | Scalar only | Token-level + Position-level |
| Reward storage | Server-side | Server-side (HTTP API) |
| Cache | None | None (HTTP-based) |
| Modes | inline, subproc, online | inline only |
| Entropy computation | Not supported | Via `/rl/compute_entropy` |
| Distillation support | Not supported | Position rewards for KL loss |

## Configuration

When initializing the training, ensure the proxy server endpoints are enabled:

```python
from areal.api.cli_args import OpenAIProxyConfig

openai_cfg = OpenAIProxyConfig(
    mode="inline",  # Required for token-level rewards
    admin_api_key="your-admin-key",
)
```

## Running the Server

```bash
python -m customized_areal.on_policy_distill.proxy.proxy_rollout_server \
    --host 0.0.0.0 \
    --port 8000 \
    --admin-api-key your-admin-key \
    --experiment-name my-exp \
    --trial-name my-trial \
    --role rollout \
    --worker-index 0
```

## Testing

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
cd customized_areal/on_policy_distill/proxy/tests
python -m pytest test_token_rewards.py -v
```

## Implementation Notes

### Thread Safety

- `TokenRewardSessionData` uses `threading.Lock` for concurrent access to reward storage
- `InteractionCache` (testing) also uses `threading.Lock`
- HTTP endpoints are stateless and session-scoped

### Reward Application

Token-level rewards are applied during `export_interactions()`:

1. Server stores rewards via `set_token_rewards()` or `set_position_rewards()`
2. When session ends, `export_interactions()` applies rewards to interactions
3. Scalar reward is preserved separately from token/position rewards
4. Position rewards are attached as Python attributes, not in `to_tensor_dict()`

### Scalar vs. Position Reward Separation

Scalar (trajectory-level) rewards and position-level rewards are stored and set
independently. This ensures:

- **Tree backup advantage** computation uses only trajectory-level scalar rewards
- **Distillation loss** uses position-level rewards (logp_model - logp_teacher)
- Setting position rewards does NOT overwrite the scalar reward

### Memory Management

- Rewards are stored on the proxy server, not in the training process
- Session data is cleaned up after `export_trajectories` or by the stale session
  cleanup task (runs every 60 seconds, removes sessions idle > `SESSION_TIMEOUT_SECONDS`)
- Use `max_concurrent_rollouts` config to limit memory usage via capacity grants
