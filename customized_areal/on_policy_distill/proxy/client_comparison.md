# Client Comparison Analysis

## Overview

| Aspect | `areal/experimental/openai/proxy/` | `customized_areal/on_policy_distill/proxy/` |
|--------|-----------------------------------|---------------------------------------------|
| **Purpose** | Standard proxy server for RL training | Extended proxy with token-level reward support |
| **Reward Types** | Scalar only | Scalar + Token-wise + Position-wise |
| **Cache Design** | Server-side cache | Server-side cache with token-level storage |
| **API** | HTTP REST API | Extended HTTP REST API |
| **Local Cache** | No | **No** - Token rewards via HTTP API |

---

## Architecture Comparison

### Before (Old Design)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Workflow                                │
│  ┌──────────────────┐        ┌──────────────────┐              │
│  │  real_client     │        │  local_client    │              │
│  │  (HTTP to server)│        │  (local cache)   │              │
│  └────────┬─────────┘        └────────┬─────────┘              │
│           │                           │                        │
│           │ 1. HTTP session           │ 3. Import interactions │
│           │    for agent LLM calls    │    from server         │
│           │                           │                        │
│           │ 2. End session            │ 4. Apply token rewards │
│           │                           │    on local cache      │
│           │                           │                        │
│           │                           │ 5. Export from cache   │
│           ▼                           ▼                        │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              Proxy Server (HTTP API)                  │     │
│  │         - Scalar rewards only                         │     │
│  └──────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### After (New Design)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Workflow                                │
│                         │                                       │
│                         │ Single unified client                 │
│                         │ (HTTP to server)                      │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              Token Reward Proxy Server                │     │
│  │  ┌────────────────────────────────────────────────┐  │     │
│  │  │  Extended HTTP API                             │  │     │
│  │  │  - POST /rl/set_reward          (scalar)       │  │     │
│  │  │  - POST /rl/set_token_rewards   (NEW)          │  │     │
│  │  │  - POST /rl/set_position_rewards (NEW)         │  │     │
│  │  │  - POST /rl/compute_entropy     (NEW)          │  │     │
│  │  └────────────────────────────────────────────────┘  │     │
│  │                                                          │     │
│  │  ┌────────────────────────────────────────────────┐  │     │
│  │  │  TokenRewardSessionData (server-side)          │  │     │
│  │  │  - _token_rewards: dict[str, list[float]]      │  │     │
│  │  │  - _position_rewards: dict[str, list[PR]]      │  │     │
│  │  └────────────────────────────────────────────────┘  │     │
│  └──────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Changes

### 1. No Local Cache

| Aspect | Before | After |
|--------|--------|-------|
| **Cache Location** | Local client + Server | Server only |
| **Token Rewards** | Applied locally after import | Sent via HTTP during session |
| **Import Step** | Required `import_from_server()` | **Eliminated** |
| **Complexity** | Two-phase workflow | Single-phase workflow |

### 2. New HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rl/set_reward` | POST | Scalar reward (inherited) |
| `/rl/set_token_rewards` | POST | **NEW** - Token-wise rewards |
| `/rl/set_position_rewards` | POST | **NEW** - Position-wise rewards |
| `/rl/compute_entropy` | POST | **NEW** - Compute entropy from logprobs |

### 3. New Data Models

```python
# Request model for token-wise rewards
class SetTokenRewardsRequest(BaseModel):
    interaction_id: str | None  # None = last interaction
    token_rewards: list[float]  # One per output token

# Request model for position-wise rewards
class SetPositionRewardsRequest(BaseModel):
    interaction_id: str | None
    position_rewards: list[PositionRewardInfo]

# Position reward info with candidates
class PositionRewardInfo(BaseModel):
    position: int
    candidates: list[str]
    candidate_token_ids: list[int]
    logprobs: list[float] | None
    rewards: list[float]
    chosen_index: int
```

### 4. New Server Components

```python
# Extended session data with token-level storage
class TokenRewardSessionData(SessionData):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self._token_rewards: dict[str, list[float]] = {}
        self._position_rewards: dict[str, list[PositionRewardInfo]] = {}

    def set_token_rewards(self, interaction_id: str, token_rewards: list[float]):
        # Store token rewards and update scalar reward as sum

    def set_position_rewards(self, interaction_id: str, position_rewards: list[PositionRewardInfo]):
        # Store position rewards, extract chosen rewards

    def compute_entropy(self, interaction_id: str) -> tuple[list[float], float]:
        # Compute entropy from position logprobs

    def export_interactions(self, discount: float, style: str):
        # Apply token rewards to interactions before export
```

---

## Workflow Comparison

### Before (Two-Phase)

```python
# Phase 1: HTTP session for agent
async with client:
    rewards = await agent.run(api_key=client.session_api_key)

# Phase 2: Import from server to local cache
await client.import_from_server()

# Phase 3: Apply token-level rewards
await client.set_rewards("comp-1", [0.5, 0.3, 0.2])

# Phase 4: Export from local cache
interactions = await client.export_interactions_with_rewards()
```

### After (Single-Phase)

```python
# Single phase: Everything within HTTP session
async with client:
    # Agent runs with session_api_key
    rewards = await agent.run(api_key=client.session_api_key)
    # Set token-level rewards via HTTP
    await client.set_rewards("comp-1", [0.5, 0.3, 0.2])

# Export after session ends
interactions = await client.export_interactions()
```

---

## Usage Example

### Server Startup

```bash
# Run the token reward proxy server
python -m customized_areal.on_policy_distill.proxy.proxy_rollout_server \
    --host 0.0.0.0 \
    --port 8000 \
    --admin-api-key my-admin-key
```

### Client Usage

```python
import aiohttp
from customized_areal.on_policy_distill.proxy.client import OpenAIProxyClient

async with aiohttp.ClientSession() as http_session:
    client = OpenAIProxyClient(
        session=http_session,
        base_url="http://localhost:8000",
        task_id="my-task",
        admin_api_key="my-admin-key",
    )

    async with client:
        # Agent uses session_api_key for LLM calls
        completion = await agent_llm_call(api_key=client.session_api_key)

        # Set scalar reward
        await client.set_reward(completion.id, 1.0)

        # Set token-wise rewards
        await client.set_rewards(completion.id, [0.5, 0.3, 0.2])

        # Set position-wise rewards
        from .server import PositionRewardInfo
        await client.set_position_rewards(
            completion.id,
            [
                PositionRewardInfo(
                    position=0,
                    candidates=["a", "b"],
                    rewards=[0.1, 0.5],
                    chosen_index=1,
                )
            ]
        )

    # Export after session ends
    interactions = await client.export_interactions()
```

---

## Benefits of New Design

| Benefit | Description |
|---------|-------------|
| **Simpler** | Single client, no local cache management |
| **Consistent** | All rewards go through HTTP API |
| **Scalable** | Server handles all state, easier to distribute |
| **Cleaner** | No need for `import_from_server()` step |
| **Compatible** | Still works with existing scalar reward workflows |

---

## Files Changed

| File | Change |
|------|--------|
| `proxy/server.py` | **NEW** - Extended server models and session data |
| `proxy/proxy_rollout_server.py` | **NEW** - FastAPI server with token-level endpoints |
| `proxy/client.py` | **UPDATED** - Removed local cache, use HTTP API |
| `proxy/workflow.py` | **UPDATED** - Simplified, no import_from_server() |

---

## Migration Guide

### From Old Design

1. **Start new server**:
   ```bash
   python -m customized_areal.on_policy_distill.proxy.proxy_rollout_server
   ```

2. **Update client imports** (no change needed):
   ```python
   from customized_areal.on_policy_distill.proxy.client import OpenAIProxyClient
   ```

3. **Remove `import_from_server()` calls** - no longer needed

4. **Call `set_rewards()` within session context**:
   ```python
   async with client:
       # ... run agent ...
       await client.set_rewards(comp_id, token_rewards)  # HTTP call
   ```
