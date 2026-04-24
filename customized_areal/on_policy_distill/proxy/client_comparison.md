# Client Comparison Analysis

## Overview

| Aspect           | `areal/experimental/openai/proxy/`         | `customized_areal/on_policy_distill/proxy/`                 |
| ---------------- | ------------------------------------------ | ----------------------------------------------------------- |
| **Purpose**      | Standard proxy server for RL training      | Extended proxy with token-level reward support              |
| **Reward Types** | Scalar only                                | Scalar + Token-wise + Position-wise                         |
| **Cache Design** | Server-side cache only                     | Server-side cache + local `InteractionCache`                |
| **API**          | HTTP REST API                              | Extended HTTP REST API                                      |
| **Local Cache**  | No                                         | Yes (`InteractionCache` in `cache.py`)                      |
| **Gateway**      | Yes (`proxy_gateway.py`)                   | No standalone gateway                                       |
| **Online Agent** | Yes (`online_agent.py`)                    | No                                                          |
| **Types Module** | Uses base `InteractionWithTokenLogpReward` | Dedicated `types.py` with `InteractionWithTokenLevelReward` |

______________________________________________________________________

## File Structure Comparison

| File                      | `areal/.../proxy/`                                    | `customized_areal/.../proxy/`                                |
| ------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| `__init__.py`             | Exports `OpenAIProxyClient`, `OpenAIProxyWorkflow`    | (not present)                                                |
| `client_session.py`       | `OpenAIProxyClient` + retry helpers                   | —                                                            |
| `client.py`               | —                                                     | `OpenAIProxyClient` (extends base)                           |
| `online_agent.py`         | `_OnlineAgent` (wait for external session)            | —                                                            |
| `proxy_gateway.py`        | `create_proxy_gateway_app()` + `CompletedSessionInfo` | —                                                            |
| `proxy_rollout_server.py` | FastAPI server (base endpoints)                       | FastAPI server (base + token-level endpoints)                |
| `server.py`               | `SessionData`, request/response models                | `TokenRewardSessionData` (extends `SessionData`), new models |
| `workflow.py`             | `OpenAIProxyWorkflow` (3 modes)                       | `OpenAIProxyWorkflow` (extended with `_process_rewards`)     |
| `types.py`                | —                                                     | `InteractionWithTokenLevelReward`, `TokenRewardInteractions` |
| `cache.py`                | —                                                     | `InteractionCache`, `PositionRewardInfo` (dataclass)         |
| `tests/`                  | —                                                     | 11 test files                                                |

______________________________________________________________________

## Class-by-Class Comparison

### Client: `OpenAIProxyClient`

| Method                      | Base (`client_session.py`)                    | Extended (`client.py`)                         | Notes                                       |
| --------------------------- | --------------------------------------------- | ---------------------------------------------- | ------------------------------------------- |
| `__init__`                  | `(session, base_url, task_id, admin_api_key)` | Same signature                                 | Extended inherits from base                 |
| `session_api_key`           | property                                      | inherited                                      |                                             |
| `set_reward`                | `async (completion_id, reward)`               | inherited                                      | Scalar reward                               |
| `set_last_reward`           | `async (reward)`                              | inherited                                      |                                             |
| `set_rewards`               | —                                             | `async (completion_id, token_rewards)`         | **NEW** - Token-wise rewards via HTTP       |
| `set_last_rewards`          | —                                             | `async (token_rewards)`                        | **NEW** - Token rewards for last completion |
| `set_position_rewards`      | —                                             | `async (completion_id, position_rewards)`      | **NEW** - Position-wise rewards via HTTP    |
| `set_last_position_rewards` | —                                             | `async (position_rewards)`                     | **NEW**                                     |
| `compute_entropy`           | —                                             | `async (completion_id) -> list[float]`         | **NEW**                                     |
| `get_entropies`             | —                                             | `async (completion_id) -> list[float] \| None` | **NEW**                                     |
| `export_interactions`       | `async (discount, style) -> dict`             | `async (discount, style) -> dict`              | Same signature, extended serialization      |
| `get_last_interaction`      | —                                             | `async () -> Any`                              | **NEW**                                     |
| `__aenter__` / `__aexit__`  | Yes                                           | inherited                                      |                                             |

### Standalone helper functions (base only)

| Function                      | Base | Extended | Notes                           |
| ----------------------------- | ---- | -------- | ------------------------------- |
| `post_json`                   | Yes  | —        | Low-level HTTP POST             |
| `post_json_with_retry`        | Yes  | —        | POST with tenacity retry        |
| `should_retry`                | Yes  | —        | Retry decision for HTTP errors  |
| `log_retry`                   | Yes  | —        | Retry logging callback          |
| `get_retry_strategy`          | Yes  | —        | Create tenacity retry config    |
| `_set_reward`                 | Yes  | —        | Internal reward setter          |
| `set_interaction_reward`      | Yes  | —        | Set reward by interaction_id    |
| `set_last_interaction_reward` | Yes  | —        | Set reward for last interaction |
| `_start_session`              | Yes  | —        | Start session with retry        |

### Server: Session Data

| Method / Field         | `SessionData` (base)                        | `TokenRewardSessionData` (extended)                |
| ---------------------- | ------------------------------------------- | -------------------------------------------------- |
| `session_id`           | Yes                                         | inherited                                          |
| `interactions`         | `dict[str, InteractionWithTokenLogpReward]` | inherited                                          |
| `_token_rewards`       | —                                           | `dict[str, list[float]]` **NEW**                   |
| `_position_rewards`    | —                                           | `dict[str, list[PositionRewardInfo]]` **NEW**      |
| `_entropies`           | —                                           | `dict[str, list[float]]` **NEW**                   |
| `set_token_rewards`    | —                                           | **NEW**                                            |
| `set_position_rewards` | —                                           | **NEW**                                            |
| `compute_entropy`      | —                                           | **NEW**                                            |
| `export_interactions`  | Yes                                         | **Override** - applies token rewards before export |

### Server: Pydantic Models

| Model                        | Base | Extended  | Notes                                                                                  |
| ---------------------------- | ---- | --------- | -------------------------------------------------------------------------------------- |
| `StartSessionRequest`        | Yes  | inherited |                                                                                        |
| `StartSessionResponse`       | Yes  | inherited |                                                                                        |
| `SetRewardRequest`           | Yes  | inherited |                                                                                        |
| `ExportTrajectoriesRequest`  | Yes  | inherited |                                                                                        |
| `ExportTrajectoriesResponse` | Yes  | inherited |                                                                                        |
| `WaitForSessionRequest`      | Yes  | inherited |                                                                                        |
| `WaitForSessionResponse`     | Yes  | inherited |                                                                                        |
| `SetTokenRewardsRequest`     | —    | **NEW**   | `interaction_id: str \| None`, `token_rewards: list[float]`                            |
| `SetPositionRewardsRequest`  | —    | **NEW**   | `interaction_id: str \| None`, `position_rewards: list[PositionRewardInfo]`            |
| `ComputeEntropyRequest`      | —    | **NEW**   | `interaction_id: str`                                                                  |
| `ComputeEntropyResponse`     | —    | **NEW**   | `entropies: list[float]`, `avg_entropy: float`                                         |
| `PositionRewardInfo`         | —    | **NEW**   | `position`, `candidates`, `candidate_token_ids`, `logprobs`, `rewards`, `chosen_index` |

### Workflow: `OpenAIProxyWorkflow`

| Method / Field                 | Base (`workflow.py`)                                                                                        | Extended (`workflow.py`)                                                  |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `__init__`                     | `(mode, agent, proxy_addr, admin_api_key, discount, export_style, subproc_max_workers, proxy_gateway_addr)` | `(agent, proxy_addr, admin_api_key, discount, export_style)` - simplified |
| `arun_episode`                 | Yes                                                                                                         | **Override** - adds reward processing                                     |
| `_run_agent`                   | —                                                                                                           | **NEW** - runs agent with proxy client injection                          |
| `_process_rewards`             | —                                                                                                           | **NEW** - processes rewards from agent output                             |
| `_wrap_run_subproc`            | —                                                                                                           | **NEW** - subprocess wrapper with env vars                                |
| `_get_executor`                | —                                                                                                           | **NEW** - thread pool executor singleton                                  |
| `mode` (inline/subproc/online) | Yes                                                                                                         | — (removed)                                                               |
| `proxy_gateway_addr`           | Yes                                                                                                         | — (removed)                                                               |

### Unique to Base Only

| Component                    | Description                                                                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `proxy_gateway.py`           | Stateless gateway routing requests to backend workers                                                                                        |
| `create_proxy_gateway_app()` | Creates FastAPI gateway with `/health`, `/rl/start_session`, `/chat/completions`, `/responses`, `/v1/messages`, `/internal/wait_for_session` |
| `CompletedSessionInfo`       | Dataclass with `session_api_key`, `session_id`, `worker_addr`                                                                                |
| `_OnlineAgent`               | Waits for external user to complete a session (online mode)                                                                                  |

### Unique to Extended Only

| Component                         | Description                                                                                                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `types.py`                        | `InteractionWithTokenLevelReward` extending base interaction                                               |
| `cache.py`                        | `InteractionCache` (OrderedDict-based) + `PositionRewardInfo` dataclass                                    |
| `InteractionWithTokenLevelReward` | Extends `InteractionWithTokenLogpReward` with `token_rewards`, `token_reward_mask`                         |
| `InteractionCache`                | Local cache with `set_rewards`, `set_position_rewards`, `compute_and_store_entropy`, `export_interactions` |

______________________________________________________________________

## HTTP Endpoint Comparison

| Endpoint                        | Base | Extended | Notes                           |
| ------------------------------- | ---- | -------- | ------------------------------- |
| `GET /health`                   | Yes  | Yes      |                                 |
| `POST /alloc_ports`             | Yes  | Yes      |                                 |
| `POST /configure`               | Yes  | Yes      |                                 |
| `POST /set_env`                 | Yes  | Yes      |                                 |
| `POST /create_engine`           | Yes  | Yes      |                                 |
| `POST /call`                    | Yes  | Yes      |                                 |
| `POST /rl/start_session`        | Yes  | Yes      |                                 |
| `POST /rl/end_session`          | Yes  | Yes      |                                 |
| `POST /rl/set_reward`           | Yes  | Yes      | Scalar reward                   |
| `POST /chat/completions`        | Yes  | Yes      |                                 |
| `POST /responses`               | Yes  | Yes      |                                 |
| `POST /v1/messages`             | Yes  | Yes      |                                 |
| `POST /grant_capacity`          | Yes  | Yes      |                                 |
| `POST /export_trajectories`     | Yes  | Yes      |                                 |
| `POST /rl/set_token_rewards`    | —    | **NEW**  | Token-wise rewards              |
| `POST /rl/set_position_rewards` | —    | **NEW**  | Position-wise candidate rewards |
| `POST /rl/compute_entropy`      | —    | **NEW**  | Compute entropy from logprobs   |

______________________________________________________________________

## Architecture Comparison

### Base Architecture (areal)

```
┌──────────────────────────────────────────────────────────────────┐
│                        Workflow (3 modes)                        │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐    │
│  │  inline mode   │  │  subproc mode │  │  online mode     │    │
│  │  (direct call) │  │  (subprocess) │  │  (external user) │    │
│  └───────┬───────┘  └───────┬───────┘  └────────┬─────────┘    │
│          │                  │                    │              │
│          └──────────────────┼────────────────────┘              │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              OpenAIProxyClient                        │      │
│  │  - set_reward(completion_id, reward)                 │      │
│  │  - set_last_reward(reward)                           │      │
│  │  - export_interactions(discount, style)              │      │
│  └──────────────────────┬───────────────────────────────┘      │
│                         │ HTTP                                   │
│  ┌──────────────────────▼───────────────────────────────┐      │
│  │         Proxy Rollout Server (SessionData)            │      │
│  │  - Scalar rewards only                                │      │
│  │  - serialize/deserialize_interactions                 │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Optional: Proxy Gateway (stateless router)                     │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  create_proxy_gateway_app()                           │      │
│  │  - Routes /chat/completions, /responses, /v1/messages │      │
│  │  - /internal/wait_for_session                         │      │
│  │  - _OnlineAgent waits for external completion         │      │
│  └──────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### Extended Architecture (customized_areal)

```
┌──────────────────────────────────────────────────────────────────┐
│                   OpenAIProxyWorkflow (simplified)               │
│                         │                                       │
│                         │ _run_agent + _process_rewards          │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │         OpenAIProxyClient (extended)                  │      │
│  │  Inherited:                                          │      │
│  │  - set_reward, set_last_reward                       │      │
│  │  NEW:                                                │      │
│  │  - set_rewards(completion_id, token_rewards)         │      │
│  │  - set_position_rewards(completion_id, pos_rewards)  │      │
│  │  - compute_entropy(completion_id)                    │      │
│  │  - get_entropies(completion_id)                      │      │
│  │  - get_last_interaction()                            │      │
│  └──────────────────────┬───────────────────────────────┘      │
│                         │ HTTP                                   │
│  ┌──────────────────────▼───────────────────────────────┐      │
│  │    Token Reward Proxy Server (TokenRewardSessionData) │      │
│  │  Inherited endpoints:                                │      │
│  │  - /rl/set_reward (scalar)                           │      │
│  │  NEW endpoints:                                      │      │
│  │  - /rl/set_token_rewards                             │      │
│  │  - /rl/set_position_rewards                          │      │
│  │  - /rl/compute_entropy                               │      │
│  │  NEW serialization:                                  │      │
│  │  - serialize/deserialize_interactions_with_position_ │      │
│  │    rewards                                           │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Additional modules (client-side):                              │
│  ┌──────────────────────┐  ┌───────────────────────────────┐   │
│  │ InteractionCache     │  │ InteractionWithTokenLevelReward│   │
│  │ (cache.py)           │  │ (types.py)                    │   │
│  │ - set_rewards()      │  │ - token_rewards, token_reward │   │
│  │ - set_position_      │  │   _mask                       │   │
│  │   rewards()          │  │ - set_token_rewards()         │   │
│  │ - compute_and_store_ │  │ - set_sparse_token_rewards()  │   │
│  │   entropy()          │  │ - compute_entropy_from_       │   │
│  │ - export_interactions│  │   logprobs()                  │   │
│  └──────────────────────┘  └───────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## Key Differences Summary

### 1. Reward Granularity

| Aspect               | Base                    | Extended                                                      |
| -------------------- | ----------------------- | ------------------------------------------------------------- |
| **Scalar reward**    | `set_reward(id, float)` | Inherited                                                     |
| **Token rewards**    | Not supported           | `set_rewards(id, list[float])` via HTTP                       |
| **Position rewards** | Not supported           | `set_position_rewards(id, list[PositionRewardInfo])` via HTTP |
| **Entropy**          | Not supported           | `compute_entropy(id)` via HTTP                                |

### 2. Data Models

| Aspect                   | Base                             | Extended                                                     |
| ------------------------ | -------------------------------- | ------------------------------------------------------------ |
| **Interaction type**     | `InteractionWithTokenLogpReward` | `InteractionWithTokenLevelReward` (extends base)             |
| **Token reward storage** | Not supported                    | `token_rewards: list[float]`, `token_reward_mask: list[int]` |
| **Position info**        | Not supported                    | `PositionRewardInfo` with candidates, logprobs, rewards      |
| **Serialization**        | `serialize_interactions()`       | `serialize_interactions_with_position_rewards()`             |

### 3. Workflow Simplification

| Aspect              | Base                              | Extended                        |
| ------------------- | --------------------------------- | ------------------------------- |
| **Modes**           | inline, subproc, online           | Single mode (no gateway/online) |
| **Gateway**         | `proxy_gateway.py` routing        | Not included                    |
| **Online agent**    | `_OnlineAgent` for external users | Not included                    |
| **Reward handling** | Manual `set_reward()` calls       | Automatic `_process_rewards()`  |
| **Constructor**     | 8 parameters                      | 5 parameters (simplified)       |

### 4. Local Cache (Extended Only)

The extended version adds a local `InteractionCache` (`cache.py`) that mirrors the
server-side `TokenRewardSessionData`:

| Cache Method                  | Server-Side Equivalent                          | Purpose                     |
| ----------------------------- | ----------------------------------------------- | --------------------------- |
| `set_rewards()`               | `TokenRewardSessionData.set_token_rewards()`    | Store token rewards         |
| `set_position_rewards()`      | `TokenRewardSessionData.set_position_rewards()` | Store position rewards      |
| `compute_and_store_entropy()` | `TokenRewardSessionData.compute_entropy()`      | Compute & cache entropy     |
| `export_interactions()`       | `TokenRewardSessionData.export_interactions()`  | Export with rewards applied |

______________________________________________________________________

## Workflow Usage Comparison

### Base Workflow

```python
from areal.experimental.openai.proxy import OpenAIProxyClient, OpenAIProxyWorkflow

# Three modes available
workflow = OpenAIProxyWorkflow(
    mode="inline",  # or "subproc" or "online"
    agent=my_agent,
    proxy_addr="http://localhost:8000",
    admin_api_key="key",
    discount=1.0,
    export_style="individual",
    proxy_gateway_addr=None,  # for online mode
)

# Scalar reward only
async with client:
    await client.set_reward(completion.id, 1.0)

interactions = await client.export_interactions()
```

### Extended Workflow

```python
from customized_areal.on_policy_distill.proxy.client import OpenAIProxyClient
from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow

# Simplified constructor, no mode selection
workflow = OpenAIProxyWorkflow(
    agent=my_agent,
    proxy_addr="http://localhost:8000",
    admin_api_key="key",
    discount=1.0,
    export_style="individual",
)

# Token-level reward support
async with client:
    # Scalar (inherited)
    await client.set_reward(completion.id, 1.0)

    # Token-wise (NEW)
    await client.set_rewards(completion.id, [0.5, 0.3, 0.2])

    # Position-wise (NEW)
    await client.set_position_rewards(completion.id, [
        PositionRewardInfo(position=0, candidates=["a", "b"],
                          rewards=[0.1, 0.5], chosen_index=1)
    ])

    # Entropy computation (NEW)
    entropies = await client.compute_entropy(completion.id)

interactions = await client.export_interactions()
```

______________________________________________________________________

## Migration Path

From base to extended:

1. **Client**: Replace `from areal.experimental.openai.proxy import OpenAIProxyClient`
   with `from customized_areal.on_policy_distill.proxy.client import OpenAIProxyClient`.
   Existing `set_reward()` / `export_interactions()` calls work unchanged.

1. **Workflow**: Replace base `OpenAIProxyWorkflow` with extended version. Remove `mode`
   and `proxy_gateway_addr` params. Add `_process_rewards` hook if agent returns
   structured reward data.

1. **Server**: Use `customized_areal.on_policy_distill.proxy.proxy_rollout_server`
   instead of base. All existing endpoints are preserved; 3 new endpoints are added.

1. **No gateway/online mode**: If using `proxy_gateway.py` or `_OnlineAgent`, these are
   not available in the extended version.

1. **New types**: Use `InteractionWithTokenLevelReward` (from `types.py`) when you need
   token-level rewards. It's a drop-in extension of `InteractionWithTokenLogpReward`.
