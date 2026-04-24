# Workflow Comparison: Base vs. Customized Token-Reward Variant

## Files Compared

| File                                                   | Purpose                                                                                                                                                                                    |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `areal/experimental/openai/proxy/workflow.py`          | Base `OpenAIProxyWorkflow` — supports inline, subproc, and online modes with scalar rewards only.                                                                                          |
| `customized_areal/on_policy_distill/proxy/workflow.py` | Customized `OpenAIProxyWorkflow` — adds token-level and position-level rewards via HTTP API, hardcodes inline mode, and returns a concatenated tensor dict with position_rewards attached. |

______________________________________________________________________

## High-Level Structural Differences

| Aspect                              | Base Workflow                                                                            | Customized Workflow                                                                                                                                                                          |
| ----------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Inheritance**                     | Extends `RolloutWorkflow` directly.                                                      | Extends `BaseOpenAIProxyWorkflow` (the base class above).                                                                                                                                    |
| **Supported Modes**                 | `inline`, `subproc`, `online` (runtime switchable).                                      | Hardcodes `mode="inline"` in `__init__`; other modes are dead code.                                                                                                                          |
| **Process-pool helpers**            | Module-level `_get_executor`, `_shutdown_executor`, `_wrap_run`.                         | Static methods `_get_executor` and `_wrap_run_subproc` that delegate back to the module-level helpers in the base file.                                                                      |
| **Client used in `arun_episode`**   | Single `OpenAIProxyClient` (HTTP) from `areal.experimental.openai.proxy.client_session`. | Single `OpenAIProxyClient` (extended) from `..proxy.client` — inherits from base client and adds `set_rewards`, `set_position_rewards`, `compute_entropy`, and custom `export_interactions`. |
| **Reward types**                    | `float` or `dict[str, float]` (scalar only).                                             | `float`, `dict[str, float]`, `dict[str, list[float]]` (token-level), or `dict[str, dict]` with `"position_rewards"` / `"scalar_reward"`.                                                     |
| **Agent kwarg injection**           | Passes `base_url`, `http_client`, `api_key`.                                             | Additionally injects `proxy_client=<extended client>` so the agent can call reward APIs directly during the session.                                                                         |
| **Return type of `arun_episode`**   | `dict[str, InteractionWithTokenLogpReward]` — raw interaction objects.                   | `dict[str, Any]` — concatenated tensor dict via `concat_padded_tensors`, with `position_rewards` list attached as a separate key.                                                            |
| **Reward processing**               | Inline in `arun_episode`: loops over `rewards.items()` and calls `set_reward`.           | Delegated to `_process_rewards()` method which handles all four reward types and calls the appropriate HTTP API endpoint.                                                                    |
| **Serialization / Deserialization** | Base `serialize_interactions` / `deserialize_interactions`.                              | Custom `serialize_interactions_with_position_rewards` / `deserialize_interactions_with_position_rewards` in `proxy_rollout_server.py` — preserves `position_rewards` through HTTP transport. |
| **Example agent**                   | None in the workflow file.                                                               | `TokenRewardExampleAgent` moved to `examples/token_reward_examples.py` (was previously in the workflow file).                                                                                |

______________________________________________________________________

## `arun_episode` Flow Comparison

### Base Workflow

1. Create `OpenAIProxyClient`.
1. `async with proxy_client:` (starts RL session on server).
1. Run agent inline/subproc/online via `_run_agent`.
1. Apply scalar rewards inline: loop over `rewards.items()` and call
   `proxy_client.set_reward`.
1. Exit context (`__aexit__` ends RL session on server).
1. `proxy_client.export_interactions()` fetches trajectory.
1. Record stats and return raw interaction dict.

### Customized Workflow

1. Create extended `OpenAIProxyClient`.
1. `async with client:` (starts RL session on server).
1. Run agent via `_run_agent`, passing `proxy_client=client` so the agent can call
   reward APIs directly.
1. `_process_rewards(client, rewards)` sends rewards to server via HTTP — supports
   scalar, token-level, and position-level rewards.
1. Exit context.
1. `client.export_interactions()` fetches trajectory with custom deserialization that
   reconstructs `position_rewards`.
1. Convert interactions to concatenated tensor dict via `concat_padded_tensors`.
1. Collect `position_rewards` from all interactions, assign `sample_index` to each, and
   attach as `tensor_dict["position_rewards"]`.
1. Record stats and return tensor dict.

### Key Flow Difference: Return Type

The base workflow returns raw `dict[str, InteractionWithTokenLogpReward]`, leaving
tensor conversion to `workflow_executor`. The customized workflow performs tensor
conversion inside `arun_episode` so it can attach `position_rewards` as a separate key —
this avoids `concat_padded_tensors` key consistency issues when some interactions have
`position_rewards` and others don't (e.g., multi-turn conversations).

______________________________________________________________________

## Architecture Diagrams

### Base Architecture

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

### Customized Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   OpenAIProxyWorkflow (inline only)               │
│                         │                                       │
│           ┌─────────────┼─────────────┐                         │
│           │ _run_agent  │             │ _process_rewards         │
│           │ (injects    │             │ (scalar / token /        │
│           │  proxy_     │             │  position)               │
│           │  client)    │             │                          │
│           └─────┬───────┘             └──────────┐               │
│                 │                                │               │
│                 ▼                                ▼               │
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
│  │  Override:                                           │      │
│  │  - export_interactions (custom deserialization)      │      │
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
│  │  - serialize_interactions_with_position_rewards      │      │
│  │  - deserialize_interactions_with_position_rewards    │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  After export:                                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  arun_episode post-processing                        │      │
│  │  1. concat_padded_tensors → tensor_dict              │      │
│  │  2. Collect position_rewards with sample_index       │      │
│  │  3. Attach as tensor_dict["position_rewards"]        │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Testing/development only (not in production path):             │
│  ┌──────────────────────┐  ┌───────────────────────────────┐   │
│  │ InteractionCache     │  │ InteractionWithTokenLevelReward│   │
│  │ (cache.py — mock)    │  │ (types.py)                    │   │
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

## Bugs Fixed in the Customized Workflow

### 1. Double `__aexit__` + Local Client Closed Too Early

**Status**: Obsoleted. The previous dual-client architecture (real client + local cache
client) has been replaced with a single HTTP-only client. The bug no longer applies
because there is no local client to close prematurely.

### 2. Missing `loss_mask` Guard

**Status**: Obsoleted. The fallback to `client.set_rewards` when `interaction._cache`
lacks `"loss_mask"` was needed for the local cache path. With the HTTP-only
architecture, rewards are sent directly to the server, which handles token-level rewards
independently of the local cache's `_cache` dict.

### 3. Silent Acceptance of `None` Rewards

**Status**: Fixed. The workflow now explicitly raises when the agent fails (the `except`
block in `arun_episode` re-raises). When `rewards is None` after a successful agent run,
the `_process_rewards` call is simply skipped (no reward assignment), which is correct —
the agent chose not to assign rewards.

### 4. Missing `@trace_session` Decorator

**Status**: Fixed. `@trace_session("run_agent")` is present on the overridden
`_run_agent` method.

### 5. Length Validation Mismatch Between Custom and Cache Paths

**Status**: Obsoleted. The local cache's `set_rewards` length validation mismatch is
irrelevant in the HTTP-only path. The server validates token reward length against
`output_tokens` in the `/rl/set_token_rewards` endpoint.

### 6. Example Agent Embedded in Production Workflow File

**Status**: Fixed. `TokenRewardExampleAgent` is in `examples/token_reward_examples.py`.

### 7. Agent kwarg `http_client` Not Passed to `run_backend`

**Status**: Fixed. The agent's `run` method now receives `http_client` via
`extra_kwargs` in inline mode, ensuring LLM calls route through AReaL's proxy session
for token-level tracking.

______________________________________________________________________

## Remaining Potential Bugs & Risks in the Customized Workflow

### 1. Dead Code Paths for `subproc` and `online` Modes

**Location**: `customized_areal/on_policy_distill/proxy/workflow.py:128–163`

The constructor hardcodes `mode="inline"`, yet `_run_agent` still contains branches for
`subproc` and `online`. These are unreachable in normal usage. The `subproc` branch
cannot inject the local `proxy_client`, and the `online` branch passes `proxy_client` to
an agent that is never instantiated via this constructor.

**Recommendation**: Remove the dead branches or document that `proxy_client` injection
is only supported in inline mode.

### 2. `export_interactions` Called After Context Exit — Intentional but Fragile

**Location**: `customized_areal/on_policy_distill/proxy/workflow.py:294–297`

This matches the base-class pattern, but it creates a hard dependency on `session_id`
being retained on the Python object after `__aexit__`. If the real client is ever
refactored to clear `session_id` in `__aexit__`, this will break.

**Recommendation**: Add a code comment explaining why the export happens outside the
`async with` block.

### 3. `position_rewards` Sample Index Assumes `export_style="individual"`

**Location**: `customized_areal/on_policy_distill/proxy/workflow.py:330–336`

When assigning `sample_index` to each `PositionRewardInfo`, the code iterates
`enumerate(interactions.values())`. This is correct for `export_style="individual"`
where each interaction becomes a separate batch item. For `export_style="concat"`,
positions from subsequent interactions would need cumulative offset adjustment based on
prior interactions' output lengths.

**Recommendation**: Add a runtime guard or warning when `export_style="concat"` is used
with position rewards, or document this limitation in the constructor docstring.

### 4. `PositionRewardInfo` Defined in Two Places

**Location**: `server.py` (Pydantic `BaseModel`) and `cache.py` (dataclass with
`sample_index` field)

The Pydantic version in `server.py` is used for HTTP request/response serialization and
server-side storage. The dataclass version in `cache.py` adds a `sample_index` field
used during batch processing. The two are not interchangeable — the server's
`set_position_rewards` endpoint converts Pydantic models to dataclass instances before
storing. This dual definition could cause confusion and subtle bugs if the schemas
drift.

**Recommendation**: Consolidate into a single definition (e.g., the dataclass in
`cache.py`) and derive the Pydantic model from it, or add a shared base.

______________________________________________________________________

## Bugs Fixed in Agents / Scripts

### `customized_areal/on_policy_distill/core/agent.py`

#### A. `OnPolicyDistillAgent.run` Silently Swallowed Exceptions and Returned `0.0`

**Status**: Fixed. The `except` block now re-raises the exception instead of returning
`0.0`. The workflow can now properly reject failed trajectories.

#### B. `http_client` from `extra_kwargs` Was Never Passed to `run_backend`

**Status**: Fixed. `http_client=http_client` is now passed in the `run_backend` call so
LLM calls route through AReaL's proxy session for token-level tracking.

#### C. Non-Deterministic `completion_id` Fallback Used `hash()`

**Status**: Fixed. Replaced `str(hash(str(metadata)))` with
`hashlib.md5(str(metadata).encode()).hexdigest()[:16]` for stable, deterministic IDs
across runs and workers.

#### D. `_convert_to_position_rewards` Skipped Positions with Empty `top_k_rewards`

**Status**: Fixed. Instead of skipping empty `top_k_rewards`, the converter now emits a
fallback `PositionRewardInfo` using the actual token as the sole candidate, preserving
contiguous positions.

#### E. `candidates` Field Populated with Stringified Token IDs Instead of Token Strings

**Status**: Fixed. `candidates` now uses `tkr.get("token", tkr.get("token_id", ""))` so
actual token text is preferred over stringified IDs.

#### F. `run_backend` Called with `task_file_path=[]`

**Status**: Unchanged. Still passing `task_file_path=[]`. This remains a potential
contract issue if `run_backend` expects a string or `None`.

______________________________________________________________________

### `customized_areal/on_policy_distill/scripts/train_with_agent.py`

#### G. Typo: `ser=tokenizer` Instead of `tokenizer=tokenizer`

**Status**: Fixed. Corrected the keyword argument from `ser=tokenizer` to
`tokenizer=tokenizer` in the `get_custom_dataset` call for the validation set.

______________________________________________________________________

### `customized_areal/on_policy_distill/examples/token_reward_examples.py`

#### H. Unsupported Reward Format in `SparseRewardAgent`

**Status**: Fixed. `SparseRewardAgent` now returns `{completion_id: token_rewards}`
directly instead of the unsupported nested `{"rewards": ..., "mask": ...}` dict.

#### I. Broken Import Path in `main()`

**Status**: Fixed. Updated
`from customized_areal.token_reward import OpenAIProxyWorkflow` to
`from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow`.

#### J. `TokenRewardExampleAgent` Embedded in Production Workflow File

**Status**: Fixed. Moved `TokenRewardExampleAgent` from `proxy/workflow.py` into
`examples/token_reward_examples.py`.

______________________________________________________________________

## Summary Table of All Issues

### Workflow Issues (Post-Fix — Remaining)

| #   | Issue                                                    | Severity | Category    |
| --- | -------------------------------------------------------- | -------- | ----------- |
| W1  | Dead code for subproc/online modes                       | Low      | Maintenance |
| W2  | Export after context exit (needs comment)                | Low      | Fragility   |
| W3  | `position_rewards` sample_index assumes individual style | Medium   | Correctness |
| W4  | `PositionRewardInfo` defined in two places               | Medium   | Maintenance |

### Workflow Issues (Post-Fix — Obsoleted by Architecture Change)

| #   | Issue                                          | Original Severity | Reason Obsoleted                             |
| --- | ---------------------------------------------- | ----------------- | -------------------------------------------- |
| O1  | Double `__aexit__` + local client closed early | High              | Dual-client replaced with single HTTP client |
| O2  | Missing `loss_mask` guard in local cache       | Medium            | No local cache fallback in production path   |
| O3  | Length validation mismatch (cache vs server)   | Medium            | Server validates length; no local cache path |

### Agent / Script Issues (Post-Fix — Remaining)

| #   | Issue                                    | Severity | Category     |
| --- | ---------------------------------------- | -------- | ------------ |
| F   | `task_file_path=[]` may violate contract | Low      | API Contract |

______________________________________________________________________

## Migration Path (Base → Customized)

1. **Client**: Replace `from areal.experimental.openai.proxy import OpenAIProxyClient`
   with `from customized_areal.on_policy_distill.proxy.client import OpenAIProxyClient`.
   Existing `set_reward()` / `export_interactions()` calls work unchanged. New methods
   (`set_rewards`, `set_position_rewards`, `compute_entropy`) are available immediately.

1. **Workflow**: Replace base `OpenAIProxyWorkflow` with extended version. Remove `mode`
   and `proxy_gateway_addr` params. The return type changes from
   `dict[str, InteractionWithTokenLogpReward]` to `dict[str, Any]` (concatenated tensor
   dict with `position_rewards`).

1. **Server**: Use `customized_areal.on_policy_distill.proxy.proxy_rollout_server`
   instead of base. All existing endpoints are preserved; 3 new endpoints are added
   (`/rl/set_token_rewards`, `/rl/set_position_rewards`, `/rl/compute_entropy`).
   Serialization uses `serialize_interactions_with_position_rewards` /
   `deserialize_interactions_with_position_rewards` to preserve `position_rewards`
   through HTTP transport.

1. **No gateway/online mode**: If using `proxy_gateway.py` or `_OnlineAgent`, these are
   not available in the customized version.

1. **Local cache (`cache.py`)**: Available for testing and development only. The
   production workflow path does not use `InteractionCache` — all reward operations go
   through HTTP to the proxy server.

1. **New types**: Use `InteractionWithTokenLevelReward` (from `types.py`) when you need
   token-level rewards. It's a drop-in extension of `InteractionWithTokenLogpReward`,
   adding `token_rewards`, `token_reward_mask`, and helper methods.
