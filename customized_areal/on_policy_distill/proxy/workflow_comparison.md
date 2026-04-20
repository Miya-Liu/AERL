# Workflow Comparison: Base vs. Customized Token-Reward Variant

## Files Compared

| File                                                   | Purpose                                                                                                                                     |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/experimental/openai/proxy/workflow.py`          | Base `OpenAIProxyWorkflow` — supports inline, subproc, and online modes with scalar rewards only.                                           |
| `customized_areal/on_policy_distill/proxy/workflow.py` | Customized `OpenAIProxyWorkflow` — adds token-level and position-level rewards, hardcodes inline mode, and uses a dual-client architecture. |

______________________________________________________________________

## High-Level Structural Differences

| Aspect                             | Base Workflow                                                                            | Customized Workflow                                                                                                                                                                                                   |
| ---------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Inheritance**                    | Extends `RolloutWorkflow` directly.                                                      | Extends `BaseOpenAIProxyWorkflow` (the base class above).                                                                                                                                                             |
| **Supported Modes**                | `inline`, `subproc`, `online` (runtime switchable).                                      | Hardcodes `mode="inline"` in `__init__`; other modes are dead code.                                                                                                                                                   |
| **Process-pool helpers**           | Module-level `_get_executor`, `_shutdown_executor`, `_wrap_run`.                         | Static methods `_get_executor` and `_wrap_run_subproc` that delegate back to the module-level helpers in the base file.                                                                                               |
| **Clients used in `arun_episode`** | Single `OpenAIProxyClient` (HTTP) from `areal.experimental.openai.proxy.client_session`. | **Two clients**:<br>1. `RealOpenAIProxyClient` — talks to the proxy server over HTTP.<br>2. `OpenAIProxyClient` (local) from `..proxy.client` — holds an in-memory `InteractionCache` for reward tensor manipulation. |
| **Reward types**                   | `float` or `dict[str, float]` (scalar only).                                             | `float`, `dict[str, float]`, `dict[str, list[float]]` (token-level), or `dict[str, dict]` with `"position_rewards"`.                                                                                                  |
| **Agent kwarg injection**          | Passes `base_url`, `http_client`, `api_key`.                                             | Additionally injects `proxy_client=<local client>` so the agent can call reward APIs directly.                                                                                                                        |
| **Example agent**                  | None in the workflow file.                                                               | `TokenRewardExampleAgent` moved to `examples/token_reward_examples.py` (was previously in the workflow file).                                                                                                         |

______________________________________________________________________

## `arun_episode` Flow Comparison

### Base Workflow

1. Create `OpenAIProxyClient`.
1. `async with proxy_client:` (starts RL session).
1. Run agent inline/subproc/online.
1. Apply scalar rewards via `proxy_client.set_reward` / `set_last_reward`.
1. Exit context (`__aexit__` ends RL session on server).
1. `proxy_client.export_interactions()` fetches trajectory.
1. Record stats and return.

### Customized Workflow

1. Create `RealOpenAIProxyClient`.
1. `async with real_client:` (starts RL session on proxy server).
1. Create a local `OpenAIProxyClient` and call `await client.__aenter__()`.
1. Run agent, passing both the real session key and the local `proxy_client`.
1. Exit `real_client` context.
1. `real_client.export_interactions()` fetches the real trajectory.
1. Copy every interaction from the real response into `client._cache` by direct
   assignment.
1. Process agent-returned rewards (token-level, scalar, or position-level) on the local
   cache.
1. `client.export_interactions()` runs discounting/filtering and returns the final dict.
1. Close the local client in a `finally` block.

______________________________________________________________________

## Bugs Fixed in the Customized Workflow

### 1. Double `__aexit__` + Local Client Closed Too Early

**Status**: Fixed **What changed**: Restructured `arun_episode` so the local
`OpenAIProxyClient` stays open until after `_process_rewards` and `export_interactions`
finish. Removed the duplicate `__aexit__` call in the `except` block.

### 2. Missing `loss_mask` Guard

**Status**: Fixed **What changed**: `_process_rewards` now falls back to
`client.set_rewards` when `interaction._cache` exists but does **not** contain
`"loss_mask"`, preventing a `KeyError` crash.

### 3. Silent Acceptance of `None` Rewards

**Status**: Fixed **What changed**: The workflow now explicitly raises
`ValueError("Agent returned None rewards")` instead of silently skipping reward
assignment. This matches the base class behavior and prevents training on unassigned
rewards.

### 4. Missing `@trace_session` Decorator

**Status**: Fixed **What changed**: Added `@trace_session("run_agent")` back to the
overridden `_run_agent` method.

### 5. Length Validation Mismatch Between Custom and Cache Paths

**Status**: Fixed **What changed**: `InteractionCache.set_rewards` now pads/truncates
token rewards to match `expected_len` (derived from `model_response.output_tokens`)
instead of raising `ValueError`. This makes the fallback path in `_process_rewards`
behave consistently with `_set_token_level_rewards`.

### 6. Example Agent Embedded in Production Workflow File

**Status**: Fixed **What changed**: Moved `TokenRewardExampleAgent` from
`proxy/workflow.py` to `examples/token_reward_examples.py`.

______________________________________________________________________

## Remaining Potential Bugs & Risks in the Customized Workflow

### 1. Dead Code Paths for `subproc` and `online` Modes

**Location**: `customized_areal/on_policy_distill/proxy/workflow.py:102–147`

The constructor hardcodes `mode="inline"`, yet `_run_agent` still contains branches for
`subproc` and `online`. These are unreachable in normal usage. The `subproc` branch
cannot inject the local `proxy_client`, and the `online` branch passes `proxy_client` to
an agent that is never instantiated via this constructor.

**Recommendation**: Remove the dead branches or document that `proxy_client` injection
is only supported in inline mode.

### 2. `messages` Validation on Import

**Location**: `customized_areal/on_policy_distill/proxy/workflow.py:351–352`

When importing real interactions into the local cache, `InteractionCache.__setitem__`
requires `value.messages` to be non-None. If the proxy server ever returns an
interaction without `messages`, the assignment will raise `ValueError`.

**Recommendation**: Add a defensive check when iterating over `real_interactions` and
skip or warn on malformed entries.

### 3. `real_client.export_interactions` Called After Context Exit — Intentional but Fragile

**Location**: `customized_areal/on_policy_distill/proxy/workflow.py:336–342`

This matches the base-class pattern, but it creates a hard dependency on `session_id`
being retained on the Python object after `__aexit__`. If the real client is ever
refactored to clear `session_id` in `__aexit__`, this will break.

**Recommendation**: Add a code comment explaining why the export happens outside the
`async with` block.

______________________________________________________________________

## Bugs Fixed in Agents / Scripts

### `customized_areal/on_policy_distill/core/agent.py`

#### A. `OnPolicyDistillAgent.run` Silently Swallowed Exceptions and Returned `0.0`

**Status**: Fixed **What changed**: The `except` block now re-raises the exception
instead of returning `0.0`. The workflow can now properly reject failed trajectories.

#### B. `http_client` from `extra_kwargs` Was Never Passed to `run_backend`

**Status**: Fixed **What changed**: Uncommented `http_client=http_client` in the
`run_backend` call so LLM calls route through AReaL's proxy session for token-level
tracking.

#### C. Non-Deterministic `completion_id` Fallback Used `hash()`

**Status**: Fixed **What changed**: Replaced `str(hash(str(metadata)))` with
`hashlib.md5(str(metadata).encode()).hexdigest()[:16]` for stable, deterministic IDs
across runs and workers.

#### D. `_convert_to_position_rewards` Skipped Positions with Empty `top_k_rewards`

**Status**: Fixed **What changed**: Instead of skipping empty `top_k_rewards`, the
converter now emits a fallback `PositionRewardInfo` using the actual token as the sole
candidate, preserving contiguous positions.

#### E. `candidates` Field Populated with Stringified Token IDs Instead of Token Strings

**Status**: Fixed **What changed**: `candidates` now uses
`tkr.get("token", tkr.get("token_id", ""))` so actual token text is preferred over
stringified IDs.

#### F. `run_backend` Called with `task_file_path=[]`

**Status**: Unchanged **What changed**: None — still passing `task_file_path=[]`. This
remains a potential contract issue if `run_backend` expects a string or `None`.

______________________________________________________________________

### `customized_areal/on_policy_distill/scripts/train_with_agent.py`

#### G. Typo: `ser=tokenizer` Instead of `tokenizer=tokenizer`

**Status**: Fixed **What changed**: Corrected the keyword argument from `ser=tokenizer`
to `tokenizer=tokenizer` in the `get_custom_dataset` call for the validation set.

______________________________________________________________________

### `customized_areal/on_policy_distill/examples/token_reward_examples.py`

#### H. Unsupported Reward Format in `SparseRewardAgent`

**Status**: Fixed **What changed**: `SparseRewardAgent` now returns
`{completion_id: token_rewards}` directly instead of the unsupported nested
`{"rewards": ..., "mask": ...}` dict.

#### I. Broken Import Path in `main()`

**Status**: Fixed **What changed**: Updated
`from customized_areal.token_reward import OpenAIProxyWorkflow` to
`from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow`.

#### J. `TokenRewardExampleAgent` Embedded in Production Workflow File

**Status**: Fixed **What changed**: Moved `TokenRewardExampleAgent` from
`proxy/workflow.py` into `examples/token_reward_examples.py`.

______________________________________________________________________

## Summary Table of All Issues

### Workflow Issues (Post-Fix — Remaining)

| #   | Issue                                     | Severity | Category      |
| --- | ----------------------------------------- | -------- | ------------- |
| W1  | Dead code for subproc/online modes        | Low      | Maintenance   |
| W2  | `messages` validation on import           | Medium   | Runtime Error |
| W3  | Export after context exit (needs comment) | Low      | Fragility     |

### Agent / Script Issues (Post-Fix — Remaining)

| #   | Issue                                    | Severity | Category     |
| --- | ---------------------------------------- | -------- | ------------ |
| F   | `task_file_path=[]` may violate contract | Low      | API Contract |
