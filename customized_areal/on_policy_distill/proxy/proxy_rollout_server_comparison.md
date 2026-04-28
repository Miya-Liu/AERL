# Proxy Rollout Server Comparison

Base: `areal/experimental/openai/proxy/proxy_rollout_server.py` Custom:
`customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`

## High-Level Summary

The custom server is a **token-level reward extension** of the base proxy rollout
server. It retains all base functionality (inference proxy, engine management, cluster
integration, OpenAI/Anthropic-compatible endpoints) and adds token-level and
position-level reward endpoints for on-policy distillation training. The core
architectural difference is the use of `TokenRewardSessionData` (from the local
`server.py`) instead of `SessionData`, plus custom serialization that preserves
`token_rewards` and `position_rewards` through HTTP transport.

## Architecture Differences

| Aspect             | Base                          | Custom                                                             |
| ------------------ | ----------------------------- | ------------------------------------------------------------------ |
| Primary role       | Inference proxy + RL sessions | Inference proxy + RL sessions + token-level reward server          |
| Session data class | `SessionData`                 | `TokenRewardSessionData` (from `.server`, extends `SessionData`)   |
| Serialization      | `serialize_interactions`      | `serialize_interactions_with_position_rewards` (custom, see below) |
| Deserialization    | `deserialize_interactions`    | `deserialize_interactions_with_position_rewards` (custom)          |
| FastAPI app        | `FastAPI()`                   | `FastAPI(title="Token Reward Proxy Server", lifespan=lifespan)`    |
| Logger name        | `ProxyRolloutServer`          | `TokenRewardProxyServer`                                           |
| Lines of code      | ~1033                         | ~1263                                                              |

## Session Data

| Aspect              | Base          | Custom                                                                                                                     |
| ------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Session data class  | `SessionData` | `TokenRewardSessionData` (extends `SessionData`)                                                                           |
| Token rewards       | Not supported | `set_token_rewards()` per interaction                                                                                      |
| Position rewards    | Not supported | `set_position_rewards()` with candidate info per position                                                                  |
| Entropy computation | Not supported | `compute_entropy()` from position logprobs                                                                                 |
| Reward separation   | N/A           | Scalar reward (set via `set_reward`) is preserved separately from token/position rewards (used only for distillation loss) |

## Endpoints

### Present in Both (Identical Logic)

| Endpoint                 | Notes                                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `/health`                | Identical                                                                                                                  |
| `/alloc_ports`           | Identical                                                                                                                  |
| `/configure`             | Identical                                                                                                                  |
| `/set_env`               | Identical                                                                                                                  |
| `/create_engine`         | Identical                                                                                                                  |
| `/call`                  | Identical                                                                                                                  |
| `/chat/completions`      | Identical (streaming + non-streaming)                                                                                      |
| `/responses`             | Identical                                                                                                                  |
| `/v1/messages`           | Identical (streaming + non-streaming, LiteLLM adapter)                                                                     |
| `rl/set_reward`          | Custom adds `is_completed` guard (returns HTTP 409 if session already finished) and logs the reward value                  |
| `rl/grant_capacity`      | Base returns `{"capacity": N}`. Custom returns `{"message": "success", "capacity": N}`                                     |
| `rl/export_trajectories` | Custom uses `serialize_interactions_with_position_rewards` instead of `serialize_interactions`; also adds logging of count |

### Present in Both (Behavioral Differences)

| Endpoint           | Base                                                    | Custom                                                                      |
| ------------------ | ------------------------------------------------------- | --------------------------------------------------------------------------- |
| `rl/start_session` | `dependencies=[Depends(_require_admin_key)]`            | `admin_key: str = Depends(_require_admin_key)` parameter                    |
|                    | On-demand `_cleanup_stale_sessions()` with `_lock` held | No on-demand cleanup (handled by background task)                           |
|                    | Session ID: `{task_id}-{idx}`                           | Session ID: `tr-{uuid.uuid4().hex[:16]}`                                    |
|                    | API key: `secrets.token_urlsafe(32)` + collision loop   | API key: `tr-session-{secrets.token_urlsafe(32)}` via `_generate_api_key()` |
|                    | `SessionData(session_id=session_id)`                    | `TokenRewardSessionData(session_id)` (no keyword arg)                       |
| `rl/end_session`   | Returns `interaction_count`, does NOT remove API keys   | Returns `interaction_count`, ALSO removes API key mappings after `finish()` |

### Only in Custom

| Endpoint                  | Purpose                                                                           |
| ------------------------- | --------------------------------------------------------------------------------- |
| `rl/set_token_rewards`    | Set per-token rewards for an interaction (validates length matches output tokens) |
| `rl/set_position_rewards` | Set per-position candidate rewards (converts Pydantic models to dataclasses)      |
| `rl/compute_entropy`      | Compute entropy from position reward logprobs                                     |

## Authentication

Identical in both servers:

| Aspect             | Implementation                                                              |
| ------------------ | --------------------------------------------------------------------------- |
| Bearer token       | Case-insensitive `Authorization: Bearer ...` per RFC 6750                   |
| `x-api-key` header | Supported (Anthropic SDK compat)                                            |
| Token comparison   | `hmac.compare_digest` (timing-attack resistant)                             |
| Admin key init     | Random `secrets.token_urlsafe(32)`, overwritten by `_setup_openai_client()` |

## Stale Session Cleanup

| Aspect                    | Base                                                             | Custom                                                                    |
| ------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Trigger                   | On-demand during `start_session` (at most once per minute)       | Async background task every 60s (via `lifespan`)                          |
| Implementation            | Synchronous `_cleanup_stale_sessions()` called with `_lock` held | `asyncio.create_task` loop in `lifespan` context manager                  |
| Orphaned key sweep        | Yes                                                              | Yes                                                                       |
| Session finish on cleanup | No (only removes from cache)                                     | Yes (calls `session_data.finish()`)                                       |
| Lifecycle management      | Manual (called at start of `start_session`)                      | Automatic via `lifespan` (task created on startup, cancelled on shutdown) |

The custom approach (background task) is cleaner — stale sessions are cleaned
proactively even if no new sessions are started. The base approach only cleans when new
sessions arrive.

## Serialization

| Aspect                       | Base                                                           | Custom                                                                                                                                                                   |
| ---------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Export format                | `serialize_interactions` (tensor_dict, reward, interaction_id) | `serialize_interactions_with_position_rewards` (adds token_rewards, position_rewards with candidates, token_ids, logprobs, rewards, chosen_index, sample_index)          |
| Deserialization              | `deserialize_interactions`                                     | `deserialize_interactions_with_position_rewards` (reconstructs `InteractionWithTokenLevelReward` objects, `PositionRewardInfo` dataclasses attached as Python attribute) |
| Position rewards in pipeline | N/A                                                            | Stored as Python attribute (not injected into `_cache`) to avoid `concat_padded_tensors` key consistency issues                                                          |
| Token rewards handling       | N/A                                                            | Set via `InteractionWithTokenLevelReward.token_rewards` typed dataclass field so `to_tensor_dict()` includes it after cache invalidation                                 |

## CLI & Startup

| Aspect            | Base                            | Custom                                                       |
| ----------------- | ------------------------------- | ------------------------------------------------------------ |
| `--admin-api-key` | Not present (set via engine)    | Present (CLI arg, sets `_admin_api_key` before engine setup) |
| Other args        | Identical                       | Identical                                                    |
| name_resolve      | Identical                       | Identical                                                    |
| Slurm support     | Identical                       | Identical                                                    |
| Port handling     | Identical                       | Identical                                                    |
| uvicorn config    | Identical                       | Identical                                                    |
| Shutdown          | `cleanup_engine()` in `finally` | `cleanup_engine()` via `lifespan`                            |

## Import Differences

| Aspect                       | Base                                    | Custom                                                                                                                                                                   |
| ---------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `from contextlib import ...` | Not imported                            | `asynccontextmanager` imported                                                                                                                                           |
| `import time`                | Present                                 | Not imported (cleanup uses `asyncio.sleep` instead)                                                                                                                      |
| Server constants             | From `.server` (relative import)        | From `areal.experimental.openai.proxy.server` (absolute import)                                                                                                          |
| `serialize_interactions`     | From `.server`                          | Not imported (custom serialization functions instead)                                                                                                                    |
| `SESSION_TIMEOUT_SECONDS`    | Not imported                            | Imported from base server (used by background cleanup task)                                                                                                              |
| Local `.server` imports      | `SessionData`, `SetRewardRequest`, etc. | `TokenRewardSessionData`, `SetTokenRewardsRequest`, `SetPositionRewardsRequest`, `ComputeEntropyRequest`, `ComputeEntropyResponse`, `PositionRewardInfo`, path constants |

## Summary of Custom Extensions

The custom server adds three capabilities on top of the base:

1. **Token-level rewards** — `set_token_rewards` endpoint validates per-token reward
   lengths and stores them via `TokenRewardSessionData.set_token_rewards()`.

1. **Position-level rewards** — `set_position_rewards` endpoint stores candidate token
   rewards per position (used for distillation loss). Converts Pydantic request models
   to dataclass `PositionRewardInfo` objects for storage.

1. **Entropy computation** — `compute_entropy` endpoint computes per-position entropy
   from stored logprobs in position rewards.

1. **Custom serialization** — `serialize_interactions_with_position_rewards` and
   `deserialize_interactions_with_position_rewards` ensure token_rewards and
   position_rewards survive HTTP transport. Deserialization creates
   `InteractionWithTokenLevelReward` objects so `to_tensor_dict()` correctly recomputes
   token_rewards after cache invalidation.
