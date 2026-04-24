# Proxy Rollout Server Comparison

Base: `areal/experimental/openai/proxy/proxy_rollout_server.py` Custom:
`customized_areal/on_policy_distill/proxy/proxy_rollout_server.py`

## High-Level Summary

The base server is a **full-featured OpenAI/Anthropic-compatible inference proxy** that
manages an inference engine, serves chat completions/responses/Anthropic-messages
endpoints, and handles distributed cluster registration. The customized server is a
**reward-focused subset** that strips all inference capabilities and adds token-level
and position-level reward endpoints for distillation training.

## Architecture Differences

| Aspect              | Base                                                                 | Custom                                         |
| ------------------- | -------------------------------------------------------------------- | ---------------------------------------------- |
| Primary role        | Inference proxy + RL session manager                                 | RL session manager + token-level reward server |
| Engine management   | Full (`/create_engine`, `/call`, engine lifecycle)                   | None                                           |
| Inference endpoints | Chat completions, Responses, Anthropic messages                      | None                                           |
| Cluster integration | name_resolve registration, Slurm `PROCID` detection, port allocation | None                                           |
| FastAPI title       | (default)                                                            | `"Token Reward Proxy Server"`                  |
| Lines of code       | ~1033                                                                | ~545                                           |

## Session Data

| Aspect              | Base          | Custom                                                                                                                     |
| ------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Session data class  | `SessionData` | `TokenRewardSessionData` (extends `SessionData`)                                                                           |
| Token rewards       | Not supported | `set_token_rewards()` per interaction                                                                                      |
| Position rewards    | Not supported | `set_position_rewards()` with candidate info per position                                                                  |
| Entropy computation | Not supported | `compute_entropy()` from position logprobs                                                                                 |
| Reward separation   | N/A           | Scalar reward (set via `set_reward`) is preserved separately from token/position rewards (used only for distillation loss) |

## Endpoints

### Present in Both

| Endpoint              | Differences                                                                                                                                                                 |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rl/start_session`    | Base: capacity-gated, API key reuse support, stale cleanup on-demand. Custom: no capacity check, always generates new key with `tr-session-` prefix, UUID-based session IDs |
| `rl/end_session`      | Base: returns `interaction_count`. Custom: returns only `{"message": "success"}`                                                                                            |
| `rl/set_reward`       | Functionally identical                                                                                                                                                      |
| `grant_capacity`      | Base: returns `{"capacity": N}`. Custom: returns `{"message": "success", "capacity": N}`                                                                                    |
| `export_trajectories` | Base: async `wait_for_finish()`, removes session from cache after export. Custom: returns empty if not completed (client retries), no cache removal                         |

### Only in Base

| Endpoint            | Purpose                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| `/health`           | Health check with initialization status                                |
| `/alloc_ports`      | Allocate free ports (Slurm fork_workers)                               |
| `/configure`        | Deserialize and apply config, set random seed                          |
| `/set_env`          | Set environment variables                                              |
| `/create_engine`    | Create inference engine instance                                       |
| `/call`             | Call engine methods (including `initialize`)                           |
| `/chat/completions` | OpenAI-compatible chat completions (streaming + non-streaming)         |
| `/responses`        | OpenAI-compatible responses endpoint                                   |
| `/v1/messages`      | Anthropic Messages API compatible endpoint (streaming + non-streaming) |

### Only in Custom

| Endpoint                  | Purpose                                                                           |
| ------------------------- | --------------------------------------------------------------------------------- |
| `rl/set_token_rewards`    | Set per-token rewards for an interaction (validates length matches output tokens) |
| `rl/set_position_rewards` | Set per-position candidate rewards (converts Pydantic models to dataclasses)      |
| `rl/compute_entropy`      | Compute entropy from position reward logprobs                                     |

## Authentication

| Aspect             | Base                                                            | Custom                                                            |
| ------------------ | --------------------------------------------------------------- | ----------------------------------------------------------------- |
| Bearer token       | Yes (case-insensitive per RFC 6750)                             | Yes (case-sensitive, `Authorization: Bearer ...` only)            |
| `x-api-key` header | Supported (Anthropic SDK compat)                                | Not supported                                                     |
| Token comparison   | `hmac.compare_digest` (timing-attack resistant)                 | `!=` (vulnerable to timing attacks)                               |
| Admin key init     | Random `secrets.token_urlsafe(32)`, overwritten on engine setup | Random `secrets.token_urlsafe(32)`, set via CLI `--admin-api-key` |

## Stale Session Cleanup

| Aspect                    | Base                                                                            | Custom                                                   |
| ------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Trigger                   | On-demand during `start_session` (at most once per minute)                      | Async background task every 60s                          |
| Implementation            | Synchronous `_cleanup_stale_sessions()` called with `_lock` held                | `asyncio.create_task` loop in `@app.on_event("startup")` |
| Orphaned key sweep        | Yes (handles client crash after `end_session` but before `export_trajectories`) | No                                                       |
| Session finish on cleanup | No (only removes from cache)                                                    | Yes (calls `session_data.finish()`)                      |

## Serialization

| Aspect                       | Base                                                           | Custom                                                                                                                                        |
| ---------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Export format                | `serialize_interactions` (tensor_dict, reward, interaction_id) | `serialize_interactions_with_position_rewards` (adds position_rewards with candidates, token_ids, logprobs, rewards, chosen_index)            |
| Deserialization              | `deserialize_interactions`                                     | `deserialize_interactions_with_position_rewards` (reconstructs `PositionRewardInfo` dataclasses, attaches as Python attribute on interaction) |
| Position rewards in pipeline | N/A                                                            | Stored as Python attribute (not injected into `_cache`) to avoid `concat_padded_tensors` key consistency issues                               |

## CLI & Startup

| Aspect         | Base                                                                    | Custom                      |
| -------------- | ----------------------------------------------------------------------- | --------------------------- |
| Required args  | `--experiment-name`, `--trial-name`, `--role`                           | None                        |
| Port handling  | Auto-assign via `find_free_ports` if `--port 0`, tracks allocated ports | Fixed `--port 8000` default |
| name_resolve   | Full integration (reconfigure + register)                               | None                        |
| Slurm support  | `SLURM_PROCID` overrides `--worker-index`                               | None                        |
| uvicorn config | `log_level="warning"`, `timeout_keep_alive=300`                         | Defaults only               |
| Shutdown       | `cleanup_engine()` destroys engine on exit                              | No cleanup                  |

## Notable Gaps in Custom Server

### Fixed (2026-04-23)

1. ~~**No capacity enforcement**~~ — Fixed: `start_session` now checks `_capacity` and
   returns HTTP 429 when exhausted.
1. ~~**Timing-attack vulnerability**~~ — Fixed: admin key comparison now uses
   `hmac.compare_digest`.
1. ~~**No `x-api-key` header support**~~ — Fixed: added `_extract_bearer_token` with
   `x-api-key` fallback for Anthropic SDK.
1. ~~**No orphaned key cleanup**~~ — Fixed: stale session cleanup now sweeps orphaned
   API key mappings.
1. ~~**No session removal on export**~~ — Fixed: `export_trajectories` now removes
   session from cache and cleans up keys after export.
1. ~~**Deprecated startup event**~~ — Fixed: replaced `@app.on_event("startup")` with
   `lifespan` context manager.
1. ~~**No warning deduplication**~~ — Fixed: added `_warn_once` with
   `AREAL_PROXY_WARN_ONCE` env var support.

### Remaining (by design / out of scope)

1. **No inference capability** — cannot serve as a standalone proxy; must be paired with
   a separate inference server or used only for reward/session management. This is
   intentional — the custom server is reward-focused only.
1. **No cluster integration** — no name_resolve, Slurm, or port allocation. Out of scope
   for the reward-server use case.
