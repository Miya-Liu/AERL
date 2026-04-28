# Server Comparison: Base vs. Extended (On-Policy Distill)

## File Paths

- **Base**: `areal/experimental/openai/proxy/server.py`
- **Extended**: `customized_areal/on_policy_distill/proxy/server.py`

## Overview

The extended server inherits from and re-exports the base server, adding **token-level reward** and **position-level reward** support for on-policy distillation. The base server only supports scalar (trajectory-level) rewards.

## Structural Relationship

The extended server **imports and re-exports** all public symbols from the base server via `from areal.experimental.openai.proxy.server import ...`. It then adds new models, a new `SessionData` subclass, and new endpoint path constants on top.

## Detailed Differences

### 1. Request/Response Models

| Model | Base | Extended | Notes |
|---|---|---|---|
| `StartSessionRequest` | Yes | Re-exported | Identical |
| `StartSessionResponse` | Yes | Re-exported | Identical |
| `SetRewardRequest` | Yes | Re-exported | Identical (scalar reward) |
| `ExportTrajectoriesRequest` | Yes | Re-exported | Identical |
| `ExportTrajectoriesResponse` | Yes | Re-exported | Identical |
| `WaitForSessionRequest` | Yes | Not re-exported | Base only |
| `WaitForSessionResponse` | Yes | Not re-exported | Base only |
| `PositionRewardInfo` | No | **New** | Per-position candidate tokens, logprobs, rewards, chosen_index |
| `SetTokenRewardsRequest` | No | **New** | `interaction_id` + `token_rewards: list[float]` |
| `SetPositionRewardsRequest` | No | **New** | `interaction_id` + `position_rewards: list[PositionRewardInfo]` |
| `ComputeEntropyRequest` | No | **New** | `interaction_id` only |
| `ComputeEntropyResponse` | No | **New** | `entropies: list[float]` + `avg_entropy: float` |

### 2. SessionData vs. TokenRewardSessionData

#### Base: `SessionData`

- Stores completions in `InteractionCache` (from `areal.experimental.openai.cache`)
- Interaction type: `InteractionWithTokenLogpReward`
- Only scalar reward support via `set_reward()` on the cache
- `export_interactions()` calls `apply_reward_discount()` then `export_interactions(style=...)` on the base cache
- Has `wait_for_finish()` async method for session completion signaling

#### Extended: `TokenRewardSessionData(SessionData)`

Inherits from `SessionData` and overrides/replaces key components:

| Aspect | Base `SessionData` | Extended `TokenRewardSessionData` |
|---|---|---|
| Completions cache | `InteractionCache` (base) | `ExtendedInteractionCache` (from `customized_areal/.../proxy/cache.py`) |
| Interaction type | `InteractionWithTokenLogpReward` | `InteractionWithTokenLevelReward` |
| Reward storage | Scalar only | Scalar + `_token_rewards: dict[str, list[float]]` + `_position_rewards: dict[str, list[PositionRewardInfo]]` |
| `set_token_rewards()` | N/A | Sets token-wise rewards; delegates to cache if interaction present; **preserves scalar reward** |
| `set_position_rewards()` | N/A | Sets position-wise rewards; extracts chosen rewards for token-level; delegates to cache |
| `compute_entropy()` | N/A | Computes per-position entropy from logprobs in `_position_rewards` |
| `export_interactions()` | Base behavior | **Overrides**: applies pending token/position rewards to interactions before calling `super().export_interactions()` |

#### Key Design Decision in `TokenRewardSessionData`

The scalar (trajectory-level) reward is **never overwritten** by token-level or position-level rewards. This separation is critical:
- **Scalar reward**: used by tree backup advantage computation and GAE
- **Token-level rewards**: used for distillation loss only
- **Position-level rewards**: attached as Python attribute (`interaction.position_rewards`), flow to `mb_input` for distillation

When `set_token_rewards()` is called and the interaction already has a scalar reward set, the scalar reward is **restored** after delegation to the cache.

### 3. `export_interactions()` Override

The extended `export_interactions()` performs two extra steps before calling `super().export_interactions()`:

1. **Apply pending token rewards**: Iterates `_token_rewards` dict; for each interaction present in cache, calls `interaction.set_token_rewards()` (or falls back to direct attribute assignment). If `interaction.reward is None`, fills it with `sum(token_rewards)`.

2. **Attach position rewards**: Iterates `_position_rewards` dict; for each interaction present in cache, sets `interaction.position_rewards = pos_rewards`.

This handles the case where token/position rewards are set **before** the interaction arrives in the cache (race condition between HTTP reward-setting and completion-streaming).

### 4. Path Constants

| Constant | Base | Extended |
|---|---|---|
| `RL_START_SESSION_PATHNAME` | Yes | Re-exported |
| `RL_END_SESSION_PATHNAME` | Yes | Re-exported |
| `RL_SET_REWARD_PATHNAME` | Yes | Re-exported |
| `CHAT_COMPLETIONS_PATHNAME` | Yes | Not re-exported |
| `RESPONSES_PATHNAME` | Yes | Not re-exported |
| `ANTHROPIC_MESSAGES_PATHNAME` | Yes | Not re-exported |
| `GRANT_CAPACITY_PATHNAME` | Yes | Re-exported |
| `EXPORT_TRAJECTORIES_PATHNAME` | Yes | Re-exported |
| `INTERNAL_WAIT_FOR_SESSION_PATHNAME` | Yes | Not re-exported |
| `DEFAULT_ADMIN_API_KEY` | Yes | Not re-exported |
| `RL_SET_TOKEN_REWARDS_PATHNAME` | No | **New**: `"rl/set_token_rewards"` |
| `RL_SET_POSITION_REWARDS_PATHNAME` | No | **New**: `"rl/set_position_rewards"` |
| `RL_COMPUTE_ENTROPY_PATHNAME` | No | **New**: `"rl/compute_entropy"` |

### 5. Serialization

Both `serialize_interactions` and `deserialize_interactions` are re-exported from the base server without modification. They operate on `InteractionWithTokenLogpReward` (base type). The extended type `InteractionWithTokenLevelReward` adds `token_rewards` and `token_reward_mask` keys to `to_tensor_dict()`, but serialization does not need to change because it uses `interaction.to_tensor_dict()` which is polymorphic.

### 6. Supporting Module Differences

The extended server depends on two local modules that extend the base AReaL types:

#### `cache.py` (Extended `InteractionCache`)

| Aspect | Base `InteractionCache` | Extended `InteractionCache` |
|---|---|---|
| Value type | `InteractionWithTokenLogpReward` | `InteractionWithTokenLevelReward` |
| `set_reward()` | Scalar only | Scalar only |
| `set_rewards()` | N/A | Token-wise rewards with validation |
| `set_position_rewards()` | N/A | Position-wise candidate rewards |
| `_set_rewards_internal()` | N/A | Internal helper; updates both `token_rewards_list` and `token_rewards` field |
| `_set_position_rewards_internal()` | N/A | Stores `position_rewards` on interaction; extracts chosen rewards for token-level; **does NOT overwrite scalar reward** |
| `compute_and_store_entropy()` | N/A | Computes per-position entropy from logprobs |
| `get_token_rewards()` | N/A | Getter for token-wise rewards |
| `get_position_rewards()` | N/A | Getter for position-wise rewards |
| `get_reward_stats()` | N/A | Comprehensive stats (scalar, token, position, entropy) |
| `export_with_token_rewards()` | N/A | Export all interactions with reward data |
| `apply_reward_discount()` | Invalidates `_cache` after discount | Same + **also invalidates `_cache`** to ensure `to_tensor_dict()` recalculates |
| `PositionRewardInfo` | N/A | Defined here with `__post_init__` validation, `chosen_token`, `chosen_reward`, `chosen_logprob` properties |

#### `types.py` (Extended `InteractionWithTokenLevelReward`)

| Aspect | Base `InteractionWithTokenLogpReward` | Extended `InteractionWithTokenLevelReward` |
|---|---|---|
| Reward field | `reward: float \| None` | Inherited + `token_rewards: list[float] \| None`, `token_reward_mask: list[int] \| None` |
| `set_token_rewards()` | N/A | Validates length, sets `token_rewards`, invalidates cache |
| `set_sparse_token_rewards()` | N/A | Sets rewards for specific token indices only |
| `to_tensor_dict()` | Returns `rewards` (scalar) | Returns base + `token_rewards` and `token_reward_mask` keys; **`rewards` remains scalar** |
| `get_reward_stats()` | N/A | Token-level stats (mean, max, min, sum, sparsity) |
| `get_output_logprobs()` | N/A | Extracts output logprobs from `model_response` |
| `compute_entropy_from_logprobs()` | N/A | Approximate entropy as `-logprob` |
| `get_token_level_logp_stats()` | N/A | Statistical summary of logprobs and entropy |
| `save_logp_and_entropy()` | N/A | Bundles logprobs, entropy, and stats for serialization |

## API Endpoint Summary

### Base Endpoints (available in both)

| Path | Method | Purpose |
|---|---|---|
| `rl/start_session` | POST | Create a new RL session |
| `rl/end_session` | POST | End a session |
| `rl/set_reward` | POST | Set scalar reward for an interaction |
| `export_trajectories` | POST | Export session trajectories |
| `grant_capacity` | POST | Grant inference capacity |

### Extended Endpoints (on-policy distill only)

| Path | Method | Purpose |
|---|---|---|
| `rl/set_token_rewards` | POST | Set per-token rewards for an interaction |
| `rl/set_position_rewards` | POST | Set per-position candidate rewards (for KL-based distillation) |
| `rl/compute_entropy` | POST | Compute entropy from position reward logprobs |

## Architectural Summary

```
Base Server (areal)
  SessionData ──────────────────── InteractionCache ────────────── InteractionWithTokenLogpReward
      │                                │                                    │
      │ scalar reward only             │ scalar reward only                 │ reward: float
      │                                │                                    │ to_tensor_dict() → {rewards}
      │                                │                                    │
      ▼                                ▼                                    ▼
Extended Server (on_policy_distill)
  TokenRewardSessionData ─────── ExtendedInteractionCache ────── InteractionWithTokenLevelReward
      │                                │                                    │
      │ + _token_rewards dict          │ + set_rewards()                    │ + token_rewards: list[float]
      │ + _position_rewards dict       │ + set_position_rewards()           │ + token_reward_mask: list[int]
      │ + set_token_rewards()          │ + compute_and_store_entropy()      │ + set_token_rewards()
      │ + set_position_rewards()       │ + get_reward_stats()              │ + set_sparse_token_rewards()
      │ + compute_entropy()            │ + export_with_token_rewards()      │ + to_tensor_dict() → {rewards, token_rewards, token_reward_mask}
      │ + export_interactions() ↑      │                                    │ + get_reward_stats()
      │   (override: apply pending     │                                    │ + compute_entropy_from_logprobs()
      │    token/position rewards      │                                    │ + save_logp_and_entropy()
      │    before super())             │                                    │
```
