# Cache Implementation Comparison

**Last Updated:** 2026-04-24

## Files Compared

| File                                                | Purpose                                                          |
| --------------------------------------------------- | ---------------------------------------------------------------- |
| `areal/experimental/openai/cache.py`                | Original OpenAI-compatible cache for multi-turn RL               |
| `customized_areal/on_policy_distill/proxy/cache.py` | Extended cache with token-level and position-wise reward support |

______________________________________________________________________

## 1. Feature Comparison

| Feature                      | OpenAI Cache                                       | Proxy Cache                                         | Notes                                            |
| ---------------------------- | -------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------ |
| **Base Class**               | `OrderedDict[str, InteractionWithTokenLogpReward]` | `OrderedDict[str, InteractionWithTokenLevelReward]` | Different interaction types                      |
| **Thread Safety**            | Lock in `set_reward()` only                        | Lock in most methods                                | Proxy is more thread-safe                        |
| **Token-wise Rewards**       | No                                                 | Yes                                                 | Proxy adds `set_rewards()` for per-token rewards |
| **Position-wise Rewards**    | No                                                 | Yes                                                 | Proxy adds `PositionRewardInfo` class            |
| **Candidate Rewards**        | No                                                 | Yes                                                 | Proxy supports multiple candidates per position  |
| **Entropy Computation**      | No                                                 | Yes                                                 | Proxy has `compute_and_store_entropy()`          |
| **Double-call guard**        | Yes                                                | Yes                                                 | Both have `_apply_reward_discount_called` guard  |
| **Reward Stats**             | No                                                 | Yes                                                 | Proxy has `get_reward_stats()`                   |
| **Parent-Child Relations**   | Yes                                                | Yes                                                 | Both build conversation trees                    |
| **Export Styles**            | concat, individual                                 | concat, individual                                  | Same export modes                                |
| **Prefix Detection Warning** | Yes                                                | Yes                                                 | Both have `_is_similar_on_last_message` warning  |
| **Empty Cache Check**        | No — `StopIteration` on empty                      | Yes — raises `KeyError`                             | OpenAI still lacks empty-check guard             |

______________________________________________________________________

## 2. Type Hierarchy

```
InteractionWithTokenLogpReward          (areal/experimental/openai/types.py)
    ├── reward: float | None
    ├── model_response: ModelResponse | None
    ├── messages, output_message_list, parent, ...
    └── to_tensor_dict()  →  {input_ids, loss_mask, logprobs, versions,
                               attention_mask, rewards}
        └── InteractionWithTokenLevelReward  (customized_areal/.../proxy/types.py)
                + token_rewards: list[float] | None
                + token_reward_mask: list[int] | None
                + set_token_rewards()
                + set_sparse_token_rewards()
                + get_reward_stats()
                + get_output_logprobs()
                + compute_entropy_from_logprobs()
                + get_token_level_logp_stats()
                + save_logp_and_entropy()
                + to_tensor_dict()  (override: adds token_rewards +
                                     token_reward_mask keys)
```

The customized type inherits everything from upstream and adds per-token reward fields
and methods. The `to_tensor_dict()` override adds two new tensor keys (`token_rewards`,
`token_reward_mask`) while preserving the scalar `rewards` key for tree backup / GAE.

______________________________________________________________________

## 3. Class Structure Comparison

### OpenAI Cache

```python
class InteractionCache(OrderedDict[str, InteractionWithTokenLogpReward]):
    - _apply_reward_discount_called: bool
    - _total_reward: float
    - _lock: threading.Lock
```

### Proxy Cache

```python
@dataclass
class PositionRewardInfo:
    - position: int
    - candidates: list[str]
    - candidate_token_ids: list[int]
    - logprobs: list[float] | None     # logp_model for each candidate
    - rewards: list[float]
    - chosen_index: int
    - sample_index: int
    # Properties: chosen_token, chosen_reward, chosen_logprob
    # Methods: get_reward_for_token(token)

class InteractionCache(OrderedDict[str, InteractionWithTokenLevelReward]):
    - _apply_reward_discount_called: bool
    - _total_reward: float
    - _total_list_reward: list[float] | None  # Element-wise sum of token rewards
    - _lock: threading.Lock
```

______________________________________________________________________

## 4. Method Comparison

### Common Methods (with differences)

| Method                  | OpenAI                               | Proxy                                    | Key Differences                                                 |
| ----------------------- | ------------------------------------ | ---------------------------------------- | --------------------------------------------------------------- |
| `__init__`              | Simple init                          | Simple init                              | Proxy adds `_total_list_reward`                                 |
| `__setitem__`           | Prefix matching + warning (unlocked) | Prefix matching + warning (lock-wrapped) | Proxy wraps parent-finding in `self._lock`; helpers are methods |
| `set_reward`            | Lock-protected ("usually no need")   | Lock-protected + `KeyError` if missing   | Proxy validates existence                                       |
| `set_last_reward`       | Direct call                          | Direct call                              | Same logic                                                      |
| `apply_reward_discount` | No lock; warns only for `i == 0`     | Lock-wrapped; warns for every missing;   | Proxy is thread-safe, more verbose, and invalidates `_cache`    |
|                         |                                      | invalidates `interaction._cache = None`  |                                                                 |
| `export_interactions`   | `style` required (no default)        | `style` defaults to `"individual"`       | Proxy provides explicit default                                 |
| `last_interaction_id`   | No empty check (`StopIteration`)     | Raises `KeyError` if empty               | Proxy has empty-check guard; OpenAI does not                    |

### Proxy-Only Methods

| Method                                                            | Purpose                                                                                                                |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `_set_rewards_internal(completion_id, token_rewards)`             | Lock-free core of `set_rewards`                                                                                        |
| `_set_position_rewards_internal(completion_id, position_rewards)` | Lock-free core of `set_position_rewards`                                                                               |
| `set_rewards(completion_id, token_rewards)`                       | Set per-token reward list; validates length against output tokens; updates `_total_list_reward` and `_total_reward`    |
| `set_last_rewards(token_rewards)`                                 | Convenience: calls `set_rewards` on last interaction                                                                   |
| `set_position_rewards(completion_id, position_rewards)`           | Store `PositionRewardInfo` list; extracts chosen-index rewards; does NOT overwrite scalar `interaction.reward`         |
| `get_token_rewards(completion_id)`                                | Retrieve `token_rewards_list` attribute                                                                                |
| `get_position_rewards(completion_id)`                             | Retrieve `position_rewards` attribute                                                                                  |
| `compute_and_store_entropy(completion_id)`                        | Compute Shannon entropy per position from `PositionRewardInfo.logprobs`; stores `position_entropies` and `avg_entropy` |
| `get_entropies(completion_id)`                                    | Retrieve `position_entropies` attribute                                                                                |
| `get_reward_stats(completion_id)`                                 | Aggregate stats dict: scalar reward, token reward stats, position reward stats, entropy stats                          |
| `export_with_token_rewards()`                                     | Export all interactions' reward stats as `{id: stats_dict}`                                                            |
| `_is_prefix(a, b)`                                                | Extracted from upstream's nested local                                                                                 |
| `_is_similar_on_last_message(a, b)`                               | Extracted from upstream's nested local                                                                                 |

______________________________________________________________________

## 5. Key Implementation Differences

### 5.1 Locking Strategy

**OpenAI:**

```python
# Only set_reward locks (with comment "usually no need, but just in case")
def set_reward(self, interaction_id: str, reward: float):
    with self._lock:
        ...
```

**Proxy:**

```python
# Most mutating methods lock
- __setitem__                  # Locks during parent search
- set_rewards                  # Uses _set_rewards_internal within lock
- set_reward                   # Locks
- set_position_rewards         # Uses _set_position_rewards_internal within lock
- apply_reward_discount        # Locks
- compute_and_store_entropy    # Locks
```

### 5.2 Reward Architecture

**OpenAI:** Single scalar `reward: float | None` per interaction. The `rewards` key in
`to_tensor_dict()` is always a 1-element tensor `[float(reward)]`.

**Proxy:** Dual reward system:

- **Trajectory-level scalar** (`interaction.reward` / `rewards` tensor key): Used by
  tree backup and GAE. Set via `set_reward()`.
- **Token-level per-position** (`token_rewards` / `token_reward_mask` tensor keys): Used
  by distillation loss. Set via `set_rewards()` or `set_position_rewards()`.
- `set_position_rewards()` intentionally avoids overwriting `interaction.reward` to keep
  the trajectory-level reward intact for tree backup.
- `_total_list_reward` tracks an element-wise sum of all token reward lists (padded to
  equal length).

### 5.3 `apply_reward_discount` Differences

| Aspect                 | OpenAI                                | Proxy                                                |
| ---------------------- | ------------------------------------- | ---------------------------------------------------- |
| Locking                | None                                  | `self._lock`                                         |
| Missing reward warning | Only warns for `i == 0` (most recent) | Warns for every interaction missing a reward         |
| Cache invalidation     | None                                  | Sets `interaction._cache = None` after reward update |

### 5.4 `__setitem__` Differences

| Aspect           | OpenAI                                      | Proxy                                                               |
| ---------------- | ------------------------------------------- | ------------------------------------------------------------------- |
| Helper functions | Nested local functions inside `__setitem__` | Extracted to `_is_prefix` and `_is_similar_on_last_message` methods |
| Thread safety    | No lock on parent-finding loop              | Wrapped in `self._lock`                                             |

### 5.5 `last_interaction_id` Empty Check

| Aspect      | OpenAI                                             | Proxy                                         |
| ----------- | -------------------------------------------------- | --------------------------------------------- |
| Empty cache | Raises `StopIteration` from `next(reversed(self))` | Raises `KeyError("No interactions in cache")` |

### 5.6 `export_interactions` Differences

| Aspect                           | OpenAI                         | Proxy          |
| -------------------------------- | ------------------------------ | -------------- |
| `style` default                  | None (required positional arg) | `"individual"` |
| Incomplete interaction filtering | Same logic                     | Same logic     |

______________________________________________________________________

## 6. Remaining Divergences (Not Yet Fixed)

| #   | Issue                                      | OpenAI                          | Proxy                                     | Status                                        |
| --- | ------------------------------------------ | ------------------------------- | ----------------------------------------- | --------------------------------------------- |
| 1   | Empty cache guard on `last_interaction_id` | Missing — `StopIteration`       | Has `KeyError` guard                      | OpenAI should add the check                   |
| 2   | `export_interactions` style default        | No default (required)           | Defaults to `"individual"`                | Minor API difference                          |
| 3   | `apply_reward_discount` cache invalidation | No `_cache = None`              | Sets `interaction._cache = None`          | OpenAI should invalidate cache after discount |
| 4   | `apply_reward_discount` thread safety      | No lock                         | Lock-wrapped                              | OpenAI should add lock for consistency        |
| 5   | `apply_reward_discount` warning verbosity  | Only warns for last interaction | Warns for all missing-reward interactions | Style difference, both valid                  |
| 6   | Helper functions in `__setitem__`          | Nested locals                   | Extracted to methods                      | Cosmetic; proxy approach is cleaner           |

______________________________________________________________________

## 7. Code Duplication

Both files still duplicate:

- `_is_prefix` function logic (nested local vs method, but identical behavior)
- `_is_similar_on_last_message` function logic
- Parent-child relationship building in `__setitem__`
- Export interaction filtering logic
- Leaf node detection in concat mode
- `apply_reward_discount` backward propagation algorithm

**Recommendation:** Extract common functionality to a shared base class or utility
module.

______________________________________________________________________

## 8. Summary

The customized `InteractionCache` extends the upstream version in three directions:

1. **Token-level rewards**: Adds `set_rewards()`, `set_position_rewards()`,
   `PositionRewardInfo`, and corresponding getters. The
   `InteractionWithTokenLevelReward` type adds `token_rewards` / `token_reward_mask`
   fields and a `to_tensor_dict()` override that emits them as separate tensor keys
   alongside the scalar `rewards`.

1. **Candidate-wise rewards for distillation**: `PositionRewardInfo` stores
   per-candidate logprobs and rewards at each position, enabling KL-divergence or
   logp-difference based distillation losses. Entropy computation from the full
   candidate distribution is supported via `compute_and_store_entropy()`.

1. **Thread safety**: All mutation paths (`set_reward`, `__setitem__`,
   `apply_reward_discount`, `set_rewards`, `set_position_rewards`,
   `compute_and_store_entropy`) are wrapped in `self._lock`, whereas upstream only has
   partial locking.

The core logic (parent-child tree building via prefix matching, `apply_reward_discount`
backward propagation, `export_interactions` concat/individual filtering) is shared
between both, with the customized version extracting helpers to methods and adding
defensive improvements (`KeyError` on empty, cache invalidation after discount, more
verbose missing-reward warnings).
