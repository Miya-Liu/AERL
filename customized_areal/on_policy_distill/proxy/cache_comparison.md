# Cache Implementation Comparison

**Last Updated:** 2026-04-02 (all bugs fixed)

## Files Compared

| File                                                | Purpose                                                          |
| --------------------------------------------------- | ---------------------------------------------------------------- |
| `areal/experimental/openai/cache.py`                | Original OpenAI-compatible cache for multi-turn RL               |
| `customized_areal/on_policy_distill/proxy/cache.py` | Extended cache with token-level and position-wise reward support |

______________________________________________________________________

## 1. Feature Comparison

| Feature                      | OpenAI Cache                                       | Proxy Cache                                         | Notes                                               |
| ---------------------------- | -------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| **Base Class**               | `OrderedDict[str, InteractionWithTokenLogpReward]` | `OrderedDict[str, InteractionWithTokenLevelReward]` | Different interaction types                         |
| **Thread Safety**            | Lock in `set_reward()` only                        | Lock in most methods                                | Proxy is more thread-safe                           |
| **Token-wise Rewards**       | ❌ No                                              | ✅ Yes                                              | Proxy adds `set_rewards()` for per-token rewards    |
| **Position-wise Rewards**    | ❌ No                                              | ✅ Yes                                              | Proxy adds `PositionRewardInfo` class               |
| **Candidate Rewards**        | ❌ No                                              | ✅ Yes                                              | Proxy supports multiple candidates per position     |
| **Entropy Computation**      | ❌ No                                              | ✅ Yes                                              | Proxy has `compute_and_store_entropy()`             |
| **Double-call guard**        | ✅ Yes                                             | ✅ **FIXED**                                        | Both now have `_apply_reward_discount_called` guard |
| **Reward Stats**             | ❌ No                                              | ✅ Yes                                              | Proxy has `get_reward_stats()`                      |
| **Parent-Child Relations**   | ✅ Yes                                             | ✅ Yes                                              | Both build conversation trees                       |
| **Export Styles**            | concat, individual                                 | concat, individual                                  | Same export modes                                   |
| **Prefix Detection Warning** | ✅ Yes                                             | ✅ **FIXED**                                        | Proxy now has `_is_similar_on_last_message` warning |
| **Empty Cache Check**        | ✅ **FIXED**                                       | ✅ Yes                                              | Both now check for empty cache                      |

______________________________________________________________________

## 2. Class Structure Comparison

### OpenAI Cache

```python
class InteractionCache(OrderedDict[str, InteractionWithTokenLogpReward]):
    - _apply_reward_discount_called: bool  # Prevents double application
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
    - logprobs: list[float] | None
    - rewards: list[float]
    - chosen_index: int

class InteractionCache(OrderedDict[str, InteractionWithTokenLevelReward]):
    - _apply_reward_discount_called: bool  # ✅ ADDED
    - _total_reward: float
    - _total_list_reward: list[float] | None  # New: element-wise sum
    - _lock: threading.Lock
```

______________________________________________________________________

## 3. Method Comparison

### Common Methods (with differences)

| Method                  | OpenAI                          | Proxy                           | Key Differences                                                  |
| ----------------------- | ------------------------------- | ------------------------------- | ---------------------------------------------------------------- |
| `__init__`              | Simple init                     | Simple init                     | Proxy adds `_total_list_reward`, `_apply_reward_discount_called` |
| `__setitem__`           | Prefix matching + warning logic | Prefix matching + warning logic | ✅ NOW MATCHES - Proxy has `_is_similar_on_last_message` warning |
| `set_reward`            | Lock-protected                  | Lock-protected + KeyError check | Proxy validates existence                                        |
| `set_last_reward`       | Direct call                     | Direct call                     | Same logic                                                       |
| `apply_reward_discount` | One-time with flag              | One-time with flag              | ✅ NOW MATCHES - Both have guard                                 |
| `export_interactions`   | Filters incomplete              | Filters incomplete              | ✅ NOW MATCHES - Both use same logic                             |
| `last_interaction_id`   | ✅ Empty check added            | Same + empty check              | ✅ NOW MATCHES - Both raise KeyError if empty                    |

### Proxy-Only Methods

```python
_set_rewards_internal(completion_id, token_rewards)   # Internal method without lock
set_rewards(completion_id, token_rewards)             # Token-wise reward setting
set_last_rewards(token_rewards)                       # Set rewards for last interaction
set_position_rewards(completion_id, position_rewards) # Position-wise candidate rewards
get_token_rewards(completion_id) -> list[float]       # Retrieve token rewards
get_position_rewards(completion_id)                   # Retrieve position rewards
compute_and_store_entropy(completion_id)              # Compute entropy from logprobs
get_entropies(completion_id)                          # Get computed entropies
get_reward_stats(completion_id) -> dict               # Comprehensive stats
export_with_token_rewards() -> dict                   # Export all with rewards
```

______________________________________________________________________

## 4. Key Implementation Differences

### 4.1 Locking Strategy

**OpenAI:**

```python
# Only set_reward locks
def set_reward(self, interaction_id: str, reward: float):
    with self._lock:  # Lock only here
        ...
```

**Proxy:**

```python
# Most mutating methods lock
- __setitem__  # Locks during parent search
- set_rewards  # Uses _set_rewards_internal within lock
- set_reward
- set_position_rewards  # Uses _set_rewards_internal to avoid recursive lock
- apply_reward_discount
- compute_and_store_entropy
```

### 4.2 Reward Discount Behavior (✅ FIXED)

**Both implementations now have the guard:**

```python
def apply_reward_discount(self, turn_discount: float = 1.0):
    if self._apply_reward_discount_called:
        raise RuntimeError("apply_reward_discount should only be called once.")
    self._apply_reward_discount_called = True
    ...
```

### 4.3 Debug Warning Logic (✅ FIXED)

**Both implementations now have the warning:**

```python
elif self._is_prefix(parent.messages, value.messages):
    is_similar, diff_a, diff_b = self._is_similar_on_last_message(
        parent_data, value.messages
    )
    if is_similar:
        logger.warning(
            "Found a parent interaction with similar last message content, "
            "but not a strict prefix match..."
        )
```

### 4.4 Incomplete Interaction Detection (✅ FIXED)

**Both now use the same logic:**

```python
if (
    interaction.interaction_id is None
    or interaction.output_message_list is None
):
    # Skip incomplete
```

### 4.5 Empty Cache Check (✅ FIXED)

**Both now check for empty cache:**

```python
@property
def last_interaction_id(self) -> str:
    if not self:
        raise KeyError("No interactions in cache")
    return next(reversed(self))
```

______________________________________________________________________

## 5. Bug Fixes Summary

### ✅ FIXED: Bug 1 - Missing Guard Against Double Discount (Proxy)

**Status:** FIXED

**Fix:** Added `_apply_reward_discount_called` flag to `__init__` and check in
`apply_reward_discount()`.

______________________________________________________________________

### ✅ FIXED: Bug 2 - Missing Parent Warning Logic (Proxy)

**Status:** FIXED

**Fix:** Added `_is_similar_on_last_message()` helper method and warning logic in
`__setitem__`.

______________________________________________________________________

### ✅ FIXED: Bug 3 - Math Import Inside Loop (Proxy)

**Status:** FIXED

**Fix:** Moved `import math` to top of file with other imports.

______________________________________________________________________

### ✅ FIXED: Bug 4 - Recursive Lock Acquisition (Proxy)

**Status:** FIXED

**Fix:** Created `_set_rewards_internal()` method that doesn't acquire lock, and
refactored:

- `set_rewards()` now calls `_set_rewards_internal()` within `with self._lock`
- `set_position_rewards()` now calls `_set_rewards_internal()` instead of
  `set_rewards()`

This avoids potential deadlock from calling a lock-acquiring method while holding the
lock.

______________________________________________________________________

### ✅ FIXED: Bug 5 - Inconsistent Incomplete Interaction Logic (Proxy vs OpenAI)

**Status:** FIXED

**Fix:** Standardized Proxy's `export_interactions()` to match OpenAI's logic:

```python
# Before (Proxy-specific logic with has_cache check)
has_cache = getattr(interaction, "_cache", None) is not None
if (...) and not has_cache:

# After (matches OpenAI)
if interaction.interaction_id is None or interaction.output_message_list is None:
```

______________________________________________________________________

### ✅ FIXED: Bug 6 - Empty Cache Error (OpenAI)

**Status:** FIXED

**Fix:** Added empty check to OpenAI's `last_interaction_id`:

```python
@property
def last_interaction_id(self) -> str:
    if not self:
        raise KeyError("No interactions in cache")
    return next(reversed(self))
```

______________________________________________________________________

## 6. Code Duplication Opportunities

Both files still duplicate:

- `_is_prefix` function logic
- `_is_similar_on_last_message` function logic (now in both)
- Parent-child relationship building
- Export interaction filtering logic (now standardized)
- Leaf node detection in concat mode

**Recommendation:** Consider extracting common functionality to a shared base class or
utility module.

______________________________________________________________________

## 7. Summary of All Fixes Applied

| Bug | Description                               | Status   | Fix Details                                  |
| --- | ----------------------------------------- | -------- | -------------------------------------------- |
| 1   | Missing double-discount guard             | ✅ FIXED | Added `_apply_reward_discount_called` flag   |
| 2   | Missing parent warning logic              | ✅ FIXED | Added `_is_similar_on_last_message()` helper |
| 3   | Math import inside loop                   | ✅ FIXED | Moved to top of file                         |
| 4   | Recursive lock acquisition                | ✅ FIXED | Created `_set_rewards_internal()` method     |
| 5   | Inconsistent incomplete interaction logic | ✅ FIXED | Standardized to match OpenAI                 |
| 6   | Empty cache error (OpenAI)                | ✅ FIXED | Added empty check                            |

______________________________________________________________________

## 8. Recommendations (Remaining)

### Medium Priority

1. **Review `_total_list_reward` padding logic** - Edge cases with variable lengths
1. **Add validation for dynamically set attributes** - Runtime safety for
   `interaction.xxx = yyy  # type: ignore`

### Low Priority

3. **Standardize error types and messages** - API consistency
1. **Consider extracting common code to base class** - Reduce duplication
1. **Add unit tests for edge cases** - Both implementations need more test coverage

______________________________________________________________________

## 9. Key Behavioral Changes After Fixes

### Before Fixes

- Proxy could have `apply_reward_discount()` called multiple times (corrupting rewards)
- Proxy lacked debug warnings for conversation tree construction
- Proxy used different incomplete interaction detection
- OpenAI raised `StopIteration` on empty cache access

### After Fixes

- Both caches throw `RuntimeError` on double discount
- Both caches warn on similar-but-not-prefix message matches
- Both caches use same incomplete interaction detection
- Both caches raise `KeyError` on empty cache access
