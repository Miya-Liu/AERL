# Bug Report: customized_areal/tree_search/trainer.py

## File: customized_areal/tree_search/trainer.py

______________________________________________________________________

## Critical Bugs

### 1. Crash when cache enabled but tree backup mode is OFF

**Lines:** 466, 287-290

**Description:** `__init__` only initializes tree components when BOTH conditions are
true:

```python
if (self.cache_config.enabled
    and self.tree_backup_config.mode != TreeBackupMode.OFF):
    self._init_tree_components()
```

But `train()` only checks `self.cache_config.enabled` (line 466):

```python
if not self.cache_config.enabled:
    return super().train(...)
```

**Impact:** If `cache_config.enabled=True` with `tree_backup_config.mode=OFF`, `train()`
will monkey-patch `self.actor.prepare_batch` to `_cache_aware_prepare_batch`, which
immediately accesses `self._batch_builder` and `self.tree_store` — both uninitialized.
This causes `AttributeError`.

**Fix:**

```python
if not (self.cache_config.enabled
        and self.tree_backup_config.mode != TreeBackupMode.OFF):
    return super().train(...)
```

______________________________________________________________________

### 2. group_size parameter completely ignored

**Lines:** 358, 412

**Description:** The method accepts `group_size` but hardcodes it to
`self.cache_config.n_samples`:

```python
def _cache_aware_prepare_batch(..., group_size=1, ...):
    ...
    new_trajs = self.actor.rollout_batch(
        all_prompts,
        workflow=workflow,
        workflow_kwargs=workflow_kwargs,
        group_size=n_samples,  # BUG: ignores the group_size parameter
    )
```

**Impact:** If the caller passes a different `group_size` (e.g., the base trainer passes
`config.gconfig.n_samples`), the cache-aware path will use the wrong grouping size,
potentially causing shape mismatches or incorrect batching.

**Fix:** Use the `group_size` parameter instead of `n_samples`.

______________________________________________________________________

### 3. should_accept_fn never forwarded

**Lines:** 357, 408-413

**Description:** The method accepts `should_accept_fn` but never passes it to
`rollout_batch`. Additionally, when `not need_gen_items` (all prompts cached), it
returns cached trajectories directly without consulting `should_accept_fn`.

**Impact:** Dynamic filtering for stale/off-policy rollouts is completely bypassed. A
cached trajectory that should be rejected will be silently accepted.

**Fix:**

- Forward `should_accept_fn` to `self.actor.rollout_batch()`
- Apply `should_accept_fn` filtering to cached trajectories as well

______________________________________________________________________

### 4. Query ID injection fails in single-controller mode

**Lines:** 66-106, 337

**Description:** The `_patch_wrap_openai_agent_for_query_id` function patches
`_wrap_openai_agent` on `self.rollout`. In single-controller mode, `self.rollout` is a
`RolloutController`, which does NOT have `_wrap_openai_agent`. The function detects
this, logs a warning, and returns without patching.

**Impact:**

- Trajectories never receive `_mcts_query_id`
- `MCTSTreeStore.insert_batch` falls back to MD5 hash of prompt tokens
- If the dataset provided `query_id`, cache lookup uses that string, but tree insertion
  computes an MD5 hash — they never match
- This completely breaks cache consistency and tree training history

**Fix:** The patch needs to target the remote worker engines via collective RPC, or
`query_id` needs to be propagated through task input data.

______________________________________________________________________

## Medium Bugs

### 5. Bypasses dispatcher async/staleness control

**Lines:** 381-418

**Description:** The base `RolloutController.prepare_batch` uses
`dispatcher.active_submit_and_wait` with a persistent generator to maintain controlled
staleness (keeping at least 2 batches pending). The cache-aware version:

- Creates its own isolated iterator (`_cache_dataloader_iter`)
- Calls `self.actor.rollout_batch` directly (bypassing the dispatcher)

**Impact:** Loses async rollout-training overlap and staleness enforcement, hurting
throughput.

______________________________________________________________________

### 6. Missing record_training_step call

**Lines:** 352-448

**Description:** `_cache_aware_prepare_batch` never calls
`self.tree_store.record_training_step(global_step, trajs)`.

**Impact:** `_training_history` is never populated. This breaks:

- Training history tracking
- CROSS_TRAINING analysis
- Curriculum learning features

______________________________________________________________________

## Minor Bugs

### 7. dynamic_bs parameter ignored

**Lines:** 359, 493

**Description:** The method accepts `dynamic_bs` but never uses it. The base
`prepare_batch` forwards this to `dispatcher.active_submit_and_wait` for dynamic batch
sizing.

**Impact:** Dynamic batch sizing is silently disabled in cache-aware mode.

______________________________________________________________________

### 8. Potential crash with empty checkpoint directory

**Lines:** 307

**Description:**

```python
self.tree_checkpoint_manager = TreeCheckpointManager(
    self.cache_config.cache_dir or self.tree_backup_config.checkpoint_dir
)
```

If both fields are `None`, `os.path.join(None, "mcts_trees")` raises `TypeError`. With
empty strings (the current dataclass defaults), it creates directories in the current
working directory.

**Fix:** Add validation:

```python
save_dir = self.cache_config.cache_dir or self.tree_backup_config.checkpoint_dir
if not save_dir:
    raise ValueError("Tree checkpoint directory not configured")
```

______________________________________________________________________

## Summary

| #   | Issue                                                         | Severity |
| --- | ------------------------------------------------------------- | -------- |
| 1   | train() checks only cache_config.enabled, missing mode != OFF | Critical |
| 2   | group_size parameter ignored                                  | Critical |
| 3   | should_accept_fn never forwarded                              | Critical |
| 4   | \_wrap_openai_agent patch fails in single-controller mode     | Critical |
| 5   | Bypasses dispatcher staleness/async overlap                   | Medium   |
| 6   | Missing record_training_step                                  | Medium   |
| 7   | dynamic_bs parameter ignored                                  | Minor    |
| 8   | Potential crash with empty checkpoint directory               | Minor    |
