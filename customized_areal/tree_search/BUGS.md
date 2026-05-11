# Bugs Found in customized_areal/tree_search

## Critical / High Priority

### 1. Wrong Object for Rollout (trainer.py:318)
**Location**: `trainer.py` line 318
```python
new_trajs = self.actor.rollout_batch(
    gen_prompts,
    workflow=workflow,
    ...
)
```
**Issue**: `self.actor` is the **training actor** (PPOActor/MultiCandidateFSDPPPOActor), but `rollout_batch` is an **inference engine** method. Unless the actor delegates to the rollout engine internally, this calls the wrong object and will likely crash or silently fail.
**Fix**: Use `self.rollout` (the inference engine) instead of `self.actor`.

### 2. Inverted Logic for Tree Training (engine/fsdp_engine.py:271-285)
**Location**: `engine/fsdp_engine.py` lines 271-285
```python
if self.enable_tree_training:
    return super()._compute_logprobs_and_loss(...)  # standard path
else:
    # multi-candidate code...
```
**Issue**: When `enable_tree_training=True` (non-critic), it falls back to the **standard** loss path. When `False`, it uses **multi-candidate** logprobs. This is backwards if multi-candidate gathering is meant for tree training/distillation.
**Fix**: Swap the conditions or rename the flag to reflect the actual behavior.

### 3. GPU-CPU Sync in Hot Path (training/loss.py:371)
**Location**: `training/loss.py` line 371
```python
output_len = int(loss_mask.sum().item())
```
**Issue**: `.item()` forces a GPU-CPU synchronization inside the loss computation hot path. Per AReaL conventions, this should be avoided.
**Fix**: Use `loss_mask.sum().int()` or keep it as a tensor operation.

## Medium Priority

### 4. Inconsistent Variance Normalization (advantage.py:74-91)
**Location**: `advantage.py` lines 74-91
**Issue**: 
- Q-values (line 75): divides by `n` (population variance)
- Rewards (line 91): divides by `n-1` (sample variance with Bessel's correction)

This inconsistency causes advantages and returns to be normalized on different scales.
**Fix**: Use consistent normalization (either both population or both sample variance).

### 5. Default node_id=0 Collision (mcts_tree_store.py:42,231)
**Location**: `mcts_tree_store.py` lines 42, 231
```python
node_id: int = 0  # default
if existing_id != 0 and existing_id in self._node_id_to_key:
```
**Issue**: Nodes with the default `node_id=0` are treated as "no existing ID" and get **re-inserted**, potentially causing duplicates and double-counting visits via `_backup()`.
**Fix**: Use `None` as the sentinel value instead of `0`, or track which nodes have been explicitly assigned IDs.

### 6. rollout_batch Violates 1:1 Contract (workflow_executor.py:141-143,163)
**Location**: `workflow_executor.py` lines 141-143, 163
**Issue**: `wait()` flattens `list[Node]` results via `extracted.extend()`, so `rollout_batch` can return **more items than the input batch size**. Downstream code in `_cache_aware_prepare_batch` may expect a 1:1 mapping between prompts and trajectories.
**Fix**: Document the many-to-one relationship or wrap results to preserve the batch dimension.

### 7. Return Type Mismatch (core/agent.py:119,235)
**Location**: `core/agent.py` lines 119, 235
```python
async def run(...) -> float:
    ...
    return {completion_id: reward}  # returns dict, not float
```
**Issue**: Method declares return type as `float` but can return a `dict`.
**Fix**: Change return type to `float | dict` or ensure consistent return type.

### 8. Overwrites Metadata on Cached Nodes (grouped_workflow.py:58-61)
**Location**: `grouped_workflow.py` lines 58-61
```python
node.episode_id = episode_id
node.query_id = query_id
node.turn_idx = turn_idx
```
**Issue**: If nodes were loaded from cache with pre-existing metadata, this blindly overwrites them.
**Fix**: Check if metadata is already set before overwriting, or document the expected behavior.

## Low Priority

### 9. Wrong Default for _next_node_id (checkpoint.py:88-89)
**Location**: `checkpoint.py` lines 88-89
```python
store._next_node_id = metadata.get("next_node_id", metadata.get("next_seq_id", 0))
```
**Issue**: Falls back to `0`, but `MCTSTreeStore.__init__` starts at `1`. A checkpoint with missing metadata would reset IDs to `0`, causing the next insert to get `node_id=1` (duplicate of existing).
**Fix**: Change default from `0` to `1`.

### 10. Fragile Result Type Check (proxy_workflow.py:166-168)
**Location**: `proxy_workflow.py` lines 166-168
```python
if isinstance(result, dict) and all(
    isinstance(v, InteractionWithTokenLogpReward) for v in result.values()
):
```
**Issue**: If `super().arun_episode()` returns a list (not a dict), this silently returns `None` instead of converting it.
**Fix**: Add handling for list results or log a warning when result type is unexpected.

### 11. Potential Shape Mismatch in Stats Tracking (training/actor.py:55-63)
**Location**: `training/actor.py` lines 55-63
```python
reward_score = data.get("rewards")
if reward_score is not None and isinstance(reward_score, torch.Tensor):
    attn_mask = data.get("attention_mask")
    if attn_mask is not None:
        stats_tracker.stat(
            task_reward=reward_score.float(),
            denominator="n_seqs",
        )
```
**Issue**: `reward_score` might be a scalar or have unexpected shape. `stats_tracker.stat(task_reward=reward_score.float(), denominator="n_seqs")` assumes a specific tensor shape that may not hold.
**Fix**: Validate tensor shape before passing to stats_tracker or handle scalars appropriately.
