# Bug Report and Fix Plan for tree_search

## Summary

This document catalogs all bugs found during code review of `customized_areal/tree_search/`, verified against the current source, along with occurrence conditions and fix plans.

---

## CRITICAL

### Bug #1: Type Mismatch Crash in `trainer.py:_cache_aware_prepare_batch`

**Location:** `trainer.py:307-382`

**Verification:** CONFIRMED

**Occurrence Condition:**
- Cache is enabled (`cache_config.enabled=True`)
- At least one prompt in the batch has `>= n_samples` cached trajectories
- `_cache_aware_prepare_batch` is called

**Root Cause:**
```python
# Line 307-310: cached_nodes = list[dict] (tensor dicts from _node_to_tensor_dict)
cached_nodes: list = []
if cached_items:
    cached_nodes = list(
        self._batch_builder.load_cached_trajectories(cached_items)
    )

# Line 314-330: generated_nodes = list[Node] (from rollout_batch)
generated_nodes: list = []
if need_gen_items:
    ...
    if new_trajs:
        generated_nodes = new_trajs

# Line 332: MIXED list — dicts + Nodes
nodes = cached_nodes + generated_nodes
```

Subsequent operations crash when they encounter dict items:
- Line 348: `self.tree_store.insert_batch(nodes)` — `getattr(dict, "node_id", 0)` returns 0 (skips dedup check), then `node.outcome_reward` raises `AttributeError`
- Line 353: `self.tree_advantage_computer.compute(nodes)` — `traj.query_id` raises `AttributeError`
- Line 359: `_mark_batch_trained(self.tree_store, nodes)` — `traj.node_id` raises `AttributeError`
- Line 373: `node.episode_id` — raises `AttributeError`

**Impact:** Training crashes whenever cache produces any hits. The rollout cache feature is completely non-functional.

**Fix Plan:**
`load_cached_trajectories` should return `list[Node]` instead of `list[dict]`. Move `_node_to_tensor_dict` conversion to the final step in `_cache_aware_prepare_batch`, after all tree operations:

```python
# In _CacheAwareBatchBuilder: load Node objects, convert later
def load_cached_trajectories(self, cached_prompts):
    all_nodes = []
    for item in cached_prompts:
        query_id = item["query_id"]
        if not query_id:
            continue
        nodes = self.tree_store.load_trajectories(query_id, self.n_samples)
        all_nodes.extend(nodes)
    return all_nodes  # list[Node], not list[dict]

# In _cache_aware_prepare_batch: convert after tree operations
nodes = cached_nodes + generated_nodes
# ... tree operations on nodes ... (all expecting Node objects)
# THEN convert:
for node in nodes:
    converted_trajs.append(_node_to_tensor_dict(node, ...))
```

---

### Bug #2: Stale References After Checkpoint Load

**Location:** `trainer.py:211-241`

**Verification:** CONFIRMED

**Occurrence Condition:**
- `cache_config.enabled=True` AND `tree_backup_config.mode == CacheMode.CROSS_TRAINING`
- Checkpoint exists and is loaded successfully

**Root Cause:**
```python
# Line 211: empty store created
self.tree_store = MCTSTreeStore()

# Line 212: advantage computer references EMPTY store
self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)

# Line 217-220: store REPLACED — but advantage computer still holds old reference
if self.tree_backup_config.mode == CacheMode.CROSS_TRAINING:
    if self.tree_checkpoint_manager.exists():
        self.tree_store = self.tree_checkpoint_manager.load()  # new object

# Line 239: batch builder created AFTER load — gets CORRECT store
self._batch_builder = _CacheAwareBatchBuilder(self.tree_store, ...)
```

`TreeAdvantageComputer.__init__` stores `self.tree_store = tree_store` (Python reference). When `self.tree_store` is reassigned to a new object, `self.tree_advantage_computer.tree_store` still points to the original empty store. (`_batch_builder` at line 239 is fine since it's created after the load.)

**Impact:** Tree advantage computation and all advantage-related operations use an empty store, ignoring loaded checkpoint data. Silent incorrect behavior — no crash, just wrong advantages.

**Fix Plan:**
Reorder initialization so all components are created AFTER checkpoint loading:
```python
def _init_tree_components(self) -> None:
    self.tree_checkpoint_manager = TreeCheckpointManager(...)

    # Load first, then create components
    if self.tree_backup_config.mode == CacheMode.CROSS_TRAINING and \
       self.tree_checkpoint_manager.exists():
        self.tree_store = self.tree_checkpoint_manager.load()
    else:
        self.tree_store = MCTSTreeStore()

    # Now all components reference the correct (possibly loaded) store
    self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
    self._batch_builder = _CacheAwareBatchBuilder(self.tree_store, ...)

    # Restore trained flags
    ...
```

---

### Bug #3: Dead Code in `_resolve_proximal_logp`

**Location:** `training/loss.py:178-196`

**Verification:** CONFIRMED (NEW — not in previous report)

**Root Cause:**
```python
def _resolve_proximal_logp(prox_logp_gt, prox_logp_method, old_logp, logprobs,
                           versions, current_version):
    if prox_logp_gt is not None:
        return prox_logp_gt
    if prox_logp_method == "recompute":
        return old_logp
    if versions is not None and current_version is not None:
        logprobs = logprobs[versions == current_version]  # ← filtered but DISCARDED
    return old_logp  # ← ALWAYS returns old_logp regardless
```

Line 193 filters `logprobs` to only keep entries matching `current_version`, but the filtered value is assigned to a local variable that is never returned. The function unconditionally returns `old_logp` on line 196.

**Impact:** When `prox_clip` uses version-based alignment, proximal logprobs are never filtered by version. The version-filtering logic is completely dead.

**Fix Plan:**
The intent was likely to return the version-filtered logprobs:
```python
if versions is not None and current_version is not None:
    return logprobs[versions == current_version]
return old_logp
```

---

## CORRECTNESS

### Bug #4: Vocab-Parallel Logprob Gathering Returns `-inf`

**Location:** `training/logprobs.py:159-173`

**Verification:** CONFIRMED

**Occurrence Condition:**
- Tensor parallelism enabled (`tp_size > 1`)
- At least one candidate token ID falls outside the current TP rank's vocab partition

**Root Cause:**
```python
log_probs_labels[labels_mask] = float("-inf")  # line 172
dist.all_reduce(log_probs_labels, op=dist.ReduceOp.SUM, group=tp_group)  # line 173
```

For a token in rank 0's partition but not rank 1's:
- Rank 0: computes real logprob (e.g., -2.3), `labels_mask` is False, value stays at -2.3
- Rank 1: `labels_mask` is True, value set to `-inf`
- After SUM: `-2.3 + (-inf) = -inf`

**Impact:** Complete loss of gradient signal for multi-candidate distillation loss under tensor parallelism. Any candidate token spanning vocab partitions gets `-inf` logprob.

**Fix Plan:**
Set out-of-range values to 0.0 before all-reduce, so only the owning rank contributes numerically:
```python
log_probs_labels[labels_mask] = 0.0  # was: float("-inf")
dist.all_reduce(log_probs_labels, op=dist.ReduceOp.SUM, group=tp_group)
```

---

### Bug #5: Wrong Token Used for Standard GRPO Loss in Multi-Candidate Mode

**Location:** `training/loss.py:95`

**Verification:** CONFIRMED

**Occurrence Condition:**
- `logprobs.dim() == 2` (multi-candidate gathered)
- Standard GRPO loss path is executed

**Root Cause:**
```python
chosen_logprobs = logprobs if logprobs.dim() == 1 else logprobs[:, 0]
```

The candidate list is built from top-k logprobs ordered by descending probability (`reward_compute.py:84`):
```python
candidates = [tid for tid, _ in pos_logprobs[:top_k]]
```

Index 0 is the **most probable** token, NOT the **actually generated** token. The actually-chosen token's index within candidates is tracked by `PositionRewardInfo.chosen_index` but is never referenced here.

**Impact:** The PPO importance ratio `exp(new_logp - old_logp)` uses the most probable token's logprob instead of the actually-generated token's logprob. This produces an incorrect policy gradient.

**Fix Plan:**
Track which column corresponds to the chosen token and use that index:
```python
# In PositionRewardInfo, ensure candidate_token_ids[0] == chosen token
# OR pass chosen_index to the loss function:
chosen_col = input_data.get("_chosen_column", 0)
chosen_logprobs = logprobs if logprobs.dim() == 1 else logprobs[:, chosen_col]
```

Simpler approach: when building `candidate_token_ids`, always put the chosen token at index 0:
```python
# In _compute_token_rewards:
chosen_tid = student_output_ids[i]
candidates = [chosen_tid] + [tid for tid, _ in pos_logprobs[:top_k] if tid != chosen_tid]
```

---

### Bug #6: Q-Value vs Reward Inconsistency (Fragile / Latent)

**Location:** `advantage.py:61`

**Verification:** CONFIRMED as latent bug

**Occurrence Condition:**
- `AdvantageMode.TREE` is used
- Nodes are ever re-inserted (currently each node is inserted exactly once, so `get_reward` and `get_q_value` are identical)

**Root Cause:**
```python
q_values = [self.tree_store.get_reward(nid) for nid in node_ids]
```

Method comment says "GRPO normalization of Q-values" but `get_reward()` returns raw static `outcome_reward` (set once at `_insert_single:217`), while `get_q_value()` returns MCTS-backed running average (updated by `_backup`). Currently `_backup` is called once per node, so both return the same value. But if tree backup were ever extended to update a node multiple times, this would silently use stale rewards.

**Impact:** Currently none (latent). Would become a real bug if multi-visit backup is implemented.

**Fix Plan:**
Use `get_q_value()` to match the documented intent:
```python
q_values = [self.tree_store.get_q_value(nid) for nid in node_ids]
```

---

### Bug #7: Non-Candidate Positions Filled with Token ID 0

**Location:** `engine/fsdp_engine.py:229-235`

**Verification:** PARTIALLY CONFIRMED (wastes compute, not incorrect)

**Occurrence Condition:**
- Multi-candidate mode active (`position_rewards` is not None)
- `seq_len > num_output_positions` (always, since prompt tokens exist)

**Root Cause:**
```python
labels = torch.zeros((seq_len, max_candidates), dtype=torch.long, device=device)
# Only output positions get real candidates:
for pr in position_rewards:
    labels[position, :num_candidates] = torch.tensor(pr.candidate_token_ids, ...)
```

Prompt positions stay as token ID 0. The engine computes logprobs for token 0 at these positions. For the standard GRPO loss, `loss_mask` excludes prompt positions. For the position-level distillation loss, only positions in `position_rewards` are used (indexed by `positions_t`).

**Impact:** Wasted computation (computing logprobs for prompt positions that are discarded). No correctness issue for the loss. The `entropy` tensor may include prompt-position entropy if used elsewhere.

**Fix Plan:**
Either fill with a clearly invalid sentinel and mask, or pass only output positions to the gather function:
```python
labels = torch.full((seq_len, max_candidates), -100, dtype=torch.long, device=device)
```

---

### Bug #8: Filename Collision in Checkpoint Save

**Location:** `checkpoint.py:17-20, 38-47`

**Verification:** CONFIRMED

**Occurrence Condition:**
- Two different query IDs sanitize to the same filename (e.g., `"a/b"` → `"a_b"` and `"a_b"` → `"a_b"`)
- Both queries have trajectories in the same checkpoint

**Root Cause:**
```python
def _sanitize_filename(query_id: str) -> str:
    return re.sub(r"[^\w\-.]", "_", query_id)
```

Collision example: `query_id="abc/def"` and `query_id="abc_def"` both map to `"query_abc_def.json"`. The second query overwrites the first. Additionally, `query_id_to_file` reverse mapping loses the first query:
```python
file_to_query = {v: k for k, v in query_id_to_file.items()}  # last wins
```

**Impact:** Silent data loss for colliding query IDs.

**Fix Plan:**
Include a hash of the original query_id to ensure uniqueness:
```python
import hashlib
def _sanitize_filename(query_id: str) -> str:
    sanitized = re.sub(r"[^\w\-.]", "_", query_id)
    query_hash = hashlib.md5(query_id.encode()).hexdigest()[:8]
    return f"{sanitized}_{query_hash}"
```

---

### Bug #9: Partial Patch Leak

**Location:** `patches.py:271-278`

**Verification:** CONFIRMED (low risk, narrow window)

**Occurrence Condition:**
- `self._loss_mode != LossMode.GRPO`
- Exception occurs between `patch_ppo_actor_class_to_use_distill_loss()` succeeding and `self._distill_undo = ...` assignment

**Root Cause:**
```python
if self._loss_mode != LossMode.GRPO:
    ...
    patch_ppo_actor_class_to_use_distill_loss()  # modifies PPOActor globally
    self._distill_undo = unpatch_ppo_actor_distill_loss  # saved AFTER
```

The `distill_undo` assignment is on the line immediately after `patch_fn()`, so the window is a single Python statement. But defensive code should assign the undo function first.

**Impact:** If `self._distill_undo` assignment fails (extremely unlikely — would require memory corruption), PPOActor would remain permanently patched. `restore()` can't unpatch because `_distill_undo` is None.

**Fix Plan:**
Assign undo function before calling patch:
```python
if self._loss_mode != LossMode.GRPO:
    self._distill_undo = unpatch_ppo_actor_distill_loss
    patch_ppo_actor_class_to_use_distill_loss()
```

---

### Bug #10: Potential Tensor Indexing Issue

**Location:** `training/actor.py:136-141`

**Verification:** CONFIRMED (depends on `forward_indices` type)

**Occurrence Condition:**
- `forward_indices` is a `torch.Tensor` instead of a Python list
- `_distribute_position_rewards` is called

**Root Cause:**
```python
orig_idx = forward_indices[offset + j]  # may return 0-d tensor
mb_assignment[orig_idx] = i              # list[int] expects Python int
```

**Impact:** `TypeError: list indices must be integers or slices, not Tensor`.

**Fix Plan:**
```python
orig_idx = int(forward_indices[offset + j])
```

---

### Bug #11: Mixed Return Types from `OnPolicyDistillAgent.run`

**Location:** `core/agent.py:234-241`

**Verification:** CONFIRMED

**Occurrence Condition:**
- `OnPolicyDistillAgent.run()` completes
- `position_rewards` is non-empty

**Root Cause:**
```python
if completion_id is not None and position_rewards:
    return {completion_id: {"position_rewards": position_rewards, "scalar_reward": reward}}
return reward  # bare float
```

Returns `dict` when position rewards exist, `float` otherwise. Upstream code (e.g., `OpenAIProxyWorkflow` reward processing) may not handle both types.

**Impact:** Type inconsistency; potential crash in reward processing depending on upstream expectations.

**Fix Plan:**
Always return a consistent structure:
```python
return {
    "reward": reward,
    "position_rewards": position_rewards or None,
    "completion_id": completion_id,
}
```

---

### Bug #12: `should_accept_fn` Parameter Ignored

**Location:** `trainer.py:264-395`

**Verification:** CONFIRMED

**Occurrence Condition:**
- `should_accept_fn` is passed to `_cache_aware_prepare_batch`
- Generated trajectories need filtering

**Root Cause:**
```python
def _cache_aware_prepare_batch(..., should_accept_fn=None, ...):
    ...
    new_trajs = self.actor.rollout_batch(
        gen_prompts,
        workflow=workflow,
        workflow_kwargs=workflow_kwargs,
        group_size=group_size,
        # should_accept_fn NOT passed
    )
```

`should_accept_fn` is received as a parameter but never forwarded to `rollout_batch()`.

**Impact:** Trajectory acceptance filtering is silently disabled for generated trajectories.

**Fix Plan:**
Forward the parameter:
```python
new_trajs = self.actor.rollout_batch(
    gen_prompts,
    workflow=workflow,
    workflow_kwargs=workflow_kwargs,
    group_size=group_size,
    should_accept_fn=should_accept_fn,
)
```

---

## ROBUSTNESS / MINOR

### Bug #13: Variance Uses `n-1` Instead of `n`

**Location:** `advantage.py:75`

**Verification:** CONFIRMED (NEW)

**Root Cause:**
```python
var_q = sum((q - mean_q) ** 2 for q in q_values) / max(len(q_values) - 1, 1)
```

Uses Bessel's correction (`n-1`, unbiased sample variance). Standard GRPO uses biased variance (divide by `n`). For small group sizes (e.g., `n=4`), this produces measurably different advantage magnitudes.

**Impact:** Minor numerical difference. Advantages are slightly larger with `n-1` (Bessel's correction inflates variance estimate for small n).

**Fix Plan:**
```python
var_q = sum((q - mean_q) ** 2 for q in q_values) / max(len(q_values), 1)
```

---

### Bug #14: GPU-CPU Sync on `loss_mask.sum()` Comparison

**Location:** `training/loss.py:375`

**Verification:** CONFIRMED (the comment says "Bug 4 fix" but the fix is incomplete)

**Root Cause:**
```python
# Bug 4 fix: avoid .item() GPU-CPU sync by keeping computation on GPU
output_len = loss_mask.sum()
n_loss = loss_per_position.shape[0]
if n_loss < output_len:  # ← Python int vs GPU tensor → implicit .item()
```

Comparing a Python int with a 0-d GPU tensor triggers implicit `.item()` which forces GPU→CPU synchronization. The comment indicates this was a known issue but the "fix" didn't actually eliminate the sync.

**Impact:** Small GPU→CPU stall on every training step where distillation loss is active. Cumulative slowdown.

**Fix Plan:**
Explicitly convert once, or keep both sides on the same device:
```python
n_loss_t = torch.tensor(n_loss, device=loss_mask.device, dtype=torch.int64)
output_len = loss_mask.sum().to(torch.int64)
if n_loss_t < output_len:
    ...
```

---

### Bug #15: Position Clamping Silently Corrupts Distillation Data

**Location:** `training/loss.py:287-298`

**Verification:** CONFIRMED (NEW)

**Root Cause:**
```python
if position >= logprobs.shape[0]:
    logger.warning(
        "Position %d + prompt_len=%d = %d exceeds logprobs length %d, clamping...",
        pr.position, pl, position, logprobs.shape[0],
    )
    position = logprobs.shape[0] - 1  # clamps to last position
```

When position + prompt_len exceeds logprobs length (due to miscomputed prompt_len), the position is clamped to the last token. This means the loss is computed using the **wrong position's** candidate logprobs, silently producing incorrect gradients.

**Impact:** Silent data corruption in distillation training when prompt_len is wrong.

**Fix Plan:**
Skip the position instead of clamping:
```python
if position >= logprobs.shape[0] or position < 0:
    logger.warning("Skipping position %d: out of bounds", pr.position)
    continue
```

---

### Bug #16: Deprecated Tensor Truthiness in Prompt Length Detection

**Location:** `engine/fsdp_engine.py:220-222`

**Verification:** CONFIRMED (NEW)

**Root Cause:**
```python
for i in range(lm_flat.shape[0]):
    if lm_flat[i]:  # 0-d tensor truthiness — deprecated in PyTorch 2.x
        prompt_len = i
        break
```

Direct truthiness evaluation of 0-d tensors generates deprecation warnings and will be removed in future PyTorch.

**Impact:** Deprecation warnings; will break in future PyTorch versions.

**Fix Plan:**
```python
if lm_flat[i].item():
```

---

### Bug #17: Concat Fallback Silently Produces Garbage

**Location:** `proxy_workflow.py:108-111`

**Verification:** CONFIRMED (NEW)

**Root Cause:**
```python
else:  # resp.input_len <= parent_len — shouldn't normally happen
    logprobs = [0.0] * resp.input_len + resp.output_logprobs
    loss_mask = [0] * resp.input_len + [1] * resp.output_len
    versions = [-1] * resp.input_len + resp.output_versions
```

When `resp.input_len <= parent_len` in concat mode, all prompt-token logprobs are set to 0.0 and versions to -1, discarding parent conversation history. The condition is unexpected (input_len should grow with turns), so this fallback silently produces garbage training data.

**Impact:** If hit, trajectory has incorrect logprobs/versions for all prompt tokens, leading to wrong PPO updates. No crash, no visible error beyond maybe degraded training.

**Fix Plan:**
Raise an explicit error or at minimum log at ERROR level:
```python
else:
    logger.error(
        "concat mode: resp.input_len (%d) <= parent_len (%d) — "
        "expected monotonic growth. Producing zero-lled prompt context.",
        resp.input_len, parent_len
    )
    logprobs = [0.0] * resp.input_len + resp.output_logprobs
    ...
```

---

## Fix Priority

| Priority | Bug | Impact |
|----------|-----|--------|
| P0 | #1 (Type mismatch) | Crash whenever cache is used |
| P0 | #2 (Stale references) | Silent incorrect behavior with checkpoints |
| P0 | #3 (Dead code `_resolve_proximal_logp`) | Version-filtered proximal logprobs never used |
| P1 | #4 (Vocab parallel `-inf`) | Complete signal loss under TP |
| P1 | #5 (Wrong chosen token) | Incorrect policy gradient |
| P2 | #6 (Q-value vs reward) | Fragile / latent — no current impact |
| P2 | #7 (Token ID 0 padding) | Wasted compute, no correctness issue |
| P2 | #8 (Filename collision) | Rare data loss |
| P2 | #9 (Partial patch leak) | Extremely narrow window; defensive fix |
| P2 | #12 (should_accept_fn ignored) | Missing trajectory filter |
| P2 | #14 (GPU-CPU sync) | Minor performance regression |
| P3 | #10 (Tensor indexing) | Crash only if forward_indices is a tensor |
| P3 | #11 (Mixed return types) | Type inconsistency in agent |
| P3 | #13 (n-1 vs n variance) | Minor numerical difference |
| P3 | #15 (Position clamping) | Silent corruption — edge case |
| P3 | #16 (Tensor truthiness) | Deprecation warning |
| P3 | #17 (Concat fallback garbage) | Edge case — shouldn't occur in practice |
