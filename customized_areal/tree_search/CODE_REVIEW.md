# Code Review: Tree Search Training Pipeline

Review of the MCTS tree backup + rollout caching training flow, covering
`trainer.py`, `mcts_tree_store.py`, `advantage.py`, `checkpoint.py`,
`grouped_workflow.py`, `proxy_workflow.py`, `workflow_executor.py`,
`tpfc_agent.py`, `tpfc_dataset.py`, and related training modules
(`training/loss.py`, `training/actor.py`, `training/logprobs.py`,
`engine/actor.py`).

---

## Critical

### 1. `insert_batch` does not skip already-inserted nodes

**File:** `mcts_tree_store.py:204-212`

The docstring says:

> Each Node is inserted directly. Nodes that already have a node_id
> assigned (loaded from cache) are skipped.

But the code never checks:

```python
def insert_batch(self, trajectories: list[Node]) -> None:
    for node in trajectories:
        query_id = getattr(node, "query_id", None) or ""
        self._insert_single(query_id, node)  # always assigns a NEW seq_id
```

`_cache_aware_prepare_batch` (`trainer.py:583`) calls `insert_batch` on
**every** batch — including batches assembled entirely from cache. Every time
a cached trajectory is loaded and reused, it is re-inserted with a fresh
`seq_id`, creating duplicates in all internal dicts.

**Consequences:**

- **Inflated MCTS visit counts** — the same trajectory counted multiple
  times.
- **Biased Q-values** — the same reward averaged in multiple times.
- **Unbounded memory growth** — `_next_seq_id`, `_rewards`, `_trained`,
  `_visit_counts`, `_total_values`, `_q_values`, and `_query_seq_ids` all
  grow without bound across training steps.
- **Incorrect GRPO normalization** — `TreeAdvantageComputer.compute()`
  normalizes Q-values using only seq_ids from the current batch, but
  `_rewards` now spans both original and duplicated entries, making per-query
  mean/std inconsistent across steps.

**Suggested fix:**

```python
def insert_batch(self, trajectories: list[Node]) -> None:
    for node in trajectories:
        existing_id = getattr(node, "node_id", 0)
        if existing_id != 0 and existing_id in self._seq_id_to_key:
            continue  # already inserted (loaded from cache)
        query_id = getattr(node, "query_id", None) or ""
        self._insert_single(query_id, node)
```

---

## High

### 2. Population variance (not Bessel-corrected) in GRPO normalization

**File:** `advantage.py:80`

```python
var_q = sum((q - mean_q) ** 2 for q in q_values) / len(q_values)
```

This uses population variance (divides by N). GRPO advantage normalization
typically uses sample variance (divides by N−1) for unbiased estimates. With
`n_samples=4`, the difference between `/4` and `/3` underestimates variance by
~25%, compressing the advantage signal.

**Fix:**

```python
var_q = sum((q - mean_q) ** 2 for q in q_values) / max(len(q_values) - 1, 1)
```

Also note that `_compute_position_level_grpo_loss` in `training/loss.py:352`
uses the same population variance — apply the fix there too.

---

### 3. `query_id` lost on checkpoint deserialization

**File:** `checkpoint.py:95-131`

`_serialize_record` does not save `query_id` (it is a non-dataclass attribute
set via `object.__setattr__` during `_insert_single`). `_deserialize_record`
does not restore it. Nodes loaded from a checkpoint via
`TreeCheckpointManager.load()` → `load_trajectories()` will return nodes
where `getattr(node, "query_id", None)` returns `None`.

This breaks `TreeAdvantageComputer._get_query_id()` (`advantage.py:42`),
which falls back to `None` and silently skips those trajectories during
advantage computation. The `query_id` is later re-set by `_insert_single`
(partially masking the problem), but any code path that reads
`node.query_id` between `load_trajectories` and `insert_batch` sees `None`.

**Fix:** Either store `query_id` in `_serialize_record`/restore it in
`_deserialize_record`, or re-derive it from the checkpoint filename
(`query_<id>.json`) at load time.

---

## Medium

### 4. `split_prompts` fallback checks the same key twice

**File:** `trainer.py:317`

```python
query_id = prompt.get("query_id") or prompt.get("query_id") or ""
```

Both branches check `"query_id"`. The docstring at lines 303-306 lists two
identical fallback descriptions. Likely a copy-paste error — the second
branch may have been intended for a different key. Simplify to:

```python
query_id = prompt.get("query_id") or ""
```

---

### 6. No safeguard if `_tree_advantages` lost in the tensor pipeline

**File:** `trainer.py:254-276` and `trainer.py:587-624`

The stash path is: `Node._tree_advantages` (via `object.__setattr__`) →
`tensor_dict["_tree_advantages"]` (via `_node_to_tensor_dict`) → survives
`concat_padded_tensors` → popped in the patched `compute_advantages`.

If any intermediate step strips unknown keys, tree advantages are silently
lost and GAE advantages are used instead — with no warning. Add a diagnostic:

```python
if advantage_mode == AdvantageMode.TREE:
    restored = 0
    for traj in result:
        tree_adv = traj.pop("_tree_advantages", None)
        tree_ret = traj.pop("_tree_returns", None)
        if tree_adv is not None:
            traj["advantages"] = tree_adv
            traj["returns"] = tree_ret
            restored += 1
    if restored < len(result):
        logger.warning(
            f"Tree advantages missing for {len(result) - restored}/{len(result)} "
            f"trajectories in TREE mode — fell back to GAE"
        )
```

---

### 7. Class-level monkey-patches leak on crash

**File:** `trainer.py:460-491` and `trainer.py:704-718`

`_init_patches` applies four class-level monkey-patches:

- `PPOActor.compute_advantages` → tree backup version
- `engine._wrap_openai_agent` → `TreeSearchGroupedRolloutWorkflow`
- `engine.workflow_executor` → `TreeSearchWorkflowExecutor`
- `PPOActor._ppo_update` → distill loss version (conditional)

While `prepare_batch` is patched inside a `try/finally` in `train()`, these
other patches are only cleaned up in `close()`. If training crashes between
`_init_patches` and `close()`, the class-level patches persist, corrupting
subsequent trainer instances in the same process.

**Fix:** Wrap all patches in a context manager that guarantees cleanup, or
move the patching into the `train()` method's `try/finally` block.

---

### 8. `episode_id` 重复：空 `query_id` 和跨 epoch 两种场景

**File:** `grouped_workflow.py:49`

```python
episode_id = f"{query_id}_{group_idx}" if query_id else f"{group_idx}"
```

**场景 A — `query_id` 为空时跨 query 重复：**

当 `query_id` 为空字符串时，`episode_id` 退化为 `"0"`, `"1"`, `"2"`, `"3"`。
如果多个 query 的 `query_id` 都为空，它们的 episode 会产生完全相同的 ID
集合（`"0"`, `"1"`, ...），在 tree store 里无法区分不同 query 的 episode。

**场景 B — CROSS_TRAINING 跨 epoch 重复：**

在 CROSS_TRAINING 模式下，同一个 query 在每个 epoch 都会重新 rollout。由于
Bug #1（`insert_batch` 不跳过已插入节点），旧 epoch 和新 epoch 的 Node
**同时存在**于 `trajectories[query_id]` 中，产生同名冲突：

```
epoch 1: query1 → Node(episode_id="query1_0", seq_id=0, trained=True)
                  Node(episode_id="query1_1", seq_id=1, trained=True)
epoch 2: query1 → Node(episode_id="query1_0", seq_id=16, trained=False)  ← 同名！
                  Node(episode_id="query1_1", seq_id=17, trained=False)  ← 同名！
```

两条不同的 trajectory 共享同一个 `episode_id`，仅靠 `seq_id` 区分。当前代码
没有按 `episode_id` 做查询或分组，因此暂不触发逻辑 bug，但会破坏：
- **调试可追溯性**：无法从 `episode_id` 判断轨迹来自哪个 epoch
- **`_turn_nodes` 映射**：multi-turn 场景下 turn_id → seq_id 可能被新 epoch
  覆盖（如果有多 turn 节点）
- **未来扩展风险**：按 `episode_id` 聚合/可视化会拿到混合 epoch 的数据

**Fix:**

```python
import uuid

query_id = data.get("query_id") or ""
if not query_id:
    logger.warning(
        "query_id is empty; episode_id will not be unique across queries"
    )

for group_idx, result in enumerate(valid_results):
    # 加入 uuid 保证跨 query 和跨 epoch 唯一性
    episode_id = f"{query_id}_{group_idx}_{uuid.uuid4().hex[:8]}"
    ...
```

注意：`episode_id` 目前只存在于 `Node` 对象上，`_node_to_tensor_dict` 不将它
带入 tensor dict，所以下游 PPO pipeline 不受影响。但 checkpoint 会序列化
`episode_id`（`checkpoint.py:103`），跨 epoch 的重名节点在 checkpoint 中也会
保留冲突的 ID。

---

## Low

### 9. `_tree_search_wrap` returns None silently on missing agent config

**File:** `trainer.py:108-114`

```python
def _tree_search_wrap(agent: Any, proxy_addr: str):
    agent_cfg = engine.config.agent
    if agent_cfg is None:
        logger.warning(...)
        return  # implicit None — causes opaque error later
```

If `config.agent` is `None`, the patched function returns `None`. Later code
will fail with an opaque `AttributeError` deep in the rollout pipeline. Raise
an explicit error instead.

---

### 10. `_patch_workflow_executor` copies private attributes by name

**File:** `trainer.py:201-210`

```python
tree_search_executor._staleness_manager = original_executor._staleness_manager
tree_search_executor._expected_trajectory_keys = ...
tree_search_executor._task_id_generator = ...
# ... 10 more lines of individual attribute copies
```

This is brittle — if `WorkflowExecutor` adds or renames a private attribute
upstream, the copy silently misses it. Use a `__dict__` update pattern
instead.

---

### 11. Cache-aware dataloader iterator stored as bare attribute

**File:** `trainer.py:538-539`

```python
if not hasattr(self, "_cache_dataloader_iter"):
    self._cache_dataloader_iter = iter(cycle_dataloader(dataloader))
```

If the `finally` block in `train()` doesn't execute cleanly, a stale iterator
persists on the trainer instance and would be reused on the next `train()`
call.

**Fix:** Reset the iterator explicitly at the start of each `train()` call,
or key it by dataloader identity.

---

### 12. Plaintext API key in config YAML

**File:** `configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml:219`

```yaml
swanlab:
    api_key: WVB9KhdtDjWVAozYQgHu5
```

API keys committed in config files risk credential exposure. Use environment
variable references or a secret manager.

---

## Readability

### 13. Unused `query_id` parameter in `set_trained`

**File:** `mcts_tree_store.py:232`

```python
def set_trained(self, query_id: str, seq_id: int, trained: bool = True) -> None:
    self._trained[seq_id] = trained  # query_id never used
```

The `query_id` parameter is accepted but never read. Either remove it from
the signature or use it for assertion/validation.

---

### 14. `dict[int, None]` used as a set

**File:** `advantage.py:59`

```python
query_seq_sets: dict[str, dict[int, None]] = {}
```

Use `dict[str, set[int]]` for clarity.

---

### 15. Misleading comment: "flat list of per-episode dicts"

**File:** `trainer.py:569`

```python
# TreeSearchWorkflowExecutor already returns flat list of per-episode dicts
trajs = new_trajs if new_trajs else []
```

These are `Node` objects, not dicts. Conversion to tensor dicts happens later
at lines 619-624.

---

### 16. Inconsistent naming: `seq_id` vs `node_id`

The same concept (unique trajectory identifier) is called `seq_id` in
`MCTSTreeStore` internals (`_seq_id_to_key`, `_visit_counts[seq_id]`,
`_rewards[seq_id]`) but `node_id` on `Node` objects and in trajectory dicts.
Unify to one name.

---

### 17. `_node_to_tensor_dict` is large and repetitive

**File:** `mcts_tree_store.py:87-150`

This 64-line function handles core tensor conversion, optional field
serialization, advantage passthrough, tree advantage stashing, and turn
metadata injection. The optional-field section has six near-identical
`if X is not None` blocks. Consider extracting a helper for the optional
tensor fields.

---

### 18. Deeply nested monkey-patching is hard to follow

See the detailed refactoring proposal below.

---

## Refactoring: Consolidate Monkey-Patching into `TreeSearchPatches`

### Problem

`_init_patches` (`trainer.py:460-491`) applies four interdependent
monkey-patches across three targets:

| # | Target | Patch | Applied by | Reversed by |
|---|--------|-------|------------|-------------|
| 1 | `PPOActor.compute_advantages` | Tree backup after GAE | `patch_ppo_actor_for_tree_backup` | `unpatch_ppo_actor` |
| 2 | `engine._wrap_openai_agent` | `TreeSearchGroupedRolloutWorkflow` | `_patch_wrap_openai_agent_for_tree_search` | `_unpatch_wrap_openai_agent` |
| 2b | `engine._resolve_workflow` | Strip outer `GroupedRolloutWorkflow` | Inside `_patch_wrap_openai_agent_for_tree_search` | Inside `_unpatch_wrap_openai_agent` |
| 3 | `engine.workflow_executor` | `TreeSearchWorkflowExecutor` | `_patch_workflow_executor` | `_unpatch_workflow_executor` |
| 4 | `PPOActor._ppo_update` | Distill loss fn | `patch_ppo_actor_class_to_use_distill_loss` | `unpatch_ppo_actor_distill_loss` |

These are spread across 6 top-level functions (`_patch_*`, `_unpatch_*`,
`patch_ppo_actor_for_tree_backup`, `unpatch_ppo_actor`) totaling ~140 lines.
The problems:

1. **No atomicity.** Patches 1-4 are applied sequentially in `_init_patches`
   with no rollback if patch 3 fails after patches 1-2 succeed. On crash,
   `_init_patches` leaves a partially-patched system.

2. **No lifecycle guarantee.** Patches 1, 2, 3, 4 are applied in
   `__init__` → `_init_patches` but only reversed in `close()`. The
   `train()` method adds yet another patch (`prepare_batch`) inside a
   `try/finally`, creating two different patch lifecycles in the same class.
   If `train()` crashes, `prepare_batch` is restored but the other 4 patches
   remain.

3. **Double-wrapping prevention is hidden.** Patch 2b is nested *inside*
   patch 2 because `_resolve_workflow` would otherwise wrap the already-grouped
   `TreeSearchGroupedRolloutWorkflow` in an outer `GroupedRolloutWorkflow`.
   This is a subtle coupling: patch 2b exists only because of a side effect
   of the upstream `_resolve_workflow` implementation (lines 560-562 in
   `remote_inf_engine.py`). The relationship is only documented in comments,
   not enforced structurally.

4. **Private attribute copy is brittle.** Patch 3 copies 7 private attributes
   by name from the original `WorkflowExecutor` to `TreeSearchWorkflowExecutor`.
   If upstream adds/renames one, this silently breaks.

5. **Duplicated engine-unwrapping logic.** `_get_underlying_engine` is called
   at the start of both `_patch_wrap_openai_agent_for_tree_search` and
   `_patch_workflow_executor`, and again in both `_unpatch_*` counterparts.

6. **Idempotency is ad-hoc.** `patch_ppo_actor_for_tree_backup` checks
   `_original_compute_advantages` to avoid stacking, but the other patches
   have no such guard.

### Proposed Design: `TreeSearchPatches` context manager

Consolidate all patching into a single class that:
- Tracks every original value before overwriting
- Provides `apply()` / `restore()` as atomic-ish operations
- Is usable as a context manager (`with TreeSearchPatches(...) as p:`)
- Guarantees partial rollback on failure
- Makes the double-wrapping prevention explicit as a first-class concern

```python
class TreeSearchPatches:
    """Manages all monkey-patches needed for tree search training.

    Usage:
        patches = TreeSearchPatches(
            rollout_engine=self.rollout,
            advantage_mode=...,
            loss_mode=...,
            group_size=...,
        )
        patches.apply()
        try:
            ...
        finally:
            patches.restore()

    Or as a context manager:
        with TreeSearchPatches(...) as patches:
            ...
    """

    def __init__(
        self,
        rollout_engine: Any,
        advantage_mode: AdvantageMode,
        loss_mode: LossMode,
        group_size: int,
    ):
        self._engine = self._unwrap_engine(rollout_engine)
        self._advantage_mode = advantage_mode
        self._loss_mode = loss_mode
        self._group_size = group_size

        # Stores (target_obj, attr_name, original_value) for every patch
        self._saved: list[tuple[Any, str, Any]] = []
        self._applied = False

    # ------------------------------------------------------------------
    # Engine unwrapping (single copy)
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_engine(engine: Any) -> Any:
        """Unwrap decorators (e.g. RemoteSGLangEngine) to RemoteInfEngine."""
        if not hasattr(engine, "_wrap_openai_agent") and hasattr(engine, "_engine"):
            return engine._engine
        return engine

    # ------------------------------------------------------------------
    # Low-level patch primitives
    # ------------------------------------------------------------------

    def _save_and_set(self, obj: Any, attr: str, new_value: Any) -> None:
        """Save the current value of obj.attr, then replace it."""
        original = getattr(obj, attr)
        self._saved.append((obj, attr, original))
        setattr(obj, attr, new_value)

    def _save_and_set_method(self, obj: Any, attr: str, new_method: Any) -> None:
        """Save and replace a method (binds new_method to obj)."""
        original = getattr(obj, attr)
        self._saved.append((obj, attr, original))
        setattr(obj, attr, new_method.__get__(obj, type(obj)))

    # ------------------------------------------------------------------
    # Individual patch builders (return the replacement value)
    # ------------------------------------------------------------------

    def _build_tree_backup_compute_advantages(self):
        """Build patched compute_advantages that restores tree advantages."""
        # Get the true original (avoid stacking if apply() is called twice)
        if hasattr(PPOActor, "_original_compute_advantages"):
            original = PPOActor._original_compute_advantages
        else:
            original = PPOActor.compute_advantages
        advantage_mode = self._advantage_mode

        def _patched(self_actor, data):
            result = original(self_actor, data)
            if advantage_mode == AdvantageMode.TREE:
                restored = 0
                for traj in result:
                    tree_adv = traj.pop("_tree_advantages", None)
                    tree_ret = traj.pop("_tree_returns", None)
                    if tree_adv is not None:
                        traj["advantages"] = tree_adv
                        traj["returns"] = tree_ret
                        restored += 1
                if restored < len(result):
                    logger.warning(
                        f"Tree advantages missing for "
                        f"{len(result) - restored}/{len(result)} "
                        f"trajectories in TREE mode — fell back to GAE"
                    )
            return result

        return _patched

    def _build_tree_search_wrap(self):
        """Build patched _wrap_openai_agent that returns
        TreeSearchGroupedRolloutWorkflow."""
        engine = self._engine
        group_size = self._group_size
        original_wrap = engine._wrap_openai_agent

        def _tree_search_wrap(agent, proxy_addr):
            agent_cfg = engine.config.agent
            if agent_cfg is None:
                raise RuntimeError(
                    "config.agent is None; tree search workflow requires "
                    "agent configuration. Set agent.mode in the config."
                )
            inner = QueryIDProxyWorkflow(
                mode=agent_cfg.mode,
                agent=agent,
                proxy_addr=proxy_addr,
                admin_api_key=agent_cfg.admin_api_key,
                discount=agent_cfg.turn_discount,
                export_style=agent_cfg.export_style,
                subproc_max_workers=agent_cfg.subproc_max_workers,
                proxy_gateway_addr=getattr(engine, "_proxy_gateway_addr", None),
            )
            return TreeSearchGroupedRolloutWorkflow(
                workflow=inner,
                group_size=group_size,
                logger=logger,
            )

        return _tree_search_wrap

    def _build_patched_resolve(self):
        """Build patched _resolve_workflow that strips outer
        GroupedRolloutWorkflow when the inner workflow is already
        TreeSearchGroupedRolloutWorkflow.

        This prevents double-wrapping: the upstream _resolve_workflow
        unconditionally wraps with GroupedRolloutWorkflow when
        group_size > 1 (see remote_inf_engine.py:560-562), but
        TreeSearchGroupedRolloutWorkflow already handles grouping
        internally.
        """
        engine = self._engine
        original_resolve = engine._resolve_workflow

        def _patched_resolve(self_engine, wf, wf_kwargs=None, gs=1):
            resolved = original_resolve(wf, wf_kwargs, gs)
            if isinstance(resolved, GroupedRolloutWorkflow) and isinstance(
                resolved.workflow, TreeSearchGroupedRolloutWorkflow
            ):
                logger.debug(
                    "Skipping outer GroupedRolloutWorkflow wrapper "
                    "(TreeSearchGroupedRolloutWorkflow already handles grouping)"
                )
                return resolved.workflow
            return resolved

        return _patched_resolve

    def _build_tree_search_executor(self):
        """Build a TreeSearchWorkflowExecutor that replaces the original."""
        engine = self._engine
        original = engine.workflow_executor

        new_executor = TreeSearchWorkflowExecutor(
            config=engine.config,
            inference_engine=engine,
        )

        # Copy all internal state from the original executor.
        # Using vars() instead of listing attributes by name so that
        # upstream additions are automatically picked up.
        for attr, value in vars(original).items():
            if not attr.startswith("__") and attr != "config" and attr != "inference_engine":
                setattr(new_executor, attr, value)
        new_executor._initialized = True

        return new_executor

    # ------------------------------------------------------------------
    # Apply / Restore
    # ------------------------------------------------------------------

    def apply(self) -> None:
        """Apply all patches. On failure, roll back any already-applied patches."""
        if self._applied:
            logger.warning("TreeSearchPatches.apply() called twice; skipping")
            return

        try:
            # Patch 1: PPOActor.compute_advantages (class-level)
            new_compute_adv = self._build_tree_backup_compute_advantages()
            # Save original at class level for idempotency tracking
            if not hasattr(PPOActor, "_original_compute_advantages"):
                PPOActor._original_compute_advantages = PPOActor.compute_advantages
            PPOActor.compute_advantages = new_compute_adv
            self._saved.append(
                (PPOActor, "compute_advantages", PPOActor._original_compute_advantages)
            )

            # Patch 2: engine._wrap_openai_agent
            self._save_and_set(
                self._engine,
                "_wrap_openai_agent",
                self._build_tree_search_wrap(),
            )

            # Patch 2b: engine._resolve_workflow (double-wrapping prevention)
            if hasattr(self._engine, "_resolve_workflow"):
                self._save_and_set_method(
                    self._engine,
                    "_resolve_workflow",
                    self._build_patched_resolve(),
                )

            # Patch 3: engine.workflow_executor
            new_executor = self._build_tree_search_executor()
            self._save_and_set(self._engine, "workflow_executor", new_executor)

            # Patch 4 (conditional): PPOActor._ppo_update distill loss
            if self._loss_mode != LossMode.GRPO:
                from customized_areal.tree_search.training.actor import (
                    patch_ppo_actor_class_to_use_distill_loss,
                    unpatch_ppo_actor_distill_loss,
                )
                patch_ppo_actor_class_to_use_distill_loss()
                # Store the unpatch function for restore
                self._saved.append(
                    (None, "_distill_loss_unpatch", unpatch_ppo_actor_distill_loss)
                )

            self._applied = True
            logger.info(
                f"Applied tree search patches "
                f"(advantage={self._advantage_mode.value}, "
                f"loss={self._loss_mode.value}, "
                f"group_size={self._group_size})"
            )

        except Exception:
            # Roll back any patches that were already applied
            self.restore()
            raise

    def restore(self) -> None:
        """Restore all original values in reverse order."""
        if not self._applied and not self._saved:
            return

        # Restore in reverse order (LIFO) so inner patches are undone first
        for obj, attr, original in reversed(self._saved):
            if obj is None and attr == "_distill_loss_unpatch":
                # Call the unpatch function for distill loss
                original()
                continue
            setattr(obj, attr, original)

        # Clean up idempotency marker
        if hasattr(PPOActor, "_original_compute_advantages"):
            del PPOActor._original_compute_advantages

        self._saved.clear()
        self._applied = False
        logger.info("Restored all tree search patches")

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()
        return False
```

### How `CacheAwarePPOTrainer` changes

```python
class CacheAwarePPOTrainer(PPOTrainer):

    def __init__(self, config, cache_config=None, tree_backup_config=None,
                 train_dataset=None, valid_dataset=None):
        self.cache_config = cache_config or RolloutCacheConfig()
        self.tree_backup_config = tree_backup_config or TreeBackupConfig()
        super().__init__(config, train_dataset, valid_dataset)

        self._patches: TreeSearchPatches | None = None

        if (self.cache_config.enabled
                and self.tree_backup_config.mode != TreeBackupMode.OFF):
            self._init_tree_components()
            # Defer patching to train() for crash safety
            self._patches = TreeSearchPatches(
                rollout_engine=self.rollout,
                advantage_mode=self.tree_backup_config.advantage_mode,
                loss_mode=self.tree_backup_config.loss_mode,
                group_size=self.cache_config.n_samples,
            )

    def train(self, workflow=None, eval_workflow=None,
              workflow_kwargs=None, eval_workflow_kwargs=None,
              dynamic_filter_fn=None, total_epochs=None):
        if not self.cache_config.enabled:
            return super().train(...)

        # Apply patches + prepare_batch override inside a single try/finally
        assert self._patches is not None
        original_prepare_batch = self.actor.prepare_batch
        self._patches.apply()

        def _prepare_batch_fn(dataloader, workflow, workflow_kwargs=None,
                              should_accept_fn=None, group_size=1, dynamic_bs=False):
            return self._cache_aware_prepare_batch(...)

        self.actor.prepare_batch = _prepare_batch_fn

        try:
            return super().train(...)
        finally:
            self.actor.prepare_batch = original_prepare_batch
            self._patches.restore()
            if hasattr(self, "_cache_dataloader_iter"):
                del self._cache_dataloader_iter

    def close(self):
        # Patches are already restored by train()'s finally block,
        # but guard against the case where train() was never called.
        if self._patches is not None:
            self._patches.restore()
        super().close()
```

### What this solves

| Problem | Before | After |
|---------|--------|-------|
| **No atomicity** | `_init_patches` has no rollback on failure | `apply()` wraps in `try/except` with full rollback |
| **No lifecycle guarantee** | Patches live from `__init__` to `close()` | Patches scoped to `train()`'s `try/finally` |
| **Double-wrapping hidden** | Patch 2b nested inside patch 2 function | `build_patched_resolve()` is a separate builder with its own docstring explaining the upstream coupling |
| **Brittle attr copy** | 7 attributes copied by name | `vars(original)` loop picks up new attrs automatically |
| **Duplicated unwrapping** | `_get_underlying_engine` called 4 times | Called once in `__init__`, stored as `self._engine` |
| **No idempotency guard** | Only `patch_ppo_actor` checks for double-apply | `apply()` returns early if already applied |
| **Silent None on config error** | `_tree_search_wrap` returns `None` on missing config | Raises `RuntimeError` with actionable message |
| **Missing tree advantage warning** | Silent fallback to GAE | Logs warning with count of missing trajectories |

### Migration steps

1. Create `patches.py` (or add `TreeSearchPatches` to `trainer.py`) with the
   class above.
2. Remove the 6 top-level functions: `_patch_wrap_openai_agent_for_tree_search`,
   `_unpatch_wrap_openai_agent`, `_patch_workflow_executor`,
   `_unpatch_workflow_executor`, `patch_ppo_actor_for_tree_backup`,
   `unpatch_ppo_actor`. Also remove `_get_underlying_engine`.
3. Simplify `_init_patches` to just instantiate `TreeSearchPatches` and store
   it as `self._patches` (don't call `apply()` yet).
4. Move `apply()` into `train()`'s `try/finally` alongside the `prepare_batch`
   patch.
5. Simplify `close()` to call `self._patches.restore()` as a safety net.
6. Add a test that verifies patches are fully restored after `train()` exits
   (both normally and on exception).

---

## Advice

### Move `cache_dir` validation into the trainer

**File:** `trainer.py:388-397` and `scripts/train_tpfc_tree_search.py:62-67`

`cache_dir` is validated only in the training script. If the trainer is
constructed programmatically, the error surfaces deep in training. Move the
validation into `CacheAwarePPOTrainer.__init__`.

### Add `torch.no_grad()` to advantage computation

**File:** `advantage.py:49-99`

`compute()` creates tensors and does arithmetic without `torch.no_grad()`.
While these tensors aren't on a computation graph in practice, wrapping with
`@torch.no_grad()` makes intent explicit and prevents accidental gradient
tracking.

### Add a unit test for the `insert_batch` skip

A test that inserts a Node, loads it via `load_trajectories`, re-inserts it,
and verifies the store has exactly one copy would catch regressions of bug #1.

### Consider `query_id` deduplication in `MCTSTreeStore`

If multiple trajectories legitimately share the same `query_id`, the store
handles them correctly. But if the same trajectory is inserted twice
(different `Node` objects with the same content), there's no protection.
Adding a content hash check in `_insert_single` could prevent silent
duplication.

---

## Priority Summary

| Severity   | #   | Issue                                          | Impact                                   |
| ---------- | --- | ---------------------------------------------- | ---------------------------------------- |
| Critical   | 1   | Cached trajectories re-inserted with new IDs   | Unbounded memory growth, biased Q-values |
| High       | 2   | Population variance instead of sample variance | ~25% variance underestimate at n=4       |
| High       | 3   | `query_id` lost on checkpoint deserialization  | Silent skip during advantage compute     |
| Medium     | 4   | Duplicate condition in `split_prompts`         | Dead code / misleading                   |
| Medium     | 5   | No safeguard if `_tree_advantages` lost        | Silent GAE fallback                      |
| Medium     | 6   | Class-level patches leak on crash              | Cross-trainer corruption                 |
| Medium     | 7   | `episode_id` cross-query and cross-epoch dup   | Traceability, future-use risk            |
| Low        | 8   | `_tree_search_wrap` returns None silently      | Opaque error downstream                  |
| Low        | 9   | Private attribute copy by name                 | Breaks on upstream refactors             |
| Low        | 10  | Stale dataloader iterator on retry             | Wrong data after failed run              |
| Security   | 11  | Plaintext API key in config                    | Credential exposure                      |

---

## Alternative Refactoring: Approach B — Lightweight `PatchRegistry`

### Rationale

Approach A (`TreeSearchPatches`, above) is a ~200-line class with builder methods
per patch, config-driven construction, and mixed-value undo stack. It's thorough
but introduces substantial new infrastructure for 6 patches. Approach B takes
the opposite trade: a ~30-line `PatchRegistry` that does one thing — a LIFO stack
of undo actions — and leaves the rest to the existing patch/unpatch functions.

### Core Class

```python
from collections.abc import Callable
from typing import Any


class PatchRegistry:
    """LIFO stack of undo actions for monkey-patches.

    Each ``save_and_set`` or ``add_undo`` call pushes one undo action.  When
    ``restore()`` is called, actions are replayed in **reverse** order so that
    inner patches are undone before outer ones.  The context-manager protocol
    makes it usable inside a ``with`` block.

    Usage::

        registry = PatchRegistry()
        registry.save_and_set(engine, "_wrap_openai_agent", patched_fn)
        registry.add_undo(lambda: delattr(engine, "_sentinel"))
        ...
        registry.restore()

        # Prefer the context manager in train():
        with PatchRegistry() as registry:
            apply_patches(registry)
            ...
    """

    def __init__(self) -> None:
        self._undos: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def save_and_set(self, obj: Any, attr: str, new_value: Any) -> None:
        """Replace *obj.attr* with *new_value*; push a ``setattr(original)`` undo.

        The undo closure captures the original value by eager evaluation
        (default-argument binding), so loop variables are safe.
        """
        original = getattr(obj, attr)
        self._undos.append(lambda orig=original: setattr(obj, attr, orig))
        setattr(obj, attr, new_value)

    def save_method_and_set(
        self, obj: Any, attr: str, new_method: Any
    ) -> None:
        """Like :meth:`save_and_set`, but binds *new_method* to *obj* first.

        Use this when the replacement is a function that needs to receive
        ``self`` as its first argument (e.g. patching instance methods).
        """
        original = getattr(obj, attr)
        self._undos.append(lambda orig=original: setattr(obj, attr, orig))
        setattr(obj, attr, new_method.__get__(obj, type(obj)))

    def add_undo(self, undo_fn: Callable[[], None]) -> None:
        """Push a custom undo action.

        Use this for cleanup that isn't a simple ``setattr``: deleting
        sentinel attributes, calling multi-step unpatch functions, etc.
        """
        self._undos.append(undo_fn)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def restore(self) -> None:
        """Replay all undo actions in reverse order, then clear the stack.

        Safe to call multiple times — second call is a no-op.
        """
        for undo in reversed(self._undos):
            undo()
        self._undos.clear()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PatchRegistry":
        return self

    def __exit__(self, *args: Any) -> bool:
        self.restore()
        return False  # don't suppress exceptions
```

**Why `lambda orig=original` instead of `functools.partial`?**

Default-argument binding eagerly evaluates `original` at call time.
`functools.partial(setattr, obj, attr, original)` would capture `obj`
and `original` by reference — if either is rebound later the undo breaks.
The lambda pattern is also one fewer import.

### Refactored Patch Functions

Each existing function gains a `registry: PatchRegistry` first parameter.
Instead of doing raw `obj.attr = new_value` and stashing `_original_*`
sentinels, it calls `registry.save_and_set()` for the attribute and
`registry.add_undo()` for sentinel cleanup.

```python
# --- Patch 1: PPOActor.compute_advantages ---

def patch_ppo_actor_for_tree_backup(
    registry: PatchRegistry,
    advantage_mode: AdvantageMode = AdvantageMode.TREE,
) -> None:
    """Patch PPOActor.compute_advantages to restore tree advantages after GAE.

    Registers undo actions with *registry* so that a single
    ``registry.restore()`` call reverts everything.
    """
    if hasattr(PPOActor, "_original_compute_advantages"):
        original = PPOActor._original_compute_advantages
    else:
        original = PPOActor.compute_advantages

    # Idempotency sentinel — keeps the "true original" stable across
    # patching / unpatching cycles within the same process lifetime.
    PPOActor._original_compute_advantages = original
    registry.add_undo(lambda: (
        delattr(PPOActor, "_original_compute_advantages")
        if hasattr(PPOActor, "_original_compute_advantages")
        else None
    ))

    def _patched(self_actor, data):
        result = original(self_actor, data)
        if advantage_mode == AdvantageMode.TREE:
            restored = 0
            for traj in result:
                tree_adv = traj.pop("_tree_advantages", None)
                tree_ret = traj.pop("_tree_returns", None)
                if tree_adv is not None:
                    traj["advantages"] = tree_adv
                    traj["returns"] = tree_ret
                    restored += 1
            if restored < len(result):
                logger.warning(
                    f"Tree advantages missing for "
                    f"{len(result) - restored}/{len(result)} "
                    f"trajectories in TREE mode — fell back to GAE"
                )
        return result

    registry.save_and_set(PPOActor, "compute_advantages", _patched)


# --- Patch 2: engine._wrap_openai_agent ---

def patch_wrap_openai_agent_for_tree_search(
    registry: PatchRegistry,
    rollout_engine: Any,
    group_size: int,
) -> None:
    """Patch engine._wrap_openai_agent to use TreeSearchGroupedRolloutWorkflow."""
    engine = _get_underlying_engine(rollout_engine)
    if not hasattr(engine, "_wrap_openai_agent"):
        logger.warning(
            "Engine has no _wrap_openai_agent method; "
            "tree search workflow will not be available"
        )
        return

    registry.save_and_set(
        engine,
        "_wrap_openai_agent",
        _build_tree_search_wrap(engine, group_size),
    )

    # Patch 2b: _resolve_workflow (double-wrapping prevention).
    # The upstream _resolve_workflow unconditionally wraps with
    # GroupedRolloutWorkflow when group_size > 1
    # (remote_inf_engine.py:560-562), but
    # TreeSearchGroupedRolloutWorkflow already handles grouping internally.
    # We strip the outer wrapper when the inner is already
    # TreeSearchGroupedRolloutWorkflow.
    if hasattr(engine, "_resolve_workflow"):
        registry.save_method_and_set(
            engine,
            "_resolve_workflow",
            _build_patched_resolve(engine),
        )


# --- Patch 3: engine.workflow_executor ---

def patch_workflow_executor(
    registry: PatchRegistry,
    rollout_engine: Any,
) -> None:
    """Replace engine.workflow_executor with TreeSearchWorkflowExecutor."""
    engine = _get_underlying_engine(rollout_engine)
    if not hasattr(engine, "workflow_executor"):
        logger.warning(
            "Engine has no workflow_executor attribute; "
            "tree search workflow executor will not be available"
        )
        return

    original_executor = engine.workflow_executor
    new_executor = TreeSearchWorkflowExecutor(
        config=engine.config,
        inference_engine=engine,
    )

    # Copy internal state from original using vars() so upstream
    # additions are automatically picked up (no attribute-name list).
    for attr, value in vars(original_executor).items():
        if attr.startswith("__"):
            continue
        if attr in ("config", "inference_engine"):
            continue
        setattr(new_executor, attr, value)
    new_executor._initialized = True

    registry.save_and_set(engine, "workflow_executor", new_executor)


# --- Patch 4 (conditional): PPOActor._ppo_update distill loss ---

def patch_distill_loss(registry: PatchRegistry) -> None:
    """Patch PPOActor._ppo_update to use grpo_distill_loss_fn."""
    from customized_areal.tree_search.training.actor import (
        patch_ppo_actor_class_to_use_distill_loss,
        unpatch_ppo_actor_distill_loss,
    )

    patch_ppo_actor_class_to_use_distill_loss()
    registry.add_undo(unpatch_ppo_actor_distill_loss)
```

### How `CacheAwarePPOTrainer` Changes

```python
class CacheAwarePPOTrainer(PPOTrainer):

    def __init__(self, config, cache_config=None, tree_backup_config=None,
                 train_dataset=None, valid_dataset=None):
        self.cache_config = cache_config or RolloutCacheConfig()
        self.tree_backup_config = tree_backup_config or TreeBackupConfig()
        super().__init__(config, train_dataset, valid_dataset)

        if (self.cache_config.enabled
                and self.tree_backup_config.mode != TreeBackupMode.OFF):
            self._init_tree_components()

    # _init_patches() is REMOVED — patches are created and applied
    # inside train(), not in __init__().

    def _apply_all_patches(self, registry: PatchRegistry) -> None:
        """Register every tree-search patch with *registry*.

        Called inside train()'s try/finally.  The registry guarantees
        that all patches are undone on exit, even if one of the later
        patches fails.
        """
        patch_ppo_actor_for_tree_backup(
            registry,
            advantage_mode=self.tree_backup_config.advantage_mode,
        )
        patch_wrap_openai_agent_for_tree_search(
            registry,
            self.rollout,
            group_size=self.cache_config.n_samples,
        )
        patch_workflow_executor(registry, self.rollout)

        if self.tree_backup_config.loss_mode != LossMode.GRPO:
            patch_distill_loss(registry)

    def train(self, workflow=None, eval_workflow=None,
              workflow_kwargs=None, eval_workflow_kwargs=None,
              dynamic_filter_fn=None, total_epochs=None):
        if not self.cache_config.enabled:
            return super().train(...)

        original_prepare_batch = self.actor.prepare_batch
        registry = PatchRegistry()

        try:
            self._apply_all_patches(registry)

            # prepare_batch override (single-attr, same lifecycle)
            registry.save_and_set(
                self.actor,
                "prepare_batch",
                _prepare_batch_fn,
            )

            return super().train(...)
        finally:
            # Single restore() undoes everything in correct LIFO order
            registry.restore()
            self.actor.prepare_batch = original_prepare_batch
            if hasattr(self, "_cache_dataloader_iter"):
                del self._cache_dataloader_iter

    def close(self):
        # Patches are scoped to train()'s finally block, so there is
        # nothing to clean up here.  Keep the method for API compatibility.
        super().close()
```

### What changes from the status quo

| Aspect | Before (status quo) | After (Approach B) |
|---|---|---|
| **New code** | — | ~30-line `PatchRegistry` class |
| **Top-level functions** | 6 (`_patch_*` × 3, `_unpatch_*` × 3) → removed. `patch_ppo_actor_for_tree_backup`, `unpatch_ppo_actor` → refactored | 4 functions, each accepting `registry` as first param |
| **`_init_patches()`** | Called in `__init__`, applies patches immediately | Removed. Patch application deferred to `train()` |
| **Patch lifecycle** | `__init__` → `close()` (entire trainer lifetime) | `train()` enter → `train()` finally (single training call) |
| **Crash safety** | Only `prepare_batch` is scoped to `try/finally`; other 4 patches persist until `close()` | All patches undone by `registry.restore()` in `finally` |
| **Rollback on failure** | None — if patch 3 fails after 1-2, half-patched system until `close()` | Full. Patches 1-2 push undos before patch 3 is attempted. If patch 3 raises, `finally` → `registry.restore()` undoes 1-2 in LIFO order. No half-patched state. |
| **`_get_underlying_engine` calls** | 4 (apply + restore for 2 functions) | 2 (inside `patch_wrap_*` and `patch_workflow_executor`). Restore is a blind `setattr` that doesn't need unwrapping. |
| **Idempotency** | `patch_ppo_actor` checks sentinel; others have no guard | `PatchRegistry` is a new object per `train()` call, so double-application is impossible by construction. The sentinel on `PPOActor` remains because it's a *class-level* attribute (shared across instances). |

### Why not Approach A?

Approach A builds a ~200-line builder-pattern class where each patch is
constructed by a dedicated `_build_*` method and the config (advantage mode,
loss mode, group size) is threaded through `__init__`. This couples the
registry to the config and introduces two flavors of undo (setattr-based and
function-based) mixed in a single list discriminated by `obj is None`.

Approach B treats the registry as a **dumb LIFO stack** of undo callables. It
doesn't know what a "tree backup" or "distill loss" is — the patch functions
retain that knowledge. Registration is uniform: every undo is a zero-argument
callable, whether it's a `setattr` or a multi-step unpatch. This is less
abstract but more composable: you can register *any* undo without the registry
needing a special case for it.

### Migration steps

1. Add `PatchRegistry` class to `trainer.py` (or a new `patches.py`).
2. Refactor the four patch functions to accept `registry` as first parameter
   and use `save_and_set` / `save_method_and_set` / `add_undo`.
3. Remove the 3 `_unpatch_*` top-level functions and `unpatch_ppo_actor`.
   The undo lambdas + `add_undo` calls replace them.
4. Replace `_init_patches` with `_apply_all_patches(registry)`.
5. In `train()`, create a `PatchRegistry`, call `_apply_all_patches`, add the
   `prepare_batch` patch to the same registry, and call `registry.restore()` in
   the `finally` block.
6. Simplify `close()` — it becomes a no-op for patching (kept for API compat).
7. Add a unit test that verifies all patches are fully restored after `train()`
   exits (both normally and on exception).
