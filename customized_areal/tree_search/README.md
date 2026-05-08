# Tree Search: MCTS Tree Backup for PPO Training

This module replaces GAE advantage computation with MCTS tree backup Q-values, enabling
rollout caching across training steps. It also supports on-policy distillation with a
teacher model. It is a customization layer on top of AReaL's `PPOTrainer`.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    CacheAwarePPOTrainer                          │
│  (extends PPOTrainer with rollout caching + tree backup)         │
│                                                                  │
│  train()                                                         │
│   └─ _cache_aware_prepare_batch()  ← replaces prepare_batch     │
│        ├─ split_prompts()          → cached / need-generation   │
│        ├─ load_cached_trajectories() from tree store             │
│        ├─ rollout_batch()           → generate missing only     │
│        ├─ tree_store.insert_batch() → store trajectories        │
│        ├─ tree_advantage_computer.compute() → stash tree adv    │
│        ├─ _mark_batch_trained()     → mark as used              │
│        └─ tree_checkpoint_manager.save() (CROSS_TRAINING)       │
│                                                                  │
│   └─ [patched] PPOActor.compute_advantages()                    │
│        ├─ original GAE pipeline (KL, scaling, normalization)    │
│        └─ restore tree advantages from _tree_advantages/_returns│
│                                                                  │
│  _save_recover_checkpoint()                                      │
│   └─ TreeCheckpointManager.save()                               │
└──────────────────────────────────────────────────────────────────┘
```

## Component Reference

### 1. Config (`config.py`)

Dataclasses controlling tree backup, caching, and advantage computation.

| Class | Field | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `TreeBackupConfig` | `mode` | `TreeBackupMode` | `OFF` | Controls when/how tree backup activates |
| | `checkpoint_dir` | `str` | `""` | Directory for MCTS tree checkpoints |
| | `advantage_mode` | `AdvantageMode` | `TREE` | TREE (Q-values) or GAE advantages |
| | `loss_mode` | `LossMode` | `GRPO` | GRPO, DISTILL, or BOTH |
| | `rl_loss_weight` | `float` | `1.0` | Weight for RL loss in BOTH mode |
| | `distill_loss_weight` | `float` | `0.005` | Weight for distillation loss |
| `RolloutCacheConfig` | `cache_dir` | `str` | `""` | Directory for rollout cache |
| | `enabled` | `bool` | `True` | Enable/disable caching |
| | `n_samples` | `int` | `1` | Number of rollout samples per prompt |

**`TreeBackupMode`** values:
- `OFF` — standard PPOTrainer, no tree backup
- `IN_TRAINING` — tree backup within a single training run (no checkpoint save/load)
- `CROSS_TRAINING` — tree persists across runs; checkpoint is saved/loaded

**`AdvantageMode`** values:
- `GAE` — standard GAE advantages (tree store is still populated for caching)
- `TREE` — MCTS Q-value advantages override GAE

**`LossMode`** values:
- `GRPO` — standard GRPO loss
- `DISTILL` — distillation loss only (rl_loss_weight=0)
- `BOTH` — combined GRPO + distillation loss

### 2. MCTS Tree Store (`mcts_tree_store.py`)

The central data structure. Manages a flat per-query list of `Node` objects, tracks MCTS
statistics per trajectory, and provides cached trajectory loading.

#### Node Dataclass

A `Node` represents one assistant response turn with its full conversation context
(all tokens from the beginning through this turn's response). Nodes are linked via
`node_id` / `parent_node_id` and grouped into episodes via `episode_id`.

| Field | Type | Description |
| --- | --- | --- |
| `input_ids` | `list[int]` | Full token sequence (prompt + response) |
| `loss_mask` | `list[int]` | 0=prompt tokens, 1=response tokens |
| `logprobs` | `list[float]` | Per-token log probabilities |
| `versions` | `list[int]` | Policy version per token (-1 on prompt) |
| `node_id` | `int` | Unique sequence ID (assigned by store) |
| `parent_node_id` | `int \| None` | Parent sequence ID (None for root) |
| `episode_id` | `str` | Groups turns into a trajectory path |
| `outcome_reward` | `float` | Trajectory-level reward |
| `topk_ids` | `list[list[int]] \| None` | Top-k candidate token IDs per response position |
| `topk_logp` | `list[list[float]] \| None` | Top-k candidate log probabilities |
| `distill_reward` | `list[list[float]] \| None` | Per-position distillation rewards |
| `teacher_logp` | `list[list[float]] \| None` | Teacher log probabilities per position |

**Turn boundaries** are derived from `loss_mask` transitions (0→1 = response start,
1→0 = response end) via `_find_turn_boundaries()`, rather than using tokenizer-specific
assistant markers.

#### Store Methods

| Method | Description |
| --- | --- |
| `insert_batch(trajectories)` | Insert trajectories from multiple formats (Node, per-turn dict, legacy tensor dict, list dict, grouped dict) |
| `get_advantages(query_id, seq_id)` | Per-token Q-value tensor expanded by turn response boundaries |
| `get_prompt_mask(query_id, seq_id)` | Boolean mask: True for response tokens |
| `set_trained(query_id, seq_id)` / `is_trained()` | Mark/check whether a trajectory has been used |
| `get_untrained_count(query_id)` | Count untrained trajectories for a query |
| `get_untrained_seq_ids(query_id, n)` | Get up to N untrained seq_ids |
| `load_trajectories(query_id, n)` | Load untrained Node objects |
| `reset_trained_flags()` | Reset all trained flags (for fresh training run) |
| `clear()` | Reset all state |

**MCTS backup** (`_backup`): Each trajectory gets a single Q-value = mean reward
(visit count = 1 currently). Stored in `_visit_counts`, `_total_values`, `_q_values`.

**Node ID assignment** (`_insert_single`): Each Node receives a globally unique
monotonic `seq_id` (stored as `node_id`). The Node's `query_id` is set via
`object.__setattr__`.

### 3. Advantage Computer (`advantage.py`)

`TreeAdvantageComputer` replaces GAE advantages with normalized MCTS Q-values.

```
tree_advantage_computer.compute(trajectories)
```

For each trajectory:
1. Collect all `(query_id, seq_id)` pairs across the batch
2. Per-query GRPO normalization: normalize Q-values to zero-mean unit-variance within
   each query group (so episodes for the same prompt are compared against each other)
3. For each trajectory, compute per-token advantages: Q-value × prompt_mask
   (Q-value on response tokens, 0 on prompt tokens)
4. Set `advantages` and `returns` (= advantages.clone()) in-place

Handles Node objects (`object.__setattr__`), grouped dicts (`node_ids` list), and
single dicts (`node_id`).

### 4. Checkpoint Manager (`checkpoint.py`)

Serializes/deserializes the full MCTS tree state to disk.

| Method | Description |
| --- | --- |
| `save(tree_store)` | Save per-query trajectory records as `query_{id}.json` + `metadata.json` (seq_id indices, MCTS stats, trained flags, rewards) |
| `load()` | Restore `MCTSTreeStore` from disk. No rebuild needed — stats keyed by int seq_id. |
| `exists()` | Check if a checkpoint directory exists |

### 5. Trainer (`trainer.py`)

#### `CacheAwarePPOTrainer`

PPO trainer with rollout caching **and** tree backup. Extends `PPOTrainer` directly.

**Initialization (`__init__`):**

When `cache_config.enabled` and `tree_backup_config.mode != OFF`:
1. Creates `MCTSTreeStore`, `TreeAdvantageComputer`, `TreeCheckpointManager`
2. On `CROSS_TRAINING` mode, loads existing tree checkpoint if available
3. Resets all trained flags for a fresh training run
4. Creates `_CacheAwareBatchBuilder` for prompt splitting
5. Patches `PPOActor.compute_advantages` via `patch_ppo_actor_for_tree_backup()`
6. Patches `_wrap_openai_agent` to use `TreeSearchGroupedRolloutWorkflow`
7. Patches `workflow_executor` to use `TreeSearchWorkflowExecutor`
8. If loss_mode is DISTILL or BOTH, patches `PPOActor._ppo_update` with distillation loss

**Training flow — per step:**

1. **Split prompts** into cached / needs-generation via `_CacheAwareBatchBuilder.split_prompts()`
   - Query ID derived from `prompt["query_id"]` (dataset-provided)
   - If all prompts have enough cached (untrained) trajectories → load from cache
   - Otherwise → generate all prompts fresh via `rollout_batch()`
2. **Tree insert** via `tree_store.insert_batch(trajs)` — while `query_id`/`node_id` are available
3. **Tree advantage compute** (TREE mode) via `tree_advantage_computer.compute(trajs)`
   - Stashes results as `_tree_advantages` / `_tree_returns` on each trajectory
4. **Mark trained** via `_mark_batch_trained()` — so rollouts aren't reused
5. **Save checkpoint** (CROSS_TRAINING mode)
6. **Convert to tensor dicts** for the downstream PPO pipeline
7. **Distillation weights** injected if loss_mode is DISTILL or BOTH
8. **GAE runs** via patched `compute_advantages()` — GAE computes normally, then tree
   advantages are restored from the stashed `_tree_advantages`/`_tree_returns`
9. **PPO update** uses tree advantages (TREE mode) or GAE advantages (GAE mode)

**Key methods:**

| Method | Description |
| --- | --- |
| `train()` | Monkey-patches `self.actor.prepare_batch` with cache-aware version; restores on exit |
| `_cache_aware_prepare_batch()` | Cache-aware batch preparation with tree operations |
| `_save_recover_checkpoint()` | Saves MCTS tree checkpoint on CROSS_TRAINING mode |
| `close()` | Restores all patches (compute_advantages, workflow, executor, distill loss) |

#### Patching Mechanism

`patch_ppo_actor_for_tree_backup()` monkey-patches `PPOActor.compute_advantages`:

1. Calls original GAE pipeline (KL rewards, scaling, normalization) — preserved for logging
2. Restores advantages/returns from `_tree_advantages`/`_tree_returns` (TREE mode only)
3. Removes temporary stash keys

The patch is idempotent — if `PPOActor._original_compute_advantages` already exists,
it reuses the true original instead of stacking patches.

`_CacheAwareBatchBuilder` splits prompts by cache availability and loads cached
trajectories. Query ID comes from `prompt["query_id"]` (dataset-provided string).

### 6. Grouped Rollout Workflow (`grouped_workflow.py`)

`TreeSearchGroupedRolloutWorkflow` extends `GroupedRolloutWorkflow` to run multiple
rollouts per query and collect all per-turn `Node` objects into a flat `list[Node]`.

- `arun_episode()`: runs `group_size` parallel rollouts via `asyncio.gather`, tags
  each Node with `episode_id` and `query_id`, returns flat `list[Node]`

### 7. Proxy Workflow (`proxy_workflow.py`)

`QueryIDProxyWorkflow` extends `OpenAIProxyWorkflow` to inject dataset `query_id` into
trajectories and convert `InteractionWithTokenLogpReward` objects to `list[Node]`.

- `_interactions_to_nodes()`: converts interaction dict to Node objects, reconstructing
  full-sequence `logprobs`/`loss_mask`/`versions` by concatenating parent context with
  new response tokens
- `arun_episode()`: calls parent, converts result to `list[Node]` with `query_id` injected

### 8. Workflow Executor (`workflow_executor.py`)

`TreeSearchWorkflowExecutor` extends `WorkflowExecutor` to handle `list[dict]` returns
from `arun_episode` (the standard executor expects a single dict).

- `_create_workflow_task()`: wraps workflow execution, handles three return types
  (list[dict], dict, InteractionWithTokenLogpReward dict), applies acceptance filtering
- `wait()`: extracts `_TreeSearchRolloutResult` trajectories, flattening into a single list
- `rollout_batch()`: submits all items and returns flattened results

### 9. Distillation Support

#### `distill_types.py`

| Class | Description |
| --- | --- |
| `PositionRewardInfo` | Per-position candidate tokens, logprobs, and rewards |
| `InteractionWithTokenLevelReward` | Extended interaction with `token_rewards` and `token_reward_mask` |

#### `core/` — On-Policy Distillation Core

| File | Purpose |
| --- | --- |
| `core/config.py` | `OnPolicyDistillConfig` (extends PPOConfig) and `AgentConfig` |
| `core/agent.py` | `OnPolicyDistillAgent` — agent class for distillation training |
| `core/reward_compute.py` | `_compute_token_rewards()` — student vs teacher logprob comparison |
| `core/teacher_client.py` | `TeacherClient` — async client for remote teacher model inference |

#### `engine/` — Multi-Candidate Engine

| File | Purpose |
| --- | --- |
| `engine/fsdp_engine.py` | `MultiCandidateFSDPEngine` — FSDP engine with multi-candidate logprob gathering |
| `engine/actor.py` | `MultiCandidateFSDPPPOActor` — PPO actor wrapping `MultiCandidateFSDPEngine` |

#### `training/` — Distillation Training

| File | Purpose |
| --- | --- |
| `training/loss.py` | `grpo_distill_loss_fn()` — combined GRPO + position-level distillation loss |
| `training/actor.py` | `patch_ppo_actor_class_to_use_distill_loss()` — patches PPOActor to use distill loss |
| `training/logprobs.py` | `gather_logprobs_entropy_multi_candidates()` — multi-candidate logprob gathering |

## Data Flow

### Cache-Aware Training

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        CacheAwarePPOTrainer.train()                      │
│                                                                          │
│  monkey-patches self.actor.prepare_batch → _cache_aware_prepare_batch() │
│  monkey-patches PPOActor.compute_advantages → restore tree adv after GAE │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  │  per training step
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  1. BATCH PREPARATION  (_cache_aware_prepare_batch)                      │
│                                                                          │
│  ┌──────────────┐                                                       │
│  │  Dataloader   │  raw prompts (list[dict])                             │
│  └──────┬───────┘                                                       │
│         │                                                                │
│  ┌──────▼─────────────────────────────────┐                              │
│  │  _CacheAwareBatchBuilder.split_prompts │                              │
│  │                                        │                              │
│  │  For each prompt:                      │                              │
│  │   • prompt["query_id"]                 │                              │
│  │   • tree_store.get_untrained_count()   │── check cached rollouts      │
│  │                                        │                              │
│  │   fully cached (≥ n_samples)  ──────► │  cached_items[]               │
│  │   not fully cached            ──────► │  need_gen_items[]             │
│  └──┬──────────────────────────────────┬─┘                              │
│     │                                   │                                │
│     ▼                                   ▼                                │
│  ┌──────────────────────┐  ┌──────────────────────┐                     │
│  │ load_cached_trajs()  │  │  rollout_batch()     │                     │
│  │                      │  │  (inference engine)   │                     │
│  │ tree_store.load_     │  │                      │                     │
│  │  trajectories(qid,n) │  │ TreeSearchGrouped    │                     │
│  │                      │  │ RolloutWorkflow runs │                     │
│  │ Returns list[Node]   │  │ group_size episodes  │                     │
│  └──────┬───────────────┘  │                      │                     │
│         │                  │ Returns flat list    │                     │
│         │                  │ of per-episode dicts │                     │
│         │                  └──────────┬───────────┘                     │
│         │                             │                                 │
│         └──────────┬──────────────────┘                                 │
│                    ▼                                                    │
│  ┌──────────────────────────────────────┐                               │
│  │  Tree Operations                     │                               │
│  │                                      │                               │
│  │  1. tree_store.insert_batch(trajs)   │                               │
│  │     For each trajectory:             │                               │
│  │      • Assign global seq_id         │                               │
│  │      • Store Node record            │                               │
│  │      • MCTS backup (Q = reward)     │                               │
│  │     Sets traj["query_id"]           │                               │
│  │     Sets traj["node_id"] or         │                               │
│  │     traj["node_ids"] (grouped)      │                               │
│  │                                      │                               │
│  │  2. tree_advantage_computer.compute  │                               │
│  │     (TREE mode)                      │                               │
│  │     • Per-query GRPO normalize      │                               │
│  │     • Stash _tree_advantages        │                               │
│  │       and _tree_returns              │                               │
│  │                                      │                               │
│  │  3. _mark_batch_trained()            │                               │
│  │                                      │                               │
│  │  4. Save checkpoint (CROSS_TRAINING) │                               │
│  └──────────────────┬───────────────────┘                               │
│                     │                                                    │
│  ┌──────────────────▼───────────────────┐                               │
│  │  Convert to tensor dicts             │                               │
│  │  • Node → _node_to_tensor_dict()     │                               │
│  │  • List dict → _list_dict_to_tensor()│                               │
│  │  • Tensor dict → pass through        │                               │
│  │                                      │                               │
│  │  Inject distillation loss weights:   │                               │
│  │  • DISTILL: rl_loss_weight = 0.0     │                               │
│  │  • BOTH:    rl_loss_weight = config  │                               │
│  └──────────────────┬───────────────────┘                               │
│                     │                                                    │
└─────────────────────┼────────────────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  2. ADVANTAGE COMPUTATION  (patched compute_advantages)                  │
│                                                                          │
│  ┌─────────────────────────────────────┐                                 │
│  │  Step A: Original GAE pipeline      │                                 │
│  │  • Compute KL rewards              │                                 │
│  │  • Scale rewards                   │                                 │
│  │  • GAE λ-returns → advantages      │                                 │
│  │  • Compute loss_mask, logprobs     │                                 │
│  │                                     │                                 │
│  │  Preserved for logging: kl_rewards, │                                 │
│  │  tot_rewards, loss_mask, logprobs   │                                 │
│  └──────────────────────┬──────────────┘                                 │
│                         │                                                │
│  ┌──────────────────────▼──────────────┐                                 │
│  │  Step B: Restore tree advantages    │                                 │
│  │  (TREE mode only)                   │                                 │
│  │                                     │                                 │
│  │  For each trajectory:               │                                 │
│  │   Pop _tree_advantages → advantages │                                 │
│  │   Pop _tree_returns → returns       │                                 │
│  └─────────────────────────────────────┘                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3. PPO UPDATE                                                           │
│                                                                          │
│  • GRPO loss: clipped loss on advantages (tree Q-values in TREE mode)   │
│  • Value loss on returns                                                 │
│  • Distillation loss (DISTILL/BOTH modes):                               │
│    position-level GRPO using teacher logprobs                            │
│  • KL metadata available for logging                                     │
└──────────────────────────────────────────────────────────────────────────┘

After PPO update (CROSS_TRAINING mode):

┌──────────────────────────────────────────────┐
│  _save_recover_checkpoint()                  │
│  └─ TreeCheckpointManager.save(tree_store)   │
│     • Each query → query_{id}.json           │
│     • metadata.json (seq_id indices,         │
│       MCTS stats, trained flags)             │
└──────────────────────────────────────────────┘
```

### Metadata Propagation

Key metadata fields attached to trajectory dicts throughout the pipeline:

| Field | Attached by | Type | Used by |
| --- | --- | --- | --- |
| `query_id` | `insert_batch()` / `QueryIDProxyWorkflow` | `str` | Tree lookup, cache splitting, advantage compute |
| `node_id` | `insert_batch()` (single traj) | `int` | Advantage lookup, mark trained |
| `node_ids` | `insert_batch()` (grouped traj) | `list[int]` | Per-sample advantage in grouped dict |

## Usage Example

```python
from customized_areal.tree_search.config import (
    AdvantageMode,
    LossMode,
    RolloutCacheConfig,
    TreeBackupConfig,
    TreeBackupMode,
)
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

cache_config = RolloutCacheConfig(
    cache_dir="/path/to/tree_cache",
    enabled=True,
    n_samples=8,
)

tree_backup_config = TreeBackupConfig(
    mode=TreeBackupMode.CROSS_TRAINING,
    checkpoint_dir="/path/to/tree_cache",
    advantage_mode=AdvantageMode.TREE,
    loss_mode=LossMode.GRPO,
)

with CacheAwarePPOTrainer(
    config,
    cache_config=cache_config,
    tree_backup_config=tree_backup_config,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
) as trainer:
    trainer.train(
        workflow=config.workflow,
        eval_workflow=config.eval_workflow,
        workflow_kwargs=workflow_kwargs,
        eval_workflow_kwargs=eval_workflow_kwargs,
    )
```

## File Index

| File | Purpose |
| --- | --- |
| `config.py` | `TreeBackupConfig`, `RolloutCacheConfig`, `TreeBackupMode`, `AdvantageMode`, `LossMode` |
| `mcts_tree_store.py` | `MCTSTreeStore`, `Node` — flat trajectory store with MCTS statistics |
| `advantage.py` | `TreeAdvantageComputer` — GRPO-normalized tree Q-value advantages |
| `checkpoint.py` | `TreeCheckpointManager` — serialize/deserialize tree state to JSON |
| `trainer.py` | `CacheAwarePPOTrainer` — PPO trainer with rollout caching + tree backup |
| `grouped_workflow.py` | `TreeSearchGroupedRolloutWorkflow` — runs group_size episodes per query |
| `proxy_workflow.py` | `QueryIDProxyWorkflow` — injects query_id, converts interactions to Nodes |
| `workflow_executor.py` | `TreeSearchWorkflowExecutor` — handles list[dict] returns from arun_episode |
| `distill_types.py` | `PositionRewardInfo`, `InteractionWithTokenLevelReward` |
| `core/agent.py` | `OnPolicyDistillAgent` — agent for distillation training |
| `core/config.py` | `OnPolicyDistillConfig`, `AgentConfig` |
| `core/reward_compute.py` | Student vs teacher logprob reward computation |
| `core/teacher_client.py` | `TeacherClient` — async teacher model inference client |
| `engine/fsdp_engine.py` | `MultiCandidateFSDPEngine` — multi-candidate logprob gathering |
| `engine/actor.py` | `MultiCandidateFSDPPPOActor` — PPO actor for distillation |
| `training/loss.py` | `grpo_distill_loss_fn` — combined GRPO + distillation loss |
| `training/actor.py` | Patch to use distillation loss in PPOActor |
| `training/logprobs.py` | Multi-candidate logprob/entropy gathering utilities |
