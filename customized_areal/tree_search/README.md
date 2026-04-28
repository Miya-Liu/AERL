# Tree Search: MCTS Tree Backup for PPO Training

This module replaces GAE advantage computation with MCTS tree backup Q-values, enabling
rollout caching across training steps. It is a customization layer on top of AReaL's
`PPOTrainer`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CacheAwarePPOTrainer                         │
│  (extends PPOTrainer with rollout caching + tree backup)        │
│                                                                 │
│  train()                                                        │
│   ├─ _cache_aware_prepare_batch()  ← replaces prepare_batch    │
│   │    ├─ split_prompts()          → cached / need-generation  │
│   │    ├─ load_cached_trajectories()                           │
│   │    ├─ rollout_batch()           → generate missing only    │
│   │    └─ _split_grouped_trajectories()                        │
│   │                                                            │
│   └─ [patched] PPOActor.compute_advantages()                   │
│        ├─ original GAE pipeline (KL, scaling, normalization)   │
│        ├─ tree_store.insert_batch()  → insert into trie        │
│        ├─ tree_advantage_computer.compute() → overwrite adv    │
│        ├─ _mark_batch_trained()       → mark as used           │
│        └─ tree_store.record_training_step()                    │
│                                                                │
│  _save_recover_checkpoint()                                    │
│   └─ TreeCheckpointManager.save()                              │
└─────────────────────────────────────────────────────────────────┘
```

## Component Reference

### 1. Config (`config.py`)

Dataclasses controlling tree backup and caching behavior.

| Class                | Field              | Type             | Default | Description                                               |
| -------------------- | ------------------ | ---------------- | ------- | --------------------------------------------------------- |
| `TreeBackupConfig`   | `mode`             | `TreeBackupMode` | `OFF`   | Controls when/how tree backup activates                   |
|                      | `assistant_marker` | `str`            | `""`    | Token marker for assistant turns (auto-detected if empty) |
|                      | `checkpoint_dir`   | `str`            | `""`    | Directory for MCTS tree checkpoints                       |
| `RolloutCacheConfig` | `cache_dir`        | `str`            | `""`    | Directory for rollout cache                               |
|                      | `enabled`          | `bool`           | `True`  | Enable/disable caching                                    |
|                      | `n_samples`        | `int`            | `1`     | Number of rollout samples per prompt                      |

**`TreeBackupMode`** values:

- `OFF` — standard PPOTrainer, no tree backup
- `IN_TRAINING` — tree backup within a single training run
- `CROSS_TRAINING` — tree persists across training runs; checkpoint is saved/loaded

### 2. Turn Splitter (`turn_splitter.py`)

Splits a flat token sequence into structured `Turn` objects at assistant role marker
boundaries.

```
Input:  [user tokens... <|im_start|>assistant response1 <|im_start|>assistant response2]
Output: [Turn(prompt=[assistant marker], response=[response1]),
         Turn(prompt=[assistant marker], response=[response2])]
```

- `Turn` — dataclass with `prompt_tokens` (shared context) and `response_tokens`
  (branching point)
- `make_turn_splitter(tokenizer, assistant_marker)` — returns a
  `split(input_ids) -> list[Turn]` function
- Auto-detects the assistant marker from the tokenizer's chat template (supports
  Qwen/ChatML, Llama-3, Gemma)

### 3. Trie Node (`trie_node.py`)

A compressed trie node for turn-level MCTS path indexing. Each node stores one full
turn's tokens (prompt + response). Children are keyed by the first response token,
enabling prefix sharing across trajectories.

**Key methods:**

| Method                        | Description                                                                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `add_turn(turn, seq_id)`      | Add a child node for a turn, keyed by first response token. Returns child (cursor for next turn). Reuses existing child if same response prefix. |
| `get_path_nodes(seq_id)`      | Return root-to-leaf node list for a given sequence ID                                                                                            |
| `get_turn_boundaries(seq_id)` | Return cumulative token positions where turns start/end                                                                                          |

**Key fields:**

| Field                   | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| `tokens`                | Full turn tokens (prompt + response)                  |
| `prompt_len`            | Number of prompt tokens in this turn                  |
| `sequence_ids`          | IDs of trajectories passing through this node         |
| `logprobs` / `versions` | Per-token metadata                                    |
| `training_steps`        | Global steps where this node's trajectory was trained |

### 4. MCTS Tree Store (`mcts_tree_store.py`)

The central data structure. Manages one trie per query (keyed by `query_id` = MD5 of
prompt tokens), tracks MCTS statistics per node, and provides a cursor-based API for
incrementally building trajectories.

**Cursor-based trajectory building:**

```
seq_id = tree_store.start_sequence(query_id)   # create root if needed, assign seq_id
tree_store.add_turn(query_id, seq_id, turn_1)  # advance cursor
tree_store.add_turn(query_id, seq_id, turn_2)  # advance cursor
tree_store.finish_sequence(query_id, seq_id, reward)  # run MCTS backup, clear cursor
```

**Convenience methods:**

| Method                                           | Description                                                                                                                              |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `insert_trajectory(query_id, input_ids, reward)` | Split into turns, then start/add/finish automatically                                                                                    |
| `insert_batch(trajectories)`                     | Batch version; handles grouped (batch > 1) dicts. Attaches `_mcts_seq_id` / `_mcts_seq_ids` and `_mcts_query_id` to each trajectory dict |

**MCTS backup** (`_backup`): Walks from leaf to root, incrementing visit counts and
averaging Q-values at each node. Q-value = mean reward over all trajectories passing
through that node.

**Advantage computation:**

| Method                              | Description                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| `get_advantages(query_id, seq_id)`  | Returns per-token advantage tensor from Q-values, expanded by turn boundaries |
| `get_prompt_mask(query_id, seq_id)` | Boolean mask: True for response tokens, False for prompt tokens               |

**Caching and training tracking:**

| Method                                            | Description                                                                    |
| ------------------------------------------------- | ------------------------------------------------------------------------------ |
| `set_trained` / `is_trained`                      | Mark/check whether a trajectory has been used in training                      |
| `get_untrained_count(query_id)`                   | Count untrained trajectories for a query                                       |
| `load_trajectories(query_id, n_samples)`          | Load up to N untrained trajectories as training dicts                          |
| `load_trajectory_by_seq_id(query_id, seq_id)`     | Load a single trajectory by exact seq_id (ignores trained flag)                |
| `record_training_step(global_step, trajectories)` | Record training order; appends step to leaf node's `training_steps`            |
| `build_training_history()`                        | Reconstruct `_training_history` from leaf nodes (fallback for old checkpoints) |

**Query ID derivation:**

- `_get_query_id(traj)` — MD5 of prompt tokens from trajectory's `loss_mask == 0` region
- `get_query_id_from_messages(messages, tokenizer)` — MD5 of tokenized messages
  (produces same ID as above, usable before rollout)

### 5. Advantage Computer (`advantage.py`)

`TreeAdvantageComputer` replaces GAE advantages with MCTS Q-values.

```
tree_advantage_computer.compute(trajectories)
```

For each trajectory:

1. Look up Q-values from `MCTSTreeStore.get_advantages(query_id, seq_id)`
1. Zero out prompt tokens using `get_prompt_mask`
1. Overwrite `traj["advantages"]` and `traj["returns"]` in-place

Handles both single trajectories (with `_mcts_seq_id`) and grouped trajectories (with
`_mcts_seq_ids` list).

### 6. Checkpoint Manager (`checkpoint.py`)

Serializes/deserializes the full MCTS tree state to disk.

| Method                | Description                                                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `save(tree_store)`    | Save each tree as `query_{id}.json` + `metadata.json` (trained flags, rewards, training history)                             |
| `load(turn_splitter)` | Restore `MCTSTreeStore` from disk. Calls `rebuild_mcts_stats()` after load because `id(node)` values change across processes |

### 7. Trainer (`trainer.py`)

#### `CacheAwarePPOTrainer`

PPO trainer with rollout caching **and** tree backup. Extends `PPOTrainer` directly.

**Initialization (`__init__`):**

When `cache_config.enabled` and `tree_backup_config.mode != OFF`:

1. Creates `MCTSTreeStore` with a turn splitter derived from the tokenizer
1. Creates `TreeAdvantageComputer` wrapping the tree store
1. Creates `TreeCheckpointManager` for the cache/checkpoint directory
1. On `CROSS_TRAINING` mode, loads an existing tree checkpoint if one exists
1. Resets all trained flags for a fresh training run
1. Patches `PPOActor.compute_advantages` via `patch_ppo_actor_for_tree_backup()`

**Training flow — cache-aware mode (per step):**

1. **Split prompts** into cached / needs-generation via
   `_CacheAwareBatchBuilder.split_prompts()`
1. **Load cached** trajectories from tree store (untrained rollouts for same query)
1. **Generate missing** trajectories via `rollout_batch()` (only for prompts without
   enough cached rollouts)
1. **Split** grouped trajectories via `_split_grouped_trajectories()` (preserves
   per-sample `_mcts_seq_id` / `_mcts_query_id` metadata)
1. **Tree backup** runs via patched `compute_advantages()` (insert into trie, compute
   Q-value advantages, mark trained)
1. **Save checkpoint** (on `CROSS_TRAINING` mode)

**Other methods:**

| Method                       | Description                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| `train()`                    | Monkey-patches `self.actor.prepare_batch` with cache-aware version; restores on exit           |
| `_save_recover_checkpoint()` | Saves MCTS tree checkpoint on `CROSS_TRAINING` mode (via `TreeCheckpointManager`)              |
| `_mark_batch_trained()`      | Marks rollout trajectories as trained (single or grouped via `_mcts_seq_id` / `_mcts_seq_ids`) |
| `close()`                    | Calls `unpatch_ppo_actor()` to restore original `PPOActor`, then `super().close()`             |

#### Patching mechanism

`patch_ppo_actor_for_tree_backup()` monkey-patches `PPOActor.compute_advantages`:

1. Calls original GAE pipeline (KL rewards, scaling, normalization)
1. Inserts trajectories into tree with raw rewards
1. Overwrites advantages/returns with tree Q-values
1. Marks trajectories as trained
1. Records training step order

The patch is idempotent — if `PPOActor._original_compute_advantages` already exists
(from a prior patch), it reuses the true original instead of stacking patches.

`unpatch_ppo_actor()` restores the original method. Called in
`CacheAwarePPOTrainer.close()`.

## Data Flow

### Cache-Aware Training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CacheAwarePPOTrainer.train()                         │
│                                                                             │
│  monkey-patches self.actor.prepare_batch → _cache_aware_prepare_batch()    │
│  monkey-patches PPOActor.compute_advantages → _tree_backup_compute_advantages() │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  │  per training step
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. BATCH PREPARATION  (_cache_aware_prepare_batch)                         │
│                                                                             │
│  ┌──────────────┐                                                          │
│  │  Dataloader   │  raw prompts (list[dict])                                │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│  ┌──────▼────────────────────────────────┐                                  │
│  │  _CacheAwareBatchBuilder.split_prompts│                                  │
│  │                                       │                                  │
│  │  For each prompt:                     │                                  │
│  │   • get_query_id_from_messages()      │── MD5(tokenize(messages))        │
│  │   • tree_store.get_untrained_count()  │── check cached rollouts          │
│  │                                       │                                  │
│  │   fully cached (≥ n_samples)  ──────► │  cached_items[]                  │
│  │   partially cached           ──────► │  cached_items[] (+ need_gen)      │
│  │   not cached                 ──────► │  need_gen_items[]                 │
│  └──┬───────────────────────────────┬───┘                                  │
│     │                               │                                       │
│     │  cached prompts               │  prompts needing generation           │
│     ▼                               ▼                                       │
│  ┌──────────────────────┐  ┌──────────────────────┐                        │
│  │ load_cached_trajs()  │  │  rollout_batch()     │                        │
│  │                      │  │  (inference engine)   │                        │
│  │ tree_store.load_     │  │                      │                        │
│  │  trajectories(qid,n) │  │ Returns grouped dicts │                        │
│  │                      │  │ shape [group, seq]    │                        │
│  │ Returns individual   │  │ + _mcts_seq_ids list  │                        │
│  │ dicts shape [1,seq]  │  │ + _mcts_query_id      │                        │
│  └──────┬───────────────┘  └──────────┬───────────┘                        │
│         │                             │                                      │
│         └──────────┬──────────────────┘                                      │
│                    ▼                                                         │
│  ┌──────────────────────────────────────┐                                   │
│  │  _split_grouped_trajectories()       │                                   │
│  │                                      │                                   │
│  │  • batch=1: kept as-is               │                                   │
│  │  • batch>1: split into individual    │                                   │
│  │    individual dicts, extracting      │                                   │
│  │    _mcts_seq_id from _mcts_seq_ids   │                                   │
│  │                                      │                                   │
│  │  Result: flat list of per-sample     │                                   │
│  │  dicts, each with _mcts_query_id     │                                   │
│  │  and _mcts_seq_id                    │                                   │
│  └──────────────────┬───────────────────┘                                   │
│                     │                                                        │
│                     │  merged trajectories                                   │
└─────────────────────┼────────────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. ADVANTAGE COMPUTATION  (patched compute_advantages)                     │
│                                                                             │
│  ┌──────────────────────────────────────────┐                              │
│  │  Step A: Original GAE pipeline           │                              │
│  │  (PPOActor.compute_advantages)           │                              │
│  │                                          │                              │
│  │  • Compute KL rewards (ref vs policy)    │                              │
│  │  • Scale rewards                         │                              │
│  │  • GAE λ-returns → advantages, returns   │                              │
│  │  • Compute loss_mask, logprobs           │                              │
│  │                                          │                              │
│  │  Preserved for logging: kl_rewards,      │                              │
│  │  tot_rewards, loss_mask, logprobs        │                              │
│  └──────────────────────┬───────────────────┘                              │
│                         │                                                   │
│  ┌──────────────────────▼───────────────────┐                              │
│  │  Step B: Insert into MCTS tree           │                              │
│  │  (tree_store.insert_batch)               │                              │
│  │                                          │                              │
│  │  For each trajectory:                    │                              │
│  │   • _get_query_id() ─ MD5(prompt tokens) │                              │
│  │   • turn_splitter(input_ids) → [Turn]    │                              │
│  │   • start_sequence(query_id) → seq_id    │                              │
│  │   • add_turn(query_id, seq_id, turn)     │                              │
│  │     for each turn (cursor walks trie)    │                              │
│  │   • finish_sequence(query_id, seq_id,    │                              │
│  │     reward) → triggers MCTS backup       │                              │
│  │                                          │                              │
│  │  MCTS _backup (leaf → root):             │                              │
│  │   • Increment visit_count at each node   │                              │
│  │   • Update total_value += traj reward    │                              │
│  │   • Q_value = total_value / visit_count  │                              │
│  │                                          │                              │
│  │  Attaches to each traj dict:             │                              │
│  │   • _mcts_query_id (str)                 │                              │
│  │   • _mcts_seq_id (int) or                │                              │
│  │     _mcts_seq_ids (list[int])            │                              │
│  └──────────────────────┬───────────────────┘                              │
│                         │                                                   │
│  ┌──────────────────────▼───────────────────┐                              │
│  │  Step C: Overwrite advantages            │                              │
│  │  (tree_advantage_computer.compute)       │                              │
│  │                                          │                              │
│  │  For each trajectory:                    │                              │
│  │   • get_advantages(qid, seq_id)          │                              │
│  │     → per-token Q-values expanded by     │                              │
│  │       turn boundaries (same Q for all    │                              │
│  │       tokens in a turn's response)       │                              │
│  │   • Zero out prompt tokens via           │                              │
│  │     get_prompt_mask() (True=response)    │                              │
│  │   • traj["advantages"] = Q * mask        │                              │
│  │   • traj["returns"]     = advantages     │                              │
│  │     (clone, since V=Q for tree backup)   │                              │
│  └──────────────────────┬───────────────────┘                              │
│                         │                                                   │
│  ┌──────────────────────▼───────────────────┐                              │
│  │  Step D: Mark trained                    │                              │
│  │  (_mark_batch_trained)                   │                              │
│  │                                          │                              │
│  │  tree_store.set_trained(query_id,        │                              │
│  │    seq_id, True) for each trajectory     │                              │
│  │                                          │                              │
│  │  → future split_prompts() will skip      │                              │
│  │    these and not load them again         │                              │
│  └──────────────────────┬───────────────────┘                              │
│                         │                                                   │
│  ┌──────────────────────▼───────────────────┐                              │
│  │  Step E: Record training step            │                              │
│  │  (tree_store.record_training_step)       │                              │
│  │                                          │                              │
│  │  _training_history[global_step] =        │                              │
│  │    [(query_id, seq_id), ...]             │                              │
│  │                                          │                              │
│  └──────────────────────┬───────────────────┘                              │
│                         │                                                   │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. PPO UPDATE  (standard PPOTrainer, unchanged)                            │
│                                                                             │
│  • PPO clipped loss on tree-Q advantages                                   │
│  • Value function loss on tree-Q returns                                   │
│  • KL metadata (kl_rewards, tot_rewards) available for logging              │
│  • Optimizer step, gradient accumulation, etc.                              │
└─────────────────────────────────────────────────────────────────────────────┘

After PPO update (on CROSS_TRAINING mode):

┌──────────────────────────────────────────────┐
│  _save_recover_checkpoint()                  │
│  └─ TreeCheckpointManager.save(tree_store)   │
│     • Each tree → query_{id}.json            │
│     • metadata.json (trained flags, rewards, │
│       training history)                      │
└──────────────────────────────────────────────┘
```

### Metadata Propagation

Key metadata fields attached to trajectory dicts throughout the pipeline:

| Field            | Attached by                          | Type        | Used by                                         |
| ---------------- | ------------------------------------ | ----------- | ----------------------------------------------- |
| `_mcts_query_id` | `insert_batch()` / `split_prompts()` | `str`       | Tree lookup, cache splitting, advantage compute |
| `_mcts_seq_id`   | `insert_batch()` (single)            | `int`       | Advantage lookup, mark trained                  |
| `_mcts_seq_ids`  | `insert_batch()` (grouped)           | `list[int]` | Per-sample advantage in grouped dict            |
| `_global_step`   | PPOTrainer (base)                    | `int`       | Training history recording                      |

`_split_grouped_trajectories()` converts `_mcts_seq_ids` → individual `_mcts_seq_id`
when splitting grouped dicts (batch > 1), ensuring downstream code always has per-sample
access.

## Usage Example

```python
from customized_areal.tree_search.config import RolloutCacheConfig, TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

cache_config = RolloutCacheConfig(
    cache_dir="/path/to/tree_cache",
    enabled=True,
    n_samples=8,
)

tree_backup_config = TreeBackupConfig(
    mode=TreeBackupMode.CROSS_TRAINING,
    assistant_marker="",  # auto-detect from tokenizer
    checkpoint_dir="/path/to/tree_cache",
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

See `customized_areal/tpfc/scripts/train_tpfc_tree_search.py` for a complete end-to-end
training script.

## File Index

| File                 | Purpose                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| `config.py`          | `TreeBackupConfig`, `RolloutCacheConfig`, `TreeBackupMode` dataclasses   |
| `turn_splitter.py`   | `Turn` dataclass, `make_turn_splitter()` factory                         |
| `trie_node.py`       | `TrieNode` — compressed trie node for turn-level path indexing           |
| `mcts_tree_store.py` | `MCTSTreeStore` — trie-backed MCTS tree with cursor API, backup, caching |
| `advantage.py`       | `TreeAdvantageComputer` — replaces GAE advantages with MCTS Q-values     |
| `checkpoint.py`      | `TreeCheckpointManager` — serialize/deserialize tree state to JSON       |
| `trainer.py`         | `CacheAwarePPOTrainer` — PPO trainer with rollout caching + tree backup  |
