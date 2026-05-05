# Plan: Fix Individual Export for Tree Search — Episode Reconstruction and GRPO Normalization

## Problem

When `export_interactions(style="individual")` is used with `CacheAwarePPOTrainer`,
multi-turn episodes are flattened into independent batch rows. Each turn becomes a
separate `[1, seq_len]` dict with its own reward. This breaks:

1. **Tree insertion**: `insert_batch` assigns one `seq_id` per turn, not per episode
1. **Advantage computation**: No episode-level Q-value; per-turn Q-values are just
   per-turn rewards
1. **GRPO normalization**: No group normalization across episodes of the same query
1. **Variable turn counts**: Different episodes have different turn counts

## Design Overview

Change `rollout_batch` to return per-turn dicts with episode metadata so that
`insert_batch` can reconstruct full episodes. Add GRPO normalization in
`TreeAdvantageComputer`.

### Key Change: Per-Turn Dict Format

`rollout_batch` returns a `list[dict]` where each dict is a single turn `[1, seq_len]`
with these metadata keys:

| Key                    | Type                 | Description                    |
| ---------------------- | -------------------- | ------------------------------ |
| `_mcts_query_id`       | `str`                | Query identifier               |
| `_episode_idx`         | `int`                | Episode index within the group |
| `_turn_idx_in_episode` | `int`                | Turn index within the episode  |
| `_parent_turn_id`      | `str \| None`        | Parent interaction ID          |
| `_turn_reward`         | `float`              | Reward for this turn           |
| `_outcome_reward`      | `float`              | Outcome reward for the episode |
| `input_ids`            | `Tensor[1, seq_len]` | Token IDs                      |
| `response_ids`         | —                    | (reserved for future use)      |

______________________________________________________________________

## Step 1: Create `TreeSearchGroupedRolloutWorkflow`

**File:** `customized_areal/tree_search/grouped_workflow.py` (NEW)

Subclass `GroupedRolloutWorkflow`. Override `arun_episode` to:

1. Run inner workflow `group_size` times via `asyncio.gather` (same as base)
1. For each result (a `dict[str, InteractionWithTokenLogpReward]`): a. Sort interactions
   by creation order (cache insertion order) b. Call `to_tensor_dict()` on each →
   individual-style `[1, seq_len]` per turn c. Collect turn metadata: `interaction_id`,
   `parent.interaction_id`, `reward` d. Determine outcome reward: last turn's reward (or
   sum, configurable)
1. `concat_padded_tensors` to stack all turns into `[total_turns, max_seq_len]`
1. Add episode-level metadata as list-valued keys (post-concat, not relying on concat to
   pass them through)

**Metadata keys on the stacked dict:**

| Key                        | Type                    | Description                                    |
| -------------------------- | ----------------------- | ---------------------------------------------- |
| `_mcts_query_id`           | `str`                   | Query identifier                               |
| `_episode_num_turns`       | `list[int]`             | Turn count per episode, length = group_size    |
| `_episode_turn_offsets`    | `list[int]`             | Cumulative turn offsets: `[0, n0, n0+n1, ...]` |
| `_episode_turn_ids`        | `list[list[str]]`       | Interaction IDs per episode                    |
| `_episode_parent_turn_ids` | `list[list[str\|None]]` | Parent interaction IDs per episode             |
| `_episode_turn_rewards`    | `list[list[float]]`     | Per-turn rewards per episode                   |
| `_episode_outcome_reward`  | `list[float]`           | Outcome reward per episode                     |

Variable turn counts: shorter episodes are padded with zero rows. `_episode_num_turns`
and `_episode_turn_offsets` allow slicing back into per-episode turn groups.

______________________________________________________________________

## Step 2: Create `_split_to_turn_dicts` function

**File:** `customized_areal/tree_search/grouped_workflow.py` (in same file)

After `rollout_batch` returns a list of stacked dicts (one per query), split each into a
flat list of per-turn dicts:

```python
EPISODE_LEVEL_METADATA_KEYS = {
    "_episode_num_turns", "_episode_turn_offsets",
    "_episode_turn_ids", "_episode_parent_turn_ids",
    "_episode_turn_rewards", "_episode_outcome_reward",
}

def _split_to_turn_dicts(trajs: list[dict]) -> list[dict]:
    """Split stacked trajectory dicts into flat list of per-turn dicts."""
    flat = []
    for traj in trajs:
        offsets = traj["_episode_turn_offsets"]
        num_turns_list = traj["_episode_num_turns"]

        for ep_idx, num_turns in enumerate(num_turns_list):
            start = offsets[ep_idx]
            for local_turn_idx in range(num_turns):
                t = start + local_turn_idx
                turn_dict = {
                    k: v[t : t + 1] if isinstance(v, torch.Tensor) else v
                    for k, v in traj.items()
                    if k not in EPISODE_LEVEL_METADATA_KEYS
                }
                turn_dict["_mcts_query_id"] = traj["_mcts_query_id"]
                turn_dict["_episode_idx"] = ep_idx
                turn_dict["_turn_idx_in_episode"] = local_turn_idx
                turn_dict["_parent_turn_id"] = traj["_episode_parent_turn_ids"][ep_idx][local_turn_idx]
                turn_dict["_turn_reward"] = traj["_episode_turn_rewards"][ep_idx][local_turn_idx]
                turn_dict["_outcome_reward"] = traj["_episode_outcome_reward"][ep_idx]
                flat.append(turn_dict)
    return flat
```

______________________________________________________________________

## Step 3: Extend `TrajectoryRecord` and update `MCTSTreeStore`

**File:** `customized_areal/tree_search/mcts_tree_store.py`

### 3a: Extend `TrajectoryRecord`

Add new fields:

```python
@dataclass
class TrajectoryRecord:
    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    versions: list[int]
    reward: float                       # outcome reward (for Q-value)
    turn_response_starts: list[int]     # unchanged
    turn_response_ends: list[int]       # unchanged
    # New fields:
    turn_ids: list[str]                 # interaction ID per turn
    parent_turn_ids: list[str | None]   # parent interaction ID per turn
    turn_rewards: list[float]           # per-turn reward
    outcome_reward: float               # outcome reward (separate from reward)
```

### 3b: Update `insert_batch` — Group per-turn dicts into episodes

Change `insert_batch` to accept per-turn dicts (from `_split_to_turn_dicts`). Group
consecutive turn dicts with the same `_mcts_query_id` and `_episode_idx` into a single
episode record:

1. Group turns by `(_mcts_query_id, _episode_idx)`
1. For each episode group:
   - Concatenate turn `input_ids`, `loss_mask`, `logprobs`, `versions` into a single
     sequence (concat-style within episode)
   - Build `turn_response_starts`/`turn_response_ends` from `loss_mask` transitions
   - Set `reward = outcome_reward`, `outcome_reward = _outcome_reward`
   - Store `turn_ids`, `parent_turn_ids`, `turn_rewards` from metadata
1. Insert via `_insert_single`
1. Register `turn_id → seq_id` mappings in `_turn_nodes` for shared-node MCTS

### 3c: Add `_turn_nodes` mapping

```python
self._turn_nodes: dict[str, int]  # turn_id → seq_id
```

In `_insert_single`:

```python
for turn_id in record.turn_ids:
    if turn_id not in self._turn_nodes:
        self._turn_nodes[turn_id] = seq_id
```

### 3d: Update `load_trajectories`

Reconstruct per-turn dicts from stored episode records. Each episode is split back into
per-turn dicts using `turn_response_starts`/`turn_response_ends` to slice the stored
sequence. Metadata keys populated from `TrajectoryRecord`.

### 3e: Backward compatibility for existing `insert_batch` path

Keep the existing path for dicts without `_episode_idx` key (legacy individual-style
trajectories). These use the current logic without episode reconstruction.

______________________________________________________________________

## Step 4: Update `TreeAdvantageComputer` with GRPO normalization

**File:** `customized_areal/tree_search/advantage.py`

### 4a: Add `_normalized_advantages` dict to `MCTSTreeStore`

```python
self._normalized_advantages: dict[int, float] = {}
```

### 4b: GRPO normalization in `compute`

1. Group episodes by `query_id`
1. Compute episode Q-value: `Q = outcome_reward` (from `_rewards[seq_id]`)
1. Per-query GRPO normalization:

```python
for query_id, seq_ids in query_groups.items():
    q_values = [self.tree_store._rewards[sid] for sid in seq_ids]
    mean_q = sum(q_values) / len(q_values)
    std_q = (sum((q - mean_q) ** 2 for q in q_values) / len(q_values)) ** 0.5
    for sid, q in zip(seq_ids, q_values):
        normalized_q = (q - mean_q) / (std_q + eps)
        self.tree_store._normalized_advantages[sid] = normalized_q
```

4. Assign per-token advantages: `normalized_q` on response tokens, 0 on prompt tokens

### 4c: Update `_compute_single`

Use normalized Q-value instead of raw Q-value:

```python
def _compute_single(self, traj, query_id, seq_id, seq_len):
    normalized_q = self.tree_store._normalized_advantages.get(seq_id, 0.0)
    # Fall back to raw Q-value if no normalization available
    if normalized_q == 0.0 and seq_id in self.tree_store._q_values:
        normalized_q = self.tree_store._q_values[seq_id]
    advantages = torch.zeros(seq_len, dtype=torch.float32)
    # Fill response token positions with normalized_q
    ...
```

### 4d: GAE passthrough unchanged

When `advantage_mode=GAE`, `TreeAdvantageComputer.compute` is skipped entirely (existing
behavior). No normalization applied.

______________________________________________________________________

## Step 5: Update trainer integration

**File:** `customized_areal/tree_search/trainer.py`

### 5a: Replace `_patch_wrap_openai_agent_for_query_id` with tree search workflow patch

```python
def _patch_wrap_openai_agent_for_tree_search(rollout_engine, group_size):
    original_wrap = rollout_engine._wrap_openai_agent

    def _tree_search_wrap(agent, proxy_addr):
        from areal.api.cli_args import OpenAIProxyConfig
        openai_cfg = rollout_engine.config.openai or OpenAIProxyConfig()
        inner = original_wrap(agent, proxy_addr)
        return TreeSearchGroupedRolloutWorkflow(
            workflow=inner,
            group_size=group_size,
            logger=...,
        )

    rollout_engine._wrap_openai_agent = _tree_search_wrap
```

### 5b: Update `_init_patches`

```python
def _init_patches(self):
    patch_ppo_actor_for_tree_backup(advantage_mode=self.tree_backup_config.advantage_mode)
    _patch_wrap_openai_agent_for_tree_search(
        self.rollout,
        group_size=self.cache_config.n_samples,
    )
```

### 5c: Update `_cache_aware_prepare_batch`

After `rollout_batch`, split stacked dicts into flat per-turn dicts:

```python
new_trajs = self.actor.rollout_batch(...)
trajs = _split_to_turn_dicts(new_trajs)  # NEW
# Then proceed with existing tree operations:
self.tree_store.insert_batch(trajs)
```

______________________________________________________________________

## Step 6: Update `clear()` in `MCTSTreeStore`

Add cleanup for new fields:

```python
self._turn_nodes.clear()
self._normalized_advantages.clear()
```

______________________________________________________________________

## Step 7: Update existing tests and add new tests

### 7a: Update existing tests in `tests/test_tree_search/test_mcts_tree_store.py`

- Update `TrajectoryRecord` construction to include new fields
- Update `_make_traj` helper to accept per-turn metadata when testing new insert path
- Ensure backward-compatible path still works (dicts without `_episode_idx`)

### 7b: New test: `TreeSearchGroupedRolloutWorkflow`

- Mock `InteractionWithTokenLogpReward` with 2-turn and 3-turn episodes
- Verify output shape and metadata keys
- Verify variable turn count padding

### 7c: New test: `_split_to_turn_dicts`

- Verify correct slicing with variable turn counts and padding
- Verify per-turn metadata is correctly populated

### 7d: New test: `insert_batch` with per-turn dicts

- Verify episode grouping: one `seq_id` per episode
- Verify turn_id registration in `_turn_nodes`
- Verify backward compatibility with old-style dicts

### 7e: New test: GRPO normalization

- Verify Q-values normalized within query groups, not across queries
- Test with multiple episodes per query

### 7f: New test: variable turn counts

- 2 episodes with 2 turns + 1 episode with 3 turns for same query
- Verify normalization is correct

### 7g: New test: shared turn_id

- Two episodes sharing turn 0 but diverging at turn 1
- Verify `_turn_nodes` mapping

______________________________________________________________________

## Implementation Order

1. Step 1: `TreeSearchGroupedRolloutWorkflow` + Step 2: `_split_to_turn_dicts` (new
   file)
1. Step 3: `TrajectoryRecord` extension + `MCTSTreeStore` changes (existing file)
1. Step 4: `TreeAdvantageComputer` GRPO normalization (existing file)
1. Step 5: Trainer integration (existing file)
1. Step 6: `MCTSTreeStore.clear()` update (existing file)
1. Step 7: Tests

______________________________________________________________________

## Files Changed

| File                                               | Action | Description                                                                                     |
| -------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------- |
| `customized_areal/tree_search/grouped_workflow.py` | NEW    | `TreeSearchGroupedRolloutWorkflow` + `_split_to_turn_dicts`                                     |
| `customized_areal/tree_search/mcts_tree_store.py`  | MODIFY | `TrajectoryRecord` extension, `insert_batch` rewrite, `_turn_nodes`, `load_trajectories` update |
| `customized_areal/tree_search/advantage.py`        | MODIFY | GRPO normalization in `compute` + `_compute_single`                                             |
| `customized_areal/tree_search/trainer.py`          | MODIFY | New patch function, `_init_patches` update, `_cache_aware_prepare_batch` update                 |
| `tests/test_tree_search/test_mcts_tree_store.py`   | MODIFY | Update existing tests, add new tests                                                            |
| `tests/test_tree_search/test_advantage.py`         | MODIFY | Add GRPO normalization tests                                                                    |
| `tests/test_tree_search/test_grouped_workflow.py`  | NEW    | Tests for `TreeSearchGroupedRolloutWorkflow` and `_split_to_turn_dicts`                         |
