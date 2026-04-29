# Tree Search List[Dict] Workflow Executor

## Problem

The tree search pipeline currently passes data as stacked tensor dicts
(shape `[num_turns, seq_len]`) through `arun_episode → concat_padded_tensors`.
This requires later splitting back into per-turn dicts for MCTS tree insertion,
and the tensor format doesn't align with `TrajectoryRecord` which stores
unpadded Python lists.

The base `WorkflowExecutor` also asserts `traj is None or isinstance(traj, dict)`,
blocking `list[dict]` returns from `arun_episode`.

## Design

Change the tree search pipeline so that `arun_episode` returns `list[dict]`
directly, where each dict uses Python lists (matching `TrajectoryRecord`)
instead of stacked tensors. A custom `TreeSearchWorkflowExecutor` handles the
new return type.

### Per-dict structure

Every dict at every level (per-turn, per-episode, per-query) has the same
schema:

```python
{
    "input_ids": list[int],           # unpadded token IDs
    "loss_mask": list[int],           # 0=prompt, 1=response
    "logprobs": list[float],          # token log probabilities
    "versions": list[int],            # policy version per token
    "reward": float,                  # outcome reward (for Q-value)
    "turn_response_starts": list[int], # response start indices per turn
    "turn_response_ends": list[int],   # response end indices per turn
    "turn_ids": list[str],            # interaction ID per turn
    "parent_turn_ids": list[str | None], # parent interaction ID per turn
    "turn_rewards": list[float],      # per-turn reward
    "outcome_reward": float,          # outcome reward (separate from reward)
    "response_ids": list[int],        # response token IDs (input_ids where loss_mask=1)
}
```

Only `individual` export style is supported.

### Data flow

| Level | Source | Returns | Semantics |
|-------|--------|---------|-----------|
| Per-turn | `proxy_workflow.arun_episode` | `list[dict]` | Each dict = one turn of one episode. Single-turn episodes have `len=1`. Multi-turn episodes have `len=N` (N turns). |
| Per-episode | `grouped_workflow.arun_episode` | `list[dict]` | Each dict = one complete episode (turns merged into episode-level lists). One dict per group_size episode. |
| Per-query | `rollout_batch` | `list[dict]` | Flat list of per-episode dicts across all queries. |

### Components

#### 1. `TreeSearchWorkflowExecutor` (new file: `customized_areal/tree_search/workflow_executor.py`)

Subclasses `WorkflowExecutor`. Overrides:

- **`_create_workflow_task`**: Accept `list[dict]` from `arun_episode`. Skip
  `InteractionWithTokenLogpReward` conversion (already handled in workflow).
  Skip `assert isinstance(traj, dict)`. Store result in
  `_TreeSearchRolloutResult.trajectories: list[dict]`.

- **`rollout_batch`**: Override to flatten `list[list[dict]]` → `list[dict]`
  from the wait results.

- **`_TreeSearchRolloutResult`** (new dataclass):
  ```python
  @dataclass
  class _TreeSearchRolloutResult:
      task_id: int
      trajectories: list[dict[str, Any]]
  ```

- **`wait`**: Override to extract `trajectories` from
  `_TreeSearchRolloutResult` instead of `trajectory` from `_RolloutResult`.

#### 2. `proxy_workflow.py` changes

`QueryIDProxyWorkflow.arun_episode` returns `list[dict] | None` instead of
`dict | None`.

For each `InteractionWithTokenLogpReward` in the result:
- Convert to a per-turn dict with Python lists
- Compute `turn_response_starts`, `turn_response_ends` from `loss_mask`
- Extract `turn_ids`, `parent_turn_ids`, `turn_rewards` from the interaction
- Compute `response_ids` from `input_ids` where `loss_mask=1`
- Set `reward` and `outcome_reward`
- Inject `_mcts_query_id` into each dict

#### 3. `grouped_workflow.py` changes

`TreeSearchGroupedRolloutWorkflow.arun_episode` returns `list[dict] | None`
instead of `dict | None`.

- Calls inner workflow's `arun_episode` `group_size` times
- Each inner call returns `list[dict]` (per-turn dicts from proxy_workflow)
- Merges per-turn dicts into per-episode dicts: combine `input_ids`,
  `loss_mask`, `logprobs`, `versions` lists; compute episode-level
  `turn_response_starts/ends`, `turn_ids`, `parent_turn_ids`,
  `turn_rewards`, `outcome_reward`
- Returns flat `list[dict]` (one dict per episode)

#### 4. `trainer.py` changes

- Patch `_wrap_openai_agent` to also swap the engine's
  `workflow_executor` with a `TreeSearchWorkflowExecutor`
- Remove `_split_to_turn_dicts` call (no longer needed — data is already
  per-episode dicts)
- Remove `_split_to_turn_dicts` function if no longer used
- `MCTSTreeStore.insert_batch` and `TreeAdvantageComputer.compute` receive
  per-episode dicts directly (may need minor adjustments for Python lists
  vs tensors)

#### 5. `mcts_tree_store.py` changes

`insert_batch` and related methods need to handle the new dict format:
- `input_ids` is `list[int]` instead of `torch.Tensor`
- Compute `turn_response_starts/ends` from `loss_mask` transitions
  if not already present
- Build `TrajectoryRecord` directly from the dict

### Out of scope

- Changes to the base `WorkflowExecutor` or `RolloutWorkflow` contract
- `concat` export style support (individual only)
- Changes to `workflow_executor._dump_trajectory` (not used in tree search)
