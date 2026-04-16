# MCTS Tree Backup Design

## Overview

Add MCTS (Monte Carlo Tree Search) tree backup to the RL training loop. Rollout
paths for the same query are inserted into a shared compressed trie at
turn-level. MCTS backup propagates rewards up the tree to compute Q-values,
which replace GAE as the advantage signal for PPO update.

Two persistence modes are controlled by a single enum config:
- `in_training` — trees accumulate across rollout steps within one training run
  (in-memory only)
- `cross_training` — trees are checkpointed to JSON files and loaded on resume,
  accumulating across training runs

## Architecture

### Current Flow

```
dataloader → rollout → enrich (critic/ref/logp) → GAE advantages → PPO update
```

### New Flow (tree mode on)

```
dataloader → rollout → add turns one-by-one to MCTS tree → finish_sequence (backup) → tree advantages → PPO update
```

When tree mode is off, the existing GAE path runs unchanged.

## Components

All new code lives in `customized_areal/tree_search/`.

| Component | File | Responsibility |
|---|---|---|
| `Turn` | `turn_splitter.py` | Dataclass: prompt_tokens (shared) + response_tokens (branching) |
| `TrieNode` | `trie_node.py` | Compressed trie node: turn tokens, children, path indexing (no MCTS stats) |
| `MCTSTreeStore` | `mcts_tree_store.py` | Per-query tree manager: insert paths, run backup, extract advantages; holds MCTS stats |
| `TreeAdvantageComputer` | `advantage.py` | Replaces GAE: reads Q-values from tree, produces advantage tensors |
| `TreeCheckpointManager` | `checkpoint.py` | Save/load MCTS trees to/from JSON files |
| `TreeBackupConfig` | `config.py` | Enum `TreeBackupMode` and config dataclass |
| `make_turn_splitter` | `turn_splitter.py` | Role-marker-based turn splitting producing structured Turn objects |
| `__init__.py` | `__init__.py` | Public exports |

## Turn — Structured Turn with Prompt/Response Split

```python
@dataclass
class Turn:
    prompt_tokens: list[int]     # shared context tokens (no branching)
    response_tokens: list[int]   # assistant output tokens (branching point)
```

**Design decisions:**

- **Prompt vs response**: each turn is split into a shared prefix (prompt —
  identical across all rollouts for this turn) and an assistant output
  (response — where different rollouts diverge).
- **Branching at first response token**: the trie creates branches keyed by
  the first token of `response_tokens`. Different rollouts that generate
  different first tokens take different branches.
- **Turn boundary**: determined by role markers in the chat template (e.g.,
  `<|im_start|>assistant`). The splitter identifies these boundaries and
  produces `Turn` objects with the prompt/response split.

**Example:**

Input: `[<im_start>user, a, b, <im_start>assistant, e1, f1, <im_start>user, h, <im_start>assistant, l1]`

```
Turn 1: prompt=[<im_start>user, a, b, <im_start>assistant], response=[e1, f1]
Turn 2: prompt=[<im_start>user, h, <im_start>assistant], response=[l1]
```

## TrieNode — Turn-Level Compressed Trie (Path Indexing Only)

```python
@dataclass
class TrieNode:
    tree_id: int                           # identifier of the tree this node belongs to
    start_idx: int = -1                    # start index in flattened tree rep (-1 for root)
    end_idx: int = -1                      # end index (inclusive) in flattened tree rep (-1 for root)
    tokens: list[int] = field(default_factory=list)  # token IDs in this node (empty for root)
    sequence_ids: list[int] = field(default_factory=list)  # sequences passing through this node
    children: dict[int, TrieNode] = field(default_factory=dict)  # first response token -> child
    ancestors: list[TrieNode] = field(default_factory=list)  # root-to-parent chain (empty for root)
    nodes: list[TrieNode] = field(default_factory=list)  # all descendants in pre-order (root only)
```

**Design decisions:**

- **Pure path indexing**: TrieNode stores only structural data — no MCTS
  statistics. Backup stats (visit_count, total_value, q_value) live in
  `MCTSTreeStore`.
- **One turn per node**: each node stores a full turn's tokens. Paths diverge
  at the first response token of each turn.
- **Compressed trie**: children are keyed by the first token of the child
  turn's response. Shared prefixes across rollouts of the same query share
  nodes.
- **Root node**: `tokens = []` (empty sentinel), no ancestors. Represents the
  shared prompt. `start_idx = end_idx = -1`.
- **Flattened indices**: `start_idx`/`end_idx` track each node's position in
  a flattened representation of the tree, enabling efficient batch operations.
- **Ancestors list**: pre-computed path from root to parent, avoiding
  recursive parent-pointer walks during backup.
- **Pre-order nodes**: root holds all descendant nodes in pre-order traversal,
  enabling iteration over all nodes without recursion.

**Core methods:**

| Method | Description |
|---|---|
| `add_turn(turn: Turn, seq_id: int) -> TrieNode` | Add a single turn as a child, keyed by first response token. Returns the child node (cursor for next turn). Tags node with `seq_id`. |
| `get_path_nodes(seq_id: int) -> list[TrieNode]` | Return the non-root nodes on the path for `seq_id`, root-to-leaf order |
| `get_turn_boundaries(seq_id: int) -> list[int]` | Return cumulative token positions where turns start/end |

**Branching example:**

```
Root
└── Turn1: prompt shared [a,b,c,d,<|assistant|] → response branches
    ├── key=e1: response [e1,f1,g1] → Turn2: prompt [h,i,j,k,<|assistant|] → response [l1,m1,n1]
    ├── key=e2: response [e2,f2,g2] → Turn2: prompt [h2,i2,j2,k2,<|assistant|] → response [l2,m2,n2]
    └── key=e3: response [e3,f3,g3]
```

Note: Turn 2's prompt tokens differ across branches because the conversation
history differs based on which Turn 1 response was chosen. This is why
consecutive turns can have no shared prefix — but they are still connected
as parent-child in the tree.

## MCTSTreeStore — Per-Query Tree Manager (Trie + MCTS Stats)

```python
class MCTSTreeStore:
    def __init__(self, turn_splitter: Callable[[list[int]], list[Turn]]):
        self.trees: dict[str, TrieNode] = {}  # query_id -> root node
        self.turn_splitter = turn_splitter
        self._next_seq_id: int = 0

        # Cursor state — tracks current position per (query_id, seq_id)
        self._cursors: dict[tuple[str, int], TrieNode] = {}

        # MCTS statistics — keyed by (query_id, node_id) -> per-node stats
        self._visit_counts: dict[tuple[str, int], int] = {}    # (query_id, node_id) -> count
        self._total_values: dict[tuple[str, int], float] = {}  # (query_id, node_id) -> total
        self._q_values: dict[tuple[str, int], float] = {}      # (query_id, node_id) -> q
```

**Design decisions:**

- **MCTS stats in TreeStore, not TrieNode**: keeps the trie as pure path
  indexing and the backup logic as a separate concern. Stats are keyed by
  `(query_id, node_id)` where `node_id` is the node's position in the root's
  `nodes` list (pre-order index).
- **Cursor-based insertion**: turns are added one at a time, matching the
  sequential generation order of multi-turn rollouts. The cursor tracks the
  current node position for each active sequence.
- **Backup traverses the trie path**: backup walks the `ancestors` list from
  leaf to root, updating stats at each node on the path.

**Methods:**

| Method | Description |
|---|---|
| `start_sequence(query_id) -> int` | Create root if needed, assign a `seq_id`, set cursor at root. Return `seq_id`. |
| `add_turn(query_id, seq_id, turn: Turn)` | Add a single turn at the cursor position, advance cursor to the new child node. |
| `finish_sequence(query_id, seq_id, reward: float)` | Run MCTS backup along the completed path, clear cursor. |
| `insert_trajectory(query_id, input_ids, reward)` | Convenience: split → start_sequence → add_turn loop → finish_sequence. |
| `insert_batch(trajectories: list[dict])` | Batch version — group trajectories by query, insert each group. |
| `get_advantages(query_id, seq_id) -> torch.Tensor` | Get Q-values per turn, expand to per-token advantages aligned with the trajectory's `input_ids` |
| `clear()` | Reset all trees, stats, and cursors (used when `in_training` mode at start) |

**Turn-by-turn insertion flow:**

```python
seq_id = store.start_sequence("q1")
for turn in turns:
    store.add_turn("q1", seq_id, turn)
store.finish_sequence("q1", seq_id, reward=1.0)
```

**What happens step by step:**

1. `start_sequence("q1")`:
   - `root = self.trees.setdefault("q1", TrieNode(tree_id=...))`
   - `seq_id = self._next_seq_id; self._next_seq_id += 1`
   - `self._cursors[("q1", seq_id)] = root`
   - Return `seq_id`

2. `add_turn("q1", seq_id, turn)`:
   - `cursor = self._cursors[("q1", seq_id)]`
   - `child = cursor.add_turn(turn, seq_id)` — creates/walks child keyed by `turn.response_tokens[0]`
   - `self._cursors[("q1", seq_id)] = child` — advance cursor

3. `finish_sequence("q1", seq_id, reward)`:
   - `self._backup(query_id, seq_id, reward)` — walk ancestors from leaf to root
   - `del self._cursors[("q1", seq_id)]` — clear cursor

**Convenience `insert_trajectory` flow:**

1. `turns: list[Turn] = self.turn_splitter(input_ids)`
2. `seq_id = self.start_sequence(query_id)`
3. For each turn: `self.add_turn(query_id, seq_id, turn)`
4. `self.finish_sequence(query_id, seq_id, reward)`
5. Return `seq_id`

**Backup flow:**

1. Get path nodes from root for `seq_id` (using `ancestors` + current node)
2. For each node on the path:
   - Increment `visit_counts[(query_id, node_idx)]`
   - Add reward to `total_values[(query_id, node_idx)]`
   - Recompute `q_values[(query_id, node_idx)] = total / visits`
3. All nodes on the path receive the same reward. No discounting for now.

**Get advantages flow:**

1. Get path nodes for `seq_id` — `[num_turns]` nodes
2. For each node `i`, look up `q_values[(query_id, node_i_idx)]`
3. Compute `boundaries` from cumulative `len(node.tokens)` for each turn
4. `advantages = torch.zeros(total_len)`
5. For each turn `i`: `advantages[boundaries[i]:boundaries[i+1]] = q_values[i]`
6. Return `advantages` — shape `[seq_len]`, one Q-value per token within each turn

**Query ID source**: derived from the prompt tokens (hash of the portion where
`loss_mask == 0`).

## TreeAdvantageComputer — Replaces GAE

```python
class TreeAdvantageComputer:
    def __init__(self, tree_store: MCTSTreeStore):
        self.tree_store = tree_store
```

**`compute` method:**

```python
def compute(self, trajectories: list[dict]) -> None:
    """Replace GAE advantages with tree Q-values. Mutates trajectories in-place."""
    for traj in trajectories:
        query_id = get_query_id(traj)
        seq_id = traj["_mcts_seq_id"]  # set by on_trajectory_ready callback
        advantages = self.tree_store.get_advantages(query_id, seq_id)

        # Mask prompt tokens — advantages only for response tokens
        response_mask = traj["loss_mask"].bool()
        traj["advantages"] = advantages * response_mask.float()

        # Returns = advantages (no critic V(s) baseline in tree mode)
        traj["returns"] = traj["advantages"].clone()
```

**Key decisions:**

- **No critic values**: tree Q-values replace GAE entirely, so `returns =
  advantages`. The `critic.compute_values` step is skipped in tree mode.
- **`_mcts_seq_id`**: transient key added by `on_trajectory_ready` callback
  during streaming insert, valid only within the current training step. Not
  persisted.
- **Advantage shape**: `[1, seq_len]` (or `[group_size, seq_len]` for grouped
  rollouts) to match existing trajectory format.
- **KL tracking**: `kl_rewards` is still computed for logging/monitoring.
  `tot_rewards` = `kl_rewards` + tree-based reward.

## TreeCheckpointManager — Persistence

```python
class TreeCheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = os.path.join(save_dir, "mcts_trees")
```

**File layout:**

```
mcts_trees/
├── query_<hash1>.json
├── query_<hash2>.json
└── metadata.json
```

**Per-tree JSON structure:**

```json
{
  "root": {
    "tree_id": 0,
    "start_idx": -1,
    "end_idx": -1,
    "tokens": [],
    "sequence_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "children": {
      "1234": {
        "tree_id": 0,
        "start_idx": 0,
        "end_idx": 5,
        "tokens": [1234, 5678, 9012],
        "sequence_ids": [0, 1, 2, 3, 5, 6, 9, 11],
        "children": {}
      }
    }
  },
  "mcts_stats": {
    "visit_counts": {0: 12, 1: 8},
    "total_values": {0: 8.4, 1: 5.6},
    "q_values": {0: 0.7, 1: 0.7}
  }
}
```

Children keys are the first token of the child node (as string, since JSON keys
must be strings).

**Methods:**

| Method | Description |
|---|---|
| `save(tree_store: MCTSTreeStore)` | Serialize all trees + MCTS stats + metadata to JSON files |
| `load(turn_splitter) -> MCTSTreeStore` | Deserialize from JSON, reconstruct trie with parent/children links and MCTS stats |
| `exists() -> bool` | Check if checkpoint directory has valid data |

**Integration with RLTrainer checkpointing:**

- **Save**: called alongside existing model checkpoint save
- **Load**: called during `RLTrainer.__init__` when `mode == cross_training` and checkpoint exists
- **`in_training` mode**: trees are in-memory only, cleared on restart
- **`cross_training` mode**: trees survive restarts, accumulate across runs

**Scaling note**: one file per query is simple and enables parallel save/load. If
the number of queries grows very large, a future optimization could shard into
chunked files (e.g., 1000 queries per file).

## Config

```python
class TreeBackupMode(str, Enum):
    OFF = "off"
    IN_TRAINING = "in_training"
    CROSS_TRAINING = "cross_training"

@dataclass
class TreeBackupConfig:
    mode: TreeBackupMode = TreeBackupMode.OFF
    assistant_marker: str = ""        # role marker text, e.g., "<|im_start|>assistant"
    checkpoint_dir: str = ""
```

| Field | Default | Description |
|---|---|---|
| `mode` | `"off"` | Controls tree behavior: off, in-training accumulation, or cross-training persistence |
| `assistant_marker` | `""` | Role marker text identifying assistant turns. Tokenized at runtime to find turn boundaries and prompt/response splits. Auto-detected from tokenizer chat template if empty. |
| `checkpoint_dir` | `""` | Where to save/load tree checkpoints. Defaults to `{experiment_dir}/mcts_checkpoints/` if empty |

**Turn splitter construction:**

```python
def make_turn_splitter(tokenizer, assistant_marker: str = ""):
    # Resolve marker: explicit string, or auto-detect from tokenizer chat template
    if not assistant_marker:
        assistant_marker = _detect_assistant_marker(tokenizer)
    marker_tokens = tokenizer.encode(assistant_marker, add_special_tokens=False)

    def split(input_ids: list[int]) -> list[Turn]:
        # 1. Find all positions where marker_tokens appears in input_ids
        # 2. At each marker position:
        #    - prompt_tokens = everything from start/prev-marker-end to end of marker
        #    - response_tokens = everything from end of marker to next marker start (or end)
        # 3. Return list of Turn(prompt_tokens, response_tokens)
        ...

    return split
```

**Auto-detection**: if `assistant_marker` is empty, the splitter inspects the
tokenizer's `chat_template` to find the assistant role marker. For most chat
models this resolves to `<|im_start|>assistant` (Qwen, Llama-3) or
`<|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>` (Gemma).

## Streaming MCTS Backup Architecture

### Problem

The original design inserted trajectories and ran backup inside
`PPOActor._compute_advantages` (via monkey-patch). This means backup only runs
once all rollouts for a step are complete and enrichment (critic, ref, teacher
logp) has finished. But the inference engine generates new paths **during
rollout** — each trajectory arrives asynchronously. To keep Q-values up-to-date
as the tree grows (enabling future tree-guided rollout), backup must execute
after each new path is generated.

### Solution: Decouple Insert+Backup from Advantage Computation

**Current flow:**
```
prepare_batch → enrich → _compute_advantages (insert + backup + compute) → PPO update
```

**New flow:**
```
prepare_batch → each trajectory arrives → on_trajectory_ready callback → insert + backup
                                          ↓
              enrich → _compute_advantages (read-only: compute advantages from tree Q-values) → PPO update
```

Insert and backup happen in the `BatchTaskDispatcher` consumer thread as soon as
each trajectory arrives. The patched `_compute_advantages` is **read-only** — it
only reads Q-values already computed during streaming backup.

### BatchTaskDispatcher Callback

Add `on_trajectory_ready: Callable[[TResult], None] | None` to
`BatchTaskDispatcher.__init__`. In the `_fetch_loop` consumer thread, after
placing a result in `_pending_results`, fire the callback **outside** the
`_result_cv` lock to avoid holding the lock during callback execution:

```python
# Collect results under lock
new_results = []
with self._result_cv:
    for result in results:
        self._pending_results[result.task_id] = result
        cb_addr = self._task_callbacks.pop(result.task_id, None)
        new_results.append((result, cb_addr))
    self._result_cv.notify_all()

# Fire callbacks outside the lock
for result, cb_addr in new_results:
    if self._on_trajectory_ready is not None:
        try:
            self._on_trajectory_ready(result.data)
        except Exception:
            self.logger.error("on_trajectory_ready callback failed", exc_info=True)
    if cb_addr:
        self._send_callback(cb_addr, result.task_id, result.data)
```

### Thread-Safe MCTSTreeStore

Since the callback fires in the consumer thread while `_compute_advantages`
runs in the main thread, `MCTSTreeStore` must be thread-safe. Add a
`threading.Lock` to all public methods:

```python
class MCTSTreeStore:
    def __init__(self, turn_splitter):
        ...
        self._lock = threading.Lock()

    def insert_trajectory(self, query_id, input_ids, reward):
        with self._lock:
            ...

    def get_advantages(self, query_id, seq_id):
        with self._lock:
            ...

    def get_prompt_mask(self, query_id, seq_id):
        with self._lock:
            ...

    def insert_batch(self, trajectories):
        with self._lock:
            ...

    def clear(self):
        with self._lock:
            ...
```

Single lock, coarse-grained — tree operations are fast (dict lookups, list
appends), so contention is negligible.

### TreeBackupPPOTrainer Integration

The trainer wires up the streaming callback to the RolloutController's
dispatcher **after** rollout initialization. Direct setter on the dispatcher
(avoiding RolloutController/BatchTaskDispatcher signature changes):

```python
class TreeBackupPPOTrainer(PPOTrainer):
    def __init__(self, ...):
        super().__init__(...)
        if self.tree_backup_config.mode != TreeBackupMode.OFF:
            ...
            patch_ppo_actor_for_tree_backup(self.tree_store, self.tree_advantage_computer)

    def train(self, workflow=None, ...):
        if self.tree_backup_config.mode != TreeBackupMode.OFF:
            self._register_tree_callback()
        super().train(workflow=workflow, ...)

    def _register_tree_callback(self):
        """Register on_trajectory_ready callback on the rollout dispatcher."""
        if hasattr(self.actor, 'rollout_coordinator') and \
           self.actor.rollout_coordinator is not None:
            self.actor.rollout_coordinator.dispatcher._on_trajectory_ready = (
                self._on_trajectory_ready
            )

    def _on_trajectory_ready(self, result):
        """Callback: insert trajectory into tree and run backup.

        Called from the BatchTaskDispatcher consumer thread.
        """
        if result is None:
            return
        trajectory = result.trajectory  # _RemoteRolloutResult has .trajectory
        query_id = _get_query_id(trajectory)
        input_ids = trajectory["input_ids"].tolist()
        reward = trajectory["rewards"].item()
        seq_id = self.tree_store.insert_trajectory(query_id, input_ids, reward)
        trajectory["_mcts_seq_id"] = seq_id
        trajectory["_mcts_query_id"] = query_id
```

The patched `_compute_advantages` becomes read-only — `tree_store.insert_batch`
is removed:

```python
def _tree_backup_compute_advantages(self, data):
    # ... (same KL/reward computation) ...

    # === TREE BACKUP replaces GAE (read-only) ===
    # Trajectories already inserted+backed during rollout via streaming callback
    tree_advantage_computer.compute([data])

    # ... (same advantage normalization and output) ...
```

### Reward Timing Guarantee

The `on_trajectory_ready` callback receives trajectories that have already
completed their full workflow execution, including reward computation. Rewards
are always present when the callback fires — no fallback needed.

## RLTrainer Integration

**Minimal, surgical changes to `RLTrainer` via `TreeBackupPPOTrainer` subclass:**

```python
class TreeBackupPPOTrainer(PPOTrainer):
    def __init__(self, ...):
        super().__init__(...)

        if self.tree_backup_config.mode != TreeBackupMode.OFF:
            turn_splitter = make_turn_splitter(self.tokenizer, tree_backup_config.assistant_marker)
            self.tree_store = MCTSTreeStore(turn_splitter)
            self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
            self.tree_checkpoint_manager = TreeCheckpointManager(tree_backup_config.checkpoint_dir)

            if tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
                if self.tree_checkpoint_manager.exists():
                    self.tree_store = self.tree_checkpoint_manager.load(turn_splitter)

            # Patch PPOActor to use tree backup instead of GAE
            patch_ppo_actor_for_tree_backup(self.tree_store, self.tree_advantage_computer)

    def train(self, ...):
        # Register streaming callback after rollout initialization
        if self.tree_backup_config.mode != TreeBackupMode.OFF:
            self._register_tree_callback()
        super().train(...)

    def _save_recover_checkpoint(self, ...):
        super()._save_recover_checkpoint(...)

        if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
            self.tree_checkpoint_manager.save(self.tree_store)
```

## File Summary

**New files in `customized_areal/tree_search/`:**

| File | Contains |
|---|---|
| `__init__.py` | Public exports |
| `config.py` | `TreeBackupMode`, `TreeBackupConfig` |
| `trie_node.py` | `TrieNode` — turn-level compressed trie node (path indexing only) |
| `turn_splitter.py` | `Turn` dataclass, `make_turn_splitter` (role-marker-based) |
| `mcts_tree_store.py` | `MCTSTreeStore` — per-query tree manager + MCTS backup stats |
| `advantage.py` | `TreeAdvantageComputer` |
| `checkpoint.py` | `TreeCheckpointManager` |

**Modified files:**

| File | Change |
|---|---|
| `areal/infra/workflow_executor.py` | Add `on_trajectory_ready` callback to `BatchTaskDispatcher` |
| `customized_areal/tree_search/mcts_tree_store.py` | Add `threading.Lock` for thread safety |
| `customized_areal/tree_search/trainer.py` | Streaming callback registration, read-only `_compute_advantages` |
