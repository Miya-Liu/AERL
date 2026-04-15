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
dataloader → rollout → insert paths into MCTS tree → MCTS backup → tree advantages → PPO update
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
| `insert_path(turns: list[Turn], seq_id: int)` | Walk/create child nodes along the turn sequence, tag each with `seq_id` |
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
- **Backup traverses the trie path**: backup walks the `ancestors` list from
  leaf to root, updating stats at each node on the path.

**Methods:**

| Method | Description |
|---|---|
| `insert_trajectory(query_id, input_ids, reward)` | Split `input_ids` into `Turn` objects, insert into the per-query trie, assign a `seq_id`, run backup |
| `insert_batch(trajectories: list[dict])` | Batch version — group trajectories by query, insert each group |
| `get_advantages(query_id, seq_id) -> torch.Tensor` | Get Q-values per turn, expand to per-token advantages aligned with the trajectory's `input_ids` |
| `clear()` | Reset all trees and stats (used when `in_training` mode at start) |

**Insert trajectory flow:**

1. `turns: list[Turn] = self.turn_splitter(input_ids)`
2. `seq_id = self._next_seq_id; self._next_seq_id += 1`
3. `root = self.trees.setdefault(query_id, TrieNode(tree_id=...))`
4. `root.insert_path(turns, seq_id)` — creates/walks child nodes
5. `self._backup(query_id, seq_id, reward)` — update MCTS stats
6. Return `seq_id`

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
        seq_id = traj["_mcts_seq_id"]  # set by insert_batch
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
- **`_mcts_seq_id`**: transient key added by `insert_batch`, valid only within
  the current training step. Not persisted.
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

## RLTrainer Integration

**Minimal, surgical changes to `RLTrainer`:**

```python
class RLTrainer:
    def __init__(self, ...):
        # ... existing init ...

        # NEW: tree backup setup
        if tree_backup_config.mode != TreeBackupMode.OFF:
            turn_splitter = make_turn_splitter(self.tokenizer, tree_backup_config.assistant_marker)
            self.tree_store = MCTSTreeStore(turn_splitter)
            self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
            self.tree_checkpoint_manager = TreeCheckpointManager(tree_backup_config.checkpoint_dir)

            if tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
                if self.tree_checkpoint_manager.exists():
                    self.tree_store = self.tree_checkpoint_manager.load(turn_splitter)

    def _train_loop(self, ...):
        for step in ...:
            rollout_batch = self.actor.prepare_batch(...)

            # ... existing enrichment (ref/logp/teacher_logp) ...

            # NEW: branch on tree mode
            if self.tree_backup_config.mode != TreeBackupMode.OFF:
                # Skip critic.compute_values — tree Q-values replace GAE
                self.tree_store.insert_batch(rollout_batch)
                self.tree_advantage_computer.compute(rollout_batch)
            else:
                # Existing path unchanged
                self.critic.compute_values(rollout_batch)
                self.actor.compute_advantages(rollout_batch)

            self.actor.ppo_update(adv_batch)

    # NEW: hook into existing checkpoint save
    def _save_checkpoint(self, ...):
        # ... existing model checkpoint logic ...

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
| `areal/trainer/rl_trainer.py` | Tree backup init, train loop branching, checkpoint hook |
