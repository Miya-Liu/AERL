# Tree Search Distilling Training Pipeline

This document describes the step-by-step training process of `TreeDistillPPOTrainer`,
which combines **MCTS tree backup advantages**, **on-policy distillation**, and
**rollout caching** into a unified PPO training loop.

## Architecture Overview

```
TreeDistillPPOTrainer
├── inherits CacheAwarePPOTrainer (rollout caching + MCTS tree backup)
│   └── inherits PPOTrainer (standard RL training loop)
├── patches PPOActor._ppo_update → grpo_distill_loss_fn (GRPO + position-level GRPO)
├── overrides _create_actor → MultiCandidateFSDPPPOActor (multi-candidate logprobs)
└── uses OpenAIProxyWorkflow + TreeDistillAgent (rollout generation with position rewards)
```

______________________________________________________________________

## Step-by-Step Training Process

### Step 1: Initialization

When `TreeDistillPPOTrainer.__init__` is called, it performs three setup operations
**before** delegating to the base class:

1. **Patch the loss function**: Calls `patch_ppo_actor_class_to_use_distill_loss()`
   which replaces `PPOActor._ppo_update` with a version that uses `grpo_distill_loss_fn`
   instead of the standard `grpo_loss_fn`. This patching happens **before**
   `super().__init__()` so the patched loss is active when the actor is created.

1. **Initialize workflow and agent**: If no pre-configured workflow is provided,
   `_init_components()` is called:

   - Creates a `TreeDistillAgent` (optionally with a `TeacherConfig` if
     `teacher_model_name` is set in config)
   - Creates an `OpenAIProxyWorkflow` wrapping the agent, configured with proxy address,
     API key, discount factor, and export style

1. **Initialize base CacheAwarePPOTrainer**: Calls `super().__init__()`, which:

   - Initializes the base `PPOTrainer` (creates the training actor, dataloader, etc.)
   - Sets up the MCTS tree store, tree advantage computer, and checkpoint manager
   - Patches `PPOActor.compute_advantages` to add MCTS tree backup after GAE

______________________________________________________________________

### Step 2: Rollout Generation (with Cache Awareness)

Each training step begins with rollout generation. Because
`CacheAwarePPOTrainer.train()` monkey-patches `self.actor.prepare_batch` with
`_cache_aware_prepare_batch`, the rollout process is cache-aware:

1. **Pull a batch from the dataloader**: Gets a list of prompt items.

1. **Split prompts into cached / needs-generation**: For each prompt, derives a
   `query_id` from the prompt messages (via tokenizer + MD5 hash), then checks the
   `MCTSTreeStore` for untrained cached rollouts:

   - **Fully cached**: If untrained count >= n_samples, all trajectories come from cache
   - **Partially cached**: If some untrained trajectories exist, load those and generate
     the rest
   - **Not cached**: All trajectories need generation

1. **Load cached trajectories**: For prompts with cached rollouts, loads trajectory data
   (input_ids, logprobs, loss_mask, rewards, etc.) directly from the tree store — no
   inference needed.

1. **Generate missing trajectories**: For prompts without enough cached rollouts, calls
   `actor.rollout_batch()` which:

   - Submits prompts to the `OpenAIProxyWorkflow`
   - The workflow starts a proxy session, runs the `TreeDistillAgent`, collects rewards,
     and exports interactions
   - The agent runs `run_backend()` to generate completions, then computes
     position-level rewards
   - Returns trajectory dicts with tensors (input_ids, logprobs, loss_mask, rewards,
     etc.)

1. **Merge cached and new trajectories**: Combines both into a single list, preserving
   per-trajectory metadata (`query_id`, `_mcts_seq_id`).

______________________________________________________________________

### Step 3: Critic Value Estimation (Optional)

If a critic model is configured, runs a forward pass on the rollout batch to produce
value estimates. These are attached to each trajectory as `values`. This step is
unchanged from the base PPOTrainer.

______________________________________________________________________

### Step 4: Reference Model Log-Probs (Optional)

If a reference model exists (for KL-regularized training), computes reference
log-probabilities for KL-penalty computation. Attached as `ref_logp` to each trajectory.

______________________________________________________________________

### Step 5: Proximal Log-Probs (Conditional)

Depending on the `prox_clip` configuration, may recompute log-probabilities from the
current policy for importance sampling. This step is unchanged from the base PPOTrainer.

______________________________________________________________________

### Step 6: Compute Advantages (with MCTS Tree Backup)

This is where the tree backup patch takes effect. The patched
`PPOActor.compute_advantages` executes:

1. **Run original GAE pipeline**: The base `compute_advantages` method runs first:

   - Applies overlong reward penalty
   - Scales and clips rewards
   - Computes KL-regularized rewards (`kl_reward = -kl_ctl * KL(old_logp, ref_logp)`)
   - Computes GAE (Generalized Advantage Estimation) using `discount` and `gae_lambda`
   - Optionally normalizes advantages

1. **Insert trajectories into MCTS tree**: `tree_store.insert_batch(result)` inserts
   each trajectory into the compressed trie:

   - Splits each trajectory into turns using the turn splitter
   - Creates a root node for the query (if not exists)
   - Adds turn nodes along the path, advancing a cursor
   - Runs MCTS backup from leaf to root: increments visit counts, accumulates total
     values, updates Q-values as `Q = total_value / visit_count`

1. **Replace GAE advantages with tree Q-values**:
   `tree_advantage_computer.compute(result)` overwrites the advantages:

   - For each trajectory, looks up Q-values per turn from the tree
   - Expands per-turn Q-values to per-token advantages
   - Zeros out prompt tokens so only response tokens carry the advantage signal
   - Sets `returns = advantages.clone()`

1. **Mark trajectories as trained**: `_mark_batch_trained(tree_store, result)` sets the
   trained flag on each trajectory's sequence ID, preventing it from being loaded from
   cache again in future steps.

______________________________________________________________________

### Step 7: Actor PPO Update (with Distillation Loss)

The patched `_ppo_update_with_distill_loss` replaces the standard PPO update:

1. **Remove unused keys**: Pops `rewards`, `tot_rewards`, `kl_rewards` from the data
   dict.

1. **Split into minibatches**: Divides the batch into `ppo_n_minibatches` minibatches
   for gradient accumulation.

1. **For each minibatch, call `engine.train_batch`** with `grpo_distill_loss_fn`:

   The `grpo_distill_loss_fn` computes a **combined loss**:

   **a. Standard GRPO Loss** (weighted by `rl_loss_weight`, default 1.0):

   - Extracts log-probabilities of chosen tokens from the multi-candidate logprobs
   - Computes importance weights: `exp(new_logp - old_logp)`
   - Applies PPO clipping:
     `min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)`
   - Optionally applies dual clipping (c_clip) for negative advantages

   **b. Position-Level GRPO Loss** (weighted by `distill_loss_weight`, default 0.005):

   - Uses multi-candidate logprobs: `[seq_len, num_candidates]` (provided by
     `MultiCandidateFSDPPPOActor`)
   - For each position, retrieves `PositionRewardInfo` containing:
     - `candidate_token_ids`: token IDs for all candidates
     - `rewards`: per-candidate rewards (from teacher model or zero for student-only)
     - `logprobs`: old log-probabilities from rollout (for importance weighting)
   - Normalizes rewards within each position group (mean/std across candidates)
   - Computes importance-weighted GRPO:
     `loss = -(importance_weight * normalized_advantage * new_logprob)`
   - Averages across positions and valid tokens

   **c. Combined Loss**:
   `total_loss = rl_loss_weight * GRPO_loss + distill_loss_weight * position_GRPO_loss`

1. **Log statistics**: Tracks distill loss, importance weights, KL divergence, clip
   ratios, etc.

______________________________________________________________________

### Step 8: Critic PPO Update (Optional)

If a critic model is configured, updates the value function using the same advantage
estimates. This step is unchanged from the base PPOTrainer.

______________________________________________________________________

### Step 9: Weight Update to Inference Engine

After the training update, the updated actor weights are pushed to the rollout inference
engine:

- Pauses the rollout engine
- Updates weights (via disk checkpoint or NCCL/XCCL distributed update)
- Increments and propagates version numbers to actor, critic, and rollout engine
- Resumes the rollout engine

______________________________________________________________________

### Step 10: Save & Checkpoint

- Saves model in HuggingFace format
- Saves recoverable checkpoint (actor, critic, dataloader state)
- **Tree-specific**: If `TreeBackupMode.CROSS_TRAINING`, also saves the MCTS tree
  checkpoint (including all cached rollouts, visit counts, Q-values) so training can
  resume with the tree intact

______________________________________________________________________

### Step 11: Evaluation (Optional)

If a valid dataset and eval workflow are configured, runs evaluation. This step is
unchanged from the base PPOTrainer.

______________________________________________________________________

### Step 12: Cleanup & Logging

- Clears distributed tensor shards from workers
- Exports statistics to wandb/tensorboard
- Restores the original `prepare_batch` method (removing the cache-aware monkey-patch)
- Cleans up the dataloader iterator

______________________________________________________________________

## Key Components

### MCTSTreeStore

A compressed trie that stores all rollout trajectories for a given query. Key
operations:

- **`insert_trajectory`**: Splits a trajectory into turns, creates/extends trie nodes,
  runs MCTS backup
- **`get_advantages`**: Returns per-token Q-values by looking up Q-values per turn and
  expanding to token granularity
- **`load_trajectories`**: Reconstructs trajectory dicts from the trie for cached
  rollouts
- **`get_untrained_count`**: Counts trajectories not yet used for training (available
  for cache loading)

### TreeAdvantageComputer

Replaces GAE advantages with tree Q-values. For each trajectory:

- Looks up Q-values from the tree (computed by MCTS backup across all trajectories
  sharing the same prefix)
- Zeroes out prompt token positions
- Overwrites both `advantages` and `returns` fields

### TreeDistillAgent

Extends `OnPolicyDistillAgent` with a student-only fallback:

- **With teacher**: Computes position rewards as `student_logprob - teacher_logprob` for
  each candidate token
- **Without teacher**: Builds `PositionRewardInfo` from student's own top-k logprobs
  with zero rewards, ensuring multi-candidate logprobs are still gathered for logging

### MultiCandidateFSDPPPOActor

Overrides the standard `FSDPPPOActor` to gather log-probabilities for multiple candidate
tokens per position (not just the chosen token). This enables position-level GRPO loss
computation.

### grpo_distill_loss_fn

Combined loss function:

- Standard GRPO loss on chosen-token log-probabilities with tree-backed advantages
- Position-level GRPO loss on multi-candidate log-probabilities with per-candidate
  rewards
- Weighted sum: `rl_loss_weight * GRPO + distill_loss_weight * distill_GRPO`

______________________________________________________________________

## Cache Tree & Checkpoint: Illustrative Examples

This section provides concrete examples of how rollout trajectories are stored in the
MCTS cache tree and how the tree is persisted to disk as a checkpoint.

### Cache Tree Saving

Rollout trajectories are inserted into `MCTSTreeStore` as compressed trie paths.
Trajectories that share a prefix (common initial turns) share the same trie nodes,
enabling MCTS statistics to aggregate across all trajectories for a query.

**Example: Inserting two trajectories for the same query**

Suppose a query (e.g., a math problem) has two sampled completions that diverge after
the first turn:

```
Trajectory A:  [system+user prompt] → [assistant turn 1: "Let x ="]  → [assistant turn 2: "5"]
Trajectory B:  [system+user prompt] → [assistant turn 1: "Let x ="]  → [assistant turn 2: "10"]
```

Using the `MCTSTreeStore` API directly:

```python
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.turn_splitter import Turn, make_turn_splitter

# Assume tokenizer is loaded; assistant_marker is e.g. "<|im_start|>assistant"
turn_splitter = make_turn_splitter(tokenizer, assistant_marker="<|im_start|>assistant")
store = MCTSTreeStore(turn_splitter)

# Insert Trajectory A (reward = 1.0)
seq_a = store.insert_trajectory(
    query_id="abc123",
    input_ids=[1, 50, 100, 200, 300, 400],  # tokenized full sequence
    reward=1.0,
    logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
)

# Insert Trajectory B (reward = 0.0)
seq_b = store.insert_trajectory(
    query_id="abc123",
    input_ids=[1, 50, 100, 200, 300, 500],  # same prefix, diverges at last token
    reward=0.0,
    logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.7],
)
```

**Resulting trie structure:**

```
Root (query_id="abc123")
 └── Turn 1 [tokens: 1, 50, 100, 200]       ← shared prefix (key=200)
      ├── sequence_ids: [seq_a, seq_b]       ← both trajectories pass through
      └── Turn 2a [tokens: 300, 400]         ← Trajectory A's continuation (key=400)
      │    └── sequence_ids: [seq_a]
      └── Turn 2b [tokens: 300, 500]         ← Trajectory B's continuation (key=500)
           └── sequence_ids: [seq_b]
```

After insertion, MCTS backup runs from leaf to root for each trajectory, producing
Q-values:

| Node        | Visit Count | Total Value | Q-value |
| ----------- | ----------- | ----------- | ------- |
| Root        | 2           | 1.0         | 0.5     |
| Turn 1      | 2           | 1.0         | 0.5     |
| Turn 2a (A) | 1           | 1.0         | 1.0     |
| Turn 2b (B) | 1           | 0.0         | 0.0     |

**Loading cached trajectories for a new training step:**

```python
# Check how many untrained trajectories are available
count = store.get_untrained_count("abc123")  # → 2

# Load up to 2 trajectories from cache (no inference needed)
cached = store.load_trajectories("abc123", n_samples=2)
# Returns list of dicts with keys: input_ids, logprobs, loss_mask,
# attention_mask, rewards, versions, query_id, _mcts_seq_id
```

After training on these trajectories, they are marked as trained:

```python
store.set_trained("abc123", seq_a, trained=True)
store.set_trained("abc123", seq_b, trained=True)
# Now store.get_untrained_count("abc123") → 0
```

### Tree Checkpoint Save

When `tree_backup_config.mode` is `CROSS_TRAINING`, the entire MCTS tree (including all
cached rollouts, visit counts, Q-values, and trained flags) is persisted to disk as JSON
files so that training can resume with the tree intact across restarts.

**When checkpoints are saved:**

- Automatically during `_save_recover_checkpoint()` (called by the base PPOTrainer at
  regular intervals)
- Only when `cache_config.enabled=True` **and** `tree_backup_config.mode=CROSS_TRAINING`
- The checkpoint directory is `cache_dir` (or `tree_backup_config.checkpoint_dir`)

**On-disk format:**

```
{cache_dir}/
  mcts_trees/
    metadata.json              # Global metadata
    query_abc123.json          # Serialized trie for query "abc123"
    query_def456.json          # Serialized trie for query "def456"
    ...
```

**`metadata.json`** contains the sequence ID counter, trained flags, and rewards:

```json
{
  "next_seq_id": 2,
  "trained": {
    "abc123:0": true,
    "abc123:1": true
  },
  "rewards": {
    "abc123:0": 1.0,
    "abc123:1": 0.0
  }
}
```

**`query_abc123.json`** contains the serialized trie root with recursive children:

```json
{
  "root": {
    "tree_id": 0,
    "start_idx": -1,
    "end_idx": -1,
    "tokens": [],
    "sequence_ids": [0, 1],
    "children": {
      "200": {
        "tree_id": 0,
        "start_idx": 0,
        "end_idx": 3,
        "tokens": [1, 50, 100, 200],
        "prompt_len": 2,
        "logprobs": [-0.1, -0.2, -0.3, -0.4],
        "versions": [0, 0, 0, 0],
        "sequence_ids": [0, 1],
        "children": {
          "400": {
            "tree_id": 0,
            "start_idx": 4,
            "end_idx": 5,
            "tokens": [300, 400],
            "prompt_len": 0,
            "logprobs": [-0.5, -0.6],
            "versions": [0, 0],
            "sequence_ids": [0],
            "children": {}
          },
          "500": {
            "tree_id": 0,
            "start_idx": 4,
            "end_idx": 5,
            "tokens": [300, 500],
            "prompt_len": 0,
            "logprobs": [-0.5, -0.7],
            "versions": [0, 0],
            "sequence_ids": [1],
            "children": {}
          }
        }
      }
    }
  }
}
```

**Loading a tree checkpoint:**

When a new training run starts with `CROSS_TRAINING` mode, the trainer automatically
loads any existing checkpoint:

```python
# This happens inside CacheAwarePPOTrainer.__init__():
if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
    if self.tree_checkpoint_manager.exists():
        self.tree_store = self.tree_checkpoint_manager.load(turn_splitter)
        # After load, MCTS stats (Q-values, visit counts) are rebuilt from
        # stored rewards via store.rebuild_mcts_stats(), because node id()
        # values change after deserialization.
    self.tree_store.reset_trained_flags()
    # All trajectories become available for the new training run
```

**Manual save/load example:**

```python
from customized_areal.tree_search.checkpoint import TreeCheckpointManager

# Save
checkpoint_mgr = TreeCheckpointManager(save_dir="/path/to/cache_dir")
checkpoint_mgr.save(tree_store)
# Writes /path/to/cache_dir/mcts_trees/metadata.json + query_*.json

# Load (in a new process or after restart)
checkpoint_mgr = TreeCheckpointManager(save_dir="/path/to/cache_dir")
if checkpoint_mgr.exists():
    tree_store = checkpoint_mgr.load(turn_splitter)
    # tree_store now contains all previously cached trajectories with
    # correct MCTS Q-values. Trained flags are reset so trajectories
    # are available for the new training run.
```

**Configuration in the YAML file:**

```yaml
# Set cache_dir to enable tree checkpoint persistence
cache_dir: /path/to/cache

# Tree backup mode must be "cross_training" for checkpoint save/load
# This is set automatically by the training script:
#   tree_backup_config = TreeBackupConfig(
#       mode=TreeBackupMode.CROSS_TRAINING,
#       assistant_marker=assistant_marker,
#       checkpoint_dir=cache_dir,
#   )
```

______________________________________________________________________

## Data Flow Summary

```
                          ┌─────────────────────────┐
                          │     Dataloader Batch     │
                          └───────────┬─────────────┘
                                      │
                          ┌───────────▼─────────────┐
                          │  Cache-Aware Splitting   │
                          │  (query_id → tree store) │
                          └─────┬───────────┬───────┘
                                │           │
                     ┌──────────▼──┐  ┌─────▼──────────────────────────────────┐
                     │ Cached Trajs│  │         Generate New Rollout           │
                     │ (from tree) │  │  ┌──────────────────────────────────┐  │
                     └──────────┬──┘  │  │ 1. Student generates completion  │  │
                                │     │  │    via proxy server              │  │
                                │     │  │    (run_backend → LLM call)      │  │
                                │     │  └──────────────┬───────────────────┘  │
                                │     │  ┌──────────────▼───────────────────┐  │
                                │     │  │ 2. Fetch student interaction     │  │
                                │     │  │    output_ids + top-k logprobs   │  │
                                │     │  └──────────────┬───────────────────┘  │
                                │     │          ┌─────┴─────┐                 │
                                │     │          │ Teacher?  │                 │
                                │     │     ┌────┴────┐ ┌────┴───────────┐    │
                                │     │     │  Yes    │ │  No            │    │
                                │     │ ┌───▼──────┐ │ │ ┌────────────┐ │    │
                                │     │ │3a.Query  │ │ │ │3b.Student  │ │    │
                                │     │ │  teacher │ │ │ │  only:     │ │    │
                                │     │ │  model   │ │ │ │ zero       │ │    │
                                │     │ │  for     │ │ │ │ rewards +  │ │    │
                                │     │ │  logprobs│ │ │ │ student    │ │    │
                                │     │ │  at      │ │ │ │ logprobs   │ │    │
                                │     │ │  candi-  │ │ │ └─────┬──────┘ │    │
                                │     │ │  date    │ │ │       │        │    │
                                │     │ │  tokens  │ │ └───────┼────────┘    │
                                │     │ └───┬──────┘ │         │             │
                                │     │     │        │         │             │
                                │     │ ┌───▼────────▼─────────▼──────────┐  │
                                │     │ │4. Compute PositionRewardInfo    │  │
                                │     │ │   per position:                  │  │
                                │     │ │   reward = student_lp − teacher_lp│ │
                                │     │ │   (no teacher → reward = 0)      │  │
                                │     │ └──────────────┬──────────────────┘  │
                                │     │ ┌──────────────▼──────────────────┐  │
                                │     │ │5. Compute scalar reward          │  │
                                │     │ │   (accuracy-based, from GT)     │  │
                                │     │ └──────────────┬──────────────────┘  │
                                │     │ ┌──────────────▼──────────────────┐  │
                                │     │ │6. Set rewards on proxy server    │  │
                                │     │ │   → set_position_rewards()       │  │
                                │     │ │   → set_reward(scalar)           │  │
                                │     │ └──────────────┬──────────────────┘  │
                                │     │ ┌──────────────▼──────────────────┐  │
                                │     │ │7. Export interactions             │  │
                                │     │ │   tensor_dict + position_rewards │  │
                                │     │ └──────────────┬──────────────────┘  │
                                │     └────────────────┼─────────────────────┘
                                │                      │
                          ┌─────▼──────────────────────▼─────┐
                          │       Merged Rollout Batch        │
                          │  (cached + new, with              │
                          │   position_rewards attached)      │
                          └───────────┬──────────────────────┘
                                      │
                          ┌───────────▼─────────────┐
                          │  GAE + KL Rewards       │
                          │  (standard pipeline)    │
                          └───────────┬─────────────┘
                                      │
                          ┌───────────▼─────────────┐
                          │  MCTS Tree Insert +     │
                          │  Tree Q-value Backup    │
                          │  (advantages replaced)  │
                          └───────────┬─────────────┘
                                      │
                          ┌───────────▼─────────────┐
                          │  PPO Update with        │
                          │  grpo_distill_loss_fn   │
                          │  (GRPO + position GRPO) │
                          └───────────┬─────────────┘
                                      │
                          ┌───────────▼─────────────┐
                          │  Weight Update +        │
                          │  Tree Checkpoint Save   │
                          └─────────────────────────┘
```

______________________________________________________________________

## Configuration

Key configuration options (via `OnPolicyDistillConfig` and related):

| Parameter                 | Default                   | Description                                                         |
| ------------------------- | ------------------------- | ------------------------------------------------------------------- |
| `teacher_model_name`      | `""`                      | Teacher model for distillation. Empty = student-only mode           |
| `teacher_base_url`        | `"http://localhost:8001"` | Teacher model API endpoint                                          |
| `teacher_top_k`           | `10`                      | Number of top candidate tokens from teacher                         |
| `student_top_k`           | `10`                      | Number of top candidate tokens from student (for student-only mode) |
| `rl_loss_weight`          | `1.0`                     | Weight for standard GRPO loss                                       |
| `distill_loss_weight`     | `0.005`                   | Weight for position-level GRPO distillation loss                    |
| `proxy_base_url`          | `"http://localhost:8000"` | OpenAI proxy server address                                         |
| `turn_discount`           | `1.0`                     | Discount factor for multi-turn reward backpropagation               |
| `cache_config.enabled`    | `True`                    | Enable rollout caching                                              |
| `cache_config.n_samples`  | `1`                       | Number of trajectories to cache per prompt                          |
| `tree_backup_config.mode` | `"off"`                   | Tree backup mode: `"off"`, `"in_training"`, `"cross_training"`      |
