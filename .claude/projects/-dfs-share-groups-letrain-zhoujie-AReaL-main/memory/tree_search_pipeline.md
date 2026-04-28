---
name: Tree Search Training Pipeline
description: CacheAwarePPOTrainer architecture, MCTS tree backup patching, rollout cache, and replay mode in customized_areal/tree_search
type: project
---

## Tree Search Training Pipeline (customized_areal/tree_search)

### Architecture Overview
- `CacheAwarePPOTrainer(PPOTrainer)` — main trainer, overrides `train()` to monkey-patch `self.actor.prepare_batch`
- `MCTSTreeStore` — stores trajectories keyed by query_id, tracks trained/untrained flags
- `TreeAdvantageComputer` — overwrites GAE advantages with MCTS Q-values
- `TreeCheckpointManager` — save/load tree state for cross-training runs
- `RolloutCacheConfig` — cache settings (enabled, n_samples, replay, cache_dir)
- `TreeBackupConfig` — tree backup settings (mode: OFF/CROSS_TRAINING)

### PPOActor Patching Flow
1. `patch_ppo_actor_for_tree_backup()` wraps `PPOActor.compute_advantages`
2. Patched method: original GAE → `tree_store.insert_batch(result)` → `tree_advantage_computer.compute(result)` (overwrites advantages/returns) → `_mark_batch_trained()`
3. Original is saved as `_original_compute_advantages` to prevent stacking patches
4. `unpatch_ppo_actor()` restores original on `close()`

### Cache-Aware Rollout (_cache_aware_prepare_batch)
1. `_CacheAwareBatchBuilder.split_prompts()` — splits prompts into cached/need_gen based on `tree_store.get_untrained_count()`
2. If ALL prompts have enough cache → load from cache only
3. If ANY prompt lacks cache → regenerate ALL prompts via `self.actor.rollout_batch()`
4. `_merge_cached_and_new()` — merges cached (shape [1, seq_len] each) + new trajectories, splits batch_size>1 into individual items to preserve `_mcts_seq_ids`

### Replay Mode (_replay_prepare_batch) — 3-level fallback
1. Level 1: Replay from `tree_store._training_history[global_step]` (exact step order)
2. Level 2: Load untrained trajectories from tree store
3. Level 3: Fresh generation from dataloader, prioritizing novel query_ids not in tree store

### Key Data Flow
- `rollout_batch` returns `list[dict[str, Any]]` (see rollout_batch_format memory)
- `_merge_cached_and_new` splits grouped trajectories (batch_size>1) into individual [1, seq_len] dicts
- Trajectories carry `_mcts_query_id` and `_mcts_seq_id`/`_mcts_seq_ids` metadata for tree store tracking
- `get_query_id_from_messages()` derives query_id from messages using tokenizer

### Key source files:
- `customized_areal/tree_search/trainer.py` — CacheAwarePPOTrainer
- `customized_areal/tree_search/mcts_tree_store.py` — MCTSTreeStore
- `customized_areal/tree_search/advantage.py` — TreeAdvantageComputer
- `customized_areal/tree_search/checkpoint.py` — TreeCheckpointManager
- `customized_areal/tree_search/config.py` — RolloutCacheConfig, TreeBackupConfig
- `customized_areal/tree_search/turn_splitter.py` — turn boundary detection

### Limitation: multi-turn non-shared-prefix agent trajectories
When using agent workflows where turns don't share prefix (InteractionWithTokenLogpReward with individual style), each turn becomes an independent trajectory dict. The tree backup "complete trajectory" concept may need redefinition since one episode is split into multiple independent trajectories.
