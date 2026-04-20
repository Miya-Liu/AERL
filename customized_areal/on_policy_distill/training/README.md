# Training Components for On-Policy Distillation

This module contains training components for on-policy distillation with multi-candidate
support.

## Overview

The training module provides:

- **Loss functions**: Combined GRPO + position-level GRPO loss
- **Logprob utilities**: Multi-candidate logprob and entropy computation
- **Trainer**: `OnPolicyDistillationTrainer` extending AReaL's PPOTrainer

## Components

### 1. Loss Function (`loss.py`)

#### `grpo_distill_loss_fn()`

Combined GRPO and position-level GRPO loss function.

```python
def grpo_distill_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    config: PPOActorConfig,
    current_version: int | None = None,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> torch.Tensor:
```

**Parameters**:

- `logprobs`: Log probabilities for all candidates `[seq_len, num_candidates]` or
  `[seq_len]`
  - These are computed by the engine and have gradient information
- `entropy`: Entropy values for the current policy `[seq_len]`
- `input_data`: Dictionary containing:
  - `logprobs`: Old log probabilities from rollout (for importance sampling)
  - `advantages`: Advantage estimates
  - `loss_mask`: Mask indicating which positions to compute loss on
  - `position_rewards`: Position-wise rewards with candidate info (`PositionRewardInfo`
    list)
  - `rl_loss_weight`: Weight for GRPO loss (default: 1.0)
  - `distill_loss_weight`: Weight for distillation loss (default: 0.005)
- `config`: PPO actor configuration
- `current_version`: Current weight version for version alignment
- `vocab_min_logits`, `vocab_max_logits`: Min/max logits for numerical stability

**Returns**: Combined loss tensor (GRPO + position-level GRPO)

**Key behaviors**:

1. For standard GRPO, uses logprobs of chosen tokens only (index 0 if 2D)
1. For position-level GRPO (when `position_rewards` is present and logprobs is 2D):
   - Uses pre-computed multi-candidate logprobs (with gradients)
   - Uses old logprobs from rollout for off-policy importance weighting
   - Computes: `-E[importance_weight * reward * logp]`

#### `_compute_grpo_loss()`

Internal GRPO/PPO loss computation.

```python
def _compute_grpo_loss(
    logprobs: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_higher: float | None,
    loss_mask: torch.Tensor,
    c_clip: float | None,
    proximal_logprobs: torch.Tensor | None,
    behave_imp_weight_cap: float | None,
    importance_sampling_level: str,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
```

#### `_compute_position_level_grpo_loss()`

Position-level GRPO loss using pre-computed multi-candidate logprobs.

```python
def _compute_position_level_grpo_loss(
    position_rewards: list[PositionRewardInfo],
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    output_len: int,
) -> torch.Tensor:
```

**Algorithm** (vectorized):

1. Build padded tensors from `position_rewards`:
   - `positions`: \[num_positions\] - position indices
   - `rewards_tensor`: \[num_positions, max_candidates\] - padded rewards
   - `old_logprobs_tensor`: \[num_positions, max_candidates\] - padded old logprobs
   - `candidate_mask`: \[num_positions, max_candidates\] - valid candidate mask
1. Gather new logprobs for all positions: `logprobs[positions, :]` → \[num_positions,
   max_candidates\]
1. Compute importance weights: `exp(new_logp - old_logp)` (clipped to max 10.0)
1. Compute advantages from rewards (GRPO normalization):
   `(rewards - mean) / (std + eps)`
1. Compute weighted loss: `-mean(importance_weights * advantages * new_logprobs)` over
   candidates
1. Normalize by sum of importance weights per position
1. Average over positions weighted by `loss_mask`

**Important**: This function uses `logprobs` directly from the engine (which has
gradients) and does NOT call `gather_logprobs_entropy_multi_candidates` again.

### 2. Logprob Utilities (`logprobs.py`)

#### `gather_logprobs_entropy_multi_candidates()`

Main entry point for computing logprobs and entropy with multi-candidate support.

```python
def gather_logprobs_entropy_multi_candidates(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
```

**Parameters**:

- `logits`: Model logits with shape `[..., vocab_size]` or `[..., vocab_size/tp]` with
  TP
- `labels`: Token indices
  - Shape `[...]` for single candidate per position
  - Shape `[..., num_candidates]` for multiple candidates per position
- `temperature`: Softmax temperature scaling (default: 1.0)
- `tp_group`: Process group for tensor parallelism (optional)
- `chunk_size`: Chunk size for memory-efficient processing (default: 1024)

**Returns**: `(logprobs, entropy)`

- `logprobs`: Log probabilities at label positions (same shape as labels)
- `entropy`: Entropy of the distribution (shape without last dim of labels)

**Features**:

- Supports tensor parallelism (TP) via vocab-parallel gathering
- Memory-efficient chunked processing for long sequences
- Handles both 1D and 2D labels

#### `_gather_logprobs_entropy_multi_candidates()`

Core implementation without chunking or TP.

```python
def _gather_logprobs_entropy_multi_candidates(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
```

**Algorithm**:

1. Compute log-softmax: `log_probs = log_softmax(logits / temperature, dim=-1)`
1. Compute entropy: `entropy = -sum(exp(log_probs) * log_probs, dim=-1)`
1. Gather logprobs at label positions:
   - If labels is 1D: `log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)`
   - If labels is 2D: `log_probs.gather(dim=-1, index=labels)`

#### `_vocab_parallel_logprobs_entropy_multi_candidates()`

Vocab-parallel implementation for tensor parallelism.

```python
def _vocab_parallel_logprobs_entropy_multi_candidates(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
```

**Algorithm**:

1. Compute local softmax and logprobs on each TP rank
1. All-reduce to get global sum of exponentials
1. Gather logprobs at local label positions
1. All-reduce to combine results from all ranks
1. Mask out-of-range labels with `-inf`

#### `_chunked_apply()` and `_chunked_gather_logprobs_entropy_multi_candidates()`

Memory-efficient chunked processing for long sequences.

### 3. Trainer (`trainer.py`)

#### `OnPolicyDistillationTrainer`

Trainer for on-policy distillation using OpenAI proxy workflow.

```python
class OnPolicyDistillationTrainer(PPOTrainer):
    def __init__(
        self,
        config: Any,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
        workflow: Optional[OpenAIProxyWorkflow] = None,
        agent: Optional[Any] = None,
    ):
```

**Key behaviors**:

- Patches PPOActor class to use `grpo_distill_loss_fn` on initialization
- Initializes OpenAIProxyWorkflow with agent for token-level reward collection
- Extends AReaL's PPOTrainer with on-policy distillation components

**Usage**:

```python
from customized_areal.on_policy_distill import OnPolicyDistillationTrainer

trainer = OnPolicyDistillationTrainer(config)
with trainer:
    trainer.train()
```

## Data Flow

### Complete Training Flow

```
1. Rollout Phase:
   Agent.run() → PositionRewardInfo with candidate_token_ids, old_logprobs, rewards
                      ↓
   OpenAIProxyWorkflow → InteractionCache.set_position_rewards()

2. Training Phase:
   MultiCandidateFSDPEngine._compute_logprobs_and_loss()
                      ↓
   _prepare_multi_candidate_labels() → 2D labels [seq_len, max_candidates]
                      ↓
   gather_logprobs_entropy_multi_candidates(logits, labels)
                      ↓
   logprobs [seq_len, num_candidates] (with gradients!)
                      ↓
   grpo_distill_loss_fn(logprobs, entropy, input_data, ...)
                      ↓
   _compute_grpo_loss() for chosen tokens
   _compute_position_level_grpo_loss() for all candidates
                      ↓
   Combined loss = rl_loss_weight * grpo_loss + distill_loss_weight * position_loss
```

## Integration Points

### With Engine

```python
# MultiCandidateFSDPEngine._compute_logprobs_and_loss()
logprobs, entropy = self._compute_logprobs_entropy(logits, ctx.model_inputs, ...)
loss = loss_fn(
    logprobs,  # [seq_len, num_candidates] with gradients
    entropy,
    ctx.mb_input,  # Contains position_rewards
    ...
)
```

### With Cache

```python
# PositionRewardInfo stores:
- candidate_token_ids: list[int]  # For gathering logprobs
- logprobs: list[float]           # Old logprobs from rollout (for importance weights)
- rewards: list[float]            # Rewards for each candidate
- position: int                   # Position in sequence
```

### With Agent

```python
# OnPolicyDistillAgent._convert_to_position_rewards()
position_rewards.append(
    PositionRewardInfo(
        position=step,
        candidates=candidates,
        candidate_token_ids=candidate_token_ids,  # For engine
        logprobs=logprobs,  # Old logprobs for importance weights
        rewards=rewards,
        chosen_index=chosen_index,
    )
)
```

## Key Design Decisions

1. **Fresh logprobs with gradients**: The engine computes logprobs from current model
   logits, preserving gradient flow for backpropagation.

1. **Old logprobs for importance weights**: `PositionRewardInfo.logprobs` stores old
   logprobs from rollout, used only for computing importance sampling weights
   (`exp(new_logp - old_logp)`).

1. **Separation of concerns**:

   - Engine: Computes fresh logprobs with gradients
   - Loss function: Uses pre-computed logprobs, computes importance weights and loss

1. **Memory efficiency**: Chunked processing and tensor parallelism support for long
   sequences.

## Files

- `loss.py` - Loss functions (`grpo_distill_loss_fn`,
  `_compute_position_level_grpo_loss`)
- `logprobs.py` - Logprob utilities (`gather_logprobs_entropy_multi_candidates`)
- `trainer.py` - `OnPolicyDistillationTrainer` class
- `__init__.py` - Module exports

## Related Modules

- `engine/fsdp_engine.py` - `MultiCandidateFSDPEngine` for computing multi-candidate
  logprobs
- `core/cache.py` - `PositionRewardInfo` for storing candidate information
- `core/agent.py` - `OnPolicyDistillAgent` for generating position rewards
