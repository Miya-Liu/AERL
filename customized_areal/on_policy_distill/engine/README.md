# MultiCandidateFSDPEngine

Custom FSDP Engine with multi-candidate logprob gathering support for on-policy distillation training.

## Overview

`MultiCandidateFSDPEngine` extends the standard AReaL `FSDPEngine` to support gathering logprobs for multiple candidate tokens per position using `gather_logprobs_entropy_multi_candidates`.

## Key Features

1. **Multi-candidate support**: Gathers logprobs for multiple candidates at each position
2. **Fresh on-policy logprobs**: Computes logprobs from current model (with gradients) for training
3. **Backward compatible**: Works with standard single-candidate training (labels shape `[seq_len]`)
4. **Position-level rewards**: Integrates with `PositionRewardInfo` for candidate-wise reward computation

## Architecture

### Data Flow

```
PositionRewardInfo (with candidate_token_ids)
    ↓
_prepare_multi_candidate_labels() → 2D labels [seq_len, max_candidates]
    ↓
_compute_logprobs_entropy()
    ↓
gather_logprobs_entropy_multi_candidates() → logprobs [seq_len, num_candidates]
    ↓
loss_fn(logprobs, entropy, input_data, ...)
```

### Key Methods

#### `_prepare_multi_candidate_labels()`

Creates a 2D labels tensor from `position_rewards` for multi-candidate gathering.

```python
def _prepare_multi_candidate_labels(
    self,
    model_inputs: dict[str, Any],
    position_rewards: list[PositionRewardInfo],
    seq_len: int,
) -> torch.Tensor | None:
    # Creates labels tensor of shape [seq_len, max_candidates]
    # Filled with candidate_token_ids for each position
```

**Parameters**:
- `model_inputs`: Standard model inputs (for device reference)
- `position_rewards`: List of `PositionRewardInfo` with `candidate_token_ids`
- `seq_len`: Sequence length

**Returns**: 2D labels tensor `[seq_len, max_candidates]` or `None` if not available

#### `_compute_logprobs_entropy()`

Computes logprobs and entropy with multi-candidate support.

```python
def _compute_logprobs_entropy(
    self,
    logits: torch.Tensor,
    inputs: dict[str, Any],
    ulysses_pad_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns:
    # - logprobs: Same shape as labels (1D or 2D)
    # - entropy: [seq_len] or [batch, seq_len]
```

**Key behaviors**:
- Handles both 1D labels `[seq_len]` (single candidate) and 2D labels `[seq_len, num_candidates]` (multi-candidate)
- Uses `gather_logprobs_entropy_multi_candidates` for logprob gathering
- Supports tensor parallelism (TP) and sequence parallelism (Ulysses)

#### `_compute_logprobs_and_loss()`

Main entry point that detects multi-candidate data and routes accordingly.

```python
def _compute_logprobs_and_loss(
    self,
    logits: torch.Tensor,
    ctx: FSDPTrainContext,
    loss_fn: Callable[..., torch.Tensor],
    loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    total_loss_weight: torch.Tensor,
    loss_multiplier: float = 1.0,
) -> torch.Tensor:
```

**Logic flow**:
1. Check if `position_rewards` is in `ctx.mb_input`
2. If yes, prepare multi-candidate labels and compute multi-candidate logprobs
3. If no, use standard single-candidate path
4. Pass logprobs to `loss_fn` for loss computation

## Usage

### In Configuration

```python
from customized_areal.on_policy_distill.engine import MultiCandidateFSDPEngine

# In your training config, use the custom engine
engine = MultiCandidateFSDPEngine(config)
```

### With OnPolicyDistillationTrainer

```python
from customized_areal.on_policy_distill import OnPolicyDistillationTrainer

# The trainer automatically patches PPOActor to use grpo_distill_loss_fn
# and works with MultiCandidateFSDPEngine when configured
trainer = OnPolicyDistillationTrainer(config)
```

## Integration with Loss Function

The engine passes multi-candidate logprobs to `grpo_distill_loss_fn`:

```python
# In _compute_logprobs_and_loss:
loss = loss_fn(
    logprobs,      # [seq_len, num_candidates] with gradients
    entropy,       # [seq_len]
    ctx.mb_input,  # Contains position_rewards with old logprobs
    vocab_min_logits=vocab_min_logits,
    vocab_max_logits=vocab_max_logits,
)
```

The loss function:
1. Uses fresh logprobs (with gradients) for the policy gradient
2. Uses old logprobs from `position_rewards` for importance sampling weights
3. Computes: `loss = -E[importance_weight * reward * logp]`

## Differences from Standard FSDPEngine

| Aspect | FSDPEngine | MultiCandidateFSDPEngine |
|--------|-----------|-------------------------|
| Logprob gathering | `gather_logprobs_entropy()` | `gather_logprobs_entropy_multi_candidates()` |
| Label shape | 1D only `[seq_len]` | 1D `[seq_len]` or 2D `[seq_len, num_candidates]` |
| Multi-candidate | Not supported | Full support |
| Position rewards | Not supported | Integrated |
| Gradient flow | Standard | Preserved for all candidates |

## When to Use

Use `MultiCandidateFSDPEngine` when:
- You want multi-candidate logprob gathering at the engine level
- You're using position-level rewards with multiple candidates per position
- You need fresh on-policy logprobs for all candidates during training
- You're implementing on-policy distillation with token-level rewards

Use standard `FSDPEngine` when:
- You're doing standard single-candidate training
- You don't need multi-candidate support
- You want to minimize code changes

## Files

- `fsdp_engine.py` - `MultiCandidateFSDPEngine` class implementation
- `__init__.py` - Module exports

## Related Modules

- `training/loss.py` - `grpo_distill_loss_fn` for position-level GRPO loss
- `training/logprobs.py` - `gather_logprobs_entropy_multi_candidates` function
- `core/cache.py` - `PositionRewardInfo` for storing candidate information
