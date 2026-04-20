# Engine Integration for Multi-Candidate Training

## Overview

This document describes how multi-candidate logprob gathering integrates with the
training engine.

## Architecture

### Recommended: MultiCandidateFSDPEngine

The recommended approach is to use the custom `MultiCandidateFSDPEngine` provided in
`customized_areal/on_policy_distill/engine`:

```python
from customized_areal.on_policy_distill.engine import MultiCandidateFSDPEngine

engine = MultiCandidateFSDPEngine(config)
```

This engine:

- Uses `gather_logprobs_entropy_multi_candidates` for logprob gathering
- Prepares 2D labels `[seq_len, num_candidates]` from `position_rewards`
- Passes multi-candidate logprobs with gradients to the loss function
- Supports both single-candidate and multi-candidate modes

See `engine/README.md` for detailed documentation.

## Key Concepts

### Why Fresh Logprobs Are Needed

1. **`pr.logprobs` from rollout has NO gradients** - it's stored data from the previous
   iteration
1. **For training, we need fresh logprobs with gradients** - computed from current model
   parameters
1. **Logits have gradient information** - computing logprobs from logits maintains the
   computation graph
1. **Multi-candidate gathering** - we gather logprobs for ALL candidates, not just the
   chosen token

### Data Flow

```
Rollout Phase (no gradients needed):
    Agent generates candidates → PositionRewardInfo stores:
        - candidate_token_ids: for gathering during training
        - logprobs: old logp from rollout (for importance weighting only)
        - rewards: reward for each candidate

Training Phase (gradients required):
    MultiCandidateFSDPEngine:
        1. Prepare 2D labels from position_rewards
        2. Compute logprobs = gather_logprobs_entropy_multi_candidates(logits, labels)
           → logprobs has shape [seq_len, num_candidates] with gradients!
        3. Pass to loss_fn(logprobs, entropy, input_data, ...)

    grpo_distill_loss_fn:
        1. Compute standard GRPO loss using chosen token logprobs
        2. If position_rewards present and logprobs is 2D:
           → _compute_position_level_grpo_loss(logprobs, ...)
        3. In position-level loss:
           - new_logprobs = logprobs[position, :num_candidates]  # Uses pre-computed!
           - Vectorized over all positions:
             - Build padded tensors [num_positions, max_candidates]
             - importance_weights = exp(new_logprobs.detach() - old_logprobs)
             - advantages = (rewards - mean(rewards)) / (std(rewards) + eps)
             - loss = -mean(importance_weights * advantages * new_logprobs)
```

## Call Chain

```
MultiCandidateFSDPEngine._compute_logprobs_and_loss()
    ├── _prepare_multi_candidate_labels()
    │       └── Creates 2D labels [seq_len, max_candidates] from position_rewards
    │
    ├── _compute_logprobs_entropy()
    │       └── gather_logprobs_entropy_multi_candidates()
    │               └── _chunked_gather_logprobs_entropy_multi_candidates()
    │                       └── _gather_logprobs_entropy_multi_candidates()
    │                               ├── F.log_softmax(logits / temperature, dim=-1)
    │                               ├── entropy = -torch.sum(probs * log_probs, dim=-1)
    │                               └── log_probs.gather(dim=-1, index=labels)
    │
    └── loss_fn(logprobs, entropy, input_data, ...)
            └── grpo_distill_loss_fn()
                    ├── _compute_grpo_loss()  # For chosen tokens
                    └── _compute_position_level_grpo_loss()  # For all candidates
                            └── Uses pre-computed logprobs directly
```

## Tensor Shapes

| Tensor                        | Shape                                                             | Description                  |
| ----------------------------- | ----------------------------------------------------------------- | ---------------------------- |
| `logits`                      | `[seq_len, vocab_size]` or `[batch, seq_len, vocab_size]`         | Model outputs                |
| `labels` (single)             | `[seq_len]` or `[batch, seq_len]`                                 | Chosen token IDs             |
| `labels` (multi)              | `[seq_len, num_candidates]` or `[batch, seq_len, num_candidates]` | All candidate IDs            |
| `logprobs` (single)           | `[seq_len]` or `[batch, seq_len]`                                 | Logprobs at chosen tokens    |
| `logprobs` (multi)            | `[seq_len, num_candidates]` or `[batch, seq_len, num_candidates]` | Logprobs at all candidates   |
| `entropy`                     | `[seq_len]` or `[batch, seq_len]`                                 | Distribution entropy         |
| `old_logprobs` (from rollout) | `list[float]` per position                                        | Stored in PositionRewardInfo |
| `rewards`                     | `list[float]` per position                                        | Stored in PositionRewardInfo |

## Important Notes

1. **No gradient recomputation in loss**: The `_compute_position_level_grpo_loss`
   function receives `logprobs` as input and uses it directly. It does NOT call
   `gather_logprobs_entropy_multi_candidates` again because the logprobs already have
   gradients from the engine.

1. **Importance weight clipping**: Importance weights (`exp(new_logp - old_logp)`) are
   clipped to max 10.0 for numerical stability.

1. **Normalization**: Position-level loss is normalized by the sum of importance weights
   and then by `loss_mask.sum()`.

1. **Loss weights**: The combined loss uses `rl_loss_weight` (default 1.0) for standard
   GRPO and `distill_loss_weight` (default 0.005) for position-level GRPO.

## Related Documentation

- `engine/README.md` - MultiCandidateFSDPEngine documentation
- `training/README.md` - Loss functions and logprob utilities
- `docs/token_reward.md` - Token-level reward system overview
