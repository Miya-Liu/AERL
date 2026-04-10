"""Training components for on-policy distillation.

This module contains:
- trainer.py: OnPolicyDistillationTrainer
- loss.py: grpo_distill_loss_fn for combined GRPO + position-level loss
- logprobs.py: Log probability and entropy computation utilities
"""

from .trainer import OnPolicyDistillationTrainer
from .loss import grpo_distill_loss_fn
from .logprobs import gather_logprobs_entropy_multi_candidates

__all__ = [
    "OnPolicyDistillationTrainer",
    "grpo_distill_loss_fn",
    "gather_logprobs_entropy_multi_candidates",
]
