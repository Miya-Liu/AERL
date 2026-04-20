"""Training components for on-policy distillation.

This module contains:
- trainer.py: OnPolicyDistillationTrainer
- loss.py: grpo_distill_loss_fn for combined GRPO + position-level loss
- logprobs.py: Log probability and entropy computation utilities
"""

from .logprobs import gather_logprobs_entropy_multi_candidates
from .loss import grpo_distill_loss_fn
from .trainer import OnPolicyDistillationTrainer

__all__ = [
    "OnPolicyDistillationTrainer",
    "grpo_distill_loss_fn",
    "gather_logprobs_entropy_multi_candidates",
]
