"""Core components for on-policy distillation.

This module contains:
- config.py: OnPolicyDistillConfig
- agent.py: OnPolicyDistillAgent and reward functions
- teacher_client.py: TeacherConfig and TeacherClient
- reward_compute.py: _compute_token_rewards for teacher/student logprob comparison
"""

from .teacher_client import TeacherClient, TeacherConfig
from .reward_compute import _compute_token_rewards

__all__ = [
    "TeacherClient",
    "TeacherConfig",
    "_compute_token_rewards",
]
