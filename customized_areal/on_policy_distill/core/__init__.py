"""Core components for on-policy distillation.

This module contains:
- config.py: OnPolicyDistillConfig
- agent.py: OnPolicyDistillAgent and reward functions
- teacher_client.py: TeacherConfig and TeacherClient
- reward_compute.py: _compute_token_rewards for teacher/student logprob comparison
"""
