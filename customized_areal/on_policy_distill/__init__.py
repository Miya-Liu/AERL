"""On-Policy Distillation module for AReaL.

This module provides on-policy distillation training using components
for token-level reward workflows.

Organized structure:
- core/: Core data structures and configuration
- engine/: Custom FSDP engine with multi-candidate support
- workflow/: Workflow implementations
- rewards/: Reward computation utilities
- training/: Training scripts and loss functions
- scripts/: Executable training scripts
- configs/: Configuration files
- examples/: Example usage code
- docs/: Documentation

Example usage:
    from customized_areal.on_policy_distill import (
        OnPolicyDistillConfig,
        OnPolicyDistillationTrainer,
        OnPolicyDistillAgent,
    )
"""

__all__ = [
    # Core components
    "OnPolicyDistillConfig",
    "OnPolicyDistillationTrainer",
    "OnPolicyDistillAgent",
    "accuracy_reward",
    "on_policy_distill_reward_fn",
    "InteractionCache",
    "OpenAIProxyClient",
    "OpenAIProxyWorkflow",
    "PositionRewardInfo",
    "InteractionWithTokenLevelReward",
    # Teacher distillation
    "TeacherClient",
    "TeacherConfig",
    # Engine
    "MultiCandidateFSDPEngine",
    "MultiCandidateFSDPPPOActor",
    # Reward utilities
    "compute_token_level_rewards",
    "apply_token_reward_mask",
    "gather_logprobs_entropy_multi_candidates",
    # Workflow
    "TokenRewardExampleAgent",
]


# Core imports
def __getattr__(name):
    # Config
    if name == "OnPolicyDistillConfig":
        from .core.config import OnPolicyDistillConfig

        return OnPolicyDistillConfig

    # Trainer
    if name == "OnPolicyDistillationTrainer":
        from .training.trainer import OnPolicyDistillationTrainer

        return OnPolicyDistillationTrainer

    # Agent and reward functions
    if name == "OnPolicyDistillAgent":
        from .core.agent import OnPolicyDistillAgent

        return OnPolicyDistillAgent
    if name == "accuracy_reward":
        from .core.agent import accuracy_reward

        return accuracy_reward
    if name == "on_policy_distill_reward_fn":
        from .core.agent import on_policy_distill_reward_fn

        return on_policy_distill_reward_fn

    # Core types and cache
    if name == "InteractionCache":
        from .proxy.cache import InteractionCache

        return InteractionCache
    if name == "PositionRewardInfo":
        from .proxy.cache import PositionRewardInfo

        return PositionRewardInfo
    if name == "InteractionWithTokenLevelReward":
        from .proxy.types import InteractionWithTokenLevelReward

        return InteractionWithTokenLevelReward
    if name == "OpenAIProxyClient":
        from .proxy.client import OpenAIProxyClient

        return OpenAIProxyClient

    # Teacher distillation
    if name == "TeacherClient":
        from .core.teacher_client import TeacherClient

        return TeacherClient
    if name == "TeacherConfig":
        from .core.teacher_client import TeacherConfig

        return TeacherConfig

    # Engine
    if name == "MultiCandidateFSDPEngine":
        from .engine import MultiCandidateFSDPEngine

        return MultiCandidateFSDPEngine
    if name == "MultiCandidateFSDPPPOActor":
        from .engine import MultiCandidateFSDPPPOActor

        return MultiCandidateFSDPPPOActor

    # Workflow
    if name == "OpenAIProxyWorkflow":
        from .proxy.workflow import OpenAIProxyWorkflow

        return OpenAIProxyWorkflow
    if name == "TokenRewardExampleAgent":
        from .proxy.workflow import TokenRewardExampleAgent

        return TokenRewardExampleAgent

    # Reward utilities (now in areal.utils.token_reward_utils)
    if name == "compute_token_level_rewards":
        from areal.utils.token_reward_utils import compute_token_level_rewards

        return compute_token_level_rewards
    if name == "apply_token_reward_mask":
        from areal.utils.token_reward_utils import apply_token_reward_mask

        return apply_token_reward_mask
    if name == "gather_logprobs_entropy_multi_candidates":
        from .training.logprobs import gather_logprobs_entropy_multi_candidates

        return gather_logprobs_entropy_multi_candidates

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
