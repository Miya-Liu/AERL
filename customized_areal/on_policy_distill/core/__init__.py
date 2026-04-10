"""Core components for on-policy distillation.

This module contains:
- types.py: Data structures like InteractionWithTokenLevelReward
- cache.py: InteractionCache and PositionRewardInfo
- client.py: OpenAIProxyClient for proxy interactions
- config.py: OnPolicyDistillConfig
- agent.py: OnPolicyDistillAgent and reward functions
- actor.py: PPOActor patching for distillation loss
"""

from .types import InteractionWithTokenLevelReward, TokenRewardInteractions
from .cache import PositionRewardInfo, InteractionCache
from .client import OpenAIProxyClient

__all__ = [
    "InteractionWithTokenLevelReward",
    "TokenRewardInteractions",
    "PositionRewardInfo",
    "InteractionCache",
    "OpenAIProxyClient",
]
