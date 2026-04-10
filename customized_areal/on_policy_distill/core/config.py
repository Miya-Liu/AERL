"""Configuration for On-Policy Distillation training experiments."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports when this file is imported directly
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from areal.api.cli_args import PPOConfig


@dataclass
class OnPolicyDistillConfig(PPOConfig):
    """Configuration for On-Policy Distillation training experiments.

    Extends PPOConfig with distillation-specific settings for training
    using OpenAI proxy workflow components.

    Attributes:
        workflow: Path to the workflow class (OpenAI proxy workflow).
        eval_workflow: Path to the eval workflow class.
        cache_size: Maximum size of the interaction cache.
        proxy_base_url: Base URL for the OpenAI proxy.
        proxy_api_key: API key for the OpenAI proxy.
        proxy_model: Model name to use for generation.
        proxy_temperature: Sampling temperature.
        proxy_max_tokens: Maximum tokens per completion.
        turn_discount: Discount factor for multi-turn rewards.
        export_style: Style for exporting interactions ('concat' or 'individual').
    """

    workflow: str = field(
        default="customized_areal.on_policy_distill.OpenAIProxyWorkflow",
        metadata={"help": "Path to the OpenAI proxy workflow class."},
    )
    eval_workflow: str = field(
        default="${workflow}",
        metadata={"help": "Path to the eval workflow class."},
    )

    # Cache configuration
    cache_size: int = field(
        default=1000,
        metadata={"help": "Maximum number of interactions to cache."},
    )

    # OpenAI Proxy Client configuration
    proxy_base_url: str = field(
        default="http://localhost:8000",
        metadata={"help": "Base URL for the OpenAI proxy server."},
    )
    proxy_api_key: str = field(
        default="",
        metadata={"help": "API key for the OpenAI proxy server."},
    )
    proxy_model: str = field(
        default="qwen/qwen3-1.7b",
        metadata={"help": "Model name to use for generation."},
    )
    proxy_temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature for generation."},
    )
    proxy_max_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum tokens per completion."},
    )
    proxy_top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling parameter."},
    )

    # Workflow configuration
    turn_discount: float = field(
        default=1.0,
        metadata={"help": "Discount factor for multi-turn reward propagation."},
    )
    export_style: str = field(
        default="individual",
        metadata={"help": "Export style: 'concat' or 'individual'."},
    )

    # Distillation-specific settings
    use_reward_scaling: bool = field(
        default=True,
        metadata={"help": "Whether to apply reward scaling."},
    )
    reward_scaling_factor: float = field(
        default=10.0,
        metadata={"help": "Factor for reward scaling."},
    )
    reward_bias: float = field(
        default=-0.5,
        metadata={"help": "Bias to add to rewards."},
    )
