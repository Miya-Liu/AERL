"""Configuration for TPFC Agent training experiments."""

from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig


@dataclass
class TPFCConfig(PPOConfig):
    """Configuration for TPFC Agent training experiments.

    Extends PPOConfig with agent-specific settings for the TPFC agent
    workflow using the OpenAI-compatible proxy approach.
    """

    workflow: str = field(
        default="customized_areal.tpfc_agent.TPFCAgent",
        metadata={"help": "Path to the TPFC workflow class for training."},
    )
    eval_workflow: str = field(
        default="${workflow}",
        metadata={"help": "Path to the TPFC workflow class for evaluation."},
    )
