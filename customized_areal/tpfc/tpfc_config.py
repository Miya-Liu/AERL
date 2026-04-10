"""Configuration for TPFC Agent training experiments."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports when this file is imported directly
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
