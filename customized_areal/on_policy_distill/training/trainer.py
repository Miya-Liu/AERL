"""On-Policy Distillation Trainer for AReaL.

This module provides a trainer for on-policy distillation using
OpenAI proxy components.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from areal import PPOTrainer
from areal.api.cli_args import PPOActorConfig
from areal.api import AllocationMode
from areal.utils import logging
from areal.utils.environ import is_single_controller

from ..proxy.workflow import OpenAIProxyWorkflow

logger = logging.getLogger("OnPolicyDistillTrainer")


class OnPolicyDistillationTrainer(PPOTrainer):
    """Trainer for on-policy distillation using OpenAI proxy workflow.

    This trainer extends AReaL's PPOTrainer with components to enable
    on-policy distillation training using grpo_distill_loss_fn.

    Args:
        config: OnPolicyDistillConfig instance.
        train_dataset: Optional training dataset.
        valid_dataset: Optional validation dataset.
        workflow: Optional pre-configured workflow instance.
        agent: Optional agent instance.
    """

    def __init__(
        self,
        config: Any,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
        workflow: Optional[OpenAIProxyWorkflow] = None,
        agent: Optional[Any] = None,
    ):
        from ..training.actor import (
            patch_ppo_actor_class_to_use_distill_loss,
        )

        # Patch PPOActor class to use grpo_distill_loss_fn
        patch_ppo_actor_class_to_use_distill_loss()

        self.workflow = workflow
        self.agent = agent

        # Initialize components if workflow not provided
        if self.workflow is None:
            self._init_components()

        super().__init__(config, train_dataset, valid_dataset)

    def _init_components(self) -> None:
        """Initialize components for training."""
        logger.info("Initializing components for on-policy distillation")

        # Create agent if not provided
        if self.agent is None:
            from ..proxy.workflow import TokenRewardExampleAgent

            self.agent = TokenRewardExampleAgent()

        # Get workflow configuration from config
        proxy_base_url = getattr(self.config, "proxy_base_url", "http://localhost:8000")
        proxy_api_key = getattr(self.config, "proxy_api_key", "dummy-admin-key")
        turn_discount = getattr(self.config, "turn_discount", 1.0)
        export_style = getattr(self.config, "export_style", "individual")

        # Create workflow
        self.workflow = OpenAIProxyWorkflow(
            agent=self.agent,
            proxy_addr=proxy_base_url,
            admin_api_key=proxy_api_key,
            discount=turn_discount,
            export_style=export_style,
        )

        logger.info("Components initialized successfully")

    def _create_actor(self, actor_config: PPOActorConfig):
        """Create actor using MultiCandidateFSDPPPOActor for multi-candidate support.

        This overrides the base PPOTrainer._create_actor to use
        MultiCandidateFSDPPPOActor instead of standard FSDPPPOActor,
        enabling multi-candidate logprob gathering for position-level rewards.
        """
        from ..engine import MultiCandidateFSDPPPOActor

        if self.allocation_mode.train_backend != "fsdp":
            raise ValueError(
                f"OnPolicyDistillationTrainer only supports FSDP backend, "
                f"got: {self.allocation_mode.train_backend}"
            )

        actor_cls = MultiCandidateFSDPPPOActor

        if is_single_controller():
            from areal.infra.scheduler import Scheduler
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)

        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        logger.info("Created MultiCandidateFSDPPPOActor for on-policy distillation")
        return actor
