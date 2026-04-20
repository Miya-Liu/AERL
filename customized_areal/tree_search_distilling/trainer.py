"""Tree Search Distilling Trainer for AReaL.

Combines MCTS tree backup advantages with on-policy distillation loss
and rollout caching in a single training step.

Inherits from CacheAwarePPOTrainer and layers on distillation components
from OnPolicyDistillationTrainer:
- Patches PPOActor._ppo_update with grpo_distill_loss_fn
- Uses MultiCandidateFSDPPPOActor for multi-candidate logprob gathering
- Creates OpenAIProxyWorkflow with TreeDistillAgent for rollout generation
"""

from __future__ import annotations

from typing import Any

from customized_areal.on_policy_distill.engine import MultiCandidateFSDPPPOActor
from customized_areal.on_policy_distill.proxy.workflow import OpenAIProxyWorkflow
from customized_areal.tree_search.config import (
    RolloutCacheConfig,
    TreeBackupConfig,
)
from customized_areal.tree_search.trainer import CacheAwarePPOTrainer

from areal.api.cli_args import PPOActorConfig
from areal.utils import logging
from areal.utils.environ import is_single_controller

logger = logging.getLogger("TreeDistillPPOTrainer")


class TreeDistillPPOTrainer(CacheAwarePPOTrainer):
    """PPOTrainer combining MCTS tree backup, rollout caching, and on-policy distillation.

    This trainer inherits from CacheAwarePPOTrainer (which provides tree
    backup advantages and rollout caching) and adds on-policy distillation
    components:

    1. Patches PPOActor._ppo_update to use grpo_distill_loss_fn (from
       OnPolicyDistillationTrainer) so the training loss includes both
       GRPO with tree-backed advantages and position-level GRPO distillation.
    2. Overrides _create_actor to use MultiCandidateFSDPPPOActor, which
       gathers logprobs for multiple candidate tokens per position.
    3. Initializes OpenAIProxyWorkflow with TreeDistillAgent for rollout
       generation. TreeDistillAgent builds PositionRewardInfo from student
       top-k logprobs even when no teacher is configured.

    Args:
        config: OnPolicyDistillConfig instance.
        cache_config: RolloutCacheConfig for rollout caching.
        tree_backup_config: TreeBackupConfig for MCTS tree backup.
        train_dataset: Optional training dataset.
        valid_dataset: Optional validation dataset.
        workflow: Optional pre-configured workflow instance.
        agent: Optional pre-configured agent instance.
    """

    def __init__(
        self,
        config: Any,
        cache_config: RolloutCacheConfig | None = None,
        tree_backup_config: TreeBackupConfig | None = None,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
        workflow: OpenAIProxyWorkflow | None = None,
        agent: Any | None = None,
    ):
        from customized_areal.on_policy_distill.training.actor import (
            patch_ppo_actor_class_to_use_distill_loss,
        )

        # Patch PPOActor._ppo_update with grpo_distill_loss_fn BEFORE
        # super().__init__() so the patched loss is used when the actor
        # is created during PPOTrainer initialization.
        patch_ppo_actor_class_to_use_distill_loss()

        self.workflow = workflow
        self.agent = agent

        # Initialize components if workflow not provided
        if self.workflow is None:
            self._init_components()

        # Initialize base CacheAwarePPOTrainer, which:
        # 1. Calls PPOTrainer.__init__() (creates actor via _create_actor override)
        # 2. Sets up MCTS tree store, advantage computer, checkpoint manager
        # 3. Patches PPOActor.compute_advantages for tree backup
        super().__init__(
            config,
            cache_config=cache_config,
            tree_backup_config=tree_backup_config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )

    def _init_components(self) -> None:
        """Initialize workflow and agent for on-policy distillation."""
        logger.info("Initializing components for tree search distilling")

        # Create agent if not provided
        if self.agent is None:
            from customized_areal.tree_search_distilling.agent import TreeDistillAgent

            # Build TeacherConfig if teacher_model_name is set
            teacher_config = None
            teacher_model_name = getattr(self.config, "teacher_model_name", "")
            if teacher_model_name:
                from customized_areal.on_policy_distill.core.teacher_client import (
                    TeacherConfig,
                )

                teacher_config = TeacherConfig(
                    teacher_base_url=getattr(
                        self.config, "teacher_base_url", "http://localhost:8001"
                    ),
                    teacher_model_name=teacher_model_name,
                    teacher_top_k=getattr(self.config, "teacher_top_k", 10),
                    teacher_max_retries=getattr(self.config, "teacher_max_retries", 3),
                    teacher_timeout=getattr(self.config, "teacher_timeout", 60.0),
                    teacher_missing_logprob=getattr(
                        self.config, "teacher_missing_logprob", -23.0
                    ),
                )

            self.agent = TreeDistillAgent(
                teacher_config=teacher_config,
                student_top_k=getattr(self.config, "student_top_k", 10),
            )

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
        if self.allocation_mode.train_backend != "fsdp":
            raise ValueError(
                f"TreeDistillPPOTrainer only supports FSDP backend, "
                f"got: {self.allocation_mode.train_backend}"
            )

        actor_cls = MultiCandidateFSDPPPOActor

        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)

        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        logger.info("Created MultiCandidateFSDPPPOActor for tree search distilling")
        return actor
