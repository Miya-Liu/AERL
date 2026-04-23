"""On-Policy Distillation Trainer for AReaL.

This module provides a trainer for on-policy distillation using
OpenAI proxy components.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from areal import PPOTrainer  # noqa: E402
from areal.api.cli_args import PPOActorConfig  # noqa: E402
from areal.infra.controller.rollout_controller import RolloutController  # noqa: E402
from areal.utils import logging  # noqa: E402
from areal.utils.environ import is_single_controller  # noqa: E402

logger = logging.getLogger("OnPolicyDistillTrainer")


class TokenRewardRolloutController(RolloutController):
    """Custom RolloutController that starts the token-reward proxy server.

    Overrides the base RolloutController to:
    1. Use the custom token-reward proxy server instead of the base one.
    2. Inject per-worker ``proxy_addr`` into ``workflow_kwargs`` so that
       OpenAIProxyWorkflow instances on each worker connect to the correct
       local proxy server.
    """

    async def _async_start_proxy(self) -> None:
        """Start proxy workers with the custom token-reward proxy server."""
        command = "customized_areal.on_policy_distill.proxy.proxy_rollout_server"
        worker_ids = self.scheduler.fork_workers(
            role=self._proxy_role,
            target_role=self._worker_role,
            command=command,
        )
        logger.info(f"Token-reward proxy workers forked: {worker_ids}")

        self.proxy_workers = self.scheduler.get_workers(role=self._proxy_role)
        logger.info(f"Proxy workers: {[w.id for w in self.proxy_workers]}")

        engine_class = f"{self.inf_engine.__module__}.{self.inf_engine.__name__}"

        create_tasks = []
        for rank, worker in enumerate(self.proxy_workers):
            create_tasks.append(
                self.scheduler.create_engine(
                    worker_id=worker.id,
                    engine=engine_class,
                    engine_name=self._proxy_engine_name(rank),
                    config=self.config,
                )
            )
        await asyncio.gather(*create_tasks)
        logger.info("Token-reward proxy engines created")

        from areal.utils.network import format_hostport

        init_tasks = []
        for rank, (worker, server_info) in enumerate(
            zip(self.proxy_workers, self.server_infos, strict=True)
        ):
            init_tasks.append(
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._proxy_engine_name(rank),
                    addr=f"{server_info.host}:{server_info.port}",
                )
            )
            self.proxy_addrs.append(
                f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
            )
        await asyncio.gather(*init_tasks)

        logger.info(
            f"Token-reward proxy servers initialized. Addresses: {self.proxy_addrs}"
        )

    def _create_submit_callback(self, pending_task):
        """Override to inject per-worker ``proxy_addr`` into ``workflow_kwargs``.

        When proxy workers are started, each rollout worker has a colocated
        proxy server. The workflow on that worker must connect to the local
        proxy address. This override copies the base callback logic and adds
        injection of ``proxy_addr`` into ``workflow_kwargs`` before the task
        reaches the engine, so the OpenAIProxyWorkflow constructor receives
        the correct address.
        """
        from areal.infra.controller.rollout_controller import (
            _RemoteRolloutResult,
        )

        async def _submit_then_wait() -> _RemoteRolloutResult | None:
            worker, rank = self._choose_worker()
            engine_name = self._engine_name(rank)
            task_id = pending_task.task_id
            manager = self.staleness_manager

            try:
                future = asyncio.get_event_loop().create_future()
                with self._futures_lock:
                    self._pending_futures[task_id] = future

                proxy_addr = pending_task.proxy_addr
                if self._proxy_started and proxy_addr is None:
                    proxy_addr = self.get_proxy_addr(rank)

                # Inject proxy_addr into workflow_kwargs so the workflow
                # constructor (called on the engine worker) receives the
                # correct local proxy address.
                effective_kwargs = pending_task.workflow_kwargs
                if (
                    proxy_addr is not None
                    and effective_kwargs is not None
                    and "proxy_addr" not in effective_kwargs
                ):
                    effective_kwargs = {**effective_kwargs, "proxy_addr": proxy_addr}

                engine_task_id = await self.scheduler.async_call_engine(
                    worker.id,
                    "submit",
                    engine_name=engine_name,
                    data=pending_task.data,
                    workflow=pending_task.workflow,
                    workflow_kwargs=effective_kwargs,
                    should_accept_fn=pending_task.should_accept_fn,
                    http_timeout=self.config.request_timeout,
                    is_eval=pending_task.is_eval,
                    group_size=pending_task.group_size,
                    task_id=task_id,
                    callback_addr=f"http://{self.callback_addr}/callback/rollout_complete",
                    proxy_addr=proxy_addr,
                )

                assert task_id == engine_task_id, (task_id, engine_task_id)

                await asyncio.wait_for(future, timeout=self.config.request_timeout)

                result = await self.scheduler.async_call_engine(
                    worker.id,
                    "wait_for_task",
                    engine_name=engine_name,
                    task_id=engine_task_id,
                    timeout=0.1,
                    raise_timeout=False,
                    http_timeout=self.config.request_timeout,
                )

                traj = result
                if traj is not None:
                    manager.on_rollout_accepted()
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Finish and accept rollout. {self._rollout_stats()}"
                        )
                    return _RemoteRolloutResult(task_id=task_id, trajectory=traj)

                manager.on_rollout_rejected()
                if self.config.enable_rollout_tracing:
                    logger.info(f"Finish but reject rollout. {self._rollout_stats()}")
                return None

            except TimeoutError:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error(f"Rollout timed out after {self.config.request_timeout}s")
                return None
            except Exception as exc:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error("Workflow execution failed: %s", exc, exc_info=True)
                return None

        return _submit_then_wait


class OnPolicyDistillationTrainer(PPOTrainer):
    """Trainer for on-policy distillation using OpenAI proxy workflow.

    This trainer extends AReaL's PPOTrainer with components to enable
    on-policy distillation training using grpo_distill_loss_fn.

    Args:
        config: OnPolicyDistillConfig instance.
        train_dataset: Optional training dataset.
        valid_dataset: Optional validation dataset.
        workflow: Optional pre-configured workflow instance.
        agent: Optional agent instance or string import path.
    """

    def __init__(
        self,
        config: Any,
        train_dataset: Any | None = None,
        valid_dataset: Any | None = None,
        workflow: Any | None = None,
        agent: Any | None = None,
    ):
        from ..training.actor import (
            patch_ppo_actor_class_to_use_distill_loss,
        )

        # Patch PPOActor class to use grpo_distill_loss_fn
        patch_ppo_actor_class_to_use_distill_loss()

        self._custom_workflow = workflow
        self._custom_agent = agent

        super().__init__(config, train_dataset, valid_dataset)

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
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)

        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        logger.info("Created MultiCandidateFSDPPPOActor for on-policy distillation")
        return actor

    def _init_rollout(
        self,
        rollout_config: Any,
        is_eval: bool = False,
        lora_path: str | None = None,
    ):
        """Override to use TokenRewardRolloutController in single-controller mode.

        In single-controller mode, replaces the default RolloutController with
        TokenRewardRolloutController that starts the custom token-reward proxy server.
        """
        from copy import deepcopy

        from areal.api.cli_args import (
            SchedulingStrategy,
            SchedulingStrategyType,
        )
        from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
        from areal.engine.sglang_config import SGLangConfig
        from areal.engine.vllm_config import vLLMConfig

        if lora_path is not None and not is_single_controller():
            raise ValueError(
                "LoRA is only supported in single-controller mode. "
                "Use `python3 train.py scheduler.type=local` instead of "
                "`python3 -m areal.infra.launcher.local`."
            )

        config = deepcopy(rollout_config)
        if is_eval:
            config.max_head_offpolicyness = int(1e12)
            config.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            )
            for spec in config.scheduling_spec:
                spec.gpu = 0

        rollout_backend = self.rollout_alloc.backend
        if rollout_backend == "sglang":
            if self.config.rollout.return_routed_experts:
                self.config.sglang.enable_return_routed_experts = True
            if lora_path is not None and self.config.actor.use_lora:
                self.config.sglang.lora_paths = [
                    f"{self.config.gconfig.lora_name}-v0={lora_path}"
                ]
            engine_cls = RemoteSGLangEngine
            server_args = SGLangConfig.build_args(
                sglang_config=self.config.sglang,
                tp_size=self.rollout_alloc.parallel.tp_size,
                base_gpu_id=0,
            )
        elif rollout_backend == "vllm":
            if self.config.rollout.return_routed_experts:
                raise ValueError(
                    "return_routed_experts is not supported with vLLM backend."
                )
            if lora_path is not None and self.config.actor.use_lora:
                self.config.vllm.lora_modules = [
                    f"{self.config.gconfig.lora_name}-v0={lora_path}"
                ]
            engine_cls = RemotevLLMEngine
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
            )
        else:
            raise ValueError(
                f"Invalid backend: {rollout_backend}, expected sglang or vllm"
            )

        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.actor_alloc.parallel.dp_size
            )
            return engine

        # Single-controller mode — use TokenRewardRolloutController
        controller = TokenRewardRolloutController(engine_cls, config, self.scheduler)
        init_kwargs = dict(
            role="rollout",
            server_args=server_args,
        )
        if is_eval:
            assert len(self.rollout.server_infos) > 0
            init_kwargs["server_infos"] = self.rollout.server_infos
            init_kwargs["role"] = "eval-rollout"
        controller.initialize(**init_kwargs)
        return controller

    def _ensure_proxy_started(self) -> None:
        """Start proxy workers using TokenRewardRolloutController.

        Reuses the base PPOTrainer's proxy pattern but with the custom
        token-reward proxy server.
        """
        if self._proxy_started:
            return

        if not is_single_controller():
            raise NotImplementedError("Proxy workers not supported in SPMD mode")

        if self.config.scheduler.type == "ray":
            raise NotImplementedError("Proxy workers not supported with RayScheduler")

        from areal.infra.controller.rollout_controller import RolloutController

        assert isinstance(self.rollout, RolloutController)

        logger.info("Initializing token-reward proxy workers")
        self.rollout.start_proxy()
        if self.eval_rollout is not None:
            self.eval_rollout.start_proxy()

        # Start proxy gateway for potential online access
        openai_cfg = self.config.rollout.openai
        if openai_cfg is not None and openai_cfg.mode == "online":
            self.rollout.start_proxy_gateway()
            logger.info(
                "Proxy gateway available at %s",
                self.rollout.proxy_gateway_addr,
            )

        self._proxy_started = True

    def train(
        self,
        workflow: Any | None = None,
        eval_workflow: Any | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        eval_workflow_kwargs: dict[str, Any] | None = None,
        dynamic_filter_fn: Any = None,
        total_epochs: int | None = None,
    ):
        """Train with the custom OpenAIProxyWorkflow connected to proxy workers.

        If no workflow is provided, automatically uses the custom
        OpenAIProxyWorkflow with the proxy address from the rollout controller.
        """
        if workflow is not None or self._custom_workflow is not None:
            # Use the explicitly provided workflow
            actual_workflow = workflow or self._custom_workflow
            return super().train(
                workflow=actual_workflow,
                eval_workflow=eval_workflow,
                workflow_kwargs=workflow_kwargs,
                eval_workflow_kwargs=eval_workflow_kwargs,
                dynamic_filter_fn=dynamic_filter_fn,
                total_epochs=total_epochs,
            )

        # Auto-configure workflow using proxy infrastructure
        self._ensure_proxy_started()

        # Build workflow_kwargs for the custom OpenAIProxyWorkflow
        actual_workflow_kwargs = self._build_workflow_kwargs(workflow_kwargs)

        # Use the workflow class path from config
        workflow_str = self.config.workflow

        return super().train(
            workflow=workflow_str,
            eval_workflow=eval_workflow or self.config.eval_workflow,
            workflow_kwargs=actual_workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn=dynamic_filter_fn,
            total_epochs=total_epochs,
        )

    def _build_workflow_kwargs(
        self, base_kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Build workflow_kwargs for OpenAIProxyWorkflow.

        Merges base_kwargs with config-derived values. The ``proxy_addr``
        will be injected by TokenRewardRolloutController at task submission
        time (per-worker), so we don't set it here.
        """
        from areal.api.cli_args import OpenAIProxyConfig

        openai_cfg = self.config.rollout.openai or OpenAIProxyConfig()
        admin_api_key = openai_cfg.admin_api_key

        # Resolve agent: prefer custom agent, then config, then default
        agent = self._custom_agent
        if agent is None:
            # Use a default agent class path
            agent = getattr(self.config, "agent", None)
            if isinstance(agent, str):
                pass  # agent is already a string path
            elif hasattr(agent, "__module__") and hasattr(agent, "__qualname__"):
                # Agent is a class or instance — convert to string path
                agent = f"{agent.__module__}.{agent.__qualname__}"

        kwargs: dict[str, Any] = {
            "agent": agent,
            "proxy_addr": "",  # Placeholder — injected per-worker by controller
            "admin_api_key": admin_api_key,
            "discount": getattr(self.config, "turn_discount", 1.0),
            "export_style": getattr(self.config, "export_style", "individual"),
        }

        # Merge with any user-provided kwargs (user values take precedence)
        if base_kwargs:
            kwargs.update(base_kwargs)

        return kwargs
