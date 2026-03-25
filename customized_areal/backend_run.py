import asyncio
import json
import sys
import traceback
import httpx
import os

from litellm import uuid
from omegaconf import OmegaConf

sys.stdout.reconfigure(encoding="utf-8")  # 设置标准输出为 UTF-8

async def get_all_accounts():
    """Load all rows from the `accounts` table."""
    from core.services.supabase import DBConnection

    db = DBConnection()
    client = await db.client

    result = (
        await client.schema("basejump")
        .table("accounts")
        .select("*")
        .execute()
    )
    return result.data or []


async def run_backend(
    task_description,
    task_file_path,
    log_path="./log.json",
    task_id="",
    gt="",
    tags=None,
    user_id: str | None = None,
    model_name: str | None = None,
    server_manager=None,
    tokenizer=None,
    # agent_id=None,
    agent_id='89395eb4-dd1a-4a13-932d-4f7d3a17bca6',
    # run_from_step_new=False
    # OpenAI proxy parameters (passed by OpenAIProxyWorkflow)
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
    rebuild_llm_client: bool = True,
):

    # await get_all_accounts()
    
    from core.services.supabase import DBConnection
    from core.tasks.service import create_task

    # Initialize database connection
    db = DBConnection()
    client = await db.client

    # Step 1: Create a task to get task_id
    user_id = user_id or '13183c90-ac94-403e-893e-c53552ad429d'

    if agent_id is None:
        # Step 2: Get agent list and find the agent
        from core.agents.service import AgentService, AgentFilters, PaginationParams
        from core.agents.schemas import AgentCreateRequest

        agent_service = AgentService(client)

        spcified_agent = None

        if not spcified_agent:
            from core.agents.builtin.tpfc import TPFC_CONFIG
            # Create the Janus agent if it doesn't exist yet
            created_agent = await agent_service.create_agent(
                user_id,
                AgentCreateRequest(
                    name=TPFC_CONFIG["name"],
                    config=TPFC_CONFIG["config"],
                    is_default=TPFC_CONFIG.get("is_default", False),
                ),
            )
            agent_id = created_agent.agent_id
            # from core.agents.agent_loader import get_agent_loader

            # loader = await get_agent_loader()
            # agent_data = await loader.load_agent(agent_id, user_id, load_config=True)
            # # agent_data = None

            logger.info("Created agent", agent_id=agent_id)
        else:
            agent_id = spcified_agent.get("agent_id")
            logger.info("Found  agent", agent_id=agent_id)


    task_id = await create_task(
        client=client,
        account_id=user_id,
        agent_id=agent_id, 
        name=task_description[:100] if task_description else None,
    )
    logger.info("Task created", task_id=task_id)

    # Step 3: Start agent run via HTTP API endpoint
    # Use the unified_agent_start API for consistency with the frontend flow
    import jwt
    import time
    
    # Create a JWT token for internal API authentication
    # The decode_user_id function doesn't verify signatures, so a simple token works
    internal_token = jwt.encode(
        {"sub": user_id, "exp": time.time() + 3600},
        "internal",
        algorithm="HS256"
    )
    
    # Determine API base URL: use provided base_url or fall back to env var
    api_base_url = base_url or os.environ.get("API_BASE_URL", "http://localhost:8000")
    
    # Determine auth token: use provided api_key or fall back to internal token
    auth_token = api_key or internal_token
    
    # Use provided http_client or create a new one
    if http_client is not None:
        # Prepare multipart form data
        form_data = {
            "task_id": task_id,
            "prompt": task_description,
            "agent_id": agent_id,
            "model_name": model_name or "openrouter/qwen/qwen3-235b-a22b",
            "backend_mode": "false",
            "is_sub_agent": "false",
            "rebuild_llm_client": "true" if rebuild_llm_client else "false",
        }
        
        # Add tags if provided
        if tags:
            for tag in tags:
                form_data.setdefault("tags", [])
            # Note: For multipart, we need to handle tags differently
        
        response = await http_client.post(
            f"{api_base_url}/api/agent/start",
            headers={
                "Authorization": f"Bearer {auth_token}",
            },
            data=form_data,
        )
        
        if response.status_code != 200:
            logger.error(
                "Failed to start agent run via API",
                status_code=response.status_code,
                response=response.text,
            )
            raise RuntimeError(f"Failed to start agent run: {response.status_code} - {response.text}")
        
        result = response.json()
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Prepare multipart form data
            form_data = {
                "task_id": task_id,
                "prompt": task_description,
                "agent_id": agent_id,
                "model_name": model_name or "openrouter/qwen/qwen3-235b-a22b",
                "backend_mode": "false",
                "is_sub_agent": "false",
                "rebuild_llm_client": "true" if rebuild_llm_client else "false",
                "base_url": base_url or "",
                "api_key": api_key or "",
            }
            
            # Add tags if provided
            if tags:
                for tag in tags:
                    form_data.setdefault("tags", [])
                # Note: For multipart, we need to handle tags differently
            
            response = await client.post(
                f"{api_base_url}/api/agent/start",
                headers={
                    "Authorization": f"Bearer {auth_token}",
                },
                data=form_data,
            )
            
            if response.status_code != 200:
                logger.error(
                    "Failed to start agent run via API",
                    status_code=response.status_code,
                    response=response.text,
                )
                raise RuntimeError(f"Failed to start agent run: {response.status_code} - {response.text}")
            
            result = response.json()
    
    logger.info("Agent run started via API", result=result)

    # Wait for agent run to complete
    agent_run_id = result["agent_run_id"]
    timeout = 300  # 5 minutes
    import time
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        agent_run = (
            await client.table("agent_runs")
            .select("status, error, completed_at")
            .eq("id", agent_run_id)
            .single()
            .execute()
        )
        
        status = agent_run.data["status"]
        logger.info("Agent run status", status=status, agent_run_id=agent_run_id)
        
        if status in ["completed", "failed", "stopped"]:
            if status == "failed":
                logger.error("Agent run failed", error=agent_run.data.get("error"), agent_run_id=agent_run_id)
            else:
                logger.info("Agent run completed", status=status, agent_run_id=agent_run_id)
            break
        
        await asyncio.sleep(2)
    else:
        logger.error("Timeout waiting for agent run to complete", agent_run_id=agent_run_id)
        raise TimeoutError(f"Agent run {agent_run_id} did not complete within {timeout} seconds")

    from core.tasks.messages.service import get_llm_messages
    messages = await get_llm_messages(task_id, return_raw=True)
    
    return messages


if __name__ == "__main__":
    # task_description = "\u00ac(A \u2227 B) \u2194 (\u00acA \u2228 \u00acB)\n\u00ac(A \u2228 B) \u2194 (\u00acA \u2227 \u00acB)\n(A \u2192 B) \u2194 (\u00acB \u2192 \u00acA)\n(A \u2192 B) \u2194 (\u00acA \u2228 B)\n(\u00acA \u2192 B) \u2194 (A \u2228 \u00acB)\n\u00ac(A \u2192 B) \u2194 (A \u2227 \u00acB)\n\nWhich of the above is not logically equivalent to the rest? Provide the full statement that doesn't fit."
    task_description = "今天北京天气怎么样"
    cfg = OmegaConf.create(
        {
            "benchmark": {
                "name": "gaia-validation",
                "data": {
                    "data_dir": "core/miroflow/gaia-benchmark/gaia/2023/validation",
                    "metadata_file": "metadata.jsonl",
                    "whitelist": [],
                },
                "execution": {"max_concurrent": 1, "max_tasks": 166, "pass_at_k": 1},
            },
            "llm": {
                "provider": "openai",
                # "model_name": 'openrouter/anthropic/claude-haiku-4.5',
                "model_name": "openrouter/qwen/qwen3-235b-a22b",
                # "model_name": "openai/Qwen3_VL_8B_Training",
                # "model_name": "openai/Qwen35_397B_a17b_fp8",
                # "model_name": "deepseek-v3.2",
                # "model_name": "lenovo/basic",
                "enable_thinking": False,
                "reasoning_effort": "low",
                "stream": False,
            },
            "env": {"openai_api_key": ""},
            "level": 1,
            "user_id": '13183c90-ac94-403e-893e-c53552ad429d',
            "tags": ["debug"],
            "backend_mode": True,
        }
    )

    cfg.output_dir = (
        f"logs/{cfg.benchmark.name}/{cfg.llm.provider}_{cfg.llm.model_name}_/level_{cfg.level}"
    )

    response = asyncio.run(run_backend(task_description, [], tags=["debug"], cfg=cfg))
