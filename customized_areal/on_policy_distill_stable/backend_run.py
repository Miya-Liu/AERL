import asyncio
import sys
import httpx
import os
from pathlib import Path

# Add parent of 'customized_areal' to Python path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

from customized_areal.db_service import (
    DBConnection,
    create_task,
    get_llm_messages,
    AgentService,
    AgentFilters,
    PaginationParams,
    AgentCreateRequest,
    get_agent_loader,
)

sys.stdout.reconfigure(encoding="utf-8")  # 设置标准输出为 UTF-8

# Initialize logger
try:
    from areal.utils.logging import getLogger
    logger = getLogger("BackendRun")
except ImportError:
    import logging
    logger = logging.getLogger("BackendRun")

async def get_all_accounts():
    """Load all rows from the `accounts` table."""
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
    agent_id='8bba75cb-0d87-4efe-b566-87de77335b76',
    # agent_id='89395eb4-dd1a-4a13-932d-4f7d3a17bca6',
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
    rebuild_llm_client: bool = True,
):

    # await get_all_accounts()

    # Initialize database connection
    db = DBConnection()
    client = await db.client

    # Step 1: Create a task to get task_id
    user_id = user_id or '13183c90-ac94-403e-893e-c53552ad429d'

    if agent_id is None:
        # Step 2: Get agent list and find the agent
        agent_service = AgentService(client)

        # filters = AgentFilters(search="river", content_type="agents")
        # pagination_params = PaginationParams(page=1, page_size=100)
        # paginated_result = await agent_service.get_agents_paginated(
        #     user_id=user_id, pagination_params=pagination_params, filters=filters
        # )
        spcified_agent = None

        if not spcified_agent:
            from customized_areal.db_service.builtin import TPFC_CONFIG
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
            loader = await get_agent_loader()
            agent_data = await loader.load_agent(agent_id, user_id, load_config=True)
            # agent_data = None

            logger.info(f"Created agent: {agent_id}")
        else:
            agent_id = spcified_agent.get("agent_id")
            logger.info(f"Found agent: {agent_id}")


    task_id = await create_task(
        client=client,
        account_id=user_id,
        agent_id=agent_id, 
        name=task_description[:100] if task_description else None,
    )
    logger.info(f"Task created: {task_id}")

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
    
    # Determine API base URL for le-agent-dev backend API
    # This is different from base_url (which is the AReaL proxy URL for LLM calls)
    api_base_url = os.environ.get("LE_AGENT_API_URL", "http://localhost:8000")

    # Always use internal JWT token for backend API authentication
    # The api_key from AReaL is for proxy LLM calls, NOT for backend auth
    auth_token = internal_token
    
    # Prepare multipart form data (common for both branches)
    # proxy_base_url and proxy_api_key are passed to the agent for LLM routing through AReaL proxy
    form_data = {
        "task_id": task_id,
        "prompt": task_description,
        "agent_id": agent_id,
        "model_name": model_name or "openrouter/qwen/qwen3-235b-a22b",
        "is_sub_agent": "false",
        "using_agent": "tpfc",
        "stream": "false"
    }

    # Only include proxy settings when they have values (avoid empty strings in form data)
    if base_url != None:
        form_data["proxy_base_url"] = base_url  # AReaL proxy URL for LLM calls
    if api_key != None:
        form_data["proxy_api_key"] = api_key    # Session API key for proxy auth
    
    # Add tags if provided
    if tags:
        for tag in tags:
            form_data.setdefault("tags", [])
        # Note: For multipart, we need to handle tags differently
    
    async with httpx.AsyncClient(timeout=300.0) as http_client:
        response = await http_client.post(
            f"{api_base_url}/api/agent/start",
            headers={
                "Authorization": f"Bearer {auth_token}",
            },
            data=form_data,
        )

        if response.status_code != 200:
            logger.error(
                f"Failed to start agent run via API: status_code={response.status_code}, response={response.text}"
            )
            raise RuntimeError(f"Failed to start agent run: {response.status_code} - {response.text}")

        result = response.json()

    logger.info(f"Agent run started via API: {result}")

    # Wait for agent run to complete
    agent_run_id = result["agent_run_id"]
    timeout = 3000  # 5 minutes
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
        logger.info(f"Agent run status: {status}, agent_run_id: {agent_run_id}")
        
        if status in ["completed", "failed", "stopped"]:
            if status == "failed":
                logger.error(f"Agent run failed: error={agent_run.data.get('error')}, agent_run_id={agent_run_id}")
            else:
                logger.info(f"Agent run completed: status={status}, agent_run_id={agent_run_id}")
            break
        
        await asyncio.sleep(20)
    else:
        logger.error(f"Timeout waiting for agent run to complete: agent_run_id={agent_run_id}")
        raise TimeoutError(f"Agent run {agent_run_id} did not complete within {timeout} seconds")

    messages = await get_llm_messages(task_id, return_raw=True)

    return messages


if __name__ == "__main__":
    task_description = "今天北京天气怎么样"

    response = asyncio.run(
        run_backend(
            task_description=task_description,
            task_file_path=[],
            tags=["debug"],
            user_id="13183c90-ac94-403e-893e-c53552ad429d",
            model_name="openrouter/qwen/qwen3-235b-a22b",
            # model_name="qwen/qwen3.5-397b-a17b",
            api_key='',
            base_url='',
        )
    )
    print("Response:", response)
