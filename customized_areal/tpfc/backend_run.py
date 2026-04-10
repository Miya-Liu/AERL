import asyncio
import sys
import httpx
import json
import base64
import os
import time
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

DEFAULT_BACKEND_AUTH_TOKEN = "eyJhbGciOiJFUzI1NiIsImtpZCI6ImM2YWFjMzE0LTczMzctNDBlYS04ZmU3LTBjMDMxODA3OTJlZiIsInR5cCI6IkpXVCJ9.eyJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJwYXNzd29yZCIsInRpbWVzdGFtcCI6MTc3NTIxMDczN31dLCJhcHBfbWV0YWRhdGEiOnsicHJvdmlkZXIiOiJlbWFpbCIsInByb3ZpZGVycyI6WyJlbWFpbCJdfSwiYXVkIjoiYXV0aGVudGljYXRlZCIsImVtYWlsIjoiemhvdWppZTIyQGxlbm92by5jb20iLCJleHAiOjE3NzUyMTQzMzcsImlhdCI6MTc3NTIxMDczNywiaXNfYW5vbnltb3VzIjpmYWxzZSwiaXNzIjoiaHR0cHM6Ly93bGxpa3hpZmNrZXRhdm9pZ2lmYS5zdXBhYmFzZS5jby9hdXRoL3YxIiwicGhvbmUiOiIiLCJyb2xlIjoiYXV0aGVudGljYXRlZCIsInNlc3Npb25faWQiOiIyNWRmNzE5MC02YWI4LTQ1MjgtYmIzNi1lODY5NDA4ZDhhNTQiLCJzdWIiOiIxMzE4M2M5MC1hYzk0LTQwM2UtODkzZS1jNTM1NTJhZDQyOWQiLCJ1c2VyX21ldGFkYXRhIjp7ImVtYWlsX3ZlcmlmaWVkIjp0cnVlfX0.MiHqWwj6mR3WwzaArSo_UlSv6KFAZOf7HMcrHhLRGjuf-yZ1O3_-kgDX2Ou-Ra8tQ1WSZwjecqlTg2UylMhjgw"
DEFAULT_REFRESH_TOKEN = "4uhiohwgwp7e"

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


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without verifying signature."""
    try:
        payload_b64 = token.split(".")[1]
        padding_needed = 4 - len(payload_b64) % 4
        if padding_needed != 4:
            payload_b64 += "=" * padding_needed
        payload_json = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_json)
    except Exception as e:
        logger.warning(f"Failed to decode JWT payload: {e}")
        return {}


async def _refresh_access_token(refresh_token: str) -> str:
    """Refresh Supabase access token using refresh_token."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_anon_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set to refresh token")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{supabase_url}/auth/v1/token?grant_type=refresh_token",
            headers={
                "apikey": supabase_anon_key,
                "Content-Type": "application/json",
            },
            json={"refresh_token": refresh_token},
        )
        resp.raise_for_status()
        data = resp.json()
        new_access_token = data.get("access_token")
        if not new_access_token:
            raise RuntimeError(f"No access_token in refresh response: {data}")
        logger.info("Access token refreshed successfully")
        return new_access_token


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
    api_key: str | None = None,
    backend_auth_token: str | None = DEFAULT_BACKEND_AUTH_TOKEN,
    refresh_token: str | None = DEFAULT_REFRESH_TOKEN,
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

    # Determine API base URL for le-agent-dev backend API
    # This is different from base_url (which is the AReaL proxy URL for LLM calls)
    api_base_url = os.environ.get("LE_AGENT_API_URL", "http://localhost:8000")

    # Backend API authentication
    # The new backend (le-agent-dev2) validates JWT via Supabase JWKS using ES256.
    # Forging a local HS256 token no longer works. You must supply a real Supabase
    # access token via backend_auth_token, or obtain one via Supabase auth.
    auth_token = backend_auth_token
    if auth_token:
        payload = _decode_jwt_payload(auth_token)
        exp = payload.get("exp")
        if exp and exp < time.time() + 300:  # expires within 5 minutes
            if refresh_token:
                try:
                    auth_token = await _refresh_access_token(refresh_token)
                except Exception as e:
                    logger.error(f"Failed to refresh access token: {e}")
                    raise RuntimeError(f"Access token expired and refresh failed: {e}")
            else:
                logger.error("Access token is expired or about to expire, but no refresh_token provided")
                raise RuntimeError("Access token expired and no refresh_token available")

    if not auth_token:
        logger.warning(
            "No backend_auth_token provided. Falling back to locally-generated HS256 token. "
            "This will fail against backends that use Supabase JWKS (ES256) validation."
        )
        import jwt
        jwt_secret = os.environ.get("SUPABASE_JWT_SECRET", "internal")
        auth_token = jwt.encode(
            {"sub": user_id, "exp": time.time() + 3600, "aud": "authenticated"},
            jwt_secret,
            algorithm="HS256"
        )
    
    # Prepare multipart form data (common for both branches)
    # proxy_base_url and proxy_api_key are passed to the agent for LLM routing through AReaL proxy
    form_data = {
        "task_id": task_id,
        "prompt": task_description,
        "agent_id": agent_id,
        "model_name": model_name,
        "is_sub_agent": "false",
        "stream": "false"
    }

    # Only include proxy settings when they have values (avoid empty strings in form data)
    if base_url is not None:
        form_data["proxy_base_url"] = base_url  # AReaL proxy URL for LLM calls
    if api_key is not None:
        form_data["proxy_api_key"] = api_key    # Session API key for proxy auth

    # Add tags if provided
    if tags:
        for tag in tags:
            form_data.setdefault("tags", [])
        # Note: For multipart, we need to handle tags differently

    # Prepare files for multipart upload and make request
    async with httpx.AsyncClient(timeout=300.0) as http_client:
        files = []
        file_handles = []
        try:
            if task_file_path:
                for file_path in task_file_path:
                    if os.path.exists(file_path):
                        fh = open(file_path, "rb")
                        file_handles.append(fh)
                        files.append(("files", (os.path.basename(file_path), fh, None)))
                    else:
                        logger.warning(f"File not found: {file_path}")

            response = await http_client.post(
                f"{api_base_url}/api/agent/start",
                headers={
                    "Authorization": f"Bearer {auth_token}",
                },
                data=form_data,
                files=files if files else None,
            )
        finally:
            for fh in file_handles:
                fh.close()

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
        await asyncio.sleep(60)

        if status in ["completed", "failed", "stopped"]:
            if status == "failed":
                logger.error(f"Agent run failed: error={agent_run.data.get('error')}, agent_run_id={agent_run_id}")
            else:
                logger.info(f"Agent run completed: status={status}, agent_run_id={agent_run_id}")
            break
    else:
        logger.error(f"Timeout waiting for agent run to complete: agent_run_id={agent_run_id}")
        raise TimeoutError(f"Agent run {agent_run_id} did not complete within {timeout} seconds")

    messages = await get_llm_messages(task_id, return_raw=True)

    # Extract final_boxed_answer from the last assistant message with <answer> tags
    final_boxed_answer = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "").get("content", "")
            if isinstance(content, str):
                import re
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if match:
                    final_boxed_answer = match.group(1).strip()
                    break

    # Return values expected by benchmark_run.py: (response, final_boxed_answer, log_file_path, _trace)
    return messages, final_boxed_answer, log_path, None


if __name__ == "__main__":
    task_description = "今天北京天气怎么样"

    messages, final_answer, log_path, _trace = asyncio.run(
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
    print("Messages:", messages)
    print("Final boxed answer:", final_answer)
    print("Log path:", log_path)
