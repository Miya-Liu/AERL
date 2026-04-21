import asyncio
import base64
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

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
    AgentCreateRequest,
    AgentService,
    DBConnection,
    cleanup_sandbox_for_task,
    create_task,
    get_agent_loader,
    get_llm_messages,
)

sys.stdout.reconfigure(encoding="utf-8")

# Initialize logger
try:
    from areal.utils.logging import getLogger

    logger = getLogger("BackendRun")
except ImportError:
    import logging

    logger = logging.getLogger("BackendRun")

DEFAULT_REFRESH_TOKEN = "4uhiohwgwp7e"
DEFAULT_AGENT_ID = "b11faebe-8d6a-4467-8609-10323d9444d6"
# DEFAULT_AGENT_ID = None
DEFAULT_USER_ID = "13183c90-ac94-403e-893e-c53552ad429d"
LE_AGENT_API_URL = os.environ.get("LE_AGENT_API_URL", "http://localhost:8000")
TOKEN_REFRESH_MARGIN = 60  # seconds — refresh 1 min before expiry, not 5

_SHARED_TOKEN_FILE = Path(__file__).parent / ".shared_auth_token.json"


class _FileLock:
    """Context manager for fcntl file locking."""

    def __init__(self, fd, lock_type):
        self._fd = fd
        self._lock_type = lock_type

    def __enter__(self):
        fcntl.flock(self._fd, self._lock_type)

    def __exit__(self, *args):
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        return False


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
        logger.warning("Failed to decode JWT payload: %s", e)
        return {}


def _is_token_valid(token: str, margin: int = TOKEN_REFRESH_MARGIN) -> bool:
    """Return True if *token* is still valid for at least *margin* seconds."""
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp")
    return bool(exp and exp >= time.time() + margin)


class SharedTokenManager:
    """File-based auth token manager for multi-process sharing.

    Stores the access_token in a JSON file so multiple processors can
    read the same token. When a processor detects the token is expired,
    it refreshes the token via _refresh_access_token and writes the new
    token back to the file.

    Uses fcntl file locking to prevent race conditions between processes,
    and asyncio.Lock to prevent race conditions between tasks in the same
    process. The fcntl lock is NEVER held across an await.
    """

    _async_lock: asyncio.Lock | None = None

    def __init__(
        self,
        token_file: Path = _SHARED_TOKEN_FILE,
        refresh_token: str = DEFAULT_REFRESH_TOKEN,
    ):
        self.token_file = token_file
        self.refresh_token = refresh_token
        # One async lock per class (shared across instances in the same process)
        if SharedTokenManager._async_lock is None:
            SharedTokenManager._async_lock = asyncio.Lock()

    def read_token(self) -> str | None:
        """Read the shared access token from file."""
        try:
            with open(self.token_file) as f, _FileLock(f, fcntl.LOCK_SH):
                data = json.load(f)
            return data.get("access_token")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def write_token(self, access_token: str) -> None:
        """Write the access token to the shared file."""
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.token_file, "w") as f, _FileLock(f, fcntl.LOCK_EX):
            json.dump({"access_token": access_token, "updated_at": time.time()}, f)
        logger.info("Shared auth token updated in file: %s", self.token_file)

    async def get_valid_token(self) -> str:
        """Get a valid access token, refreshing if necessary.

        Reads the shared token file. If the token is expired or missing,
        acquires locks so only one coroutine/process refreshes the token.
        The fcntl lock is never held across an await.
        """
        auth_token = self.read_token()
        if auth_token and _is_token_valid(auth_token):
            return auth_token

        async with SharedTokenManager._async_lock:
            # Double-check after acquiring async lock.
            auth_token = self.read_token()
            if auth_token and _is_token_valid(auth_token):
                return auth_token

            logger.info("Shared auth token expired or about to expire, refreshing...")

            # Re-read under fcntl lock in case another process refreshed.
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_file, "a+") as f, _FileLock(f, fcntl.LOCK_EX):
                try:
                    f.seek(0)
                    data = json.load(f)
                    auth_token = data.get("access_token")
                except (json.JSONDecodeError, KeyError):
                    auth_token = None

                if auth_token and _is_token_valid(auth_token):
                    return auth_token

            # Refresh token WITHOUT holding the file lock (avoids blocking the event loop).
            new_token = await _refresh_access_token(self.refresh_token)

            # Write the new token under fcntl lock.
            with open(self.token_file, "w") as f, _FileLock(f, fcntl.LOCK_EX):
                json.dump({"access_token": new_token, "updated_at": time.time()}, f)
            logger.info("Shared auth token updated in file: %s", self.token_file)
            return new_token


async def _refresh_access_token(refresh_token: str) -> str:
    """Refresh Supabase access token using refresh_token."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_anon_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set to refresh token"
        )

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


def _prepare_form_data(
    task_id: str,
    task_description: str | None,
    agent_id: str,
    model_name: str | None,
    base_url: str | None,
    api_key: str | None,
    tags: list[str] | None,
) -> dict:
    """Build the multipart form data for the agent start API."""
    form_data = {
        "task_id": task_id,
        "prompt": task_description,
        "agent_id": agent_id,
    }
    if model_name is not None:
        form_data["model_name"] = model_name

    if base_url is not None:
        form_data["proxy_base_url"] = base_url
    if api_key is not None:
        form_data["proxy_api_key"] = api_key
    if tags is not None:
        form_data["tags"] = json.dumps(list(tags))

    return form_data


async def _resolve_agent_id(client, user_id: str, agent_id: str | None) -> str:
    """Return the provided agent_id or create a default one if missing."""
    if agent_id is not None:
        return agent_id

    agent_service = AgentService(client)
    from customized_areal.tpfc.config.builtin import TPFC_CONFIG

    created_agent = await agent_service.create_agent(
        user_id,
        AgentCreateRequest(
            name=TPFC_CONFIG["name"],
            config=TPFC_CONFIG["config"],
            is_default=TPFC_CONFIG.get("is_default", False),
        ),
    )
    new_agent_id = created_agent.agent_id
    loader = await get_agent_loader()
    # agent_data = await loader.load_agent(agent_id, user_id, load_config=True)
    agent_data = await loader.load_agent(new_agent_id, user_id, load_config=True)
    logger.info("Created agent: %s", new_agent_id)
    return new_agent_id


async def _start_agent_run(
    api_base_url: str,
    auth_token: str,
    form_data: dict,
    task_file_path: list[str] | None,
) -> dict:
    """Start the agent run via HTTP and return the JSON response."""
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
                        logger.warning("File not found: %s", file_path)

            response = await http_client.post(
                f"{api_base_url}/api/agent/start",
                headers={"Authorization": f"Bearer {auth_token}"},
                data=form_data,
                files=files if files else None,
            )
        finally:
            for fh in file_handles:
                fh.close()

        if response.status_code != 200:
            logger.error(
                "Failed to start agent run via API: status_code=%s, response=%s",
                response.status_code,
                response.text,
            )
            raise RuntimeError(
                f"Failed to start agent run: {response.status_code} - {response.text}"
            )

        return response.json()


TERMINAL_SSE_EVENTS = {"task_end", "error"}


async def _wait_for_agent_run(
    client,
    task_id: str,
    agent_run_id: str | None,
    api_base_url: str | None = None,
    auth_token: str | None = None,
    timeout: int = 3000,
) -> str:
    """Wait until the agent run reaches a terminal state.

    If *api_base_url* and *auth_token* are provided, attempts to consume the
    task-level SSE stream from the backend first with auto-reconnect. Falls
    back to database polling if the stream endpoint is unavailable or the
    stream ends without a terminal status.
    """
    start_time = time.time()
    status = "pending"
    streamed = False
    last_event_id: str | None = None

    def _time_left() -> float:
        return timeout - (time.time() - start_time)

    if api_base_url and auth_token:
        stream_url = (
            f"{api_base_url}/api/tasks/{task_id}/stream?token={auth_token}"
        )
        sse_retry_delay = 1.0

        while _time_left() > 0:
            headers: dict[str, str] = {}
            if last_event_id is not None:
                headers["last-event-id"] = last_event_id

            try:
                async with httpx.AsyncClient(
                    timeout=_time_left() + 10.0
                ) as http_client:
                    async with http_client.stream(
                        "GET",
                        stream_url,
                        headers=headers,
                        timeout=_time_left() + 10.0,
                    ) as response:
                        if response.status_code == 200:
                            streamed = True
                            sse_retry_delay = 1.0
                            current_event = "message"
                            current_data_parts: list[str] = []

                            async for raw_line in response.aiter_lines():
                                if _time_left() <= 0:
                                    break

                                line = raw_line.strip()
                                # SSE comment (e.g. keepalive) — ignore
                                if line.startswith(":"):
                                    continue
                                if line.startswith("id:"):
                                    last_event_id = line[3:].strip() or last_event_id
                                    continue
                                if line.startswith("event:"):
                                    current_event = line[6:].strip()
                                    continue
                                if line.startswith("data:"):
                                    current_data_parts.append(line[5:].strip())
                                    continue
                                # Empty line = end of SSE message
                                if line == "":
                                    if current_data_parts:
                                        data_str = "\n".join(current_data_parts)
                                        current_data_parts = []
                                        try:
                                            event = json.loads(data_str)
                                        except json.JSONDecodeError:
                                            event = {}

                                        # Derive status from event data when available
                                        event_status = event.get("status")
                                        if event_status:
                                            status = event_status

                                        # Terminal event types close the stream
                                        if current_event in TERMINAL_SSE_EVENTS:
                                            # task_end carries status; error is a failure
                                            if current_event == "error":
                                                status = "failed"
                                            break

                                        logger.debug(
                                            "SSE event: type=%s status=%s task_id=%s",
                                            current_event,
                                            status,
                                            task_id,
                                        )
                                    current_event = "message"

                            if status in {"completed", "failed", "stopped"}:
                                break
                            # Stream ended without terminal event — may need reconnect
                            if current_event not in TERMINAL_SSE_EVENTS:
                                logger.warning(
                                    "SSE stream ended for task_id=%s, reconnecting in %.1fs",
                                    task_id,
                                    sse_retry_delay,
                                )
                                await asyncio.sleep(min(sse_retry_delay, _time_left()))
                                sse_retry_delay = min(sse_retry_delay * 2, 30.0)
                                continue
                        else:
                            logger.warning(
                                "SSE stream endpoint returned %s, falling back to polling: %s",
                                response.status_code,
                                stream_url,
                            )
                            break
            except httpx.ConnectError as exc:
                logger.warning(
                    "SSE connect error for task_id=%s, retrying in %.1fs: %s",
                    task_id,
                    sse_retry_delay,
                    exc,
                )
                await asyncio.sleep(min(sse_retry_delay, _time_left()))
                sse_retry_delay = min(sse_retry_delay * 2, 30.0)
                continue
            except Exception as exc:
                logger.warning(
                    "SSE error for task_id=%s, retrying in %.1fs: %s",
                    task_id,
                    sse_retry_delay,
                    exc,
                )
                await asyncio.sleep(min(sse_retry_delay, _time_left()))
                sse_retry_delay = min(sse_retry_delay * 2, 30.0)
                continue

    if not streamed or status not in {"completed", "failed", "stopped"}:
        retry_delay = 1.0
        while _time_left() > 0:
            try:
                agent_run = (
                    await client.table("agent_runs")
                    .select("status, error, completed_at")
                    .eq("id", agent_run_id)
                    .single()
                    .execute()
                )
                status = agent_run.data["status"]
            except Exception as exc:
                if _time_left() <= 0:
                    break
                sleep_for = min(retry_delay, _time_left())
                logger.warning(
                    "DB poll failed for agent_run_id=%s, retrying in %.1fs: %s",
                    agent_run_id,
                    sleep_for,
                    exc,
                )
                await asyncio.sleep(sleep_for)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

            if status in {"completed", "failed", "stopped"}:
                break

            if _time_left() <= 0:
                break
            await asyncio.sleep(min(30.0, _time_left()))
            retry_delay = 1.0
        else:
            logger.error(
                "Timeout waiting for agent run to complete: agent_run_id=%s",
                agent_run_id,
            )
            raise TimeoutError(
                f"Agent run {agent_run_id} did not complete within {timeout} seconds"
            )

    return status


def _extract_final_answer(messages: list[dict]) -> str | None:
    """Extract text inside the first <answer> tag from the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", "")
        if isinstance(content, dict):
            content = content.get("content", "")
        elif isinstance(content, list):
            content = "".join(
                p.get("text", p.get("content", "")) if isinstance(p, dict) else str(p)
                for p in content
            )

        if isinstance(content, str):
            match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if match:
                return match.group(1).strip()

    return None


async def run_backend(
    task_description: str | None,
    task_file_path: list[str] | None,
    log_path: str = "./log.json",
    task_id: str = "",
    gt: str = "",
    tags: list[str] | None = None,
    user_id: str | None = None,
    model_name: str | None = None,
    server_manager=None,
    tokenizer=None,
    agent_id: str | None = DEFAULT_AGENT_ID,
    base_url: str | None = None,
    api_key: str | None = None,
    refresh_token: str | None = DEFAULT_REFRESH_TOKEN,
):
    db = DBConnection()
    client = await db.client

    user_id = user_id or DEFAULT_USER_ID
    resolved_agent_id = await _resolve_agent_id(client, user_id, agent_id)

    task_id = await create_task(
        client=client,
        account_id=user_id,
        agent_id=resolved_agent_id,
        name=task_description[:100] if task_description else None,
    )
    logger.info("Task created: %s", task_id)

    token_manager = SharedTokenManager(
        refresh_token=refresh_token or DEFAULT_REFRESH_TOKEN
    )
    auth_token = await token_manager.get_valid_token()

    form_data = _prepare_form_data(
        task_id=task_id,
        task_description=task_description,
        agent_id=resolved_agent_id,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        tags=tags,
    )

    result = await _start_agent_run(
        api_base_url=LE_AGENT_API_URL,
        auth_token=auth_token,
        form_data=form_data,
        task_file_path=task_file_path,
    )
    logger.info("Agent run started via API: %s", result)

    agent_run_id = result["agent_run_id"]
    status = await _wait_for_agent_run(
        client,
        agent_run_id,
        api_base_url=LE_AGENT_API_URL,
        auth_token=auth_token,
    )

    messages = await get_llm_messages(task_id, return_raw=True)
    final_boxed_answer = _extract_final_answer(messages)

    if status in {"completed", "failed", "stopped"}:
        await cleanup_sandbox_for_task(client, task_id)

    return messages, final_boxed_answer, log_path, None


if __name__ == "__main__":
    task_description = "今天北京天气怎么样"

    messages, final_answer, log_path, _trace = asyncio.run(
        run_backend(
            task_description=task_description,
            task_file_path=[],
            tags=["debug"],
            user_id=DEFAULT_USER_ID,
            model_name="openrouter/qwen/qwen3-235b-a22b",
            api_key="",
            base_url="",
            refresh_token=DEFAULT_REFRESH_TOKEN,
        )
    )
    print("Messages:", messages)
    print("Final boxed answer:", final_answer)
    print("Log path:", log_path)
