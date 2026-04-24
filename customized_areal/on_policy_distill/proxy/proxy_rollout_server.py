"""
Token-level reward proxy rollout server.

This server extends the base proxy rollout server to support token-level
rewards via HTTP API. It uses TokenRewardSessionData to store and manage
token-level rewards directly on the server.

Usage:
    python -m customized_areal.on_policy_distill.proxy.proxy_rollout_server \
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import inspect
import os
import secrets
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import uvicorn
from anthropic.types.message import Message
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.types.utils import ModelResponse as LitellmModelResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses import Response
from openai.types.responses.response_create_params import ResponseCreateParams

# Import from areal base server
from areal.api.cli_args import NameResolveConfig, OpenAIProxyConfig
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.proxy.server import (
    ANTHROPIC_MESSAGES_PATHNAME,
    CHAT_COMPLETIONS_PATHNAME,
    DEFAULT_ADMIN_API_KEY,
    EXPORT_TRAJECTORIES_PATHNAME,
    GRANT_CAPACITY_PATHNAME,
    RESPONSES_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    SESSION_TIMEOUT_SECONDS,
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
    serialize_interactions,
)
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils import name_resolve, names, seeding
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.logging import getLogger
from areal.utils.network import find_free_ports, gethostip

# Import from local server module
from .server import (
    RL_COMPUTE_ENTROPY_PATHNAME,
    RL_SET_POSITION_REWARDS_PATHNAME,
    RL_SET_TOKEN_REWARDS_PATHNAME,
    ComputeEntropyRequest,
    ComputeEntropyResponse,
    SetPositionRewardsRequest,
    SetTokenRewardsRequest,
    TokenRewardSessionData,
)

if TYPE_CHECKING:
    from areal.api import InferenceEngine

logger = getLogger("TokenRewardProxyServer")


# =============================================================================
# Warning Deduplication
# =============================================================================

_warn_once_enabled = os.environ.get("AREAL_PROXY_WARN_ONCE", "0") == "1"
_warned_messages: set[str] = set()
_warn_lock = threading.Lock()


def _warn_once(msg: str) -> None:
    """Log a warning message, optionally only once if AREAL_PROXY_WARN_ONCE=1."""
    if not _warn_once_enabled:
        logger.warning(msg)
        return

    with _warn_lock:
        if msg not in _warned_messages:
            _warned_messages.add(msg)
            logger.warning(msg)


# =============================================================================
# Custom Serialization with position_rewards Support
# =============================================================================


def serialize_interactions_with_position_rewards(
    interactions: dict,
) -> dict:
    """Serialize interactions including position_rewards and token_rewards for distillation.

    Extends the base serialize_interactions to include position_rewards
    and token_rewards so they can flow through HTTP transport to the training pipeline.
    """
    from areal.infra.rpc.serialization import serialize_value

    result = {}
    for key, interaction in interactions.items():
        entry = {
            "tensor_dict": interaction.to_tensor_dict(),
            "reward": interaction.reward,
            "interaction_id": interaction.interaction_id,
        }
        # Include token_rewards if available (set by
        # TokenRewardSessionData.set_token_rewards or export_interactions)
        token_rewards = getattr(interaction, "token_rewards", None)
        if token_rewards is not None:
            entry["token_rewards"] = token_rewards
        # Include position_rewards if available (set by
        # TokenRewardSessionData.set_position_rewards)
        pos_rewards = getattr(interaction, "position_rewards", None)
        if pos_rewards is not None:
            entry["position_rewards"] = [
                {
                    "position": pr.position,
                    "candidates": pr.candidates,
                    "candidate_token_ids": pr.candidate_token_ids,
                    "logprobs": pr.logprobs,
                    "rewards": pr.rewards,
                    "chosen_index": pr.chosen_index,
                    "sample_index": pr.sample_index,
                }
                for pr in pos_rewards
            ]
        result[key] = entry
    return serialize_value(result)


def deserialize_interactions_with_position_rewards(
    data: dict,
) -> dict:
    """Deserialize interactions including position_rewards and token_rewards for distillation.

    Extends the base deserialize_interactions to reconstruct position_rewards
    and token_rewards from the serialized data. Position_rewards are stored as
    a Python attribute on the interaction object; token_rewards are also stored
    as a Python attribute so they can be used downstream if needed.
    """
    from areal.experimental.openai.types import InteractionWithTokenLogpReward
    from areal.infra.rpc.serialization import deserialize_value

    data = deserialize_value(data)
    result = {}
    for key, item in data.items():
        interaction = InteractionWithTokenLogpReward()
        interaction._cache = item["tensor_dict"]
        interaction.reward = item["reward"]
        interaction.interaction_id = item["interaction_id"]

        # Reconstruct token_rewards if available
        token_rewards_data = item.get("token_rewards")
        if token_rewards_data is not None:
            interaction.token_rewards = token_rewards_data  # type: ignore

        # Reconstruct position_rewards if available and inject into the
        # cached tensor dict so they flow through the data pipeline to
        # the distillation loss function.
        pos_rewards_data = item.get("position_rewards")
        if pos_rewards_data is not None:
            from .server import PositionRewardInfo as PRI

            pos_rewards = [
                PRI(
                    position=pr["position"],
                    candidates=pr["candidates"],
                    candidate_token_ids=pr["candidate_token_ids"],
                    logprobs=pr["logprobs"],
                    rewards=pr["rewards"],
                    chosen_index=pr["chosen_index"],
                    sample_index=pr.get("sample_index", 0),
                )
                for pr in pos_rewards_data
            ]
            # Store as a Python attribute — the workflow extracts it after
            # to_tensor_dict() conversion and attaches it to the tensor dict.
            # We do NOT inject it into _cache to avoid concat_padded_tensors
            # key consistency issues when some interactions have position_rewards
            # and others don't (e.g., multi-turn conversations).
            interaction.position_rewards = pos_rewards  # type: ignore

        result[key] = interaction
    return result


# =============================================================================
# Module-Level Globals
# =============================================================================

# Session management with token-level reward support
_session_cache: dict[str, TokenRewardSessionData] = {}
_lock = threading.Lock()
_capacity = 0
_admin_api_key: str = secrets.token_urlsafe(32)
_api_key_to_session: dict[str, str] = {}
_session_to_api_key: dict[str, str] = {}

# Server config
_server_host: str = "0.0.0.0"
_server_port: int = 8000

# Engine and client (created via /create_engine and /call with method "initialize")
_engine: InferenceEngine | None = None
_openai_client: ArealOpenAI | None = None

# Port allocation tracking
_allocated_ports: set[int] = set()
_port_alloc_lock = asyncio.Lock()

# Session cleanup timing
_last_cleanup_time: float = 0
_session_timeout_seconds: int = 3600  # Overridden by _setup_openai_client()

# Name_resolve config (needed for cluster registration)
_experiment_name: str | None = None
_trial_name: str | None = None
_name_resolve_type: str = "nfs"
_nfs_record_root: str = "/tmp/areal/name_resolve"
_etcd3_addr: str = "localhost:2379"

# Anthropic request adapter
_adapter = AnthropicAdapter()


# =============================================================================
# FastAPI App with Lifespan
# =============================================================================

_cleanup_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage background tasks with proper lifecycle."""
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_stale_sessions())
    logger.info(f"Token Reward Proxy Server started on {_server_host}:{_server_port}")
    yield
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Token Reward Proxy Server", lifespan=lifespan)


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_bearer_token(request: Request) -> str:
    """Extract API token from Authorization header or x-api-key header.

    Supports both 'Authorization: Bearer <token>' (OpenAI SDK, case-insensitive
    per RFC 6750) and 'x-api-key: <token>' (Anthropic SDK) for cross-SDK
    compatibility.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    x_api_key = request.headers.get("x-api-key", "")
    if x_api_key:
        return x_api_key
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header. Expected 'Bearer <token>' or 'x-api-key: <token>'.",
    )


def _require_admin_key(request: Request) -> str:
    """Validate admin API key using constant-time comparison."""
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, _admin_api_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


def _require_session_key(request: Request) -> str:
    """Resolve session_id from the session API key."""
    token = _extract_bearer_token(request)
    with _lock:
        session_id = _api_key_to_session.get(token)
    if session_id is None:
        raise HTTPException(
            status_code=401, detail="Invalid or expired session API key."
        )
    return session_id


def _generate_api_key() -> str:
    """Generate a unique session API key."""
    return f"tr-session-{secrets.token_urlsafe(32)}"


def _remove_api_keys_for_session(session_id: str) -> None:
    """Remove the API key mapping for the given session.

    Must be called with _lock held.
    """
    api_key = _session_to_api_key.pop(session_id, None)
    if api_key:
        _api_key_to_session.pop(api_key, None)


# =============================================================================
# Admin Endpoints
# =============================================================================


@app.post(f"/{RL_START_SESSION_PATHNAME}")
def start_session(
    request: StartSessionRequest, admin_key: str = Depends(_require_admin_key)
) -> StartSessionResponse:
    """Start a new RL session with token-level reward support.

    If ``request.api_key`` is provided, reuse that key instead of generating
    a new one.  When the key already maps to a *finished* session on this
    worker, the stale mapping is cleaned up first.  If it maps to an
    *active* (unfinished) session, the request is rejected with HTTP 409.
    """
    import uuid

    global _capacity
    task_id = request.task_id

    with _lock:
        if _capacity <= 0:
            raise HTTPException(
                status_code=429,
                detail="No available capacity to start a new session",
            )

        session_id = f"tr-{uuid.uuid4().hex[:16]}"

        # Resolve session API key
        if request.api_key:
            session_api_key = request.api_key
            if session_api_key == _admin_api_key:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot use the admin API key as a session key.",
                )
            existing_sid = _api_key_to_session.get(session_api_key)
            if existing_sid is not None:
                existing_session = _session_cache.get(existing_sid)
                if existing_session is not None and not existing_session.is_completed:
                    raise HTTPException(
                        status_code=409,
                        detail=f"API key is already bound to active session {existing_sid}.",
                    )
                _remove_api_keys_for_session(existing_sid)
        else:
            session_api_key = _generate_api_key()

        _capacity -= 1
        _session_cache[session_id] = TokenRewardSessionData(session_id)
        _api_key_to_session[session_api_key] = session_id
        _session_to_api_key[session_id] = session_api_key

    logger.info(f"Started session {session_id} for task {task_id}")
    return StartSessionResponse(session_id=session_id, api_key=session_api_key)


@app.post(f"/{RL_END_SESSION_PATHNAME}")
def end_session(session_id: str = Depends(_require_session_key)):
    """End an RL session.

    Returns the number of recorded interactions so callers (e.g. the proxy
    gateway refresh path) can decide whether meaningful trajectory data
    exists.

    Marks the session as finished and removes API key mappings so no
    further writes are possible. The session data remains in cache for
    export_trajectories to retrieve. The cleanup task will eventually
    remove it if export_trajectories is never called.
    """
    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail="Session already ended or expired"
            )
        session_data = _session_cache[session_id]
        interaction_count = len(session_data.completions)

    # finish() outside lock to avoid holding lock during potential I/O
    session_data.finish()

    # Remove API key mappings so no further authenticated writes are possible
    with _lock:
        _remove_api_keys_for_session(session_id)

    logger.info(f"Ended session {session_id}")
    return {"message": "success", "interaction_count": interaction_count}


@app.post(f"/{EXPORT_TRAJECTORIES_PATHNAME}")
async def export_trajectories(
    request: ExportTrajectoriesRequest, admin_key: str = Depends(_require_admin_key)
) -> ExportTrajectoriesResponse:
    """Export trajectories for a session with token-level rewards.

    Waits for the session to complete before exporting, then removes the
    session from cache and cleans up API key mappings.
    """
    session_id = request.session_id

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )
        session_data = _session_cache[session_id]

    # Wait for session to complete (non-blocking, outside lock)
    await session_data.wait_for_finish()

    # Export with token-level rewards applied
    interactions = session_data.export_interactions(
        discount=request.discount,
        style=request.style,
    )

    # Serialize for HTTP response (includes position_rewards)
    serialized = serialize_interactions_with_position_rewards(interactions)

    # Remove session from cache and clean up API key mapping
    with _lock:
        _session_cache.pop(session_id, None)
        _remove_api_keys_for_session(session_id)

    logger.info(f"Exported {len(serialized)} interactions from session {session_id}")
    return ExportTrajectoriesResponse(interactions=serialized)


@app.post(f"/{GRANT_CAPACITY_PATHNAME}")
def grant_capacity(admin_key: str = Depends(_require_admin_key)):
    """Grant capacity for a new session."""
    global _capacity
    with _lock:
        _capacity += 1
    logger.info("Capacity granted")
    return {"message": "success", "capacity": _capacity}


# =============================================================================
# Scalar Reward Endpoints (from base server)
# =============================================================================


@app.post(f"/{RL_SET_REWARD_PATHNAME}")
def set_reward(
    request: SetRewardRequest, session_id: str = Depends(_require_session_key)
):
    """Set scalar reward for an interaction."""
    interaction_id = request.interaction_id
    reward = request.reward

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail=f"Session {session_id} already ended or expired"
            )
        session_data = _session_cache[session_id]

    if session_data.is_completed:
        raise HTTPException(
            status_code=409, detail=f"Session {session_id} is already finished"
        )

    session_data.update_last_access()

    completions = session_data.completions
    if interaction_id is None:
        # Take the last interaction id
        if len(completions) == 0:
            logger.error(f"No interactions in session {session_id}")
            raise HTTPException(status_code=400, detail="No interactions in session")
        interaction_id = completions.last_interaction_id
    elif interaction_id not in completions:
        logger.error(f"Interaction {interaction_id} not found in session {session_id}")
        raise HTTPException(
            status_code=400, detail=f"Interaction {interaction_id} not found"
        )

    session_data.completions.set_reward(interaction_id, reward)
    logger.info(f"Set scalar reward for {interaction_id}: {reward}")
    return {"message": "success"}


# =============================================================================
# Token-Level Reward Endpoints (NEW)
# =============================================================================


@app.post(f"/{RL_SET_TOKEN_REWARDS_PATHNAME}")
def set_token_rewards(
    request: SetTokenRewardsRequest, session_id: str = Depends(_require_session_key)
):
    """
    Set token-wise rewards for an interaction.

    Each reward in the list corresponds to one output token in the completion.
    """
    interaction_id = request.interaction_id
    token_rewards = request.token_rewards

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail=f"Session {session_id} already ended or expired"
            )
        session_data = _session_cache[session_id]

    if session_data.is_completed:
        raise HTTPException(
            status_code=409, detail=f"Session {session_id} is already finished"
        )

    session_data.update_last_access()

    # Resolve interaction_id if not provided
    if interaction_id is None:
        completions = session_data.completions
        if len(completions) == 0:
            logger.error(f"No interactions in session {session_id}")
            raise HTTPException(status_code=400, detail="No interactions in session")
        interaction_id = completions.last_interaction_id

    # Validate token rewards length matches output tokens
    if interaction_id in session_data.completions:
        interaction = session_data.completions[interaction_id]
        if (
            hasattr(interaction, "model_response")
            and interaction.model_response is not None
        ):
            expected_len = len(interaction.model_response.output_tokens)
            if len(token_rewards) != expected_len:
                raise HTTPException(
                    status_code=400,
                    detail=f"Token rewards length ({len(token_rewards)}) must match "
                    f"output tokens length ({expected_len})",
                )

    # Store token-level rewards
    try:
        session_data.set_token_rewards(interaction_id, token_rewards)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    logger.info(
        f"Set {len(token_rewards)} token rewards for {interaction_id}: "
        f"sum={sum(token_rewards):.3f}"
    )
    return {"message": "success", "interaction_id": interaction_id}


@app.post(f"/{RL_SET_POSITION_REWARDS_PATHNAME}")
def set_position_rewards(
    request: SetPositionRewardsRequest,
    session_id: str = Depends(_require_session_key),
):
    """
    Set position-wise rewards for an interaction.

    Each position contains candidate tokens and their rewards.
    """
    interaction_id = request.interaction_id
    position_rewards = request.position_rewards

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail=f"Session {session_id} already ended or expired"
            )
        session_data = _session_cache[session_id]

    if session_data.is_completed:
        raise HTTPException(
            status_code=409, detail=f"Session {session_id} is already finished"
        )

    session_data.update_last_access()

    # Resolve interaction_id if not provided
    if interaction_id is None:
        completions = session_data.completions
        if len(completions) == 0:
            logger.error(f"No interactions in session {session_id}")
            raise HTTPException(status_code=400, detail="No interactions in session")
        interaction_id = completions.last_interaction_id

    # Convert Pydantic models to dataclasses for storage
    from .server import PositionRewardInfo as PositionRewardInfoDataclass

    pr_dataclasses = []
    for pr in position_rewards:
        pr_dataclasses.append(
            PositionRewardInfoDataclass(
                position=pr.position,
                candidates=pr.candidates,
                candidate_token_ids=pr.candidate_token_ids,
                logprobs=pr.logprobs,
                rewards=pr.rewards,
                chosen_index=pr.chosen_index,
                sample_index=pr.sample_index,
            )
        )

    # Store position-level rewards
    try:
        session_data.set_position_rewards(interaction_id, pr_dataclasses)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    logger.info(
        f"Set position-wise rewards for {interaction_id}: "
        f"{len(position_rewards)} positions"
    )
    return {"message": "success", "interaction_id": interaction_id}


@app.post(f"/{RL_COMPUTE_ENTROPY_PATHNAME}")
def compute_entropy(
    request: ComputeEntropyRequest, session_id: str = Depends(_require_session_key)
) -> ComputeEntropyResponse:
    """Compute entropy for an interaction with position rewards."""
    interaction_id = request.interaction_id

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail=f"Session {session_id} already ended or expired"
            )
        session_data = _session_cache[session_id]

    session_data.update_last_access()

    try:
        entropies, avg_entropy = session_data.compute_entropy(interaction_id)
        return ComputeEntropyResponse(entropies=entropies, avg_entropy=avg_entropy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Cleanup Task
# =============================================================================


async def _cleanup_stale_sessions():
    """Periodically clean up stale sessions and orphaned API key mappings."""
    while True:
        await asyncio.sleep(60)  # Check every minute

        with _lock:
            stale_sessions = [
                sid
                for sid, session in _session_cache.items()
                if session.is_stale(SESSION_TIMEOUT_SECONDS)
            ]

            for sid in stale_sessions:
                logger.warning(f"Cleaning up stale session {sid}")
                session_data = _session_cache.pop(sid, None)
                if session_data is not None:
                    session_data.finish()
                _remove_api_keys_for_session(sid)

            if stale_sessions:
                logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")

            # Sweep orphaned API key mappings whose session_id is no longer
            # in cache. Handles edge case where client crashes after
            # end_session but before export_trajectories.
            orphaned_sids = [
                sid for sid in _session_to_api_key if sid not in _session_cache
            ]
            for sid in orphaned_sids:
                _remove_api_keys_for_session(sid)
            if orphaned_sids:
                logger.info(
                    f"Cleaned up {len(orphaned_sids)} orphaned API key mappings"
                )


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the token reward proxy server."""
    parser = argparse.ArgumentParser(description="Token Reward Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--admin-api-key",
        default=DEFAULT_ADMIN_API_KEY,
        help="Admin API key for management operations",
    )

    args = parser.parse_args()

    global _server_host, _server_port, _admin_api_key
    _server_host = args.host
    _server_port = args.port
    _admin_api_key = args.admin_api_key

    logger.info(f"Starting Token Reward Proxy Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
