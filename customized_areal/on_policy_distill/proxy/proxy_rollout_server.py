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
import secrets
import threading
import time

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request

# Import from areal base server
from areal.experimental.openai.proxy.server import (
    DEFAULT_ADMIN_API_KEY,
    EXPORT_TRAJECTORIES_PATHNAME,
    GRANT_CAPACITY_PATHNAME,
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
from areal.utils.logging import getLogger

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

logger = getLogger("TokenRewardProxyServer")

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


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="Token Reward Proxy Server")


# =============================================================================
# Helper Functions
# =============================================================================


def _require_admin_key(request: Request) -> str:
    """Validate admin API key from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )
    token = auth[7:]  # Remove "Bearer "
    if token != _admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin API key")
    return token


def _require_session_key(request: Request) -> str:
    """Validate session API key and return session_id."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )
    token = auth[7:]  # Remove "Bearer "

    with _lock:
        if token not in _api_key_to_session:
            raise HTTPException(status_code=403, detail="Invalid session API key")
        return _api_key_to_session[token]


def _generate_api_key() -> str:
    """Generate a unique session API key."""
    return f"tr-session-{secrets.token_urlsafe(32)}"


# =============================================================================
# Admin Endpoints
# =============================================================================


@app.post(f"/{RL_START_SESSION_PATHNAME}")
def start_session(
    request: StartSessionRequest, admin_key: str = Depends(_require_admin_key)
) -> StartSessionResponse:
    """Start a new RL session with token-level reward support."""
    import uuid

    session_id = f"tr-{uuid.uuid4().hex[:16]}"
    api_key = _generate_api_key()

    with _lock:
        _session_cache[session_id] = TokenRewardSessionData(session_id)
        _api_key_to_session[api_key] = session_id
        _session_to_api_key[session_id] = api_key

    logger.info(f"Started session {session_id} for task {request.task_id}")
    return StartSessionResponse(session_id=session_id, api_key=api_key)


@app.post(f"/{RL_END_SESSION_PATHNAME}")
def end_session(session_id: str = Depends(_require_session_key)):
    """End an RL session."""
    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail="Session already ended or expired"
            )
        session_data = _session_cache[session_id]

    session_data.finish()

    # Clean up mappings
    with _lock:
        if session_id in _session_to_api_key:
            api_key = _session_to_api_key[session_id]
            del _api_key_to_session[api_key]
            del _session_to_api_key[session_id]

    logger.info(f"Ended session {session_id}")
    return {"message": "success"}


@app.post(f"/{EXPORT_TRAJECTORIES_PATHNAME}")
def export_trajectories(
    request: ExportTrajectoriesRequest, admin_key: str = Depends(_require_admin_key)
) -> ExportTrajectoriesResponse:
    """Export trajectories for a session with token-level rewards."""
    session_id = request.session_id

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )
        session_data = _session_cache[session_id]

    # Wait for session to complete
    if not session_data.is_completed:
        # Return empty response - client should retry
        return ExportTrajectoriesResponse(interactions={})

    # Export with token-level rewards applied
    interactions = session_data.export_interactions(
        discount=request.discount,
        style=request.style,
    )

    # Serialize for HTTP response
    serialized = serialize_interactions(interactions)

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
    session_data.set_token_rewards(interaction_id, token_rewards)

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
            )
        )

    # Store position-level rewards
    session_data.set_position_rewards(interaction_id, pr_dataclasses)

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
    """Periodically clean up stale sessions."""
    while True:
        await asyncio.sleep(60)  # Check every minute

        current_time = time.time()
        with _lock:
            stale_sessions = [
                sid
                for sid, session in _session_cache.items()
                if session.is_stale(SESSION_TIMEOUT_SECONDS)
            ]

            for sid in stale_sessions:
                logger.warning(f"Cleaning up stale session {sid}")
                session_data = _session_cache[sid]
                session_data.finish()

                # Clean up mappings
                if sid in _session_to_api_key:
                    api_key = _session_to_api_key[sid]
                    del _api_key_to_session[api_key]
                    del _session_to_api_key[sid]

                del _session_cache[sid]


# =============================================================================
# Server Startup
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup."""
    asyncio.create_task(_cleanup_stale_sessions())
    logger.info(f"Token Reward Proxy Server started on {_server_host}:{_server_port}")


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
