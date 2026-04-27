"""Tests for on-policy distillation pipeline bug fixes."""

import inspect

import pytest


def test_bug1_completion_id_uses_interaction_id():
    """Bug 1: OnPolicyDistillAgent should use the interaction's actual ID
    from the proxy server, not an MD5 hash of completion_messages."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.core.agent",
            fromlist=["OnPolicyDistillAgent"],
        ).OnPolicyDistillAgent.run
    )
    assert "hashlib.md5" not in source, (
        "OnPolicyDistillAgent.run() should not use hashlib.md5 for "
        "completion_id. Use interaction.interaction_id from the proxy server."
    )
    assert "interaction_id" in source, (
        "OnPolicyDistillAgent.run() should use interaction.interaction_id "
        "from the proxy server as the completion_id."
    )


def test_bug4_stale_session_uses_configured_timeout():
    """Bug 4: _cleanup_stale_sessions should use _session_timeout_seconds
    (from config) instead of hardcoded SESSION_TIMEOUT_SECONDS."""
    source = inspect.getsource(
        __import__(
            "customized_areal.on_policy_distill.proxy.proxy_rollout_server",
            fromlist=["_cleanup_stale_sessions"],
        )._cleanup_stale_sessions
    )
    for line in source.split("\n"):
        stripped = line.strip()
        if "is_stale" in stripped and "SESSION_TIMEOUT_SECONDS" in stripped:
            pytest.fail(
                "_cleanup_stale_sessions uses hardcoded SESSION_TIMEOUT_SECONDS "
                "instead of _session_timeout_seconds. Custom timeout configs "
                "are silently ignored, causing OOM from uncleaned sessions."
            )
