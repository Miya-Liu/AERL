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
