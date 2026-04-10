"""Shared fixtures for proxy module tests."""

import pytest
from unittest.mock import Mock, MagicMock, patch

# Mock areal modules before importing proxy modules
with patch.dict(
    "sys.modules",
    {
        "areal": Mock(),
        "areal.utils": Mock(),
        "areal.utils.logging": Mock(),
        "areal.experimental": Mock(),
        "areal.experimental.openai": Mock(),
        "areal.experimental.openai.proxy": Mock(),
        "areal.experimental.openai.proxy.client_session": Mock(),
        "areal.experimental.openai.types": Mock(),
        "areal.infra": Mock(),
        "areal.infra.utils": Mock(),
        "areal.infra.utils.http": Mock(),
    },
):
    pass  # Fixtures defined below


@pytest.fixture
def mock_model_response():
    """Create a mock model response."""
    mock = Mock()
    mock.output_tokens = [100, 200, 300, 400, 500]
    mock.input_len = 10
    mock.output_len = 5
    mock.output_logprobs = [-0.5, -0.3, -1.2, -0.8, -0.1]
    return mock


@pytest.fixture
def mock_completion():
    """Create a mock completion."""
    mock = Mock()
    mock.id = "test-completion-id"
    return mock


@pytest.fixture
def sample_messages():
    """Create sample messages."""
    return [{"role": "user", "content": "Hello"}]


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session."""
    from unittest.mock import AsyncMock

    session = AsyncMock()
    return session
