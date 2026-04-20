"""Tests for client module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

# Mock the areal imports before importing the module
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
        "areal.infra": Mock(),
        "areal.infra.utils": Mock(),
        "areal.infra.utils.http": Mock(),
    },
):
    from customized_areal.on_policy_distill.proxy.client import OpenAIProxyClient


class TestOpenAIProxyClient:
    """Test OpenAIProxyClient class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def client(self, mock_session):
        """Create OpenAIProxyClient for testing."""
        with patch(
            "customized_areal.on_policy_distill.proxy.client.BaseOpenAIProxyClient.__init__",
            return_value=None,
        ):
            with patch(
                "customized_areal.on_policy_distill.proxy.client.ensure_end_with_slash",
                return_value="http://localhost:8000/",
            ):
                client = OpenAIProxyClient(
                    session=mock_session,
                    base_url="http://localhost:8000",
                    task_id="test-task",
                    admin_api_key="test-admin-key",
                )
                # Mock parent attributes
                client.session_id = "test-session-id"
                client.base_url = "http://localhost:8000/"
                client._session = mock_session
                return client

    @pytest.mark.asyncio
    async def test_set_rewards_success(self, client, mock_session):
        """Test successful set_rewards call."""
        # Mock the response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = Mock()
        mock_response.json = AsyncMock(return_value={"message": "success"})

        mock_session.post = AsyncMock(return_value=mock_response)

        # Mock parent methods
        client._session_auth_headers = Mock(
            return_value={"Authorization": "Bearer token"}
        )
        client.session_id = "test-session-id"

        with patch(
            "customized_areal.on_policy_distill.proxy.client.post_json_with_retry",
            new_callable=AsyncMock,
        ) as mock_post:
            await client.set_rewards("comp-1", [0.1, 0.2, 0.3])
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_rewards_not_started(self, client):
        """Test set_rewards when session not started raises error."""
        client.session_id = None

        with pytest.raises(RuntimeError, match="Session not started"):
            await client.set_rewards("comp-1", [0.1, 0.2, 0.3])

    @pytest.mark.asyncio
    async def test_set_position_rewards_success(self, client):
        """Test successful set_position_rewards call."""
        with patch(
            "customized_areal.on_policy_distill.proxy.client.post_json_with_retry",
            new_callable=AsyncMock,
        ) as mock_post:
            position_rewards = [
                Mock(
                    position=0,
                    candidates=["a", "b"],
                    rewards=[0.1, 0.5],
                    chosen_index=1,
                ),
                Mock(
                    position=1,
                    candidates=["c", "d"],
                    rewards=[0.2, 0.6],
                    chosen_index=0,
                ),
            ]

            await client.set_position_rewards("comp-1", position_rewards)
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_last_rewards(self, client):
        """Test set_last_rewards method."""
        with patch.object(client, "set_rewards", new_callable=AsyncMock) as mock_set:
            await client.set_last_rewards([0.1, 0.2, 0.3])
            mock_set.assert_called_once_with(
                completion_id="", token_rewards=[0.1, 0.2, 0.3]
            )

    @pytest.mark.asyncio
    async def test_compute_entropy_success(self, client, mock_session):
        """Test successful compute_entropy call."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = Mock()
        mock_response.json = AsyncMock(return_value={"entropies": [0.5, 0.6, 0.7]})

        mock_session.post = AsyncMock(return_value=mock_response)

        client.session_id = "test-session-id"

        # Mock _session_auth_headers
        client._session_auth_headers = Mock(
            return_value={"Authorization": "Bearer token"}
        )

        entropies = await client.compute_entropy("comp-1")
        assert entropies == [0.5, 0.6, 0.7]

    @pytest.mark.asyncio
    async def test_compute_entropy_not_started(self, client):
        """Test compute_entropy when session not started raises error."""
        client.session_id = None

        with pytest.raises(RuntimeError, match="Session not started"):
            await client.compute_entropy("comp-1")

    @pytest.mark.asyncio
    async def test_get_entropies_success(self, client):
        """Test successful get_entropies call."""
        with patch.object(
            client, "compute_entropy", new_callable=AsyncMock
        ) as mock_compute:
            mock_compute.return_value = [0.5, 0.6, 0.7]

            entropies = await client.get_entropies("comp-1")
            assert entropies == [0.5, 0.6, 0.7]

    @pytest.mark.asyncio
    async def test_get_entropies_failure(self, client):
        """Test get_entropies when compute_entropy fails returns None."""
        with patch.object(
            client, "compute_entropy", new_callable=AsyncMock
        ) as mock_compute:
            mock_compute.side_effect = Exception("Compute failed")

            entropies = await client.get_entropies("comp-1")
            assert entropies is None
