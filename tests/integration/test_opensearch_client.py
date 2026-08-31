"""Integration tests for OpenSearch client connection factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.integrations.opensearch_client import get_opensearch_client


@pytest.fixture(autouse=True)
def reset_client() -> None:
    """Reset the module-level singleton between tests."""
    import src.integrations.opensearch_client as mod

    mod._client = None


@patch("src.integrations.opensearch_client.get_settings")
@patch("src.integrations.opensearch_client.OpenSearch")
def test_returns_client_singleton(
    mock_os_cls: MagicMock, mock_settings: MagicMock
    ) -> None:
    mock_settings.return_value.opensearch_endpoint = "http://10.0.3.100:9200"

    client1 = get_opensearch_client()
    client2 = get_opensearch_client()

    assert client1 is client2
    mock_os_cls.assert_called_once()


@patch("src.integrations.opensearch_client.get_settings")
@patch("src.integrations.opensearch_client.OpenSearch")
def test_client_configured_with_settings(
    mock_os_cls: MagicMock, mock_settings: MagicMock
    ) -> None:
    mock_settings.return_value.opensearch_endpoint = "http://10.0.3.100:9200"

    get_opensearch_client()

    call_kwargs = mock_os_cls.call_args
    hosts = call_kwargs.kwargs.get("hosts") or call_kwargs[1].get("hosts")
    assert hosts[0]["host"] == "10.0.3.100"
    assert hosts[0]["port"] == 9200
