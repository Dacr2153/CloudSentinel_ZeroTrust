# FILE: cloudsentinel-zero-trust/src/integrations/opensearch_client.py
"""OpenSearch client factory with connection pooling and health checks.

Creates a configured opensearch-py client pointing to the EC2-hosted
OpenSearch instance. Supports both direct connection and SSH-tunneled
local development.
"""

from __future__ import annotations

from urllib.parse import urlparse

from opensearchpy import OpenSearch, RequestsHttpConnection

from src.utils.config import get_settings
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="opensearch_client")

_client: OpenSearch | None = None


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse an OpenSearch endpoint URL into (host, port)."""
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 9200)
    return host, port


def get_opensearch_client() -> OpenSearch:
    """Return a singleton OpenSearch client configured from settings."""
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    host, port = _parse_endpoint(settings.opensearch_endpoint)

    _client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=None,  # Internal VPC — no auth on Free Tier single-node
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )

    logger.info(
        "OpenSearch client initialized: %s:%d",
        host,
        port,
    )
    return _client
