# FILE: cloudsentinel-zero-trust/src/pipeline/ingester.py
"""OpenSearch bulk ingester with retry, circuit breaker, and daily rolling indices.

Design:
- Index pattern: cloudsentinel-events-YYYY.MM.DD
- Bulk API: batches of 500 documents
- Retry: exponential backoff (base 1s, max 30s, 3 attempts)
- Circuit breaker: opens after 5 consecutive failures, pauses 60s
- Thread-safe via _lock for circuit breaker state
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from src.models.ecs_event import ECSCloudTrailEvent
from src.utils.exceptions import BulkIngestionError, IngestionError
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="ingester")

# Ingestion constants
BULK_BATCH_SIZE = 500
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 30.0
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60
INDEX_PREFIX = "cloudsentinel-events"


class CircuitBreaker:
    """Simple circuit breaker to prevent hammering a failed OpenSearch."""

    def __init__(
        self,
        threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        cooldown: float = CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    ) -> None:
        self._threshold = threshold
        self._cooldown = cooldown
        self._failures = 0
        self._open_until: float = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._open_until > 0 and time.monotonic() < self._open_until:
                return True
            if self._open_until > 0 and time.monotonic() >= self._open_until:
                # Half-open → allow traffic, reset
                self._open_until = 0.0
                self._failures = 0
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._open_until = time.monotonic() + self._cooldown
                logger.error(
                    "Circuit breaker OPEN — %d consecutive failures. "
                    "Pausing ingestion for %ds.",
                    self._failures,
                    int(self._cooldown),
                )


class OpenSearchIngester:
    """Ingests enriched ECS events into OpenSearch via the Bulk API.

    Parameters:
        client: opensearchpy.OpenSearch client instance (injected for testability).
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._cb = CircuitBreaker()


    def ingest(self, events: list[ECSCloudTrailEvent]) -> dict[str, int]:
        """Ingest a list of events into daily rolling indices.

        Returns:
            dict with keys 'success_count' and 'failure_count'.
        """
        if not events:
            return {"success_count": 0, "failure_count": 0}

        success_total = 0
        failure_total = 0
        failed_items: list[dict[str, Any]] = []

        # Chunk into batches
        for i in range(0, len(events), BULK_BATCH_SIZE):
            batch = events[i : i + BULK_BATCH_SIZE]
            batch_result = self._ingest_batch(batch)
            success_total += batch_result["success"]
            failure_total += batch_result["failure"]
            failed_items.extend(batch_result.get("failed_items", []))

        if failure_total > 0:
            logger.warning(
                "Ingestion completed with failures: %d/%d succeeded",
                success_total,
                success_total + failure_total,
            )
            raise BulkIngestionError(
                message=f"Bulk ingestion: {failure_total} failures out of {success_total + failure_total}",
                failed_items=failed_items,
                success_count=success_total,
                failure_count=failure_total,
            )

        logger.info("Ingestion complete: %d documents indexed", success_total)
        return {"success_count": success_total, "failure_count": 0}


    def _ingest_batch(self, batch: list[ECSCloudTrailEvent]) -> dict[str, Any]:
        """Send a single batch via Bulk API with retries."""
        body = self._build_bulk_body(batch)

        for attempt in range(1, MAX_RETRIES + 1):
            if self._cb.is_open:
                raise IngestionError(
                    "Circuit breaker is open — ingestion paused",
                    context={"cooldown_s": CIRCUIT_BREAKER_COOLDOWN_SECONDS},
                )

            try:
                response = self._client.bulk(body=body, timeout="60s")
                return self._parse_bulk_response(response, len(batch))
            except IngestionError:
                raise
            except Exception as exc:
                self._cb.record_failure()
                sleep_time = min(
                    RETRY_MAX_SECONDS,
                    RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
                )
                logger.warning(
                    "Bulk ingest attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    sleep_time,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(sleep_time)

        raise IngestionError(
            f"Bulk ingest failed after {MAX_RETRIES} retries",
            context={"batch_size": len(batch)},
        )

    def _build_bulk_body(self, batch: list[ECSCloudTrailEvent]) -> str:
        """Build NDJSON body for _bulk API.

        Each event gets two lines:
          {"index": {"_index": "cloudsentinel-events-2024.01.15"}}
          {...document...}
        """
        lines: list[str] = []
        for event in batch:
            index_name = self._resolve_index(event)
            action = json.dumps({"index": {"_index": index_name}})
            document = json.dumps(event.to_opensearch_dict(), default=str)
            lines.append(action)
            lines.append(document)
        # Bulk API requires trailing newline
        return "\n".join(lines) + "\n"

    def _parse_bulk_response(
        self, response: dict[str, Any], expected: int
    ) -> dict[str, Any]:
        """Parse OpenSearch _bulk response and track successes/failures."""
        items = response.get("items", [])

        success = 0
        failure = 0
        failed_items: list[dict[str, Any]] = []

        for item in items:
            action_result = item.get("index", item.get("create", {}))
            status = action_result.get("status", 500)
            if 200 <= status < 300:
                success += 1
                self._cb.record_success()
            else:
                failure += 1
                self._cb.record_failure()
                failed_items.append({
                    "status": status,
                    "error": action_result.get("error", {}),
                    "id": action_result.get("_id"),
                })

        if success > 0:
            self._cb.record_success()

        return {
            "success": success,
            "failure": failure,
            "failed_items": failed_items,
        }

    @staticmethod
    def _resolve_index(event: ECSCloudTrailEvent) -> str:
        """Determine the daily rolling index name from the event timestamp."""
        ts = event.base.timestamp
        if ts:
            date_suffix = ts.strftime("%Y.%m.%d")
        else:
            date_suffix = datetime.now(timezone.utc).strftime("%Y.%m.%d")
        return f"{INDEX_PREFIX}-{date_suffix}"
