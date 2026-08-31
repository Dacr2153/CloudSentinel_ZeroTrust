# FILE: cloudsentinel-zero-trust/src/utils/exceptions.py
"""Custom exception hierarchy for CloudSentinel pipeline.

Design decisions:
- Every exception carries context dict + correlation_id for structured logging
- Hierarchy mirrors pipeline stages for granular error handling
- BulkIngestionError tracks partial failures (some events succeed, some fail)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class CloudSentinelError(Exception):
    """Base exception for all CloudSentinel errors.

    Attributes:
        message: Human-readable error description.
        context: Dictionary with additional error metadata.
        correlation_id: Pipeline correlation ID for log correlation.
        timestamp: UTC timestamp when the error occurred.
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        correlation_id: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.correlation_id = correlation_id
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception for structured logging / DLQ messages."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }


class ExtractionError(CloudSentinelError):
    """Failure reading S3 objects or parsing CloudTrail JSON.

    Raised when:
    - S3 GetObject fails (permission, not found, network)
    - Gzip decompression fails (corrupt file)
    - JSON parse fails (invalid structure)
    - CloudTrail Records array is missing
    """

    pass


class NormalizationError(CloudSentinelError):
    """Failure mapping CloudTrail raw event to ECS 8.10+ schema.

    Raised when:
    - Required fields are missing from raw event
    - Pydantic validation fails on ECS model
    - Unknown userIdentity type encountered
    """

    pass


class IngestionError(CloudSentinelError):
    """Failure writing events to OpenSearch.

    Raised when:
    - OpenSearch cluster is unreachable
    - Authentication/authorization fails
    - Index does not exist and auto-create is disabled
    """

    pass


class BulkIngestionError(IngestionError):
    """Partial failure during OpenSearch bulk API call.

    Some events were indexed successfully, others failed.

    Attributes:
        failed_items: List of dicts describing each failed document.
        success_count: Number of successfully indexed documents.
        failure_count: Number of failed documents.
    """

    def __init__(
        self,
        message: str,
        failed_items: list[dict[str, Any]] | None = None,
        success_count: int = 0,
        failure_count: int = 0,
        context: dict[str, Any] | None = None,
        correlation_id: str = "",
    ) -> None:
        super().__init__(message, context=context, correlation_id=correlation_id)
        self.failed_items = failed_items or []
        self.success_count = success_count
        self.failure_count = failure_count

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "failed_items": self.failed_items[:10],  # Cap at 10 to avoid huge payloads
            }
        )
        return base


class ModelError(CloudSentinelError):
    """Failure in ML model operations (load, inference, scoring)."""

    pass


class ModelNotFoundError(ModelError):
    """Model artifact not found in S3 at expected key."""

    pass


class ModelInferenceError(ModelError):
    """Failure during model.predict() or score computation.

    Raised when:
    - Feature array has wrong shape
    - NaN/Inf values in features
    - scikit-learn internal error
    """

    pass


class AlertError(CloudSentinelError):
    """Failure sending alert via SNS or storing in OpenSearch alerts index."""

    pass
