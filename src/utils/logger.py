# FILE: cloudsentinel-zero-trust/src/utils/logger.py
"""Structured JSON logging with correlation IDs for CloudSentinel pipeline.

Design decisions:
- Uses stdlib logging (no external deps, Lambda-friendly)
- JSON output for CloudWatch Insights querying
- correlation_id propagated from Lambda context or auto-generated
- Thread-safe context enrichment via add_context()
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for CloudWatch Insights."""

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "function_name": record.funcName,
            "module": record.module,
            "message": record.getMessage(),
        }

        # Correlation ID and AWS request ID are stored on the record by CloudSentinelLogger
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id is not None:
            log_entry["correlation_id"] = correlation_id
        aws_request_id = getattr(record, "aws_request_id", None)
        if aws_request_id is not None:
            log_entry["aws_request_id"] = aws_request_id

        # Extra context dict attached by add_context()
        context_data = getattr(record, "context_data", None)
        if context_data is not None:
            log_entry["context"] = context_data

        # Exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Exception",
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class CloudSentinelLogger:
    """Thread-safe structured JSON logger with correlation ID tracking.

    Usage:
        logger = CloudSentinelLogger(service="pipeline")
        logger.set_correlation_id("abc-123")
        logger.add_context(event_count=42, bucket="logs-bucket")
        logger.info("Processing events")
    """

    def __init__(
        self,
        service: str = "cloudsentinel",
        level: str | None = None,
    ) -> None:
        self._service = service
        self._correlation_id: str = str(uuid.uuid4())
        self._aws_request_id: str = ""
        self._context: dict[str, Any] = {}
        self._lock = threading.Lock()

        resolved_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()

        self._logger = logging.getLogger(f"cloudsentinel.{service}")
        self._logger.setLevel(getattr(logging, resolved_level, logging.INFO))
        self._logger.propagate = False

        # Avoid duplicate handlers on warm Lambda starts
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_JSONFormatter(service_name=service))
            self._logger.addHandler(handler)


    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID (typically from Lambda context.aws_request_id)."""
        with self._lock:
            self._correlation_id = correlation_id

    def set_aws_request_id(self, request_id: str) -> None:
        """Set AWS request ID from Lambda context."""
        with self._lock:
            self._aws_request_id = request_id

    def add_context(self, **kwargs: Any) -> None:
        """Enrich all subsequent log entries with additional key-value pairs."""
        with self._lock:
            self._context.update(kwargs)

    def clear_context(self) -> None:
        """Reset extra context (e.g., between Lambda invocations)."""
        with self._lock:
            self._context.clear()

    @property
    def correlation_id(self) -> str:
        return self._correlation_id


    _LOGGING_KWARGS = frozenset({"exc_info", "stack_info", "stacklevel"})

    def _log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        log_kwargs = {k: v for k, v in kwargs.items() if k in self._LOGGING_KWARGS}
        context_kwargs = {k: v for k, v in kwargs.items() if k not in self._LOGGING_KWARGS}
        extra = {
            "correlation_id": self._correlation_id,
            "aws_request_id": self._aws_request_id,
            "context_data": {**self._context, **context_kwargs},
        }
        self._logger.log(level, msg, *args, extra=extra, **log_kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log ERROR with exception traceback (must be called inside except block)."""
        kwargs["exc_info"] = kwargs.get("exc_info", True)
        self._log(logging.ERROR, msg, *args, **kwargs)
