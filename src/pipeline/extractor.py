# FILE: cloudsentinel-zero-trust/src/pipeline/extractor.py
"""CloudTrail S3 event extraction: download, decompress, parse, validate.

Design decisions:
- Streaming download (no temp files): Lambda has limited /tmp space
- gzip decompression in-memory via io.BytesIO
- Pydantic validation on each event record (fail-fast on individual events, not batch)
- Digest files (.json.gz ending in Digest/) silently skipped
"""

from __future__ import annotations

import gzip
import json
import time
from typing import Any

from botocore.exceptions import ClientError

from src.models.cloudtrail_event import CloudTrailEvent
from src.utils.exceptions import ExtractionError
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="extractor")


class CloudTrailExtractor:
    """Downloads, decompresses, and parses CloudTrail log files from S3."""

    def __init__(self, s3_client: Any) -> None:
        self._s3 = s3_client

    def extract(self, bucket: str, key: str) -> list[CloudTrailEvent]:
        """Download and parse a single CloudTrail log file from S3.

        Args:
            bucket: S3 bucket name.
            key: S3 object key (e.g. AWSLogs/123456/CloudTrail/us-east-1/...).

        Returns:
            List of validated CloudTrailEvent objects.

        Raises:
            ExtractionError: On download, decompression, or parse failure.
        """
        start = time.monotonic()

        # Skip digest files — they contain checksums, not events
        if "/CloudTrail-Digest/" in key or key.endswith("Digest.json.gz"):
            logger.info("Skipping digest file: %s", key)
            return []

        logger.info("Extracting CloudTrail events", bucket=bucket, key=key)

        try:
            raw_bytes = self._download(bucket, key)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(
                f"Failed to download s3://{bucket}/{key}: {exc}",
                context={"bucket": bucket, "key": key},
            ) from exc

        try:
            json_bytes = self._decompress(raw_bytes, key)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(
                f"Failed to decompress {key}: {exc}",
                context={"bucket": bucket, "key": key},
            ) from exc

        try:
            events = self._parse(json_bytes, key)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(
                f"Failed to parse JSON in {key}: {exc}",
                context={"bucket": bucket, "key": key},
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Extraction complete: %d events in %.1fms (%.1f KB compressed)",
            len(events),
            elapsed_ms,
            len(raw_bytes) / 1024,
        )
        return events

    def extract_batch(
        self, records: list[dict[str, Any]]
    ) -> list[CloudTrailEvent]:
        """Extract events from multiple S3 event notification records.

        Args:
            records: List of S3 event records from Lambda event['Records'].

        Returns:
            Combined list of CloudTrailEvent from all records.
        """
        all_events: list[CloudTrailEvent] = []
        for record in records:
            s3_info = record.get("s3", {})
            bucket = s3_info.get("bucket", {}).get("name", "")
            key = s3_info.get("object", {}).get("key", "")
            if not bucket or not key:
                logger.warning("Skipping record with missing bucket/key: %s", record)
                continue
            try:
                events = self.extract(bucket, key)
                all_events.extend(events)
            except ExtractionError as exc:
                logger.error("Extraction failed for s3://%s/%s: %s", bucket, key, exc.message)
                # Continue processing other records; don't let one bad file kill the batch
        return all_events


    def _download(self, bucket: str, key: str) -> bytes:
        """Download S3 object as bytes (streaming, no temp file)."""
        try:
            response = self._s3.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read()
            return body
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ("NoSuchKey", "NoSuchBucket"):
                raise ExtractionError(
                    f"S3 object not found: s3://{bucket}/{key}",
                    context={"bucket": bucket, "key": key, "error_code": error_code},
                ) from exc
            if error_code == "AccessDenied":
                raise ExtractionError(
                    f"Access denied to s3://{bucket}/{key}. Check IAM permissions.",
                    context={"bucket": bucket, "key": key, "error_code": error_code},
                ) from exc
            raise ExtractionError(
                f"S3 error ({error_code}): {exc}",
                context={"bucket": bucket, "key": key},
            ) from exc

    def _decompress(self, raw_bytes: bytes, key: str) -> bytes:
        """Decompress gzip bytes in memory."""
        if key.endswith(".gz"):
            try:
                return gzip.decompress(raw_bytes)
            except (gzip.BadGzipFile, OSError) as exc:
                raise ExtractionError(
                    f"Corrupt gzip file: {key} — {exc}",
                    context={"key": key, "compressed_size": len(raw_bytes)},
                ) from exc
        # Non-gzip files returned as-is
        return raw_bytes

    def _parse(self, json_bytes: bytes, key: str) -> list[CloudTrailEvent]:
        """Parse JSON and validate each CloudTrail event record."""
        try:
            data = json.loads(json_bytes)
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"Invalid JSON in {key}: {exc}",
                context={"key": key, "json_size": len(json_bytes)},
            ) from exc

        if not isinstance(data, dict) or "Records" not in data:
            raise ExtractionError(
                f"Missing 'Records' key in CloudTrail JSON: {key}",
                context={"key": key, "keys_found": list(data.keys()) if isinstance(data, dict) else "not-a-dict"},
            )

        raw_records = data["Records"]
        if not isinstance(raw_records, list):
            raise ExtractionError(
                f"'Records' is not a list in {key}",
                context={"key": key},
            )

        if len(raw_records) == 0:
            logger.info("Empty Records array in %s", key)
            return []

        events: list[CloudTrailEvent] = []
        for idx, raw in enumerate(raw_records):
            try:
                event = CloudTrailEvent.model_validate(raw)
                events.append(event)
            except Exception as exc:
                # Log and skip individual malformed events; don't fail the batch
                logger.warning(
                    "Failed to validate event #%d in %s: %s",
                    idx,
                    key,
                    str(exc)[:200],
                )
        return events
