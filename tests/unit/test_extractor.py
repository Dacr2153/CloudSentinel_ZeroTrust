# FILE: cloudsentinel-zero-trust/tests/unit/test_extractor.py
"""Unit tests for CloudTrailExtractor — S3 download, decompression, parsing."""

from __future__ import annotations

import gzip
import json
from typing import Any

import pytest

from src.pipeline.extractor import CloudTrailExtractor
from src.utils.exceptions import ExtractionError

TEST_BUCKET = "cloudsentinel-logs-123456789012"
TEST_KEY = "AWSLogs/123456789012/CloudTrail/us-east-1/2026/03/10/test.json.gz"


def test_extract_valid_cloudtrail_gzip(mock_s3_client: Any) -> None:
    """Extract events from a correctly formatted gzip CloudTrail file."""
    extractor = CloudTrailExtractor(s3_client=mock_s3_client)
    events = extractor.extract(TEST_BUCKET, TEST_KEY)

    assert len(events) == 3
    assert events[0].eventName == "DescribeInstances"
    assert events[1].eventName == "GetObject"
    assert events[2].eventName == "ListUsers"
    assert events[0].eventID == "evt-s3-test-001"
    assert events[0].userIdentity.type == "IAMUser"
    assert events[0].awsRegion == "us-east-1"


def test_extract_handles_empty_records(mock_s3_client: Any) -> None:
    """Returns empty list when CloudTrail file has zero records."""
    empty_data = json.dumps({"Records": []}).encode()
    gzip_bytes = gzip.compress(empty_data)
    mock_s3_client.put_object(
    Bucket=TEST_BUCKET,
    Key="AWSLogs/empty.json.gz",
    Body=gzip_bytes,
    )

    extractor = CloudTrailExtractor(s3_client=mock_s3_client)
    events = extractor.extract(TEST_BUCKET, "AWSLogs/empty.json.gz")

    assert events == []


def test_extract_handles_corrupt_gzip(mock_s3_client: Any) -> None:
    """Raises ExtractionError for corrupt gzip data."""
    mock_s3_client.put_object(
    Bucket=TEST_BUCKET,
    Key="AWSLogs/corrupt.json.gz",
    Body=b"this is not valid gzip data at all",
    )

    extractor = CloudTrailExtractor(s3_client=mock_s3_client)
    with pytest.raises(ExtractionError, match="(?i)corrupt|decompress|gzip"):
        extractor.extract(TEST_BUCKET, "AWSLogs/corrupt.json.gz")


def test_extract_handles_digest_file(mock_s3_client: Any) -> None:
    """Digest files (checksums, not events) should return empty list."""
    extractor = CloudTrailExtractor(s3_client=mock_s3_client)

    # Key containing CloudTrail-Digest
    events = extractor.extract(
    TEST_BUCKET,
    "AWSLogs/123456789012/CloudTrail-Digest/us-east-1/digest.json.gz",
    )
    assert events == []


def test_extract_batch_count(mock_s3_client: Any) -> None:
    """extract_batch() processes multiple S3 records and returns combined events."""
    # Upload a second file with 2 events
    second_records = {
    "Records": [
        {
            "eventVersion": "1.08",
            "eventTime": "2026-03-10T15:00:00Z",
            "eventSource": "s3.amazonaws.com",
            "eventName": "PutObject",
            "eventType": "AwsApiCall",
            "eventID": "evt-batch-001",
            "awsRegion": "us-east-1",
            "sourceIPAddress": "10.0.1.50",
            "userIdentity": {
                "type": "IAMUser",
                "principalId": "AIDAEXAMPLE",
                "arn": "arn:aws:iam::123456789012:user/dev",
                "accountId": "123456789012",
                "userName": "dev",
            },
            "requestParameters": None,
            "responseElements": None,
            "readOnly": False,
            "managementEvent": False,
            "recipientAccountId": "123456789012",
        },
        {
            "eventVersion": "1.08",
            "eventTime": "2026-03-10T15:01:00Z",
            "eventSource": "s3.amazonaws.com",
            "eventName": "DeleteObject",
            "eventType": "AwsApiCall",
            "eventID": "evt-batch-002",
            "awsRegion": "us-east-1",
            "sourceIPAddress": "10.0.1.50",
            "userIdentity": {
                "type": "IAMUser",
                "principalId": "AIDAEXAMPLE",
                "arn": "arn:aws:iam::123456789012:user/dev",
                "accountId": "123456789012",
                "userName": "dev",
            },
            "requestParameters": None,
            "responseElements": None,
            "readOnly": False,
            "managementEvent": False,
            "recipientAccountId": "123456789012",
        },
    ]
    }

    json_bytes = json.dumps(second_records).encode()
    gzip_bytes = gzip.compress(json_bytes)
    mock_s3_client.put_object(
    Bucket=TEST_BUCKET,
    Key="AWSLogs/123456789012/CloudTrail/us-east-1/2026/03/10/second.json.gz",
    Body=gzip_bytes,
    )

    # Build S3 event notification records (Lambda trigger format)
    s3_records = [
    {
        "s3": {
            "bucket": {"name": TEST_BUCKET},
            "object": {"key": "AWSLogs/123456789012/CloudTrail/us-east-1/2026/03/10/test.json.gz"},
        }
    },
    {
        "s3": {
            "bucket": {"name": TEST_BUCKET},
            "object": {"key": "AWSLogs/123456789012/CloudTrail/us-east-1/2026/03/10/second.json.gz"},
        }
    },
    ]

    extractor = CloudTrailExtractor(s3_client=mock_s3_client)
    all_events = extractor.extract_batch(s3_records)

    # First file has 3 events, second has 2 = 5 total
    assert len(all_events) == 5
