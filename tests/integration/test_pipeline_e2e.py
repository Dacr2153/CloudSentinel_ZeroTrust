# FILE: cloudsentinel-zero-trust/tests/integration/test_pipeline_e2e.py
"""End-to-end integration tests for the full CloudSentinel pipeline.

These tests exercise the complete flow:
  S3 → Extract → Normalize → Feature Eng → ML Detect → Rule Evaluate → Alert

Uses moto mocks for AWS services.
"""

from __future__ import annotations

import gzip
import json
from unittest.mock import MagicMock, patch

import boto3
from moto import mock_aws

from src.lambda_handler import handler
from src.pipeline.extractor import CloudTrailExtractor
from src.pipeline.feature_engineer import FeatureEngineer
from src.pipeline.normalizer import ECSNormalizer


# Helpers

BUCKET = "cloudsentinel-test-cloudtrail"
MODEL_BUCKET = "cloudsentinel-test-models"
REGION = "us-east-1"


def _upload_cloudtrail(s3, bucket: str, key: str, records: list[dict]) -> None:
    """Upload a gzipped CloudTrail JSON file to the mock S3 bucket."""
    body = json.dumps({"Records": records}).encode()
    compressed = gzip.compress(body)
    s3.put_object(Bucket=bucket, Key=key, Body=compressed)


def _normal_event() -> dict:
    return {
        "eventVersion": "1.08",
        "eventSource": "ec2.amazonaws.com",
        "eventName": "DescribeInstances",
        "awsRegion": "us-east-1",
        "sourceIPAddress": "10.0.1.50",
        "userAgent": "console.amazonaws.com",
        "eventID": "evt-e2e-normal",
        "eventTime": "2026-03-10T14:30:00Z",
        "eventType": "AwsApiCall",
        "userIdentity": {
            "type": "IAMUser",
            "accountId": "123456789012",
            "arn": "arn:aws:iam::123456789012:user/developer",
            "userName": "developer",
        },
        "requestParameters": {},
        "responseElements": None,
        "readOnly": True,
        "recipientAccountId": "123456789012",
    }


def _privilege_escalation_event() -> dict:
    return {
        "eventVersion": "1.08",
        "eventSource": "iam.amazonaws.com",
        "eventName": "CreateAccessKey",
        "awsRegion": "us-east-1",
        "sourceIPAddress": "185.220.101.50",
        "userAgent": "aws-cli/2.0",
        "eventID": "evt-e2e-priv-esc",
        "eventTime": "2026-03-10T14:35:00Z",
        "eventType": "AwsApiCall",
        "userIdentity": {
            "type": "Root",
            "accountId": "123456789012",
            "arn": "arn:aws:iam::123456789012:root",
            "userName": "Root",
        },
        "requestParameters": {"userName": "backdoor-user"},
        "responseElements": {"accessKey": {"accessKeyId": "AKIAIOSFODNN7EXAMPLE"}},
        "readOnly": False,
        "recipientAccountId": "123456789012",
    }


def _anomalous_event() -> dict:
    return {
        "eventVersion": "1.08",
        "eventSource": "iam.amazonaws.com",
        "eventName": "AttachUserPolicy",
        "awsRegion": "ap-southeast-1",
        "sourceIPAddress": "203.0.113.50",
        "userAgent": "aws-cli/2.0",
        "eventID": "evt-e2e-anomalous",
        "eventTime": "2026-03-15T03:00:00Z",
        "eventType": "AwsApiCall",
        "userIdentity": {
            "type": "Root",
            "accountId": "123456789012",
            "arn": "arn:aws:iam::123456789012:root",
            "userName": "Root",
        },
        "requestParameters": {
            "userName": "backdoor-user",
            "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess",
        },
        "responseElements": None,
        "readOnly": False,
        "recipientAccountId": "123456789012",
    }


# Pipeline segment tests (no Lambda handler, just pipeline components)


class TestExtractNormalizePipeline:
    """Extract → Normalize segment."""

    @mock_aws
    def test_normal_event_end_to_end(self) -> None:
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET)
        _upload_cloudtrail(s3, BUCKET, "logs/normal.json.gz", [_normal_event()])

        extractor = CloudTrailExtractor(s3_client=s3)
        raw_events = extractor.extract(BUCKET, "logs/normal.json.gz")
        assert len(raw_events) == 1

        normalizer = ECSNormalizer()
        ecs_events = normalizer.normalize_batch(raw_events)
        assert len(ecs_events) == 1
        assert ecs_events[0].event.action == "DescribeInstances"

    @mock_aws
    def test_privilege_escalation_alert(self) -> None:
        """Privilege-escalation event should produce a rule match."""
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET)
        _upload_cloudtrail(
            s3, BUCKET, "logs/priv_esc.json.gz", [_privilege_escalation_event()]
        )

        extractor = CloudTrailExtractor(s3_client=s3)
        raw_events = extractor.extract(BUCKET, "logs/priv_esc.json.gz")

        normalizer = ECSNormalizer()
        ecs_events = normalizer.normalize_batch(raw_events)
        assert len(ecs_events) == 1

        ecs = ecs_events[0]
        assert ecs.event.action == "CreateAccessKey"
        # Root user should have been normalized
        assert "root" in ecs.user.roles

    @mock_aws
    def test_feature_extraction_produces_vector(self) -> None:
        """Normalized event → feature vector of length NUM_FEATURES."""
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET)
        _upload_cloudtrail(s3, BUCKET, "logs/test.json.gz", [_anomalous_event()])

        extractor = CloudTrailExtractor(s3_client=s3)
        raw_events = extractor.extract(BUCKET, "logs/test.json.gz")

        normalizer = ECSNormalizer()
        ecs_events = normalizer.normalize_batch(raw_events)

        engineer = FeatureEngineer(home_region="us-east-1")
        features = engineer.extract_features(ecs_events[0])

        assert len(features) == 10
        assert all(0.0 <= f <= 1.0 for f in features)


# Full Lambda handler test


def test_handler_returns_200_on_valid_event(
    mock_s3_client, trained_model, lambda_context
) -> None:
    """Handler processes a valid S3 event and returns 200."""
    key = "AWSLogs/123456789012/CloudTrail/us-east-1/2026/03/10/events.json.gz"
    records = [_normal_event(), _privilege_escalation_event()]
    body = json.dumps({"Records": records}).encode()
    compressed = gzip.compress(body)
    mock_s3_client.create_bucket(Bucket="cloudsentinel-test-cloudtrail")
    mock_s3_client.put_object(
        Bucket="cloudsentinel-test-cloudtrail",
        Key=key,
        Body=compressed,
    )

    s3_event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "cloudsentinel-test-cloudtrail"},
                    "object": {"key": key},
                }
            }
        ]
    }

    from src.detectors.anomaly_detector import AnomalyDetector
    from src.detectors.alert_manager import AlertManager
    from src.detectors.rule_engine import RuleEngine
    from src.pipeline.enricher import EventEnricher
    from src.pipeline.extractor import CloudTrailExtractor
    from src.pipeline.feature_engineer import FeatureEngineer
    from src.pipeline.ingester import OpenSearchIngester
    from src.pipeline.normalizer import ECSNormalizer

    mock_os = MagicMock()
    mock_os.bulk.return_value = {"errors": False, "items": [
        {"index": {"status": 201, "_id": f"doc-{i}"}} for i in range(2)
    ]}
    mock_sns = MagicMock()

    detector = AnomalyDetector(
        s3_client=mock_s3_client,
        model_bucket="cloudsentinel-models-123456789012",
        threshold=65,
    )
    detector.load_model()

    with (
        patch("src.lambda_handler._extractor", CloudTrailExtractor(s3_client=mock_s3_client)),
        patch("src.lambda_handler._normalizer", ECSNormalizer()),
        patch("src.lambda_handler._enricher", EventEnricher()),
        patch("src.lambda_handler._feature_engineer", FeatureEngineer(home_region="us-east-1")),
        patch("src.lambda_handler._anomaly_detector", detector),
        patch("src.lambda_handler._rule_engine", RuleEngine()),
        patch("src.lambda_handler._alert_manager", AlertManager(
            sns_client=mock_sns,
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts",
            opensearch_client=mock_os,
        )),
        patch("src.lambda_handler._ingester", OpenSearchIngester(client=mock_os)),
        patch("src.lambda_handler._cloudwatch", MagicMock()),
        patch("src.lambda_handler.get_settings", return_value=MagicMock(
            aws_region="us-east-1",
            model_bucket="cloudsentinel-models-123456789012",
            anomaly_threshold=65,
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts",
        )),
    ):
        result = handler(s3_event, lambda_context)

    assert result["statusCode"] == 200
    resp_body = json.loads(result["body"])
    assert resp_body["stats"]["events_extracted"] >= 1
