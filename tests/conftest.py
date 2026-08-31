# FILE: cloudsentinel-zero-trust/tests/conftest.py
"""Global test fixtures for CloudSentinel test suite.

Provides:
- Sample CloudTrail events (normal and anomalous)
- Pre-normalized ECS event
- moto mocks for S3, SNS, SSM
- Trained IsolationForest model fixture
- Lambda context mock
- Settings monkeypatch
"""

from __future__ import annotations

import gzip
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Generator
from unittest.mock import MagicMock

import boto3
import joblib
import numpy as np
import pytest
from moto import mock_aws

os.environ.setdefault("CLOUDSENTINEL_ENVIRONMENT", "test")
os.environ.setdefault("CLOUDSENTINEL_AWS_REGION", "us-east-1")
os.environ.setdefault("CLOUDSENTINEL_LOG_LEVEL", "DEBUG")
os.environ.setdefault("CLOUDSENTINEL_OPENSEARCH_ENDPOINT", "http://localhost:9200")
os.environ.setdefault("CLOUDSENTINEL_SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:test-alerts")
os.environ.setdefault("CLOUDSENTINEL_MODEL_BUCKET", "cloudsentinel-models-123456789012")
os.environ.setdefault("CLOUDSENTINEL_ANOMALY_THRESHOLD", "65")


TEST_BUCKET = "cloudsentinel-logs-123456789012"
MODEL_BUCKET = "cloudsentinel-models-123456789012"
TEST_REGION = "us-east-1"
TEST_ACCOUNT_ID = "123456789012"
SNS_TOPIC_NAME = "test-alerts"


# Sample CloudTrail Events


@pytest.fixture
def sample_cloudtrail_event() -> dict[str, Any]:
    """Normal CloudTrail event: IAMUser calling DescribeInstances."""
    return {
        "eventVersion": "1.08",
        "eventTime": "2026-03-10T14:30:00Z",
        "eventSource": "ec2.amazonaws.com",
        "eventName": "DescribeInstances",
        "eventType": "AwsApiCall",
        "eventID": "evt-normal-001",
        "awsRegion": "us-east-1",
        "sourceIPAddress": "10.0.1.50",
        "userAgent": "aws-cli/2.15.0",
        "userIdentity": {
            "type": "IAMUser",
            "principalId": "AIDAEXAMPLE123456",
            "arn": "arn:aws:iam::123456789012:user/developer",
            "accountId": "123456789012",
            "userName": "developer",
        },
        "requestParameters": {"instancesSet": {"items": []}},
        "responseElements": None,
        "readOnly": True,
        "managementEvent": True,
        "recipientAccountId": "123456789012",
    }


@pytest.fixture
def sample_cloudtrail_anomalous_event() -> dict[str, Any]:
    """Anomalous CloudTrail event: Root user creating access key from external IP on weekend."""
    return {
        "eventVersion": "1.08",
        "eventTime": "2026-03-07T03:15:00Z",  # Saturday 3:15 AM
        "eventSource": "iam.amazonaws.com",
        "eventName": "CreateAccessKey",
        "eventType": "AwsApiCall",
        "eventID": "evt-anomalous-001",
        "awsRegion": "ap-southeast-1",  # Cross-region
        "sourceIPAddress": "185.220.101.50",  # External IP (Tor exit node range)
        "userAgent": "aws-sdk-python/1.34.0",
        "userIdentity": {
            "type": "Root",
            "principalId": "123456789012",
            "arn": "arn:aws:iam::123456789012:root",
            "accountId": "123456789012",
        },
        "requestParameters": {"userName": "backdoor-admin"},
        "responseElements": {
            "accessKey": {
                "accessKeyId": "AKIAIOSFODNN7EXAMPLE",
                "status": "Active",
                "userName": "backdoor-admin",
            }
        },
        "readOnly": False,
        "managementEvent": True,
        "recipientAccountId": "123456789012",
    }


@pytest.fixture
def sample_ecs_event() -> Any:
    """Pre-normalized ECS CloudTrail event for tests that skip extraction/normalization."""
    from src.models.ecs_event import (
        CloudSentinelFields,
        ECSAws,
        ECSBase,
        ECSCloud,
        ECSCloudTrailEvent,
        ECSEvent,
        ECSRelated,
        ECSSource,
        ECSSourceGeo,
        ECSUser,
    )

    return ECSCloudTrailEvent(
        base=ECSBase(
            **{  # type: ignore[arg-type]  # Pydantic alias
                "@timestamp": datetime(2026, 3, 10, 14, 30, 0, tzinfo=timezone.utc),
            },
            message="DescribeInstances by developer from 10.0.1.50",
            tags=["cloudtrail"],
        ),
        event=ECSEvent(
            kind="event",
            category=["api"],
            type=["access"],
            action="DescribeInstances",
            provider="ec2.amazonaws.com",
            outcome="success",
            id="evt-normal-001",
            severity=0,
        ),
        user=ECSUser(
            id="AIDAEXAMPLE123456",
            name="developer",
            roles=["IAMUser"],
            domain="123456789012",
        ),
        related=ECSRelated(user=["developer"], ip=["10.0.1.50"]),
        source=ECSSource(
            ip="10.0.1.50",
            address="10.0.1.50",
            geo=ECSSourceGeo(),
        ),
        cloud=ECSCloud(
            provider="aws",
            region="us-east-1",
            account={"id": "123456789012"},
            service={"name": "ec2.amazonaws.com"},
        ),
        aws=ECSAws(cloudtrail={"request_parameters": {}, "read_only": True}),
        cloudsentinel=CloudSentinelFields(
            ip_classification="internal",
            geo_risk_score=0.0,
            api_risk_score=0.1,
        ),
    )


@pytest.fixture
def sample_anomalous_ecs_event() -> Any:
    """Anomalous ECS event: Root user, external IP, weekend, cross-region."""
    from src.models.ecs_event import (
        CloudSentinelFields,
        ECSAws,
        ECSBase,
        ECSCloud,
        ECSCloudTrailEvent,
        ECSEvent,
        ECSRelated,
        ECSSource,
        ECSSourceGeo,
        ECSUser,
    )

    return ECSCloudTrailEvent(
        base=ECSBase(
            **{  # type: ignore[arg-type]  # Pydantic alias
                "@timestamp": datetime(2026, 3, 7, 3, 15, 0, tzinfo=timezone.utc),
            },
            message="CreateAccessKey by Root from 185.220.101.50",
            tags=["cloudtrail"],
        ),
        event=ECSEvent(
            kind="event",
            category=["iam"],
            type=["creation"],
            action="CreateAccessKey",
            provider="iam.amazonaws.com",
            outcome="success",
            id="evt-anomalous-001",
            severity=80,
        ),
        user=ECSUser(
            id="123456789012",
            name="Root",
            roles=["Root"],
            domain="123456789012",
        ),
        related=ECSRelated(user=["Root"], ip=["185.220.101.50"]),
        source=ECSSource(
            ip="185.220.101.50",
            address="185.220.101.50",
            geo=ECSSourceGeo(country_iso_code="DE", country_name="Germany"),
        ),
        cloud=ECSCloud(
            provider="aws",
            region="ap-southeast-1",
            account={"id": "123456789012"},
            service={"name": "iam.amazonaws.com"},
        ),
        aws=ECSAws(
            cloudtrail={
                "request_parameters": {"userName": "backdoor-admin"},
                "response_elements": {
                    "accessKey": {"accessKeyId": "AKIAIOSFODNN7EXAMPLE"}
                },
                "read_only": False,
                "user_identity": {"accountId": "123456789012"},
            }
        ),
        cloudsentinel=CloudSentinelFields(
            ip_classification="external",
            geo_risk_score=0.7,
            api_risk_score=0.9,
        ),
    )


# AWS Service Mocks


@pytest.fixture
def aws_credentials() -> None:
    """Set dummy AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = TEST_REGION


@pytest.fixture
def mock_s3_client(aws_credentials: None) -> Generator[Any, None, None]:
    """Moto S3 mock with test buckets and a sample CloudTrail gzip file."""
    with mock_aws():
        s3 = boto3.client("s3", region_name=TEST_REGION)

        # Create buckets
        s3.create_bucket(Bucket=TEST_BUCKET)
        s3.create_bucket(Bucket=MODEL_BUCKET)

        # Upload a sample CloudTrail log file (gzipped JSON)
        cloudtrail_records = {
            "Records": [
                {
                    "eventVersion": "1.08",
                    "eventTime": "2026-03-10T14:30:00Z",
                    "eventSource": "ec2.amazonaws.com",
                    "eventName": "DescribeInstances",
                    "eventType": "AwsApiCall",
                    "eventID": "evt-s3-test-001",
                    "awsRegion": "us-east-1",
                    "sourceIPAddress": "10.0.1.50",
                    "userIdentity": {
                        "type": "IAMUser",
                        "principalId": "AIDAEXAMPLE123456",
                        "arn": "arn:aws:iam::123456789012:user/developer",
                        "accountId": "123456789012",
                        "userName": "developer",
                    },
                    "requestParameters": None,
                    "responseElements": None,
                    "readOnly": True,
                    "managementEvent": True,
                    "recipientAccountId": "123456789012",
                },
                {
                    "eventVersion": "1.08",
                    "eventTime": "2026-03-10T14:31:00Z",
                    "eventSource": "s3.amazonaws.com",
                    "eventName": "GetObject",
                    "eventType": "AwsApiCall",
                    "eventID": "evt-s3-test-002",
                    "awsRegion": "us-east-1",
                    "sourceIPAddress": "10.0.1.50",
                    "userIdentity": {
                        "type": "IAMUser",
                        "principalId": "AIDAEXAMPLE123456",
                        "arn": "arn:aws:iam::123456789012:user/developer",
                        "accountId": "123456789012",
                        "userName": "developer",
                    },
                    "requestParameters": {"bucketName": "my-data-bucket", "key": "data.csv"},
                    "responseElements": None,
                    "readOnly": True,
                    "managementEvent": False,
                    "recipientAccountId": "123456789012",
                },
                {
                    "eventVersion": "1.08",
                    "eventTime": "2026-03-10T14:32:00Z",
                    "eventSource": "iam.amazonaws.com",
                    "eventName": "ListUsers",
                    "eventType": "AwsApiCall",
                    "eventID": "evt-s3-test-003",
                    "awsRegion": "us-east-1",
                    "sourceIPAddress": "10.0.1.50",
                    "userIdentity": {
                        "type": "IAMUser",
                        "principalId": "AIDAEXAMPLE123456",
                        "arn": "arn:aws:iam::123456789012:user/developer",
                        "accountId": "123456789012",
                        "userName": "developer",
                    },
                    "requestParameters": None,
                    "responseElements": None,
                    "readOnly": True,
                    "managementEvent": True,
                    "recipientAccountId": "123456789012",
                },
            ]
        }

        json_bytes = json.dumps(cloudtrail_records).encode()
        gzip_bytes = gzip.compress(json_bytes)
        s3.put_object(
            Bucket=TEST_BUCKET,
            Key="AWSLogs/123456789012/CloudTrail/us-east-1/2026/03/10/test.json.gz",
            Body=gzip_bytes,
        )

        yield s3


@pytest.fixture
def mock_sns_client(aws_credentials: None) -> Generator[Any, None, None]:
    """Moto SNS mock with a test topic."""
    with mock_aws():
        sns = boto3.client("sns", region_name=TEST_REGION)
        topic = sns.create_topic(Name=SNS_TOPIC_NAME)
        os.environ["CLOUDSENTINEL_SNS_TOPIC_ARN"] = topic["TopicArn"]
        yield sns


@pytest.fixture
def mock_ssm_client(aws_credentials: None) -> Generator[Any, None, None]:
    """Moto SSM mock with CloudSentinel parameters."""
    with mock_aws():
        ssm = boto3.client("ssm", region_name=TEST_REGION)
        params = {
            "/cloudsentinel/opensearch/endpoint": "http://localhost:9200",
            "/cloudsentinel/model/anomaly-threshold": "65",
            "/cloudsentinel/model/bucket": MODEL_BUCKET,
            "/cloudsentinel/sns/topic-arn": f"arn:aws:sns:{TEST_REGION}:{TEST_ACCOUNT_ID}:{SNS_TOPIC_NAME}",
            "/cloudsentinel/pipeline/log-level": "DEBUG",
            "/cloudsentinel/pipeline/batch-size": "500",
        }
        for name, value in params.items():
            ssm.put_parameter(Name=name, Value=value, Type="String")
        yield ssm


@pytest.fixture
def mock_opensearch_client() -> MagicMock:
    """Mock OpenSearch client with preset responses."""
    mock = MagicMock()
    mock.index.return_value = {"result": "created", "_id": "test-doc-id"}
    mock.bulk.return_value = {"errors": False, "items": []}
    mock.search.return_value = {
        "hits": {"total": {"value": 0}, "hits": []},
    }
    mock.indices.exists.return_value = True
    return mock


# ML Model Fixture


@pytest.fixture
def trained_model(mock_s3_client: Any) -> Any:
    """Train a small IsolationForest and upload to mock S3.

    Returns the fitted sklearn Pipeline for direct assertions.
    """
    from sklearn.ensemble import IsolationForest

    rng = np.random.RandomState(42)

    # 100 normal samples: clustered around [0.5, 0.3, 0, 0.1, 0.2, 0, 0, 0.1, 0.3, 0.1]
    normal = rng.normal(
        loc=[0.5, 0.3, 0.0, 0.1, 0.2, 0.0, 0.0, 0.1, 0.3, 0.1],
        scale=0.1,
        size=(100, 10),
    ).clip(0, 1)

    model = IsolationForest(
        n_estimators=50,
        contamination=0.1,
        random_state=42,
    )
    model.fit(normal)

    # Serialize and upload to mock S3
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    mock_s3_client.put_object(
        Bucket=MODEL_BUCKET,
        Key="models/isolation_forest/model.joblib",
        Body=buffer.read(),
    )

    return model


# Lambda Context Mock


@pytest.fixture
def lambda_context() -> MagicMock:
    """Mock AWS Lambda context object."""
    ctx = MagicMock()
    ctx.function_name = "cloudsentinel-pipeline"
    ctx.function_version = "$LATEST"
    ctx.memory_limit_in_mb = 512
    ctx.invoked_function_arn = (
        f"arn:aws:lambda:{TEST_REGION}:{TEST_ACCOUNT_ID}:function:cloudsentinel-pipeline"
    )
    ctx.aws_request_id = "test-request-id-12345"
    ctx.get_remaining_time_in_millis.return_value = 290_000  # 290 seconds
    return ctx


def make_ecs(
    timestamp: datetime | None = None,
    action: str = "DescribeInstances",
    user_name: str = "developer",
    user_roles: list[str] | None = None,
    source_ip: str = "10.0.1.50",
    region: str = "us-east-1",
    outcome: str = "success",
    ip_classification: str = "internal",
    geo_risk: float = 0.0,
    api_risk: float = 0.1,
    provider: str = "ec2.amazonaws.com",
    account_id: str = "123456789012",
    cloudtrail_extra: dict[str, Any] | None = None,
) -> Any:
    """Build an ECS event with controllable fields for testing."""
    from src.models.ecs_event import (
        CloudSentinelFields,
        ECSAws,
        ECSBase,
        ECSCloud,
        ECSCloudTrailEvent,
        ECSEvent,
        ECSRelated,
        ECSSource,
        ECSSourceGeo,
        ECSUser,
    )

    if timestamp is None:
        timestamp = datetime(2026, 3, 10, 14, 30, 0, tzinfo=timezone.utc)
    if user_roles is None:
        user_roles = ["IAMUser"]

    ct: dict[str, Any] = {"request_parameters": {}, "read_only": True}
    if cloudtrail_extra:
        ct.update(cloudtrail_extra)

    return ECSCloudTrailEvent(
        base=ECSBase(**{"@timestamp": timestamp}),  # type: ignore[arg-type]
        event=ECSEvent(
            action=action,
            provider=provider,
            outcome=outcome,
            id="evt-test",
        ),
        user=ECSUser(name=user_name, roles=user_roles, domain=account_id),
        related=ECSRelated(user=[user_name], ip=[source_ip]),
        source=ECSSource(ip=source_ip, address=source_ip, geo=ECSSourceGeo()),
        cloud=ECSCloud(
            region=region,
            account={"id": account_id},
            service={"name": provider},
        ),
        aws=ECSAws(cloudtrail=ct),
        cloudsentinel=CloudSentinelFields(
            ip_classification=ip_classification,
            geo_risk_score=geo_risk,
            api_risk_score=api_risk,
        ),
    )
