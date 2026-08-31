# FILE: cloudsentinel-zero-trust/tests/unit/test_normalizer.py
"""Unit tests for ECSNormalizer — CloudTrail to ECS 8.10+ mapping."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from src.models.cloudtrail_event import CloudTrailEvent
from src.pipeline.normalizer import ECSNormalizer


@pytest.fixture
def normalizer() -> ECSNormalizer:
    return ECSNormalizer()


def _make_ct_event(**overrides: Any) -> CloudTrailEvent:
    """Helper to build a CloudTrailEvent with sensible defaults and overrides."""
    base = {
        "eventVersion": "1.08",
        "eventTime": "2026-03-10T14:30:00Z",
        "eventSource": "ec2.amazonaws.com",
        "eventName": "DescribeInstances",
        "eventType": "AwsApiCall",
        "eventID": "evt-test-001",
        "awsRegion": "us-east-1",
        "sourceIPAddress": "10.0.1.50",
        "userIdentity": {
            "type": "IAMUser",
            "principalId": "AIDAEXAMPLE",
            "arn": "arn:aws:iam::123456789012:user/developer",
            "accountId": "123456789012",
            "userName": "developer",
        },
        "requestParameters": None,
        "responseElements": None,
        "readOnly": True,
        "managementEvent": True,
        "recipientAccountId": "123456789012",
    }
    base.update(overrides)
    return CloudTrailEvent.model_validate(base)


def test_normalize_root_user(normalizer: ECSNormalizer) -> None:
    event = _make_ct_event(
    userIdentity={
        "type": "Root",
        "principalId": "123456789012",
        "arn": "arn:aws:iam::123456789012:root",
        "accountId": "123456789012",
    }
    )
    ecs = normalizer.normalize(event)

    assert "Root" in ecs.user.roles or ecs.user.name == "Root"
    assert ecs.cloud.account.get("id") == "123456789012"


def test_normalize_iam_user(normalizer: ECSNormalizer) -> None:
    event = _make_ct_event()  # default has IAMUser
    ecs = normalizer.normalize(event)

    assert ecs.user.name == "developer"
    assert ecs.event.action == "DescribeInstances"
    assert ecs.event.provider == "ec2.amazonaws.com"


def test_normalize_assumed_role(normalizer: ECSNormalizer) -> None:
    event = _make_ct_event(
    userIdentity={
        "type": "AssumedRole",
        "principalId": "AROA3XFRBF23:session-name",
        "arn": "arn:aws:sts::123456789012:assumed-role/AdminRole/session-name",
        "accountId": "123456789012",
        "sessionContext": {
            "sessionIssuer": {
                "type": "Role",
                "principalId": "AROA3XFRBF23",
                "arn": "arn:aws:iam::123456789012:role/AdminRole",
                "accountId": "123456789012",
                "userName": "AdminRole",
            },
            "attributes": {
                "creationDate": "2026-03-10T14:00:00Z",
                "mfaAuthenticated": "false",
            },
        },
    }
    )
    ecs = normalizer.normalize(event)

    # Should extract session name or role name
    assert ecs.user.name in ("session-name", "AdminRole", "AROA3XFRBF23:session-name")


def test_normalize_federated_user(normalizer: ECSNormalizer) -> None:
    event = _make_ct_event(
    userIdentity={
        "type": "FederatedUser",
        "principalId": "123456789012:federated-user",
        "arn": "arn:aws:sts::123456789012:federated-user/federated-user",
        "accountId": "123456789012",
    }
    )
    ecs = normalizer.normalize(event)

    assert ecs.user.name  # Should have a non-empty name extracted


def test_normalize_timestamp_utc(normalizer: ECSNormalizer) -> None:
    event = _make_ct_event(eventTime="2026-03-10T14:30:00Z")
    ecs = normalizer.normalize(event)

    ts = ecs.base.timestamp
    assert isinstance(ts, datetime)
    assert ts.tzinfo is not None  # Must be timezone-aware
    assert ts.year == 2026
    assert ts.month == 3
    assert ts.day == 10
    assert ts.hour == 14
    assert ts.minute == 30


def test_normalize_error_outcome(normalizer: ECSNormalizer) -> None:
    event = _make_ct_event(
    errorCode="AccessDenied",
    errorMessage="User is not authorized to perform this action",
    )
    ecs = normalizer.normalize(event)

    assert ecs.event.outcome == "failure"


def test_normalize_batch_consistency(normalizer: ECSNormalizer) -> None:
    events = [
    _make_ct_event(eventName="DescribeInstances", eventID="evt-1"),
    _make_ct_event(eventName="CreateUser", eventID="evt-2"),
    _make_ct_event(eventName="GetObject", eventID="evt-3"),
    ]

    batch_results = normalizer.normalize_batch(events)
    individual_results = [normalizer.normalize(e) for e in events]

    assert len(batch_results) == len(individual_results)

    for batch_ecs, indiv_ecs in zip(batch_results, individual_results):
        assert batch_ecs.event.action == indiv_ecs.event.action
        assert batch_ecs.event.id == indiv_ecs.event.id
        assert batch_ecs.user.name == indiv_ecs.user.name
