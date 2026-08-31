# FILE: cloudsentinel-zero-trust/tests/unit/test_alert_manager.py
"""Unit tests for AlertManager: deduplication, severity, SNS dispatch, campaign correlation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock


from src.detectors.alert_manager import AlertManager
from src.detectors.rule_engine import RuleMatch
from tests.conftest import make_ecs


def _make_rule_match(
    rule_id: str = "RULE-002",
    severity: str = "high",
) -> RuleMatch:
    return RuleMatch(
        rule_id=rule_id,
        rule_name="IAM Privilege Escalation",
        severity=severity,
        mitre_tactic="Privilege Escalation",
        mitre_technique="T1098",
        evidence={"action": "CreateAccessKey"},
    )


# Deduplication tests


class TestDeduplication:
    """Same (actor + rule + resource) within 5 min → suppressed on second call."""

    def test_duplicate_alert_suppressed(self) -> None:
        sns = MagicMock()
        os_client = MagicMock()
        manager = AlertManager(
            sns_client=sns,
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts",
            opensearch_client=os_client,
        )

        event = make_ecs()
        rule_match = _make_rule_match()

        first = manager.process_event(
            event_id=event.event.id,
            event_action=event.event.action,
            user_name=event.user.name,
            source_ip=event.source.ip,
            cloud_region=event.cloud.region,
            cloud_account_id=event.cloud.account.get("id", ""),
            rule_matches=[rule_match],
            anomaly_score=30,
        )
        assert first is not None

        second = manager.process_event(
            event_id=event.event.id,
            event_action=event.event.action,
            user_name=event.user.name,
            source_ip=event.source.ip,
            cloud_region=event.cloud.region,
            cloud_account_id=event.cloud.account.get("id", ""),
            rule_matches=[rule_match],
            anomaly_score=30,
        )
        assert second is None  # suppressed by dedup

    def test_different_actor_not_suppressed(self) -> None:
        sns = MagicMock()
        os_client = MagicMock()
        manager = AlertManager(
            sns_client=sns,
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts",
            opensearch_client=os_client,
        )

        event_a = make_ecs(user_name="attacker-1")
        event_b = make_ecs(user_name="attacker-2")
        rule_match = _make_rule_match()

        first = manager.process_event(
            event_id=event_a.event.id,
            event_action=event_a.event.action,
            user_name=event_a.user.name,
            source_ip=event_a.source.ip,
            cloud_region=event_a.cloud.region,
            cloud_account_id=event_a.cloud.account.get("id", ""),
            rule_matches=[rule_match],
            anomaly_score=30,
        )
        second = manager.process_event(
            event_id=event_b.event.id,
            event_action=event_b.event.action,
            user_name=event_b.user.name,
            source_ip=event_b.source.ip,
            cloud_region=event_b.cloud.region,
            cloud_account_id=event_b.cloud.account.get("id", ""),
            rule_matches=[rule_match],
            anomaly_score=30,
        )
        assert first is not None
        assert second is not None


# Severity assignment


def test_root_user_severity_critical() -> None:
    sns = MagicMock()
    os_client = MagicMock()
    manager = AlertManager(
        sns_client=sns,
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts",
        opensearch_client=os_client,
    )

    event = make_ecs(user_name="Root", user_roles=["Root"])
    rule_match = _make_rule_match(rule_id="RULE-001", severity="critical")

    alert = manager.process_event(
        event_id=event.event.id,
        event_action=event.event.action,
        user_name=event.user.name,
        source_ip=event.source.ip,
        cloud_region=event.cloud.region,
        cloud_account_id=event.cloud.account.get("id", ""),
        rule_matches=[rule_match],
        anomaly_score=85,
    )

    assert alert is not None
    assert alert.severity == "critical"


def test_sns_message_fields() -> None:
    sns = MagicMock()
    os_client = MagicMock()
    manager = AlertManager(
        sns_client=sns,
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts",
        opensearch_client=os_client,
    )

    event = make_ecs()
    rule_match = _make_rule_match()

    alert = manager.process_event(
        event_id=event.event.id,
        event_action=event.event.action,
        user_name=event.user.name,
        source_ip=event.source.ip,
        cloud_region=event.cloud.region,
        cloud_account_id=event.cloud.account.get("id", ""),
        rule_matches=[rule_match],
        anomaly_score=72,
    )

    assert alert is not None
    msg = alert.to_sns_message()
    assert isinstance(msg, str)
    assert "CLOUDSENTINEL" in msg
    assert alert.severity.upper() in msg
    assert alert.alert_id in msg


def test_to_dict_completeness() -> None:
    sns = MagicMock()
    os_client = MagicMock()
    manager = AlertManager(
        sns_client=sns,
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts",
        opensearch_client=os_client,
    )

    event = make_ecs()
    rule_match = _make_rule_match()

    alert = manager.process_event(
        event_id=event.event.id,
        event_action=event.event.action,
        user_name=event.user.name,
        source_ip=event.source.ip,
        cloud_region=event.cloud.region,
        cloud_account_id=event.cloud.account.get("id", ""),
        rule_matches=[rule_match],
        anomaly_score=72,
    )

    assert alert is not None
    d = alert.to_dict()

    required_keys = {"alert_id", "severity", "@timestamp", "dedup_hash", "title"}
    assert required_keys.issubset(d.keys()), (
        f"Missing keys: {required_keys - d.keys()}"
    )

    json.dumps(d, default=str)
