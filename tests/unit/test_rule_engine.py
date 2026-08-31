# FILE: cloudsentinel-zero-trust/tests/unit/test_rule_engine.py
"""Unit tests for all 8 detection rules + RuleEngine orchestrator."""

from __future__ import annotations



from src.detectors.rule_engine import (
    CloudTrailTampering,
    ConsoleLoginNoMFA,
    CrossAccountAccess,
    IAMPrivilegeEscalation,
    RootAccountUsage,
    RuleEngine,
    S3BucketPolicyChange,
    SecurityGroupModification,
    UnauthorizedExternalAPI,
)
from tests.conftest import make_ecs


# RULE-001: Root Account Usage


class TestRootAccountUsage:
    def test_matches_positive(self) -> None:
        """Root user events MUST trigger RULE-001."""
        event = make_ecs(user_name="Root", user_roles=["Root"])
        rule = RootAccountUsage()
        result = rule.evaluate(event)

        assert result is not None
        assert result.rule_id == "RULE-001"
        assert result.severity == "critical"

    def test_no_match_negative(self) -> None:
        """IAMUser events must NOT trigger RULE-001."""
        event = make_ecs(user_name="developer", user_roles=["IAMUser"])
        rule = RootAccountUsage()
        result = rule.evaluate(event)

        assert result is None


# RULE-002: IAM Privilege Escalation


class TestIAMPrivilegeEscalation:
    def test_matches_positive(self) -> None:
        """CreateAccessKey MUST trigger RULE-002."""
        event = make_ecs(
            action="CreateAccessKey",
            provider="iam.amazonaws.com",
            cloudtrail_extra={
                "request_parameters": {"userName": "target-user"},
            },
        )
        rule = IAMPrivilegeEscalation()
        result = rule.evaluate(event)

        assert result is not None
        assert result.rule_id == "RULE-002"
        assert result.severity == "high"

    def test_no_match_negative(self) -> None:
        """ListUsers must NOT trigger RULE-002."""
        event = make_ecs(action="ListUsers", provider="iam.amazonaws.com")
        rule = IAMPrivilegeEscalation()
        result = rule.evaluate(event)

        assert result is None


# RULE-003: CloudTrail Tampering


class TestCloudTrailTampering:
    def test_matches_positive(self) -> None:
        """StopLogging MUST trigger RULE-003."""
        event = make_ecs(
            action="StopLogging",
            provider="cloudtrail.amazonaws.com",
        )
        rule = CloudTrailTampering()
        result = rule.evaluate(event)

        assert result is not None
        assert result.rule_id == "RULE-003"
        assert result.severity == "critical"

    def test_no_match_negative(self) -> None:
        """DescribeTrails must NOT trigger RULE-003."""
        event = make_ecs(
            action="DescribeTrails",
            provider="cloudtrail.amazonaws.com",
        )
        rule = CloudTrailTampering()
        result = rule.evaluate(event)

        assert result is None


# RULE-004: Security Group Modification


class TestSecurityGroupModification:
    def test_matches_positive(self) -> None:
        """AuthorizeSecurityGroupIngress MUST trigger RULE-004."""
        event = make_ecs(
            action="AuthorizeSecurityGroupIngress",
            provider="ec2.amazonaws.com",
            cloudtrail_extra={
                "request_parameters": {"groupId": "sg-12345"},
            },
        )
        rule = SecurityGroupModification()
        result = rule.evaluate(event)

        assert result is not None
        assert result.rule_id == "RULE-004"
        assert result.severity == "medium"

    def test_no_match_negative(self) -> None:
        """DescribeSecurityGroups must NOT trigger RULE-004."""
        event = make_ecs(
            action="DescribeSecurityGroups",
            provider="ec2.amazonaws.com",
        )
        rule = SecurityGroupModification()
        result = rule.evaluate(event)

        assert result is None


# RULE-005: Sensitive API from External IP


class TestUnauthorizedExternalAPI:
    def test_matches_positive(self) -> None:
        """GetSecretValue from external IP MUST trigger RULE-005."""
        event = make_ecs(
            action="GetSecretValue",
            source_ip="185.220.101.50",
            ip_classification="external",
        )
        rule = UnauthorizedExternalAPI()
        result = rule.evaluate(event)

        assert result is not None
        assert result.rule_id == "RULE-005"
        assert result.severity == "high"

    def test_no_match_negative(self) -> None:
        """GetSecretValue from internal IP must NOT trigger RULE-005."""
        event = make_ecs(
            action="GetSecretValue",
            source_ip="10.0.1.50",
            ip_classification="internal",
        )
        rule = UnauthorizedExternalAPI()
        result = rule.evaluate(event)

        assert result is None


# RULE-006: Console Login Without MFA


class TestConsoleLoginNoMFA:
    def test_matches_positive(self) -> None:
        """ConsoleLogin with MFAUsed=No MUST trigger RULE-006."""
        event = make_ecs(
            action="ConsoleLogin",
            provider="signin.amazonaws.com",
            cloudtrail_extra={
                "additional_event_data": {"MFAUsed": "No"},
            },
        )
        rule = ConsoleLoginNoMFA()
        result = rule.evaluate(event)

        assert result is not None
        assert result.rule_id == "RULE-006"
        assert result.severity == "high"

    def test_no_match_negative(self) -> None:
        """ConsoleLogin with MFAUsed=Yes must NOT trigger RULE-006."""
        event = make_ecs(
            action="ConsoleLogin",
            provider="signin.amazonaws.com",
            cloudtrail_extra={
                "additional_event_data": {"MFAUsed": "Yes"},
            },
        )
        rule = ConsoleLoginNoMFA()
        result = rule.evaluate(event)

        assert result is None


# RULE-007: S3 Bucket Policy Change


class TestS3BucketPolicyChange:
    def test_matches_positive(self) -> None:
        """PutBucketPolicy MUST trigger RULE-007."""
        event = make_ecs(
            action="PutBucketPolicy",
            provider="s3.amazonaws.com",
            cloudtrail_extra={
                "request_parameters": {"bucketName": "sensitive-data"},
            },
        )
        rule = S3BucketPolicyChange()
        result = rule.evaluate(event)

        assert result is not None
        assert result.rule_id == "RULE-007"
        assert result.severity == "medium"

    def test_no_match_negative(self) -> None:
        """GetBucketPolicy must NOT trigger RULE-007."""
        event = make_ecs(action="GetBucketPolicy", provider="s3.amazonaws.com")
        rule = S3BucketPolicyChange()
        result = rule.evaluate(event)

        assert result is None


# RULE-008: Cross-Account Access


class TestCrossAccountAccess:
    def test_matches_positive(self) -> None:
        """Different user account vs resource account MUST trigger RULE-008."""
        event = make_ecs(
            account_id="123456789012",
            cloudtrail_extra={
                "user_identity": {"accountId": "999888777666"},
            },
        )
        rule = CrossAccountAccess()
        result = rule.evaluate(event)

        assert result is not None
        assert result.rule_id == "RULE-008"
        assert result.severity == "medium"

    def test_no_match_negative(self) -> None:
        """Same account MUST NOT trigger RULE-008."""
        event = make_ecs(
            account_id="123456789012",
            cloudtrail_extra={
                "user_identity": {"accountId": "123456789012"},
            },
        )
        rule = CrossAccountAccess()
        result = rule.evaluate(event)

        assert result is None


# RuleEngine Integration Tests


def test_rule_engine_returns_all_matching_rules() -> None:
    """Root user calling CreateAccessKey should trigger RULE-001 + RULE-002."""
    event = make_ecs(
    action="CreateAccessKey",
    user_name="Root",
    user_roles=["Root"],
    provider="iam.amazonaws.com",
    )

    engine = RuleEngine()
    matches = engine.evaluate(event)

    rule_ids = {m.rule_id for m in matches}
    assert "RULE-001" in rule_ids  # Root account usage
    assert "RULE-002" in rule_ids  # IAM privilege escalation
    assert len(matches) >= 2


def test_rule_mitre_fields_populated() -> None:
    engine = RuleEngine()

    # Use an event that triggers at least one rule
    event = make_ecs(
    action="StopLogging",
    provider="cloudtrail.amazonaws.com",
    )
    matches = engine.evaluate(event)

    assert len(matches) > 0
    for match in matches:
        assert match.mitre_tactic, f"Empty mitre_tactic in {match.rule_id}"
        assert match.mitre_technique, f"Empty mitre_technique in {match.rule_id}"
        assert match.mitre_technique.startswith("T"), (
            f"mitre_technique should start with 'T': {match.mitre_technique}"
        )


def test_rule_match_evidence_contains_details() -> None:
    event = make_ecs(
    action="DeleteTrail",
    provider="cloudtrail.amazonaws.com",
    )

    rule = CloudTrailTampering()
    result = rule.evaluate(event)

    assert result is not None
    assert "action" in result.evidence
    assert result.evidence["action"] == "DeleteTrail"


def test_rule_severity_levels_valid() -> None:
    valid_severities = {"critical", "high", "medium", "low", "informational"}

    all_rules = [
    RootAccountUsage(),
    IAMPrivilegeEscalation(),
    CloudTrailTampering(),
    SecurityGroupModification(),
    UnauthorizedExternalAPI(),
    ConsoleLoginNoMFA(),
    S3BucketPolicyChange(),
    CrossAccountAccess(),
    ]

    for rule in all_rules:
        assert rule.severity in valid_severities, (
            f"Rule {rule.rule_id} has invalid severity: {rule.severity}"
        )
