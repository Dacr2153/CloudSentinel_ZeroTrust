# FILE: cloudsentinel-zero-trust/src/detectors/rule_engine.py
"""Deterministic rule engine with 8 detection rules mapped to MITRE ATT&CK.

Each rule implements the BaseRule interface:
- matches(event) → bool
- get_evidence(event) → dict
- severity / rule_id / mitre_tactic / mitre_technique properties

Rules:
  RULE-001: Root account usage
  RULE-002: IAM privilege escalation
  RULE-003: CloudTrail tampering
  RULE-004: Security group modification
  RULE-005: Unauthorized API from external IP
  RULE-006: Console login without MFA
  RULE-007: S3 bucket policy change
  RULE-008: Cross-account access
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.models.ecs_event import ECSCloudTrailEvent
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="rule_engine")


@dataclass
class RuleMatch:
    """Result of a rule evaluation."""
    rule_id: str
    rule_name: str
    severity: str  # critical, high, medium, low, informational
    mitre_tactic: str
    mitre_technique: str
    evidence: dict[str, Any]


class BaseRule(ABC):
    """Abstract base for all detection rules."""

    rule_id: str
    rule_name: str
    severity: str
    mitre_tactic: str
    mitre_technique: str

    @abstractmethod
    def matches(self, event: ECSCloudTrailEvent) -> bool:
        """Return True if the event triggers this rule."""

    @abstractmethod
    def get_evidence(self, event: ECSCloudTrailEvent) -> dict[str, Any]:
        """Return evidence dict explaining why the rule fired."""

    def evaluate(self, event: ECSCloudTrailEvent) -> RuleMatch | None:
        """Evaluate and return a RuleMatch or None."""
        if self.matches(event):
            return RuleMatch(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                severity=self.severity,
                mitre_tactic=self.mitre_tactic,
                mitre_technique=self.mitre_technique,
                evidence=self.get_evidence(event),
            )
        return None


# RULE-001: Root Account Usage

class RootAccountUsage(BaseRule):
    rule_id = "RULE-001"
    rule_name = "Root Account Usage"
    severity = "critical"
    mitre_tactic = "Privilege Escalation"
    mitre_technique = "T1078.004"  # Valid Accounts: Cloud Accounts

    def matches(self, event: ECSCloudTrailEvent) -> bool:
        roles = event.user.roles
        return bool(roles and any(r.lower() == "root" for r in roles))

    def get_evidence(self, event: ECSCloudTrailEvent) -> dict[str, Any]:
        return {
            "action": event.event.action,
            "source_ip": event.source.ip,
            "outcome": event.event.outcome,
            "detail": "Root account used directly — violates Zero Trust principle of least privilege",
        }


# RULE-002: IAM Privilege Escalation

_ESCALATION_APIS = {
    "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile",
    "AttachUserPolicy", "AttachRolePolicy", "AttachGroupPolicy",
    "PutUserPolicy", "PutRolePolicy", "PutGroupPolicy",
    "CreatePolicyVersion", "SetDefaultPolicyVersion",
    "AddUserToGroup", "UpdateAssumeRolePolicy",
}


class IAMPrivilegeEscalation(BaseRule):
    rule_id = "RULE-002"
    rule_name = "IAM Privilege Escalation Attempt"
    severity = "high"
    mitre_tactic = "Privilege Escalation"
    mitre_technique = "T1098"  # Account Manipulation

    def matches(self, event: ECSCloudTrailEvent) -> bool:
        return event.event.action in _ESCALATION_APIS

    def get_evidence(self, event: ECSCloudTrailEvent) -> dict[str, Any]:
        params = {}
        ct = event.aws.cloudtrail
        if ct.get("request_parameters") and isinstance(ct["request_parameters"], dict):
            params = {
                "target_user": ct["request_parameters"].get("userName", "N/A"),
                "policy": ct["request_parameters"].get("policyArn", "N/A"),
            }
        return {
            "action": event.event.action,
            "caller": event.user.name,
            "outcome": event.event.outcome,
            **params,
        }


# RULE-003: CloudTrail Tampering

_TRAIL_TAMPERING_APIS = {
    "DeleteTrail", "StopLogging", "UpdateTrail",
    "PutEventSelectors", "DeleteEventDataStore",
}


class CloudTrailTampering(BaseRule):
    rule_id = "RULE-003"
    rule_name = "CloudTrail Tampering"
    severity = "critical"
    mitre_tactic = "Defense Evasion"
    mitre_technique = "T1562.008"  # Impair Defenses: Disable Cloud Logs

    def matches(self, event: ECSCloudTrailEvent) -> bool:
        return event.event.action in _TRAIL_TAMPERING_APIS

    def get_evidence(self, event: ECSCloudTrailEvent) -> dict[str, Any]:
        return {
            "action": event.event.action,
            "caller": event.user.name,
            "source_ip": event.source.ip,
            "outcome": event.event.outcome,
            "detail": "Attempt to modify or disable CloudTrail logging",
        }


# RULE-004: Security Group Modification

_SG_MODIFICATION_APIS = {
    "AuthorizeSecurityGroupIngress", "AuthorizeSecurityGroupEgress",
    "RevokeSecurityGroupIngress", "RevokeSecurityGroupEgress",
    "CreateSecurityGroup", "DeleteSecurityGroup",
}


class SecurityGroupModification(BaseRule):
    rule_id = "RULE-004"
    rule_name = "Security Group Modification"
    severity = "medium"
    mitre_tactic = "Defense Evasion"
    mitre_technique = "T1562.007"  # Impair Defenses: Disable or Modify Cloud Firewall

    def matches(self, event: ECSCloudTrailEvent) -> bool:
        return event.event.action in _SG_MODIFICATION_APIS

    def get_evidence(self, event: ECSCloudTrailEvent) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "action": event.event.action,
            "caller": event.user.name,
            "outcome": event.event.outcome,
        }
        ct = event.aws.cloudtrail
        if ct.get("request_parameters") and isinstance(ct["request_parameters"], dict):
            params = ct["request_parameters"]
            evidence["security_group_id"] = params.get("groupId", "N/A")
            # Check for 0.0.0.0/0 — open to world
            ip_perms = params.get("ipPermissions", {})
            if isinstance(ip_perms, dict):
                items = ip_perms.get("items", [])
                for item in items if isinstance(items, list) else []:
                    for ip_range in item.get("ipRanges", {}).get("items", []):
                        if isinstance(ip_range, dict) and ip_range.get("cidrIp") == "0.0.0.0/0":
                            evidence["open_to_world"] = True
                            evidence["severity_override"] = "high"
        return evidence


# RULE-005: Unauthorized API from External IP

_SENSITIVE_APIS = _ESCALATION_APIS | _TRAIL_TAMPERING_APIS | {
    "GetSecretValue", "CreateAccessKey", "ConsoleLogin",
}


class UnauthorizedExternalAPI(BaseRule):
    rule_id = "RULE-005"
    rule_name = "Sensitive API from External IP"
    severity = "high"
    mitre_tactic = "Initial Access"
    mitre_technique = "T1078"  # Valid Accounts

    def matches(self, event: ECSCloudTrailEvent) -> bool:
        is_external = event.cloudsentinel.ip_classification == "external"
        is_sensitive = event.event.action in _SENSITIVE_APIS
        return is_external and is_sensitive

    def get_evidence(self, event: ECSCloudTrailEvent) -> dict[str, Any]:
        return {
            "action": event.event.action,
            "source_ip": event.source.ip,
            "ip_classification": event.cloudsentinel.ip_classification,
            "geo_risk": event.cloudsentinel.geo_risk_score,
            "caller": event.user.name,
        }


# RULE-006: Console Login Without MFA

class ConsoleLoginNoMFA(BaseRule):
    rule_id = "RULE-006"
    rule_name = "Console Login Without MFA"
    severity = "high"
    mitre_tactic = "Initial Access"
    mitre_technique = "T1078.004"  # Valid Accounts: Cloud Accounts

    def matches(self, event: ECSCloudTrailEvent) -> bool:
        if event.event.action != "ConsoleLogin":
            return False
        ct = event.aws.cloudtrail
        additional = ct.get("additional_event_data", {})
        if isinstance(additional, dict):
            mfa = additional.get("MFAUsed", "")
            return mfa == "No"
        return False

    def get_evidence(self, event: ECSCloudTrailEvent) -> dict[str, Any]:
        return {
            "action": "ConsoleLogin",
            "caller": event.user.name,
            "source_ip": event.source.ip,
            "mfa_used": "No",
            "outcome": event.event.outcome,
            "detail": "Console login without MFA — violates Zero Trust authentication policy",
        }


# RULE-007: S3 Bucket Policy Change

_S3_POLICY_APIS = {
    "PutBucketPolicy", "DeleteBucketPolicy",
    "PutBucketAcl", "PutBucketPublicAccessBlock",
}


class S3BucketPolicyChange(BaseRule):
    rule_id = "RULE-007"
    rule_name = "S3 Bucket Policy Change"
    severity = "medium"
    mitre_tactic = "Exfiltration"
    mitre_technique = "T1537"  # Transfer Data to Cloud Account

    def matches(self, event: ECSCloudTrailEvent) -> bool:
        return event.event.action in _S3_POLICY_APIS

    def get_evidence(self, event: ECSCloudTrailEvent) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "action": event.event.action,
            "caller": event.user.name,
            "outcome": event.event.outcome,
        }
        ct = event.aws.cloudtrail
        if ct.get("request_parameters") and isinstance(ct["request_parameters"], dict):
            evidence["bucket"] = ct["request_parameters"].get("bucketName", "N/A")
        return evidence


# RULE-008: Cross-Account Access

class CrossAccountAccess(BaseRule):
    rule_id = "RULE-008"
    rule_name = "Cross-Account Access Detected"
    severity = "medium"
    mitre_tactic = "Lateral Movement"
    mitre_technique = "T1550.001"  # Use Alternate Authentication Material

    def matches(self, event: ECSCloudTrailEvent) -> bool:
        # Cross-account: user account differs from resource account
        ct = event.aws.cloudtrail
        user_account = ct.get("user_identity", {})
        if isinstance(user_account, dict):
            user_acct_id = user_account.get("accountId", "")
            resource_acct_id = event.cloud.account.get("id", "")
            if user_acct_id and resource_acct_id:
                return user_acct_id != resource_acct_id
        return False

    def get_evidence(self, event: ECSCloudTrailEvent) -> dict[str, Any]:
        ct = event.aws.cloudtrail
        user_identity = ct.get("user_identity", {})
        user_acct = user_identity.get("accountId", "N/A") if isinstance(user_identity, dict) else "N/A"
        return {
            "action": event.event.action,
            "source_account": user_acct,
            "target_account": event.cloud.account.get("id", ""),
            "caller": event.user.name,
            "detail": "API call originated from a different AWS account",
        }


# Rule Engine (orchestrates all rules)

# Registry of all rules
ALL_RULES: list[BaseRule] = [
    RootAccountUsage(),
    IAMPrivilegeEscalation(),
    CloudTrailTampering(),
    SecurityGroupModification(),
    UnauthorizedExternalAPI(),
    ConsoleLoginNoMFA(),
    S3BucketPolicyChange(),
    CrossAccountAccess(),
]


class RuleEngine:
    """Evaluates all registered rules against an event.

    Usage:
        engine = RuleEngine()
        matches = engine.evaluate(event)
    """

    def __init__(self, rules: list[BaseRule] | None = None) -> None:
        self._rules = rules if rules is not None else ALL_RULES

    def evaluate(self, event: ECSCloudTrailEvent) -> list[RuleMatch]:
        """Evaluate all rules and return matches."""
        matches: list[RuleMatch] = []
        for rule in self._rules:
            try:
                result = rule.evaluate(event)
                if result:
                    matches.append(result)
            except Exception as exc:
                logger.warning(
                    "Rule %s failed for event %s: %s",
                    rule.rule_id,
                    event.event.id,
                    exc,
                )
        return matches

    def evaluate_batch(
        self, events: list[ECSCloudTrailEvent]
    ) -> dict[str, list[RuleMatch]]:
        """Evaluate all events and return a dict keyed by event ID."""
        results: dict[str, list[RuleMatch]] = {}
        for event in events:
            event_id = event.event.id or "unknown"
            matches = self.evaluate(event)
            if matches:
                results[event_id] = matches
                logger.info(
                    "Event %s triggered %d rules: %s",
                    event_id,
                    len(matches),
                    [m.rule_id for m in matches],
                )
        return results
