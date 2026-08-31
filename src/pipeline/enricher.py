# FILE: cloudsentinel-zero-trust/src/pipeline/enricher.py
"""Event enrichment: geolocation, IP classification, risk scoring.

Design decisions:
- Geo lookup via embedded country-level dict (no external DB dependency in Lambda;
  MaxMind GeoLite2 can be added later as a Lambda layer for city-level accuracy)
- API call risk scoring: hand-curated dict based on MITRE ATT&CK for Cloud
- IAM privilege escalation indicators: curated from known escalation paths
- Operates in-place on ECSCloudTrailEvent objects for memory efficiency
"""

from __future__ import annotations

import ipaddress

from src.models.ecs_event import ECSCloudTrailEvent, ECSSourceGeo
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="enricher")


# Score 0.0 = low risk, 1.0 = high risk
# Based on aggregated threat intelligence reports; not a geopolitical judgment.
# Countries not listed default to 0.3 (moderate baseline).

COUNTRY_RISK: dict[str, float] = {
    "US": 0.1, "GB": 0.1, "DE": 0.1, "CA": 0.1, "AU": 0.1, "JP": 0.1,
    "FR": 0.15, "NL": 0.15, "SE": 0.1, "IE": 0.1, "SG": 0.1,
    "BR": 0.4, "IN": 0.35, "NG": 0.6, "RU": 0.7, "CN": 0.6,
    "KP": 0.9, "IR": 0.7, "PK": 0.5, "UA": 0.5, "RO": 0.5,
    "TR": 0.4, "VN": 0.4, "ID": 0.35, "TH": 0.35, "PH": 0.35,
    "KR": 0.2, "IL": 0.15, "SA": 0.3, "AE": 0.2, "ZA": 0.4,
}
_DEFAULT_COUNTRY_RISK = 0.3

# Risk 0.0 = benign read, 1.0 = critical destructive/escalation action

API_RISK: dict[str, float] = {
    # Critical — direct security impact
    "DeleteTrail": 1.0,
    "StopLogging": 1.0,
    "UpdateTrail": 0.9,
    "PutEventSelectors": 0.8,
    "CreateAccessKey": 0.9,
    "DeleteAccessKey": 0.7,
    "AttachUserPolicy": 0.9,
    "AttachRolePolicy": 0.9,
    "AttachGroupPolicy": 0.8,
    "PutUserPolicy": 0.9,
    "PutRolePolicy": 0.9,
    "CreatePolicyVersion": 0.8,
    "CreateUser": 0.8,
    "CreateRole": 0.7,
    "CreateLoginProfile": 0.8,
    "UpdateLoginProfile": 0.8,
    "AssumeRole": 0.6,
    "GetSessionToken": 0.6,
    "GetFederationToken": 0.7,
    "ConsoleLogin": 0.5,
    # High — data access / exfiltration indicators
    "GetSecretValue": 0.7,
    "GetParametersByPath": 0.5,
    "GetParameter": 0.4,
    "AuthorizeSecurityGroupIngress": 0.8,
    "AuthorizeSecurityGroupEgress": 0.7,
    "RevokeSecurityGroupIngress": 0.6,
    "CreateSecurityGroup": 0.5,
    "RunInstances": 0.6,
    "TerminateInstances": 0.5,
    "ModifyInstanceAttribute": 0.6,
    "DisassociateRouteTable": 0.7,
    "CreateSnapshot": 0.5,
    "ModifySnapshotAttribute": 0.7,
    "CopySnapshot": 0.6,
    # Medium — reconnaissance
    "ListBuckets": 0.3,
    "ListUsers": 0.3,
    "ListRoles": 0.3,
    "ListAccessKeys": 0.4,
    "GetBucketAcl": 0.3,
    "GetBucketPolicy": 0.3,
    "HeadBucket": 0.2,
    # Low — routine operations
    "GetObject": 0.2,
    "PutObject": 0.2,
    "DescribeInstances": 0.1,
    "DescribeSecurityGroups": 0.1,
    "DescribeSubnets": 0.1,
    "DescribeVpcs": 0.1,
    "GetCallerIdentity": 0.05,
    "ListObjects": 0.15,
    "ListObjectsV2": 0.15,
}
_DEFAULT_API_RISK = 0.2

# Known IAM escalation paths (Rhino Security Labs / Pacu research)

PRIVILEGE_ESCALATION_APIS: set[str] = {
    "CreateAccessKey",
    "CreateLoginProfile",
    "UpdateLoginProfile",
    "AttachUserPolicy",
    "AttachRolePolicy",
    "AttachGroupPolicy",
    "PutUserPolicy",
    "PutRolePolicy",
    "PutGroupPolicy",
    "CreatePolicyVersion",
    "SetDefaultPolicyVersion",
    "AddUserToGroup",
    "UpdateAssumeRolePolicy",
    "PassRole",
    "CreateServiceLinkedRole",
    "iam:CreateVirtualMFADevice",
    "AssumeRole",
    "AssumeRoleWithSAML",
    "AssumeRoleWithWebIdentity",
}

AWS_SERVICE_SOURCES: set[str] = {
    "amazonaws.com",
    "ec2.amazonaws.com",
    "lambda.amazonaws.com",
    "cloudformation.amazonaws.com",
    "elasticmapreduce.amazonaws.com",
    "delivery.logs.amazonaws.com",
    "config.amazonaws.com",
    "guardduty.amazonaws.com",
    "AWS Internal",
}

_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
]


class EventEnricher:
    """Enriches ECS-normalized events with geo data, risk scores, and IP classification."""

    def enrich(self, event: ECSCloudTrailEvent) -> ECSCloudTrailEvent:
        """Enrich a single ECS event in-place.

        Populates: source.geo, cloudsentinel.ip_classification,
        cloudsentinel.geo_risk_score, cloudsentinel.api_risk_score.
        """
        self._classify_ip(event)
        self._enrich_geo(event)
        self._score_api_risk(event)
        self._score_privilege_escalation(event)
        return event

    def enrich_batch(self, events: list[ECSCloudTrailEvent]) -> list[ECSCloudTrailEvent]:
        """Enrich a batch of events."""
        for event in events:
            try:
                self.enrich(event)
            except Exception as exc:
                logger.warning("Enrichment failed for event %s: %s", event.event.id, exc)
        return events


    def _classify_ip(self, event: ECSCloudTrailEvent) -> None:
        """Classify source IP as internal/external/aws-service/known-threat."""
        source_addr = event.source.address or event.source.ip

        if not source_addr:
            event.cloudsentinel.ip_classification = "unknown"
            return

        # AWS service sources (not real IPs)
        if source_addr in AWS_SERVICE_SOURCES or source_addr.endswith(".amazonaws.com"):
            event.cloudsentinel.ip_classification = "aws-service"
            return

        # Check if valid IP
        try:
            ip = ipaddress.ip_address(source_addr)
        except ValueError:
            # Not a valid IP (might be an AWS service name)
            event.cloudsentinel.ip_classification = "aws-service"
            return

        # Private IP ranges
        for net in _PRIVATE_NETWORKS:
            if ip in net:
                event.cloudsentinel.ip_classification = "internal"
                return

        event.cloudsentinel.ip_classification = "external"


    def _enrich_geo(self, event: ECSCloudTrailEvent) -> None:
        """Enrich source.geo fields and compute geo risk score.

        Note: Full city-level geolocation requires MaxMind GeoLite2 DB
        (can be added as Lambda layer). For now, region-based heuristics are used.
        """
        classification = event.cloudsentinel.ip_classification

        if classification in ("aws-service", "internal", "unknown"):
            event.cloudsentinel.geo_risk_score = 0.0
            return

        # For external IPs: use cloud.region as a proxy for geo if no GeoIP DB
        # In production, integrate MaxMind here
        region = event.cloud.region
        country_code = self._region_to_country(region)
        if country_code:
            event.source.geo = ECSSourceGeo(
                country_iso_code=country_code,
            )
            event.cloudsentinel.geo_risk_score = COUNTRY_RISK.get(
                country_code, _DEFAULT_COUNTRY_RISK
            )
        else:
            event.cloudsentinel.geo_risk_score = _DEFAULT_COUNTRY_RISK

    @staticmethod
    def _region_to_country(aws_region: str) -> str:
        """Map AWS region prefix to ISO country code (best effort)."""
        region_map: dict[str, str] = {
            "us-": "US", "ca-": "CA", "eu-west-1": "IE", "eu-west-2": "GB",
            "eu-west-3": "FR", "eu-central-1": "DE", "eu-central-2": "CH",
            "eu-north-1": "SE", "eu-south-1": "IT", "eu-south-2": "ES",
            "ap-northeast-1": "JP", "ap-northeast-2": "KR", "ap-northeast-3": "JP",
            "ap-southeast-1": "SG", "ap-southeast-2": "AU", "ap-southeast-3": "ID",
            "ap-south-1": "IN", "ap-south-2": "IN", "ap-east-1": "HK",
            "sa-east-1": "BR", "me-south-1": "BH", "me-central-1": "AE",
            "af-south-1": "ZA", "il-central-1": "IL",
        }
        # Exact match first
        if aws_region in region_map:
            return region_map[aws_region]
        # Prefix match
        for prefix, code in region_map.items():
            if aws_region.startswith(prefix):
                return code
        return ""


    def _score_api_risk(self, event: ECSCloudTrailEvent) -> None:
        """Compute API call risk score based on action name."""
        action = event.event.action
        score = API_RISK.get(action, _DEFAULT_API_RISK)

        # Boost if the call failed (attackers often trigger errors during recon)
        if event.event.outcome == "failure" and action in (
            "ConsoleLogin", "AssumeRole", "GetSecretValue",
            "CreateAccessKey", "AttachUserPolicy",
        ):
            score = min(1.0, score + 0.2)

        event.cloudsentinel.api_risk_score = score


    def _score_privilege_escalation(self, event: ECSCloudTrailEvent) -> None:
        """Compute IAM privilege escalation indicator.

        A composite score based on whether the API call is a known
        escalation technique and the context of the call.
        """
        action = event.event.action
        if action not in PRIVILEGE_ESCALATION_APIS:
            return

        # Base score for escalation-type API
        score = 0.5

        # Boost: action on a different user than the caller
        if event.aws.cloudtrail.get("request_parameters"):
            params = event.aws.cloudtrail["request_parameters"]
            if isinstance(params, dict):
                target_user = params.get("userName", "")
                caller_user = event.user.name
                if target_user and caller_user and target_user != caller_user:
                    score += 0.3  # Acting on another user = higher risk

        # Boost: Root user performing escalation (highly unusual)
        if "root" in [r.lower() for r in event.user.roles]:
            score += 0.2

        event.cloudsentinel.api_risk_score = min(1.0, max(
            event.cloudsentinel.api_risk_score, score
        ))
