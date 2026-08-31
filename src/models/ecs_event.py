# FILE: cloudsentinel-zero-trust/src/models/ecs_event.py
"""Pydantic v2 models for ECS 8.10+ normalized CloudTrail events.

Design decisions:
- Follows Elastic Common Schema 8.10 field naming and nesting
- Custom cloudsentinel.* namespace for ML scores and rule matches
- to_opensearch_dict() produces a flat-ish dict optimized for OpenSearch bulk API
- event.original stores the raw JSON string for forensic investigation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ECSBase(BaseModel):
    """ECS base fields (@timestamp, message, tags, labels)."""

    timestamp: datetime = Field(
        alias="@timestamp",
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp in UTC",
    )
    message: str = Field(default="", description="Human-readable event summary")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    labels: dict[str, str] = Field(default_factory=dict, description="Key-value labels")

    model_config = {"populate_by_name": True}


class ECSEvent(BaseModel):
    """ECS event.* fields."""

    kind: str = Field(default="event", description="event | alert | metric | state")
    category: list[str] = Field(default_factory=lambda: ["api"], description="Event category array")
    type: list[str] = Field(default_factory=list, description="Event type: creation, deletion, change, access, info")
    action: str = Field(default="", description="API action name (e.g. CreateUser)")
    provider: str = Field(default="", description="Service that generated the event (e.g. iam.amazonaws.com)")
    outcome: str = Field(default="success", description="success | failure | unknown")
    id: str = Field(default="", description="Unique event ID")
    original: str = Field(default="", description="Raw JSON of original CloudTrail event")
    severity: int = Field(default=0, ge=0, le=100, description="Normalized severity 0-100")
    created: datetime | None = Field(default=None, description="When the event was ingested")


class ECSUser(BaseModel):
    """ECS user.* fields."""

    id: str = Field(default="", description="User principal ID")
    name: str = Field(default="", description="Username or role name")
    roles: list[str] = Field(default_factory=list, description="User roles")
    domain: str = Field(default="", description="AWS account ID (user's domain)")


class ECSRelated(BaseModel):
    """ECS related.* fields for pivot searches."""

    user: list[str] = Field(default_factory=list, description="All related usernames")
    ip: list[str] = Field(default_factory=list, description="All related IPs")


class ECSSourceGeo(BaseModel):
    """ECS source.geo.* fields."""

    country_iso_code: str = Field(default="", description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    city_name: str = Field(default="", description="City name")
    location: dict[str, float] | None = Field(default=None, description="Geo point {lat, lon}")
    region_name: str = Field(default="", description="Region/state name")


class ECSSource(BaseModel):
    """ECS source.* fields."""

    ip: str = Field(default="", description="Source IP address")
    geo: ECSSourceGeo = Field(default_factory=ECSSourceGeo)
    address: str = Field(default="", description="IP or hostname")


class ECSCloud(BaseModel):
    """ECS cloud.* fields."""

    provider: str = Field(default="aws", description="Cloud provider")
    region: str = Field(default="", description="AWS region")
    account: dict[str, str] = Field(default_factory=dict, description="Account ID dict")
    service: dict[str, str] = Field(default_factory=dict, description="Service name dict")


class ECSAws(BaseModel):
    """ECS aws.cloudtrail.* fields — AWS-specific extensions."""

    cloudtrail: dict[str, Any] = Field(
        default_factory=dict,
        description="CloudTrail-specific fields: request_parameters, response_elements, etc.",
    )


class CloudSentinelFields(BaseModel):
    """Custom cloudsentinel.* namespace for ML and rule engine outputs."""

    anomaly_score: float = Field(default=0.0, ge=0.0, le=100.0, description="ML anomaly score 0-100")
    is_anomaly: bool = Field(default=False, description="True if score exceeds threshold")
    contributing_features: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top contributing features from ML model (feature, impact, value)",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Model prediction confidence")
    rule_matches: list[str] = Field(default_factory=list, description="IDs of deterministic rules that matched")
    rule_severities: list[str] = Field(default_factory=list, description="Severity per matched rule")
    mitre_tactics: list[str] = Field(default_factory=list, description="MITRE ATT&CK tactic IDs matched")
    mitre_techniques: list[str] = Field(default_factory=list, description="MITRE ATT&CK technique IDs matched")
    ip_classification: str = Field(default="unknown", description="internal | external | aws-service | known-threat")
    geo_risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Geographic risk 0.0-1.0")
    api_risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="API call risk 0.0-1.0")


class ECSCloudTrailEvent(BaseModel):
    """Complete ECS 8.10+ normalized CloudTrail event with CloudSentinel extensions.

    This is the primary data model ingested into OpenSearch.
    """

    base: ECSBase = Field(default_factory=ECSBase)
    event: ECSEvent = Field(default_factory=ECSEvent)
    user: ECSUser = Field(default_factory=ECSUser)
    related: ECSRelated = Field(default_factory=ECSRelated)
    source: ECSSource = Field(default_factory=ECSSource)
    cloud: ECSCloud = Field(default_factory=ECSCloud)
    aws: ECSAws = Field(default_factory=ECSAws)
    cloudsentinel: CloudSentinelFields = Field(default_factory=CloudSentinelFields)

    model_config = {"populate_by_name": True}

    def to_opensearch_dict(self) -> dict[str, Any]:
        """Serialize to a flat-ish dict optimized for OpenSearch bulk API.

        Flattens nested ECS models using dot notation (e.g., source.ip, user.name)
        which is OpenSearch's expected format when dynamic mapping or the
        index template defines nested field types.
        """
        doc: dict[str, Any] = {}

        # Base fields
        doc["@timestamp"] = self.base.timestamp.isoformat()
        if self.base.message:
            doc["message"] = self.base.message
        if self.base.tags:
            doc["tags"] = self.base.tags
        if self.base.labels:
            doc["labels"] = self.base.labels

        # event.*
        doc["event.kind"] = self.event.kind
        doc["event.category"] = self.event.category
        doc["event.type"] = self.event.type
        doc["event.action"] = self.event.action
        doc["event.provider"] = self.event.provider
        doc["event.outcome"] = self.event.outcome
        doc["event.id"] = self.event.id
        if self.event.original:
            doc["event.original"] = self.event.original
        doc["event.severity"] = self.event.severity
        if self.event.created:
            doc["event.created"] = self.event.created.isoformat()

        # user.*
        doc["user.id"] = self.user.id
        doc["user.name"] = self.user.name
        doc["user.roles"] = self.user.roles
        doc["user.domain"] = self.user.domain

        # related.*
        if self.related.user:
            doc["related.user"] = self.related.user
        if self.related.ip:
            doc["related.ip"] = self.related.ip

        # source.*
        doc["source.ip"] = self.source.ip
        doc["source.address"] = self.source.address
        if self.source.geo.country_iso_code:
            doc["source.geo.country_iso_code"] = self.source.geo.country_iso_code
        if self.source.geo.country_name:
            doc["source.geo.country_name"] = self.source.geo.country_name
        if self.source.geo.city_name:
            doc["source.geo.city_name"] = self.source.geo.city_name
        if self.source.geo.location:
            doc["source.geo.location"] = self.source.geo.location
        if self.source.geo.region_name:
            doc["source.geo.region_name"] = self.source.geo.region_name

        # cloud.*
        doc["cloud.provider"] = self.cloud.provider
        doc["cloud.region"] = self.cloud.region
        if self.cloud.account:
            doc["cloud.account.id"] = self.cloud.account.get("id", "")
        if self.cloud.service:
            doc["cloud.service.name"] = self.cloud.service.get("name", "")

        # aws.cloudtrail.*
        for k, v in self.aws.cloudtrail.items():
            if isinstance(v, (dict, list)):
                doc[f"aws.cloudtrail.{k}"] = json.dumps(v, default=str)
            else:
                doc[f"aws.cloudtrail.{k}"] = v

        # cloudsentinel.* (ML and rule engine outputs)
        doc["cloudsentinel.anomaly_score"] = self.cloudsentinel.anomaly_score
        doc["cloudsentinel.is_anomaly"] = self.cloudsentinel.is_anomaly
        if self.cloudsentinel.contributing_features:
            doc["cloudsentinel.contributing_features"] = self.cloudsentinel.contributing_features
        doc["cloudsentinel.confidence"] = self.cloudsentinel.confidence
        if self.cloudsentinel.rule_matches:
            doc["cloudsentinel.rule_matches"] = self.cloudsentinel.rule_matches
        if self.cloudsentinel.rule_severities:
            doc["cloudsentinel.rule_severities"] = self.cloudsentinel.rule_severities
        if self.cloudsentinel.mitre_tactics:
            doc["cloudsentinel.mitre_tactics"] = self.cloudsentinel.mitre_tactics
        if self.cloudsentinel.mitre_techniques:
            doc["cloudsentinel.mitre_techniques"] = self.cloudsentinel.mitre_techniques
        doc["cloudsentinel.ip_classification"] = self.cloudsentinel.ip_classification
        doc["cloudsentinel.geo_risk_score"] = self.cloudsentinel.geo_risk_score
        doc["cloudsentinel.api_risk_score"] = self.cloudsentinel.api_risk_score

        return doc
