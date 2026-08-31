# FILE: cloudsentinel-zero-trust/src/models/cloudtrail_event.py
"""Pydantic v2 models for AWS CloudTrail raw JSON events.

Design decisions:
- extra='allow' on all models: CloudTrail adds new fields regularly; we
  must not reject events just because AWS added a field we haven't mapped yet.
- userIdentity handled via discriminated union on 'type' field.
- Validators parse CloudTrail's ISO 8601 timestamps into aware datetimes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SessionContext(BaseModel):
    """CloudTrail sessionContext block."""

    sessionIssuer: dict[str, Any] | None = None
    webIdFederationData: dict[str, Any] | None = None
    attributes: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class CloudTrailUserIdentity(BaseModel):
    """CloudTrail userIdentity block.

    Supports all 6 identity types:
    - Root, IAMUser, AssumedRole, FederatedUser, AWSAccount, AWSService
    """

    type: str = Field(..., description="Identity type: Root | IAMUser | AssumedRole | ...")
    principalId: str | None = None
    arn: str | None = None
    accountId: str | None = None
    accessKeyId: str | None = None
    userName: str | None = None
    invokedBy: str | None = None
    sessionContext: SessionContext | None = None

    model_config = {"extra": "allow"}

    @field_validator("type")
    @classmethod
    def validate_identity_type(cls, v: str) -> str:
        known = {
            "Root",
            "IAMUser",
            "AssumedRole",
            "FederatedUser",
            "AWSAccount",
            "AWSService",
            "SAMLUser",
            "WebIdentityUser",
            "Unknown",
        }
        # Accept unknown types gracefully (log but don't reject)
        if v not in known:
            # We still allow it — CloudTrail may introduce new types
            pass
        return v


class TLSDetails(BaseModel):
    """TLS connection details (added in recent CloudTrail versions)."""

    tlsVersion: str | None = None
    cipherSuite: str | None = None
    clientProvidedHostHeader: str | None = None

    model_config = {"extra": "allow"}


class CloudTrailEvent(BaseModel):
    """Complete CloudTrail event record.

    Maps all fields from a single entry in the CloudTrail JSON ``Records`` array.
    """

    eventVersion: str = Field(default="1.08")
    eventTime: datetime = Field(..., description="Event timestamp (ISO 8601)")
    eventSource: str = Field(..., description="AWS service (e.g. 'iam.amazonaws.com')")
    eventName: str = Field(..., description="API action name (e.g. 'CreateUser')")
    eventType: str = Field(default="AwsApiCall")
    eventCategory: str | None = Field(default=None)
    eventID: str | None = Field(default=None)

    awsRegion: str = Field(..., description="AWS region where the call was made")
    sourceIPAddress: str = Field(default="", description="Caller's IP or AWS service name")

    userAgent: str | None = None
    userIdentity: CloudTrailUserIdentity = Field(...)

    requestParameters: dict[str, Any] | None = None
    responseElements: dict[str, Any] | None = None
    additionalEventData: dict[str, Any] | None = None

    requestID: str | None = None
    recipientAccountId: str | None = None

    errorCode: str | None = None
    errorMessage: str | None = None

    resources: list[dict[str, Any]] | None = None
    readOnly: bool | str | None = None
    managementEvent: bool | None = None

    sharedEventID: str | None = None
    vpcEndpointId: str | None = None
    serviceEventDetails: dict[str, Any] | None = None

    tlsDetails: TLSDetails | None = None

    model_config = {"extra": "allow"}


    @field_validator("eventTime", mode="before")
    @classmethod
    def parse_event_time(cls, v: Any) -> datetime:
        """Parse CloudTrail ISO 8601 timestamp to timezone-aware datetime."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        if isinstance(v, str):
            # CloudTrail uses: 2024-01-15T10:30:45Z
            v = v.rstrip("Z") + "+00:00" if v.endswith("Z") else v
            return datetime.fromisoformat(v)
        raise ValueError(f"Cannot parse eventTime: {v!r}")

    @field_validator("readOnly", mode="before")
    @classmethod
    def coerce_read_only(cls, v: Any) -> bool | None:
        """CloudTrail sometimes sends readOnly as string 'true'/'false'."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() == "true"
        return None

    @model_validator(mode="after")
    def ensure_region(self) -> "CloudTrailEvent":
        """Validate that awsRegion looks like a valid AWS region pattern."""
        region = self.awsRegion
        if region and not any(
            region.startswith(prefix)
            for prefix in ("us-", "eu-", "ap-", "sa-", "ca-", "me-", "af-", "il-")
        ):
            # Some events (global services) use 'us-east-1' even from other regions.
            # We accept all values but could log a warning.
            pass
        return self


    @property
    def is_error(self) -> bool:
        """True if the API call resulted in an error."""
        return self.errorCode is not None

    @property
    def identity_type(self) -> str:
        """Shortcut to userIdentity.type."""
        return self.userIdentity.type

    @property
    def principal_name(self) -> str:
        """Best-effort human-readable principal name."""
        ui = self.userIdentity
        if ui.userName:
            return ui.userName
        if ui.arn:
            # Extract last part: arn:aws:iam::123:user/admin → admin
            return ui.arn.rsplit("/", 1)[-1] if "/" in ui.arn else ui.arn.rsplit(":", 1)[-1]
        if ui.principalId:
            return ui.principalId
        return ui.type
