# FILE: cloudsentinel-zero-trust/src/pipeline/normalizer.py
"""CloudTrail → ECS 8.10+ normalization.

Design decisions:
- Complete field mapping following Elastic Common Schema 8.10
- event.type derived from eventName heuristics (Create→creation, Delete→deletion, etc.)
- event.outcome derived from errorCode presence
- event.original stores the full raw JSON for forensics
- All userIdentity types handled (Root, IAMUser, AssumedRole, FederatedUser, AWSService)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.models.cloudtrail_event import CloudTrailEvent
from src.models.ecs_event import (
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
from src.utils.exceptions import NormalizationError
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="normalizer")


# Derive ECS event.type array from CloudTrail eventName prefix/suffix patterns

_EVENT_TYPE_PREFIXES: dict[str, str] = {
    "Create": "creation",
    "Put": "creation",
    "Run": "creation",
    "Start": "start",
    "Allocate": "creation",
    "Register": "creation",
    "Delete": "deletion",
    "Remove": "deletion",
    "Terminate": "deletion",
    "Deregister": "deletion",
    "Release": "deletion",
    "Update": "change",
    "Modify": "change",
    "Set": "change",
    "Attach": "change",
    "Detach": "change",
    "Enable": "change",
    "Disable": "change",
    "Authorize": "change",
    "Revoke": "change",
    "Get": "access",
    "Describe": "access",
    "List": "access",
    "Head": "access",
    "Lookup": "access",
    "Assume": "access",
    "Console": "access",
    "Login": "access",
}


def _derive_event_types(event_name: str) -> list[str]:
    """Derive ECS event.type array from CloudTrail eventName."""
    types: list[str] = []
    for prefix, ecs_type in _EVENT_TYPE_PREFIXES.items():
        if event_name.startswith(prefix):
            types.append(ecs_type)
            break
    if not types:
        types.append("info")
    return types


def _derive_event_category(event_source: str) -> list[str]:
    """Derive ECS event.category from CloudTrail eventSource."""
    source_lower = event_source.lower()
    if "iam" in source_lower or "sts" in source_lower:
        return ["iam"]
    if "s3" in source_lower:
        return ["file"]
    if "ec2" in source_lower:
        return ["host"]
    if "login" in source_lower or "signin" in source_lower:
        return ["authentication"]
    if "kms" in source_lower or "secretsmanager" in source_lower:
        return ["configuration"]
    if "guardduty" in source_lower or "securityhub" in source_lower:
        return ["intrusion_detection"]
    return ["api"]


class ECSNormalizer:
    """Transforms CloudTrail raw events into ECS 8.10+ normalized events."""

    def normalize(self, raw: CloudTrailEvent) -> ECSCloudTrailEvent:
        """Normalize a single CloudTrail event to ECS 8.10+.

        Args:
            raw: Validated CloudTrailEvent from extractor.

        Returns:
            ECSCloudTrailEvent ready for enrichment and ingestion.

        Raises:
            NormalizationError: If a critical mapping fails.
        """
        try:
            return self._map(raw)
        except NormalizationError:
            raise
        except Exception as exc:
            raise NormalizationError(
                f"Normalization failed for event {raw.eventName}: {exc}",
                context={
                    "event_name": raw.eventName,
                    "event_source": raw.eventSource,
                    "event_id": raw.eventID or "",
                },
            ) from exc

    def normalize_batch(self, events: list[CloudTrailEvent]) -> list[ECSCloudTrailEvent]:
        """Normalize a batch of events. Skips individual failures.

        Args:
            events: List of CloudTrailEvent objects.

        Returns:
            List of successfully normalized ECS events.
        """
        results: list[ECSCloudTrailEvent] = []
        for raw in events:
            try:
                ecs = self.normalize(raw)
                results.append(ecs)
            except NormalizationError as exc:
                logger.warning("Skipping event: %s", exc.message)
        return results


    def _map(self, raw: CloudTrailEvent) -> ECSCloudTrailEvent:
        """Full CloudTrail → ECS field mapping."""

        # ── event.outcome ──
        outcome = "failure" if raw.is_error else "success"

        # ── event.original (raw JSON for forensics) ──
        try:
            original_json = raw.model_dump_json(exclude_none=True)
        except Exception:
            original_json = ""

        # ── Base ──
        base = ECSBase(
            **{  # type: ignore[arg-type]  # Pydantic alias
                "@timestamp": raw.eventTime.astimezone(timezone.utc),
            },
            message=f"{raw.eventName} by {raw.principal_name} from {raw.sourceIPAddress}",
            tags=["cloudsentinel", "cloudtrail"],
        )

        # ── Event ──
        event = ECSEvent(
            kind="event",
            category=_derive_event_category(raw.eventSource),
            type=_derive_event_types(raw.eventName),
            action=raw.eventName,
            provider=raw.eventSource,
            outcome=outcome,
            id=raw.eventID or "",
            original=original_json,
            severity=0,  # Will be set by rule_engine / anomaly_detector
            created=datetime.now(timezone.utc),
        )

        # ── User ──
        user = self._map_user(raw)

        # ── Related ──
        related_users: list[str] = []
        if user.name:
            related_users.append(user.name)
        if user.id and user.id != user.name:
            related_users.append(user.id)

        related_ips: list[str] = []
        if raw.sourceIPAddress and not raw.sourceIPAddress.endswith(".amazonaws.com"):
            related_ips.append(raw.sourceIPAddress)

        related = ECSRelated(user=related_users, ip=related_ips)

        # ── Source ──
        source = ECSSource(
            ip=raw.sourceIPAddress if self._is_ip(raw.sourceIPAddress) else "",
            address=raw.sourceIPAddress,
            geo=ECSSourceGeo(),  # Populated later by enricher
        )

        # ── Cloud ──
        cloud = ECSCloud(
            provider="aws",
            region=raw.awsRegion,
            account={"id": raw.recipientAccountId or ""},
            service={"name": raw.eventSource.replace(".amazonaws.com", "")},
        )

        # ── AWS-specific ──
        cloudtrail_data: dict[str, Any] = {}
        if raw.requestParameters:
            cloudtrail_data["request_parameters"] = raw.requestParameters
        if raw.responseElements:
            cloudtrail_data["response_elements"] = raw.responseElements
        if raw.additionalEventData:
            cloudtrail_data["additional_event_data"] = raw.additionalEventData
        if raw.userAgent:
            cloudtrail_data["user_agent"] = raw.userAgent
        if raw.errorCode:
            cloudtrail_data["error_code"] = raw.errorCode
        if raw.errorMessage:
            cloudtrail_data["error_message"] = raw.errorMessage
        if raw.readOnly is not None:
            cloudtrail_data["read_only"] = raw.readOnly
        if raw.managementEvent is not None:
            cloudtrail_data["management_event"] = raw.managementEvent
        if raw.resources:
            cloudtrail_data["resources"] = raw.resources
        if raw.sharedEventID:
            cloudtrail_data["shared_event_id"] = raw.sharedEventID
        cloudtrail_data["event_type"] = raw.eventType
        cloudtrail_data["event_category"] = raw.eventCategory or ""
        cloudtrail_data["event_version"] = raw.eventVersion

        aws = ECSAws(cloudtrail=cloudtrail_data)

        return ECSCloudTrailEvent(
            base=base,
            event=event,
            user=user,
            related=related,
            source=source,
            cloud=cloud,
            aws=aws,
        )

    def _map_user(self, raw: CloudTrailEvent) -> ECSUser:
        """Map CloudTrail userIdentity to ECS user.* fields.

        Handles all identity types:
        - Root: user.name='Root', user.roles=['root']
        - IAMUser: user.name=userName
        - AssumedRole: extracts role name and session name
        - FederatedUser: extracts federated identity
        - AWSService: user.name=invokedBy (e.g. 'ec2.amazonaws.com')
        - AWSAccount: external account cross-account call
        """
        ui = raw.userIdentity

        user_name = ""
        user_id = ui.principalId or ""
        roles: list[str] = []
        domain = ui.accountId or raw.recipientAccountId or ""

        identity_type = ui.type

        if identity_type == "Root":
            user_name = "Root"
            roles = ["root"]

        elif identity_type == "IAMUser":
            user_name = ui.userName or ""

        elif identity_type == "AssumedRole":
            # ARN pattern: arn:aws:sts::123:assumed-role/RoleName/SessionName
            if ui.arn:
                parts = ui.arn.split("/")
                if len(parts) >= 2:
                    roles.append(parts[1])
                if len(parts) >= 3:
                    user_name = parts[2]  # Session name
                else:
                    user_name = parts[-1]
            # principalId pattern: AROA...:SessionName
            if not user_name and ui.principalId and ":" in ui.principalId:
                user_name = ui.principalId.split(":", 1)[1]

        elif identity_type == "FederatedUser":
            if ui.arn:
                user_name = ui.arn.rsplit("/", 1)[-1] if "/" in ui.arn else ""
            if not user_name:
                user_name = ui.principalId or ""

        elif identity_type == "AWSService":
            user_name = ui.invokedBy or "aws-service"

        elif identity_type == "AWSAccount":
            user_name = f"account:{ui.accountId}" if ui.accountId else "external-account"

        else:
            # Unknown type — best effort
            user_name = ui.userName or ui.principalId or identity_type

        return ECSUser(
            id=user_id,
            name=user_name,
            roles=roles,
            domain=domain,
        )

    @staticmethod
    def _is_ip(value: str) -> bool:
        """Check if a string looks like an IP address (v4 or v6)."""
        if not value:
            return False
        # IPv4 quick check
        if value.count(".") == 3:
            parts = value.split(".")
            try:
                return all(0 <= int(p) <= 255 for p in parts)
            except ValueError:
                return False
        # IPv6 quick check
        if ":" in value and not value.endswith(".com"):
            return True
        return False
