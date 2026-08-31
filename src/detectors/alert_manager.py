# FILE: cloudsentinel-zero-trust/src/detectors/alert_manager.py
"""Alert management: deduplication, correlation, severity calculation, and SNS dispatch.

Design:
- Deduplication: SHA-256 hash of (rule_id, user, source_ip, action) → 5-min window
- Correlation: group related alerts within 5 min into a campaign
- Severity: composite of rule severity + anomaly score + rule match count
- Dispatch: SNS topic for email + future Slack/PagerDuty hooks
- Persistence: index alerts into OpenSearch for dashboards
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any

from src.detectors.rule_engine import RuleMatch
from src.utils.exceptions import AlertError
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="alert_manager")

# Alert constants
DEDUP_WINDOW_SECONDS = 300  # 5 minutes
CORRELATION_WINDOW_SECONDS = 300
ALERT_INDEX_PREFIX = "cloudsentinel-alerts"

# Severity order for comparisons
SEVERITY_ORDER = {
    "informational": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


class Alert:
    """Structured alert produced by the alert manager."""

    def __init__(
        self,
        alert_id: str,
        dedup_hash: str,
        severity: str,
        title: str,
        description: str,
        rule_matches: list[RuleMatch],
        anomaly_score: int | None = None,
        is_anomaly: bool = False,
        contributing_features: list[dict[str, Any]] | None = None,
        event_id: str = "",
        event_action: str = "",
        user_name: str = "",
        source_ip: str = "",
        cloud_region: str = "",
        cloud_account_id: str = "",
        timestamp: datetime | None = None,
        campaign_id: str | None = None,
    ) -> None:
        self.alert_id = alert_id
        self.dedup_hash = dedup_hash
        self.severity = severity
        self.title = title
        self.description = description
        self.rule_matches = rule_matches
        self.anomaly_score = anomaly_score
        self.is_anomaly = is_anomaly
        self.contributing_features = contributing_features or []
        self.event_id = event_id
        self.event_action = event_action
        self.user_name = user_name
        self.source_ip = source_ip
        self.cloud_region = cloud_region
        self.cloud_account_id = cloud_account_id
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.campaign_id = campaign_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize for OpenSearch indexing."""
        return {
            "@timestamp": self.timestamp.isoformat(),
            "alert_id": self.alert_id,
            "dedup_hash": self.dedup_hash,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "rules": [
                {
                    "rule_id": m.rule_id,
                    "rule_name": m.rule_name,
                    "severity": m.severity,
                    "mitre_tactic": m.mitre_tactic,
                    "mitre_technique": m.mitre_technique,
                    "evidence": m.evidence,
                }
                for m in self.rule_matches
            ],
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "contributing_features": self.contributing_features,
            "event": {
                "id": self.event_id,
                "action": self.event_action,
            },
            "user": {"name": self.user_name},
            "source": {"ip": self.source_ip},
            "cloud": {
                "region": self.cloud_region,
                "account_id": self.cloud_account_id,
            },
            "campaign_id": self.campaign_id,
        }

    def to_sns_message(self) -> str:
        """Format as human-readable text for SNS email notifications."""
        rules_text = "\n".join(
            f"  - [{m.rule_id}] {m.rule_name} (severity={m.severity}, "
            f"MITRE={m.mitre_tactic}/{m.mitre_technique})"
            for m in self.rule_matches
        )

        features_text = ""
        if self.contributing_features:
            features_text = "\nTop Contributing Features:\n" + "\n".join(
                f"  - {f['feature']}: value={f.get('value', 'N/A')}, "
                f"impact={f.get('impact', 'N/A')}"
                for f in self.contributing_features[:3]
            )

        return (
            f"{'='*60}\n"
            f"  CLOUDSENTINEL ZERO-TRUST ALERT\n"
            f"{'='*60}\n"
            f"\n"
            f"Severity:   {self.severity.upper()}\n"
            f"Title:      {self.title}\n"
            f"Timestamp:  {self.timestamp.isoformat()}\n"
            f"Alert ID:   {self.alert_id}\n"
            f"Campaign:   {self.campaign_id or 'N/A'}\n"
            f"\n"
            f"{'─'*60}\n"
            f"  EVENT DETAILS\n"
            f"{'─'*60}\n"
            f"Action:     {self.event_action}\n"
            f"User:       {self.user_name}\n"
            f"Source IP:  {self.source_ip}\n"
            f"Region:     {self.cloud_region}\n"
            f"Account:    {self.cloud_account_id}\n"
            f"\n"
            f"{'─'*60}\n"
            f"  DETECTION\n"
            f"{'─'*60}\n"
            f"Anomaly Score: {self.anomaly_score if self.anomaly_score is not None else 'N/A'}/100\n"
            f"Is Anomaly:    {self.is_anomaly}\n"
            f"{features_text}\n"
            f"\n"
            f"Rules Triggered:\n{rules_text}\n"
            f"\n"
            f"Description:\n  {self.description}\n"
            f"{'='*60}\n"
        )


class AlertManager:
    """Creates, deduplicates, correlates, and dispatches security alerts.

    Parameters:
        sns_client: boto3 SNS client (injected).
        opensearch_client: opensearch-py client (injected, optional for alert indexing).
        sns_topic_arn: ARN of the SNS notification topic.
    """

    def __init__(
        self,
        sns_client: Any,
        sns_topic_arn: str,
        opensearch_client: Any | None = None,
    ) -> None:
        self._sns = sns_client
        self._sns_topic_arn = sns_topic_arn
        self._os_client = opensearch_client
        # In-memory dedup cache: hash → timestamp
        self._dedup_cache: dict[str, float] = {}
        # Campaign tracker: user → campaign_id with last-seen timestamp
        self._campaigns: dict[str, tuple[str, float]] = {}


    def process_event(
        self,
        event_id: str,
        event_action: str,
        user_name: str,
        source_ip: str,
        cloud_region: str,
        cloud_account_id: str,
        rule_matches: list[RuleMatch],
        anomaly_score: int | None = None,
        is_anomaly: bool = False,
        contributing_features: list[dict[str, Any]] | None = None,
        timestamp: datetime | None = None,
    ) -> Alert | None:
        """Process detection results and generate/dispatch an alert if warranted.

        Returns the Alert if dispatched, None if deduplicated or below threshold.
        """
        # Only alert if rules matched OR anomaly detected
        if not rule_matches and not is_anomaly:
            return None

        now = time.monotonic()
        ts = timestamp or datetime.now(timezone.utc)

        # Build dedup hash
        dedup_hash = self._compute_dedup_hash(rule_matches, user_name, source_ip, event_action)

        # Dedup check
        if self._is_duplicate(dedup_hash, now):
            logger.debug(
                "Alert deduplicated (hash=%s, event=%s)", dedup_hash[:12], event_id
            )
            return None

        # Record for dedup
        self._dedup_cache[dedup_hash] = now
        self._cleanup_dedup_cache(now)

        # Resolve campaign
        campaign_id = self._resolve_campaign(user_name, now)

        # Compute composite severity
        severity = self._compute_severity(rule_matches, anomaly_score)

        # Build title and description
        title = self._build_title(rule_matches, is_anomaly, event_action)
        description = self._build_description(
            rule_matches, anomaly_score, is_anomaly, contributing_features
        )

        # Generate alert ID
        alert_id = f"ALERT-{ts.strftime('%Y%m%d%H%M%S')}-{dedup_hash[:8]}"

        alert = Alert(
            alert_id=alert_id,
            dedup_hash=dedup_hash,
            severity=severity,
            title=title,
            description=description,
            rule_matches=rule_matches,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            contributing_features=contributing_features,
            event_id=event_id,
            event_action=event_action,
            user_name=user_name,
            source_ip=source_ip,
            cloud_region=cloud_region,
            cloud_account_id=cloud_account_id,
            timestamp=ts,
            campaign_id=campaign_id,
        )

        # Dispatch
        self._publish_sns(alert)

        # Index in OpenSearch (best effort)
        if self._os_client:
            self._index_alert(alert)

        logger.info(
            "Alert dispatched: %s severity=%s rules=%d anomaly_score=%s",
            alert_id,
            severity,
            len(rule_matches),
            anomaly_score,
        )

        return alert


    @staticmethod
    def _compute_dedup_hash(
        rule_matches: list[RuleMatch],
        user_name: str,
        source_ip: str,
        event_action: str,
    ) -> str:
        """SHA-256 hash for deduplication."""
        rule_ids = sorted(m.rule_id for m in rule_matches) if rule_matches else ["anomaly"]
        payload = f"{','.join(rule_ids)}|{user_name}|{source_ip}|{event_action}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _is_duplicate(self, dedup_hash: str, now: float) -> bool:
        """Check if an identical alert was dispatched within the dedup window."""
        last_seen = self._dedup_cache.get(dedup_hash)
        if last_seen is None:
            return False
        return (now - last_seen) < DEDUP_WINDOW_SECONDS

    def _cleanup_dedup_cache(self, now: float) -> None:
        """Evict expired dedup entries."""
        expired = [
            h for h, ts in self._dedup_cache.items()
            if (now - ts) >= DEDUP_WINDOW_SECONDS
        ]
        for h in expired:
            del self._dedup_cache[h]


    def _resolve_campaign(self, user_name: str, now: float) -> str:
        """Assign a campaign ID (group related alerts within 5 min for same user)."""
        existing = self._campaigns.get(user_name)
        if existing:
            campaign_id, last_time = existing
            if (now - last_time) < CORRELATION_WINDOW_SECONDS:
                # Same campaign, update timestamp
                self._campaigns[user_name] = (campaign_id, now)
                return campaign_id

        # New campaign
        campaign_id = f"CAMPAIGN-{user_name[:20]}-{int(time.time())}"
        self._campaigns[user_name] = (campaign_id, now)
        return campaign_id


    @staticmethod
    def _compute_severity(
        rule_matches: list[RuleMatch], anomaly_score: int | None
    ) -> str:
        """Composite severity from rule severities + anomaly score.

        Logic:
        1. Start with the maximum rule severity
        2. Boost by one level if anomaly_score >= 80
        3. Boost by one level if >= 3 rules triggered
        4. Cap at 'critical'
        """
        if not rule_matches and anomaly_score is not None:
            # Pure anomaly (no rule matches)
            if anomaly_score >= 90:
                return "critical"
            if anomaly_score >= 75:
                return "high"
            if anomaly_score >= 60:
                return "medium"
            return "low"

        # Maximum rule severity
        max_sev = max(
            (SEVERITY_ORDER.get(m.severity, 0) for m in rule_matches),
            default=0,
        )

        # Boost for high anomaly score
        if anomaly_score is not None and anomaly_score >= 80:
            max_sev = min(4, max_sev + 1)

        # Boost for multiple rule matches
        if len(rule_matches) >= 3:
            max_sev = min(4, max_sev + 1)

        # Reverse lookup
        for name, order in SEVERITY_ORDER.items():
            if order == max_sev:
                return name
        return "medium"


    @staticmethod
    def _build_title(
        rule_matches: list[RuleMatch], is_anomaly: bool, event_action: str
    ) -> str:
        if rule_matches and is_anomaly:
            primary = rule_matches[0].rule_name
            return f"[Rule+ML] {primary} — {event_action}"
        if rule_matches:
            primary = rule_matches[0].rule_name
            return f"[Rule] {primary} — {event_action}"
        return f"[ML Anomaly] Anomalous activity detected — {event_action}"

    @staticmethod
    def _build_description(
        rule_matches: list[RuleMatch],
        anomaly_score: int | None,
        is_anomaly: bool,
        contributing_features: list[dict[str, Any]] | None,
    ) -> str:
        parts: list[str] = []
        if rule_matches:
            rules_desc = ", ".join(
                f"{m.rule_id} ({m.rule_name})" for m in rule_matches
            )
            parts.append(f"Rules triggered: {rules_desc}.")
        if is_anomaly and anomaly_score is not None:
            parts.append(f"ML anomaly score: {anomaly_score}/100.")
            if contributing_features:
                top = contributing_features[:3]
                feat_desc = ", ".join(f['feature'] for f in top)
                parts.append(f"Top contributing features: {feat_desc}.")
        mitre = set()
        for m in rule_matches:
            mitre.add(f"{m.mitre_tactic}/{m.mitre_technique}")
        if mitre:
            parts.append(f"MITRE ATT&CK: {', '.join(sorted(mitre))}.")
        return " ".join(parts)


    def _publish_sns(self, alert: Alert) -> None:
        """Publish alert to SNS topic."""
        try:
            subject = f"[CloudSentinel {alert.severity.upper()}] {alert.title}"
            # SNS subject max 100 chars
            if len(subject) > 100:
                subject = subject[:97] + "..."
            self._sns.publish(
                TopicArn=self._sns_topic_arn,
                Subject=subject,
                Message=alert.to_sns_message(),
                MessageAttributes={
                    "severity": {
                        "DataType": "String",
                        "StringValue": alert.severity,
                    },
                    "alert_id": {
                        "DataType": "String",
                        "StringValue": alert.alert_id,
                    },
                },
            )
        except Exception as exc:
            logger.error("Failed to publish SNS alert: %s", exc)
            raise AlertError(
                f"SNS publish failed: {exc}",
                context={"topic_arn": self._sns_topic_arn, "alert_id": alert.alert_id},
            ) from exc

    def _index_alert(self, alert: Alert) -> None:
        """Index alert into OpenSearch (best effort)."""
        if self._os_client is None:
            return
        try:
            index_name = (
                f"{ALERT_INDEX_PREFIX}-{alert.timestamp.strftime('%Y.%m.%d')}"
            )
            self._os_client.index(
                index=index_name,
                body=alert.to_dict(),
                id=alert.alert_id,
            )
        except Exception as exc:
            logger.warning("Failed to index alert %s: %s", alert.alert_id, exc)
