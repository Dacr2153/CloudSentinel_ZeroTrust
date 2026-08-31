# FILE: cloudsentinel-zero-trust/src/pipeline/feature_engineer.py
"""Feature engineering for ML anomaly detection (Isolation Forest).

Extracts a fixed-length feature vector (10 features) from each enriched
ECSCloudTrailEvent. All features are normalized to [0, 1].

Feature Vector (10 dimensions):
───────────────────────────────
 0  hour_of_day_normalized      — sin/cos-free linear [0,1] mapping of UTC hour
 1  day_of_week_normalized      — 0=Monday .. 6=Sunday → [0, 1]
 2  is_weekend                  — 0.0 or 1.0
 3  source_geo_risk_score       — from enricher (already [0,1])
 4  api_call_risk_score         — from enricher (already [0,1])
 5  is_root_user                — 0.0 or 1.0
 6  is_cross_region             — 0.0 or 1.0 (event region ≠ home region)
 7  failed_auth_velocity        — proxy via error presence + sensitive API
 8  api_entropy_1h              — approximated by action cardinality signal
 9  privilege_escalation_score  — from enricher escalation APIs
"""

from __future__ import annotations

import numpy as np

from src.models.ecs_event import ECSCloudTrailEvent
from src.pipeline.enricher import PRIVILEGE_ESCALATION_APIS
from src.utils.config import get_settings
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="feature_engineer")

# Feature names in vector order — useful for explaining anomalies
FEATURE_NAMES: list[str] = [
    "hour_of_day_normalized",
    "day_of_week_normalized",
    "is_weekend",
    "source_geo_risk_score",
    "api_call_risk_score",
    "is_root_user",
    "is_cross_region",
    "failed_auth_velocity",
    "api_entropy_1h",
    "privilege_escalation_score",
]

NUM_FEATURES: int = len(FEATURE_NAMES)


class FeatureEngineer:
    """Converts enriched ECS events into numerical feature vectors.

    Each event produces a 1-D numpy array of shape (10,) with values in [0, 1].
    """

    def __init__(self, home_region: str | None = None) -> None:
        self._home_region = home_region or self._get_home_region()

    @staticmethod
    def _get_home_region() -> str:
        """Resolve home region from settings (defaults to us-east-1)."""
        try:
            settings = get_settings()
            return settings.aws_region
        except Exception:
            return "us-east-1"


    def extract_features(self, event: ECSCloudTrailEvent) -> np.ndarray:
        """Extract a fixed-length feature vector from a single event.

        Returns:
            np.ndarray of shape (10,) with float64 values in [0, 1].
        """
        features = np.zeros(NUM_FEATURES, dtype=np.float64)

        features[0] = self._hour_of_day(event)
        features[1] = self._day_of_week(event)
        features[2] = self._is_weekend(event)
        features[3] = self._geo_risk_score(event)
        features[4] = self._api_risk_score(event)
        features[5] = self._is_root_user(event)
        features[6] = self._is_cross_region(event)
        features[7] = self._failed_auth_velocity(event)
        features[8] = self._api_entropy_signal(event)
        features[9] = self._privilege_escalation_score(event)

        return features


    @staticmethod
    def _hour_of_day(event: ECSCloudTrailEvent) -> float:
        """UTC hour normalized to [0, 1]: 0h → 0.0, 23h → 23/23 ≈ 1.0."""
        ts = event.base.timestamp
        if ts:
            return ts.hour / 23.0
        return 0.5  # midday default

    @staticmethod
    def _day_of_week(event: ECSCloudTrailEvent) -> float:
        """Day of week normalized: Monday=0.0, Sunday≈1.0."""
        ts = event.base.timestamp
        if ts:
            return ts.weekday() / 6.0
        return 0.5

    @staticmethod
    def _is_weekend(event: ECSCloudTrailEvent) -> float:
        """1.0 if Saturday or Sunday, else 0.0."""
        ts = event.base.timestamp
        if ts:
            return 1.0 if ts.weekday() >= 5 else 0.0
        return 0.0

    @staticmethod
    def _geo_risk_score(event: ECSCloudTrailEvent) -> float:
        """Geo risk score from enricher, already [0, 1]."""
        return _clamp(event.cloudsentinel.geo_risk_score)

    @staticmethod
    def _api_risk_score(event: ECSCloudTrailEvent) -> float:
        """API risk score from enricher, already [0, 1]."""
        return _clamp(event.cloudsentinel.api_risk_score)

    @staticmethod
    def _is_root_user(event: ECSCloudTrailEvent) -> float:
        """1.0 if the acting identity is the root account."""
        roles = event.user.roles
        if roles and any(r.lower() == "root" for r in roles):
            return 1.0
        return 0.0

    def _is_cross_region(self, event: ECSCloudTrailEvent) -> float:
        """1.0 if the event's region differs from the home region."""
        region = event.cloud.region
        if region and region != self._home_region:
            return 1.0
        return 0.0

    @staticmethod
    def _failed_auth_velocity(event: ECSCloudTrailEvent) -> float:
        """Proxy for failed authentication velocity.

        In a full implementation this would aggregate failures over a time
        window (e.g., 5 min). In single-event mode we use a composite signal:
        - 0.8 if the event is an auth-related failure
        - 0.3 if any error occurred on a sensitive API
        - 0.0 otherwise
        """
        auth_actions = {
            "ConsoleLogin", "AssumeRole", "GetSessionToken",
            "GetFederationToken", "AssumeRoleWithSAML",
            "AssumeRoleWithWebIdentity",
        }
        action = event.event.action
        is_failure = event.event.outcome == "failure"

        if is_failure and action in auth_actions:
            return 0.8
        if is_failure and action in PRIVILEGE_ESCALATION_APIS:
            return 0.5
        if is_failure:
            return 0.3
        return 0.0

    @staticmethod
    def _api_entropy_signal(event: ECSCloudTrailEvent) -> float:
        """Proxy for API call entropy over 1 hour.

        True entropy requires historical aggregation. Here we use a composite
        signal based on action characteristics:
        - Unusual actions (long names with many segments) get higher score
        - Read-only operations score lower than writes
        """
        action = event.event.action
        if not action:
            return 0.0

        # Longer action names with more segments tend to be rarer
        segments = action.count(":") + action.count(".")
        length_score = min(1.0, len(action) / 40.0)
        segment_score = min(1.0, segments / 3.0)

        # Read-only gets penalized (less unusual)
        read_penalty = 0.7 if (
            action.startswith("Get") or action.startswith("List")
            or action.startswith("Describe") or action.startswith("Head")
        ) else 1.0

        return _clamp((length_score * 0.5 + segment_score * 0.5) * read_penalty)

    @staticmethod
    def _privilege_escalation_score(event: ECSCloudTrailEvent) -> float:
        """Score for IAM privilege escalation potential.

        Composite of: is_escalation_api + target_different_from_caller + is_error
        """
        action = event.event.action
        if action not in PRIVILEGE_ESCALATION_APIS:
            return 0.0

        score = 0.5  # Base: known escalation API

        # Higher score if the call succeeded (actual escalation)
        if event.event.outcome == "success":
            score += 0.2

        # Add if acting on another user
        if event.aws.cloudtrail.get("request_parameters"):
            params = event.aws.cloudtrail["request_parameters"]
            if isinstance(params, dict):
                target = params.get("userName", "")
                caller = event.user.name
                if target and caller and target != caller:
                    score += 0.2

        return _clamp(score)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))
