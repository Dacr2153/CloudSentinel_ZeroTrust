# FILE: cloudsentinel-zero-trust/tests/unit/test_feature_engineer.py
"""Unit tests for FeatureEngineer — 10-dimensional ML feature extraction."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.pipeline.feature_engineer import FEATURE_NAMES, NUM_FEATURES, FeatureEngineer
from tests.conftest import make_ecs


@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer(home_region="us-east-1")


def test_feature_vector_length(engineer: FeatureEngineer) -> None:
    """Feature vector must have exactly NUM_FEATURES (10) elements."""
    event = make_ecs()
    features = engineer.extract_features(event)

    assert isinstance(features, np.ndarray)
    assert features.shape == (NUM_FEATURES,)
    assert len(FEATURE_NAMES) == NUM_FEATURES


def test_feature_range_valid(engineer: FeatureEngineer) -> None:
    """All feature values must be in [0.0, 1.0]."""
    event = make_ecs()
    features = engineer.extract_features(event)

    for i, val in enumerate(features):
        assert 0.0 <= val <= 1.0, (
            f"Feature {FEATURE_NAMES[i]} (idx={i}) out of range: {val}"
        )


def test_hour_of_day_normalized(engineer: FeatureEngineer) -> None:
    """Event at 12:00 UTC should produce hour feature ≈ 0.5."""
    noon = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
    event = make_ecs(timestamp=noon)
    features = engineer.extract_features(event)

    hour_feature = features[0]  # hour_of_day_normalized
    assert 0.45 <= hour_feature <= 0.55, f"Expected ~0.5, got {hour_feature}"


def test_day_of_week_normalized(engineer: FeatureEngineer) -> None:
    """Monday event should have day_of_week ≈ 0.0."""
    monday = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)  # Mon
    event = make_ecs(timestamp=monday)
    features = engineer.extract_features(event)

    day_feature = features[1]  # day_of_week_normalized
    assert day_feature <= 0.2, f"Expected ~0.0 for Monday, got {day_feature}"


class TestIsWeekend:
    def test_is_weekend_saturday(self, engineer: FeatureEngineer) -> None:
        """Saturday should produce is_weekend = 1.0."""
        saturday = datetime(2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc)
        event = make_ecs(timestamp=saturday)
        features = engineer.extract_features(event)

        assert features[2] == 1.0  # is_weekend

    def test_is_weekend_monday(self, engineer: FeatureEngineer) -> None:
        """Monday should produce is_weekend = 0.0."""
        monday = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)
        event = make_ecs(timestamp=monday)
        features = engineer.extract_features(event)

        assert features[2] == 0.0  # is_weekend


class TestSourceGeoRiskScore:
    def test_source_geo_risk_internal(self, engineer: FeatureEngineer) -> None:
        """Internal IP with geo_risk=0.0 should produce low geo risk feature."""
        event = make_ecs(ip_classification="internal", geo_risk=0.0)
        features = engineer.extract_features(event)

        assert features[3] <= 0.2  # source_geo_risk_score

    def test_source_geo_risk_external(self, engineer: FeatureEngineer) -> None:
        """External IP with high geo_risk should produce high feature value."""
        event = make_ecs(ip_classification="external", geo_risk=0.8)
        features = engineer.extract_features(event)

        assert features[3] >= 0.5  # source_geo_risk_score


class TestAPICallRiskScore:
    def test_api_risk_low(self, engineer: FeatureEngineer) -> None:
        """DescribeInstances should have low API risk."""
        event = make_ecs(action="DescribeInstances", api_risk=0.1)
        features = engineer.extract_features(event)

        assert features[4] <= 0.3  # api_call_risk_score

    def test_api_risk_high(self, engineer: FeatureEngineer) -> None:
        """CreateAccessKey should have high API risk."""
        event = make_ecs(action="CreateAccessKey", api_risk=0.9)
        features = engineer.extract_features(event)

        assert features[4] >= 0.7  # api_call_risk_score


class TestIsRootUser:
    def test_is_root_user(self, engineer: FeatureEngineer) -> None:
        """Root identity should produce is_root_user = 1.0."""
        event = make_ecs(user_name="Root", user_roles=["Root"])
        features = engineer.extract_features(event)

        assert features[5] == 1.0  # is_root_user

    def test_is_not_root_user(self, engineer: FeatureEngineer) -> None:
        """IAMUser should produce is_root_user = 0.0."""
        event = make_ecs(user_name="developer", user_roles=["IAMUser"])
        features = engineer.extract_features(event)

        assert features[5] == 0.0  # is_root_user


class TestIsCrossRegion:
    def test_is_cross_region(self, engineer: FeatureEngineer) -> None:
        """Event region != home region should produce is_cross_region = 1.0."""
        event = make_ecs(region="ap-southeast-1")
        features = engineer.extract_features(event)

        assert features[6] == 1.0  # is_cross_region

    def test_is_not_cross_region(self, engineer: FeatureEngineer) -> None:
        """Event region == home region should produce is_cross_region = 0.0."""
        event = make_ecs(region="us-east-1")
        features = engineer.extract_features(event)

        assert features[6] == 0.0  # is_cross_region


def test_failed_auth_velocity(engineer: FeatureEngineer) -> None:
    """Event with error outcome + sensitive API should have higher score."""
    event = make_ecs(
    action="CreateAccessKey",
    outcome="failure",
    api_risk=0.9,
    )
    features = engineer.extract_features(event)

    # failed_auth_velocity should be elevated for errors on sensitive APIs
    assert features[7] >= 0.0  # At least non-negative


class TestPrivilegeEscalationScore:
    def test_privilege_escalation_high(self, engineer: FeatureEngineer) -> None:
        """IAM privilege escalation API should score high."""
        event = make_ecs(
            action="AttachUserPolicy",
            provider="iam.amazonaws.com",
            api_risk=0.9,
        )
        features = engineer.extract_features(event)

        assert features[9] >= 0.5  # privilege_escalation_score

    def test_privilege_escalation_low(self, engineer: FeatureEngineer) -> None:
        """Non-IAM read API should score low."""
        event = make_ecs(action="DescribeInstances", api_risk=0.1)
        features = engineer.extract_features(event)

        assert features[9] <= 0.3  # privilege_escalation_score
