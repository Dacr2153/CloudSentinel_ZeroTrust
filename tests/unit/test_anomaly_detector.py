# FILE: cloudsentinel-zero-trust/tests/unit/test_anomaly_detector.py
"""Unit tests for AnomalyDetector — IsolationForest model loading and scoring."""

from __future__ import annotations

import io
from typing import Any

import joblib
import numpy as np

from src.detectors.anomaly_detector import AnomalyDetector
from src.pipeline.feature_engineer import FEATURE_NAMES

MODEL_BUCKET = "cloudsentinel-models-123456789012"


def test_predict_normal_event_low_score(
    mock_s3_client: Any, trained_model: Any
) -> None:
    detector = AnomalyDetector(
        s3_client=mock_s3_client,
        model_bucket=MODEL_BUCKET,
        threshold=65,
    )
    detector.load_model()

    normal_features = np.array(
        [0.5, 0.3, 0.0, 0.1, 0.2, 0.0, 0.0, 0.1, 0.3, 0.1],
        dtype=np.float64,
    )

    result = detector.predict(normal_features)

    assert result["anomaly_score"] < 50
    assert result["is_anomaly"] is False
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_anomalous_event_high_score(
    mock_s3_client: Any, trained_model: Any
) -> None:
    detector = AnomalyDetector(
        s3_client=mock_s3_client,
        model_bucket=MODEL_BUCKET,
        threshold=65,
    )
    detector.load_model()

    anomalous_features = np.array(
        [0.13, 0.85, 1.0, 0.9, 0.95, 1.0, 1.0, 0.8, 0.9, 0.95],
        dtype=np.float64,
    )

    result = detector.predict(anomalous_features)

    assert result["anomaly_score"] > 60
    assert isinstance(result["is_anomaly"], bool)


def test_contributing_features_not_empty(
    mock_s3_client: Any, trained_model: Any
) -> None:
    detector = AnomalyDetector(
        s3_client=mock_s3_client,
        model_bucket=MODEL_BUCKET,
        threshold=65,
    )
    detector.load_model()

    features = np.array(
        [0.13, 0.85, 1.0, 0.9, 0.95, 1.0, 1.0, 0.8, 0.9, 0.95],
        dtype=np.float64,
    )

    result = detector.predict(features)

    contributing = result["contributing_features"]
    assert len(contributing) > 0
    assert len(contributing) <= 3  # top_k=3

    for feat in contributing:
        assert "feature" in feat
        assert "impact" in feat
        assert "value" in feat
        assert feat["feature"] in FEATURE_NAMES


def test_model_refresh_loads_new_version(
    mock_s3_client: Any, trained_model: Any
) -> None:
    from sklearn.ensemble import IsolationForest

    detector = AnomalyDetector(
        s3_client=mock_s3_client,
        model_bucket=MODEL_BUCKET,
        threshold=65,
    )
    detector.load_model()

    features = np.array([0.5, 0.3, 0.0, 0.1, 0.2, 0.0, 0.0, 0.1, 0.3, 0.1])
    detector.predict(features)

    rng = np.random.RandomState(99)
    normal = rng.normal(loc=0.5, scale=0.15, size=(100, 10)).clip(0, 1)
    model_v2 = IsolationForest(
        n_estimators=50, contamination=0.1, random_state=99
    )
    model_v2.fit(normal)
    buf = io.BytesIO()
    joblib.dump(model_v2, buf)
    buf.seek(0)
    mock_s3_client.put_object(
        Bucket=MODEL_BUCKET,
        Key="models/isolation_forest/model.joblib",
        Body=buf.read(),
    )

    detector.refresh_model()
    assert detector.is_loaded

    result_v2 = detector.predict(features)
    assert isinstance(result_v2["anomaly_score"], int)


def test_batch_predict_consistency(
    mock_s3_client: Any, trained_model: Any
) -> None:
    detector = AnomalyDetector(
        s3_client=mock_s3_client,
        model_bucket=MODEL_BUCKET,
        threshold=65,
    )
    detector.load_model()

    v1 = np.array([0.5, 0.3, 0.0, 0.1, 0.2, 0.0, 0.0, 0.1, 0.3, 0.1])
    v2 = np.array([0.13, 0.85, 1.0, 0.9, 0.95, 1.0, 1.0, 0.8, 0.9, 0.95])

    batch_results = detector.predict_batch(np.vstack([v1, v2]))
    single_r1 = detector.predict(v1)
    single_r2 = detector.predict(v2)

    assert batch_results[0]["anomaly_score"] == single_r1["anomaly_score"]
    assert batch_results[1]["anomaly_score"] == single_r2["anomaly_score"]
