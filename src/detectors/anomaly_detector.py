# FILE: cloudsentinel-zero-trust/src/detectors/anomaly_detector.py
"""ML-based anomaly detection using Isolation Forest.

Design:
- Model: scikit-learn IsolationForest, trained offline, serialized with joblib
- Storage: S3 model bucket → loaded at Lambda cold start
- Scoring: decision_function → normalized to [0, 100]
- Threshold: configurable via SSM (default 65)
- Contributing features: per-sample feature importance via score perturbation
- Thread-safe model reference via _lock

Hyperparameters (training):
    n_estimators=200, contamination=0.1, max_samples='auto', random_state=42
"""

from __future__ import annotations

import io
import threading
from typing import Any

import joblib
import numpy as np

from src.pipeline.feature_engineer import FEATURE_NAMES, NUM_FEATURES
from src.utils.config import get_settings
from src.utils.exceptions import ModelInferenceError, ModelNotFoundError
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="anomaly_detector")

# Default model path in S3
MODEL_S3_KEY = "models/isolation_forest/model.joblib"
DEFAULT_THRESHOLD = 65


class AnomalyDetector:
    """Loads an Isolation Forest model from S3 and scores events.

    Parameters:
        s3_client: boto3 S3 client (injected for testability).
        model_bucket: S3 bucket name for the model artifacts.
        threshold: Anomaly score threshold [0-100]. Scores >= threshold → anomaly.
    """

    def __init__(
        self,
        s3_client: Any,
        model_bucket: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self._s3 = s3_client
        self._model: Any | None = None
        self._lock = threading.Lock()

        settings = get_settings()
        self._bucket = model_bucket or settings.model_bucket
        self._threshold = threshold or settings.anomaly_threshold
        self._model_key = MODEL_S3_KEY


    def load_model(self) -> None:
        """Download and deserialize the model from S3."""
        logger.info("Loading model from s3://%s/%s", self._bucket, self._model_key)
        try:
            response = self._s3.get_object(
                Bucket=self._bucket, Key=self._model_key
            )
            model_bytes = response["Body"].read()
            buffer = io.BytesIO(model_bytes)
            artifact = joblib.load(buffer)

            # train_model.py saves a dict artifact: {"pipeline": ..., "threshold": ...}
            if isinstance(artifact, dict) and "pipeline" in artifact:
                model = artifact["pipeline"]
                # Always use the threshold calibrated at training time (maximises F1 with FPR < 5%)
                if "threshold" in artifact:
                    self._threshold = float(artifact["threshold"])
            else:
                model = artifact

            with self._lock:
                self._model = model

            logger.info("Model loaded successfully")
        except self._s3.exceptions.NoSuchKey:
            raise ModelNotFoundError(
                f"Model not found at s3://{self._bucket}/{self._model_key}"
            ) from None
        except Exception as exc:
            raise ModelNotFoundError(
                f"Failed to load model: {exc}",
                context={"bucket": self._bucket, "key": self._model_key},
            ) from exc

    def refresh_model(self) -> None:
        """Re-download the model (e.g., after retraining)."""
        self.load_model()

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None


    def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Score a single feature vector.

        Args:
            features: np.ndarray of shape (10,)

        Returns:
            dict with keys:
                anomaly_score: int [0-100]
                is_anomaly: bool
                contributing_features: list[dict] — top 3 features by impact
                confidence: float [0-1]
        """
        if features.shape != (NUM_FEATURES,):
            raise ModelInferenceError(
                f"Expected feature vector of shape ({NUM_FEATURES},), got {features.shape}"
            )

        model = self._get_model()

        try:
            # decision_function returns negative for anomalies, positive for normal
            raw_score = model.decision_function(features.reshape(1, -1))[0]
            # Normalize: more negative → higher anomaly score
            anomaly_score = self._normalize_score(raw_score)
            is_anomaly = anomaly_score >= self._threshold
            confidence = self._compute_confidence(anomaly_score)
            contributing = self._get_contributing_features(model, features)

            return {
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "contributing_features": contributing,
                "confidence": confidence,
            }
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(
                f"Model inference failed: {exc}",
                context={"features_shape": str(features.shape)},
            ) from exc

    def predict_batch(self, features_matrix: np.ndarray) -> list[dict[str, Any]]:
        """Score a batch of feature vectors.

        Args:
            features_matrix: np.ndarray of shape (N, 10)

        Returns:
            list of prediction dicts (same schema as predict())
        """
        if features_matrix.ndim != 2 or features_matrix.shape[1] != NUM_FEATURES:
            raise ModelInferenceError(
                f"Expected matrix of shape (N, {NUM_FEATURES}), got {features_matrix.shape}"
            )

        model = self._get_model()

        try:
            raw_scores = model.decision_function(features_matrix)
            results = []
            for i, raw_score in enumerate(raw_scores):
                anomaly_score = self._normalize_score(raw_score)
                is_anomaly = anomaly_score >= self._threshold
                confidence = self._compute_confidence(anomaly_score)
                contributing = self._get_contributing_features(
                    model, features_matrix[i]
                )
                results.append({
                    "anomaly_score": anomaly_score,
                    "is_anomaly": is_anomaly,
                    "contributing_features": contributing,
                    "confidence": confidence,
                })
            return results
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(
                f"Batch inference failed: {exc}",
                context={"batch_size": features_matrix.shape[0]},
            ) from exc


    @staticmethod
    def _normalize_score(raw_score: float) -> int:
        """Convert Isolation Forest decision_function output to [0, 100].

        decision_function: negative = anomaly, positive = normal
        Typical range: [-0.5, 0.5] (unbounded, but concentrated here)

        Mapping: -0.5 → 100 (most anomalous), 0.5 → 0 (most normal)
        """
        # Clamp to expected range
        clamped = max(-0.5, min(0.5, raw_score))
        # Linear mapping from [-0.5, 0.5] to [100, 0]
        normalized = int(round((0.5 - clamped) * 100))
        return max(0, min(100, normalized))

    @staticmethod
    def _compute_confidence(anomaly_score: int) -> float:
        """Confidence in the anomaly/normal classification.

        High confidence when the score is far from the threshold.
        """
        distance_from_threshold = abs(anomaly_score - DEFAULT_THRESHOLD) / 100.0
        return min(1.0, distance_from_threshold * 2)


    @staticmethod
    def _get_contributing_features(
        model: Any, features: np.ndarray, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Approximate per-sample feature importance via score perturbation.

        For each feature, zero it out and measure the change in anomaly score.
        The features causing the largest score drop are the most important.
        """
        baseline = model.decision_function(features.reshape(1, -1))[0]
        importances: list[tuple[str, float, float]] = []

        for i in range(NUM_FEATURES):
            perturbed = features.copy()
            perturbed[i] = 0.0
            perturbed_score = model.decision_function(perturbed.reshape(1, -1))[0]
            delta = baseline - perturbed_score
            importances.append((FEATURE_NAMES[i], abs(delta), features[i]))

        # Sort by impact (largest delta first)
        importances.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "feature": name,
                "impact": round(delta, 4),
                "value": round(value, 4),
            }
            for name, delta, value in importances[:top_k]
        ]


    def _get_model(self) -> Any:
        """Return the loaded model or raise."""
        with self._lock:
            if self._model is None:
                raise ModelNotFoundError(
                    "Model not loaded. Call load_model() first."
                )
            return self._model
