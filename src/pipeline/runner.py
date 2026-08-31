# FILE: cloudsentinel-zero-trust/src/pipeline/runner.py
"""CloudSentinel detection pipeline runner.

Executes the full detection pipeline without AWS, processing REAL CloudTrail
JSON files exported from your AWS account.

Pipeline stages:
  1. Read CloudTrail JSON events from local filesystem
  2. Normalize (CloudTrail → ECS schema)
  3. Enrich (geo-lookup, risk scoring)
  4. Feature engineering (10-dimensional vector)
  5. ML anomaly detection (Isolation Forest, trained offline)
  6. Rule-based detection (8 deterministic MITRE ATT&CK rules)
  7. Alert dispatch (stdout + tools/alerts/alerts.jsonl)

Usage:
  from src.pipeline.runner import LocalPipelineRunner

  runner = LocalPipelineRunner(opensearch_enabled=False)
  stats = runner.run()               # all files in default events dir
  stats = runner.run("event.json")   # single file

CloudTrail export:
  AWS Console → CloudTrail → Event history → Download JSON
  Or configure a CloudTrail trail to deliver to an S3 bucket and download .json.gz files.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..detectors.alert_manager import AlertManager
from ..detectors.anomaly_detector import MODEL_S3_KEY, AnomalyDetector
from ..detectors.rule_engine import RuleEngine
from ..integrations.s3_client import get_s3_client
from ..integrations.sns_client import get_sns_client
from ..models.cloudtrail_event import CloudTrailEvent
from ..utils.config import get_settings
from ..utils.logger import CloudSentinelLogger
from .enricher import EventEnricher
from .feature_engineer import FeatureEngineer
from .normalizer import ECSNormalizer

logger = CloudSentinelLogger(service="runner")



class LocalPipelineRunner:
    """Runs the full CloudSentinel detection pipeline on local event files.

    No AWS services required in LOCAL_MODE. Uses LocalS3Client (filesystem)
    and LocalSNSClient (stdout + JSONL) transparently.

    Args:
        events_dir: Directory containing real CloudTrail JSON or .json.gz exports.
        opensearch_enabled: Whether to attempt OpenSearch indexing.
    """

    DEFAULT_EVENTS_DIR = "ml/data/cloudtrail_samples"

    def __init__(
        self,
        events_dir: str = DEFAULT_EVENTS_DIR,
        opensearch_enabled: bool = False,
    ) -> None:
        self._events_dir = Path(events_dir)
        self._opensearch_enabled = opensearch_enabled
        self._settings = get_settings()
        self._init_components()

    # ── Initialisation ──────────────────────────────────────────────

    def _init_components(self) -> None:
        self._normalizer = ECSNormalizer()
        self._enricher = EventEnricher()
        self._feature_eng = FeatureEngineer()
        self._rule_engine = RuleEngine()

        s3_client = get_s3_client()
        sns_client = get_sns_client()

        model_bucket = self._settings.model_bucket
        model_local_path = (
            Path(self._settings.local_data_dir) / model_bucket / MODEL_S3_KEY
        )

        if not model_local_path.exists():
            logger.warning("Model not found at %s — ML detection disabled", model_local_path)
            self._anomaly_detector = None
        else:
            self._anomaly_detector = AnomalyDetector(
                s3_client=s3_client,
                model_bucket=model_bucket,
                threshold=self._settings.anomaly_threshold,
            )
            try:
                self._anomaly_detector.load_model()
            except Exception as exc:
                logger.warning("Failed to load model: %s", exc)
                self._anomaly_detector = None

        self._alert_manager = AlertManager(
            sns_client=sns_client,
            sns_topic_arn=self._settings.sns_topic_arn,
            opensearch_client=None,
        )

    # ── Public API ──────────────────────────────────────────────────

    @property
    def model_ready(self) -> bool:
        """True when the ML model is loaded and ready."""
        return self._anomaly_detector is not None

    @property
    def events_dir(self) -> Path:
        return self._events_dir

    def run(self, single_file: str | None = None) -> dict[str, Any]:
        """Process CloudTrail events and return aggregate statistics.

        Args:
            single_file: Path to a single CloudTrail JSON export. When *None*
                all *.json and *.json.gz files in ``events_dir`` are processed.

        Returns:
            Dict with keys: files, events_total, events_normalized,
            anomalies_detected, rules_triggered, alerts_dispatched, errors.
        """
        start = time.monotonic()
        totals: dict[str, int] = {
            "files": 0,
            "events_total": 0,
            "events_normalized": 0,
            "anomalies_detected": 0,
            "rules_triggered": 0,
            "alerts_dispatched": 0,
            "errors": 0,
        }

        if single_file:
            files = [Path(single_file)]
        else:
            self._events_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(
                list(self._events_dir.glob("*.json"))
                + list(self._events_dir.glob("*.json.gz"))
            )

        if not files:
            logger.info("No CloudTrail event files found in %s", self._events_dir)
            return dict(totals)

        for path in files:
            stats = self.process_event_file(path)
            totals["files"] += 1
            for key in totals:
                if key != "files":
                    totals[key] += stats.get(key, 0)

        totals["elapsed_seconds"] = round(time.monotonic() - start, 3)
        return dict(totals)

    def process_event_file(self, path: Path) -> dict[str, Any]:
        """Process a single CloudTrail event file through the full pipeline.

        Returns per-file statistics dict.
        """
        stats: dict[str, Any] = {
            "file": str(path),
            "events_total": 0,
            "events_normalized": 0,
            "anomalies_detected": 0,
            "rules_triggered": 0,
            "alerts_dispatched": 0,
            "errors": 0,
        }

        try:
            raw_events = self._load_event_file(path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            stats["errors"] += 1
            return stats

        stats["events_total"] = len(raw_events)
        feature_vectors: list[list[float]] = []
        ecs_events: list[Any] = []

        for raw in raw_events:
            try:
                ct_event = CloudTrailEvent(**raw)
                ecs = self._normalizer.normalize(ct_event)
                self._enricher.enrich(ecs)
                features = self._feature_eng.extract_features(ecs)
                feature_vectors.append(features.tolist())
                ecs_events.append(ecs)
                stats["events_normalized"] += 1
            except Exception as exc:
                logger.warning("Normalization error: %s", exc)
                stats["errors"] += 1

        if not ecs_events:
            return stats

        # ── ML batch scoring ────────────────────────────────────────
        anomaly_scores, is_anomaly_flags, contrib_features = self._score_batch(
            feature_vectors
        )
        stats["anomalies_detected"] = sum(1 for a in is_anomaly_flags if a)

        # ── Rule engine + alert dispatch ────────────────────────────
        for i, ecs in enumerate(ecs_events):
            try:
                rule_matches = self._rule_engine.evaluate(ecs)
                if rule_matches:
                    stats["rules_triggered"] += len(rule_matches)

                alert = self._alert_manager.process_event(
                    event_id=ecs.event.id or f"local-{i}",
                    event_action=ecs.event.action or "",
                    user_name=ecs.user.name or "unknown",
                    source_ip=ecs.source.ip or "0.0.0.0",  # nosec B104 — display value, not a socket
                    cloud_region=ecs.cloud.region or "local",
                    cloud_account_id=(
                        ecs.cloud.account.get("id", "local")
                        if ecs.cloud.account
                        else "local"
                    ),
                    rule_matches=rule_matches,
                    anomaly_score=anomaly_scores[i] if anomaly_scores else None,
                    is_anomaly=is_anomaly_flags[i] if is_anomaly_flags else False,
                    contributing_features=contrib_features[i] if contrib_features else [],
                )
                if alert:
                    stats["alerts_dispatched"] += 1
            except Exception as exc:
                logger.warning("Alert processing error: %s", exc)
                stats["errors"] += 1

        return stats

    # ── Private helpers ──────────────────────────────────────────────

    def _load_event_file(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, dict) and "Records" in data:
            return data["Records"]
        if isinstance(data, list):
            return data
        return [data]  # single event object

    def _score_batch(
        self,
        feature_vectors: list[list[float]],
    ) -> tuple[list[int], list[bool], list[list[dict[str, Any]]]]:
        """Run ML batch scoring. Returns (scores, flags, contributing_features)."""
        n = len(feature_vectors)
        zeros_int: list[int] = [0] * n
        zeros_bool: list[bool] = [False] * n
        zeros_cf: list[list[dict[str, Any]]] = [[]] * n

        if not self._anomaly_detector or not feature_vectors:
            return zeros_int, zeros_bool, zeros_cf

        try:
            X = np.array(feature_vectors)
            results = self._anomaly_detector.predict_batch(X)
            scores = [int(r.get("score", 0)) for r in results]
            flags = [bool(r.get("is_anomaly", False)) for r in results]
            cfs = [r.get("contributing_features", []) for r in results]
            return scores, flags, cfs
        except Exception as exc:
            logger.warning("Batch prediction failed: %s", exc)
            return zeros_int, zeros_bool, zeros_cf
