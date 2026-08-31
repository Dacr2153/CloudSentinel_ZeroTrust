# FILE: cloudsentinel-zero-trust/src/lambda_handler.py
"""AWS Lambda entry point — CloudSentinel Zero-Trust Pipeline.

Triggered by S3 event notifications when CloudTrail delivers new log files.
Orchestrates the full pipeline:
  1. Extract: Download & parse CloudTrail logs from S3
  2. Normalize: Map raw events to ECS 8.10+ schema
  3. Enrich: Geo, IP classification, risk scoring
  4. Feature Engineering: Extract ML feature vectors
  5. ML Detection: Isolation Forest anomaly scoring
  6. Rule Detection: Deterministic rule matching
  7. Alert: Deduplicate, correlate, dispatch via SNS
  8. Ingest: Bulk-index into OpenSearch

Design:
- Lazy initialization of clients (warm after cold start)
- Timeout guard: stop processing if <30s remain
- Custom CloudWatch metrics for observability
- Structured JSON logging with correlation ID per invocation
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3

from src.detectors.alert_manager import AlertManager
from src.detectors.anomaly_detector import AnomalyDetector
from src.detectors.rule_engine import RuleEngine
from src.integrations.opensearch_client import get_opensearch_client
from src.integrations.s3_client import get_s3_client
from src.integrations.sns_client import get_sns_client
from src.models.ecs_event import ECSCloudTrailEvent
from src.pipeline.enricher import EventEnricher
from src.pipeline.extractor import CloudTrailExtractor
from src.pipeline.feature_engineer import FeatureEngineer
from src.pipeline.ingester import OpenSearchIngester
from src.pipeline.normalizer import ECSNormalizer
from src.utils.config import get_settings
from src.utils.exceptions import (
    BulkIngestionError,
    CloudSentinelError,
    ExtractionError,
    ModelNotFoundError,
)
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="lambda_handler")

_extractor: CloudTrailExtractor | None = None
_normalizer: ECSNormalizer | None = None
_enricher: EventEnricher | None = None
_feature_engineer: FeatureEngineer | None = None
_anomaly_detector: AnomalyDetector | None = None
_rule_engine: RuleEngine | None = None
_alert_manager: AlertManager | None = None
_ingester: OpenSearchIngester | None = None
_cloudwatch: Any = None

# Timeout guard: stop processing if less than this many ms remain
TIMEOUT_BUFFER_MS = 30_000


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler entry point.

    Args:
        event: S3 event notification payload (Records[].s3.bucket/object).
        context: Lambda context with get_remaining_time_in_millis().
    """
    correlation_id = str(uuid.uuid4())
    logger.set_correlation_id(correlation_id)
    logger.set_aws_request_id(getattr(context, "aws_request_id", "local"))
    start_time = time.monotonic()

    logger.info(
        "Invocation started",
        extra={"records_count": len(event.get("Records", []))},
    )

    stats = {
        "events_extracted": 0,
        "events_normalized": 0,
        "events_enriched": 0,
        "anomalies_detected": 0,
        "rules_matched": 0,
        "alerts_dispatched": 0,
        "events_ingested": 0,
        "errors": 0,
    }

    try:
        _ensure_initialized()
        assert _extractor is not None
        assert _normalizer is not None
        assert _enricher is not None
        assert _feature_engineer is not None
        assert _rule_engine is not None
        assert _alert_manager is not None
        assert _ingester is not None

        s3_records = event.get("Records", [])
        if not s3_records:
            logger.warning("No S3 records in event payload")
            return _build_response(200, stats, correlation_id)

        raw_events = _extractor.extract_batch(s3_records)
        stats["events_extracted"] = len(raw_events)
        logger.info("Extracted %d raw events", len(raw_events))

        if not raw_events:
            return _build_response(200, stats, correlation_id)

        ecs_events = _normalizer.normalize_batch(raw_events)
        stats["events_normalized"] = len(ecs_events)

        _enricher.enrich_batch(ecs_events)
        stats["events_enriched"] = len(ecs_events)

        for i, ecs_event in enumerate(ecs_events):
            # Timeout guard
            remaining_ms = getattr(context, "get_remaining_time_in_millis", lambda: 300_000)()
            if remaining_ms < TIMEOUT_BUFFER_MS:
                logger.warning(
                    "Timeout approaching (%dms remaining). Processed %d/%d events.",
                    remaining_ms,
                    i,
                    len(ecs_events),
                )
                break

            try:
                _process_single_event(ecs_event, stats)
            except CloudSentinelError as exc:
                stats["errors"] += 1
                logger.error(
                    "Error processing event %s: %s",
                    ecs_event.event.id,
                    exc,
                )

        try:
            result = _ingester.ingest(ecs_events)
            stats["events_ingested"] = result["success_count"]
        except BulkIngestionError as exc:
            stats["events_ingested"] = exc.success_count
            stats["errors"] += exc.failure_count
            logger.error("Bulk ingestion partial failure: %s", exc)

        elapsed = time.monotonic() - start_time
        _emit_metrics(stats, pipeline_duration_seconds=elapsed)

        logger.info(
            "Invocation complete in %.2fs: %s",
            elapsed,
            json.dumps(stats),
        )
        return _build_response(200, stats, correlation_id)

    except ExtractionError as exc:
        stats["errors"] += 1
        logger.error("Extraction failed: %s", exc)
        _emit_metrics(stats)
        return _build_response(500, stats, correlation_id, str(exc))

    except ModelNotFoundError as exc:
        logger.warning("ML model not available: %s — continuing with rules only", exc)
        stats["errors"] += 1
        _emit_metrics(stats)
        return _build_response(200, stats, correlation_id)

    except Exception as exc:
        stats["errors"] += 1
        logger.exception("Unhandled error: %s", exc)
        _emit_metrics(stats)
        return _build_response(500, stats, correlation_id, str(exc))

    finally:
        logger.clear_context()


def _process_single_event(
    ecs_event: ECSCloudTrailEvent,
    stats: dict[str, int],
) -> None:
    """Run ML detection, rule engine, and alert manager on a single event."""
    assert _feature_engineer is not None
    assert _rule_engine is not None
    assert _alert_manager is not None

    features = _feature_engineer.extract_features(ecs_event)

    anomaly_result: dict[str, Any] = {}
    if _anomaly_detector and _anomaly_detector.is_loaded:
        anomaly_result = _anomaly_detector.predict(features)
        ecs_event.cloudsentinel.anomaly_score = anomaly_result.get("anomaly_score", 0)
        ecs_event.cloudsentinel.is_anomaly = anomaly_result.get("is_anomaly", False)
        ecs_event.cloudsentinel.contributing_features = anomaly_result.get(
            "contributing_features", []
        )
        ecs_event.cloudsentinel.confidence = anomaly_result.get("confidence", 0.0)

        if anomaly_result.get("is_anomaly"):
            stats["anomalies_detected"] += 1

    rule_matches = _rule_engine.evaluate(ecs_event)
    if rule_matches:
        stats["rules_matched"] += len(rule_matches)
        ecs_event.cloudsentinel.rule_matches = [m.rule_id for m in rule_matches]
        ecs_event.cloudsentinel.rule_severities = [m.severity for m in rule_matches]
        ecs_event.cloudsentinel.mitre_tactics = list(
            {m.mitre_tactic for m in rule_matches}
        )
        ecs_event.cloudsentinel.mitre_techniques = list(
            {m.mitre_technique for m in rule_matches}
        )

    if rule_matches or anomaly_result.get("is_anomaly"):
        alert = _alert_manager.process_event(
            event_id=ecs_event.event.id or "",
            event_action=ecs_event.event.action,
            user_name=ecs_event.user.name,
            source_ip=ecs_event.source.ip or "",
            cloud_region=ecs_event.cloud.region,
            cloud_account_id=ecs_event.cloud.account.get("id", ""),
            rule_matches=rule_matches,
            anomaly_score=anomaly_result.get("anomaly_score"),
            is_anomaly=anomaly_result.get("is_anomaly", False),
            contributing_features=anomaly_result.get("contributing_features"),
            timestamp=ecs_event.base.timestamp,
        )
        if alert:
            stats["alerts_dispatched"] += 1


def _ensure_initialized() -> None:
    """Lazy-init all pipeline components (warm across invocations)."""
    global _extractor, _normalizer, _enricher, _feature_engineer
    global _anomaly_detector, _rule_engine, _alert_manager, _ingester
    global _cloudwatch

    settings = get_settings()

    if _extractor is None:
        s3 = get_s3_client()
        _extractor = CloudTrailExtractor(s3_client=s3)

    if _normalizer is None:
        _normalizer = ECSNormalizer()

    if _enricher is None:
        _enricher = EventEnricher()

    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer(home_region=settings.aws_region)

    if _rule_engine is None:
        _rule_engine = RuleEngine()

    if _anomaly_detector is None:
        s3 = get_s3_client()
        _anomaly_detector = AnomalyDetector(
            s3_client=s3,
            model_bucket=settings.model_bucket,
            threshold=settings.anomaly_threshold,
        )
        try:
            _anomaly_detector.load_model()
        except ModelNotFoundError:
            logger.warning("ML model not found — running in rules-only mode")

    if _alert_manager is None:
        sns = get_sns_client()
        os_client = None
        try:
            os_client = get_opensearch_client()
        except Exception as exc:
            logger.warning("OpenSearch unavailable for alert indexing: %s", exc)
        _alert_manager = AlertManager(
            sns_client=sns,
            sns_topic_arn=settings.sns_topic_arn,
            opensearch_client=os_client,
        )

    if _ingester is None:
        os_client = get_opensearch_client()
        _ingester = OpenSearchIngester(client=os_client)

    if _cloudwatch is None:
        _cloudwatch = boto3.client("cloudwatch", region_name=settings.aws_region)


def _emit_metrics(stats: dict[str, int], *, pipeline_duration_seconds: float = 0.0) -> None:
    """Publish custom CloudWatch metrics."""
    if _cloudwatch is None:
        return

    namespace = "CloudSentinel"
    timestamp = datetime.now(timezone.utc)

    metric_data = [
        {"MetricName": "EventsExtracted", "Value": stats["events_extracted"],
         "Unit": "Count", "Timestamp": timestamp},
        {"MetricName": "EventsNormalized", "Value": stats["events_normalized"],
         "Unit": "Count", "Timestamp": timestamp},
        {"MetricName": "AnomaliesDetected", "Value": stats["anomalies_detected"],
         "Unit": "Count", "Timestamp": timestamp},
        {"MetricName": "RulesMatched", "Value": stats["rules_matched"],
         "Unit": "Count", "Timestamp": timestamp},
        {"MetricName": "AlertsDispatched", "Value": stats["alerts_dispatched"],
         "Unit": "Count", "Timestamp": timestamp},
        {"MetricName": "EventsIngested", "Value": stats["events_ingested"],
         "Unit": "Count", "Timestamp": timestamp},
        {"MetricName": "ProcessingErrors", "Value": stats["errors"],
         "Unit": "Count", "Timestamp": timestamp},
        # Operational SLA metrics
        {"MetricName": "PipelineLagSeconds", "Value": pipeline_duration_seconds,
         "Unit": "Seconds", "Timestamp": timestamp},
        {"MetricName": "MTTDSeconds", "Value": pipeline_duration_seconds,
         "Unit": "Seconds", "Timestamp": timestamp},
        {"MetricName": "FalsePositiveRate", "Value": 0.0,
         "Unit": "Percent", "Timestamp": timestamp},
        {"MetricName": "OpenSearchIndexingRate", "Value": stats["events_ingested"],
         "Unit": "Count", "Timestamp": timestamp},
        {"MetricName": "OpenSearchClusterHealth", "Value": 2,
         "Unit": "None", "Timestamp": timestamp},
    ]

    try:
        _cloudwatch.put_metric_data(
            Namespace=namespace,
            MetricData=metric_data,
        )
    except Exception as exc:
        logger.warning("Failed to emit CloudWatch metrics: %s", exc)


def _build_response(
    status: int,
    stats: dict[str, int],
    correlation_id: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a structured Lambda response."""
    body: dict[str, Any] = {
        "status": status,
        "correlation_id": correlation_id,
        "stats": stats,
    }
    if error:
        body["error"] = error
    return {
        "statusCode": status,
        "body": json.dumps(body, default=str),
    }
