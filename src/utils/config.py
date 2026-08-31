# FILE: cloudsentinel-zero-trust/src/utils/config.py
"""Centralized configuration via Pydantic BaseSettings + AWS SSM Parameter Store.

Design decisions:
- Pydantic v2 BaseSettings for validation and type coercion
- SSM parameters fetched lazily with 5-minute in-memory cache (avoids API rate limits)
- Environment variables override SSM values (12-Factor App)
- All sensitive values fetched from SSM, never hardcoded
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """CloudSentinel pipeline configuration.

    Resolution order: env vars → SSM Parameter Store → defaults.
    """

    # ── Core pipeline settings ──
    environment: str = Field(default="cloudsentinel", description="Environment/stack name prefix")
    log_level: str = Field(default="INFO", description="Logging level")
    ssm_prefix: str = Field(default="/cloudsentinel", description="SSM parameter path prefix")

    # ── OpenSearch ──
    opensearch_endpoint: str = Field(default="", description="OpenSearch HTTP endpoint (e.g. http://10.0.1.x:9200)")
    opensearch_index_prefix: str = Field(default="cloudsentinel-events", description="Index name prefix")
    opensearch_batch_size: int = Field(default=500, ge=1, le=5000, description="Bulk API batch size")

    # ── SNS ──
    sns_topic_arn: str = Field(default="", description="SNS topic ARN for alerts")

    # ── ML Model ──
    model_bucket: str = Field(default="", description="S3 bucket for ML models")
    model_key: str = Field(default="models/isolation_forest/model.joblib", description="S3 key for model artifact")
    anomaly_threshold: float = Field(default=65.0, ge=0.0, le=100.0, description="Anomaly score threshold (0-100)")

    # ── Alert ──
    alert_cooldown_minutes: int = Field(default=15, ge=0, description="Alert deduplication cooldown window")

    # ── AWS ──
    aws_region: str = Field(default="us-east-1", description="AWS region")

    # ── LOCAL MODE (no AWS required) ──
    local_mode: bool = Field(
        default=False,
        description="Skip SSM/S3/SNS; use local filesystem and stdout alerts.",
    )
    local_data_dir: str = Field(
        default="ml/data",
        description="Root directory for local model + data storage.",
    )
    local_alerts_dir: str = Field(
        default="tools/alerts",
        description="Directory for local alert JSONL files.",
    )
    local_events_dir: str = Field(
        default="ml/data/cloudtrail_samples",
        description="Directory with local CloudTrail sample JSON files.",
    )

    model_config = {"env_prefix": "CLOUDSENTINEL_", "case_sensitive": False}

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    # ── SSM Integration ──

    def load_from_ssm(self) -> None:
        """Populate empty settings from SSM Parameter Store.

        Called once during Lambda cold start. Only fetches parameters
        whose corresponding settings are empty/default.
        """
        ssm_mappings: dict[str, str] = {
            f"{self.ssm_prefix}/opensearch/endpoint": "opensearch_endpoint",
            f"{self.ssm_prefix}/opensearch/index-prefix": "opensearch_index_prefix",
            f"{self.ssm_prefix}/sns/topic-arn": "sns_topic_arn",
            f"{self.ssm_prefix}/model/bucket": "model_bucket",
            f"{self.ssm_prefix}/model/key-prefix": "model_key",
            f"{self.ssm_prefix}/model/anomaly-threshold": "anomaly_threshold",
            f"{self.ssm_prefix}/alert/cooldown-minutes": "alert_cooldown_minutes",
            f"{self.ssm_prefix}/pipeline/log-level": "log_level",
            f"{self.ssm_prefix}/pipeline/batch-size": "opensearch_batch_size",
        }

        for ssm_name, field_name in ssm_mappings.items():
            current_value = getattr(self, field_name)
            # Only fetch from SSM if the field is at its default empty value
            if current_value in ("", 0, 0.0) or (field_name == "anomaly_threshold" and current_value == 65.0):
                value = get_ssm_parameter(ssm_name, region=self.aws_region)
                if value is not None:
                    # Use Pydantic's type coercion by going through model_validate
                    field_info = self.model_fields.get(field_name)
                    if field_info and field_info.annotation:
                        try:
                            if field_info.annotation in (int,):
                                setattr(self, field_name, int(value))
                            elif field_info.annotation in (float,):
                                setattr(self, field_name, float(value))
                            else:
                                setattr(self, field_name, value)
                        except (ValueError, TypeError):
                            setattr(self, field_name, value)
                    else:
                        setattr(self, field_name, value)


# Cache: {param_name: (value, fetch_timestamp)}
_ssm_cache: dict[str, tuple[str, float]] = {}
_SSM_CACHE_TTL_SECONDS: int = 300  # 5 minutes

_ssm_client: Any = None


def _get_ssm_client(region: str) -> Any:
    """Lazy-init SSM client (reused across warm Lambda invocations)."""
    global _ssm_client
    if _ssm_client is None:
        _ssm_client = boto3.client("ssm", region_name=region)
    return _ssm_client


def get_ssm_parameter(
    name: str,
    region: str = "us-east-1",
    decrypt: bool = True,
) -> str | None:
    """Fetch a single SSM parameter with in-memory cache (TTL 5 minutes).

    Args:
        name: Full SSM parameter name (e.g. /cloudsentinel/opensearch/endpoint).
        region: AWS region.
        decrypt: Whether to decrypt SecureString parameters.

    Returns:
        Parameter value string, or None if not found.
    """
    now = time.time()

    # Check cache
    if name in _ssm_cache:
        cached_value, cached_at = _ssm_cache[name]
        if (now - cached_at) < _SSM_CACHE_TTL_SECONDS:
            return cached_value

    # Fetch from SSM
    try:
        client = _get_ssm_client(region)
        response = client.get_parameter(Name=name, WithDecryption=decrypt)
        value = response["Parameter"]["Value"]
        _ssm_cache[name] = (value, now)
        return value
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ParameterNotFound":
            return None
        raise


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton Settings factory. Loads SSM on first call.

    Use: ``settings = get_settings()``
    Resolution: env vars → SSM (skipped in LOCAL_MODE) → defaults.
    """
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    settings = Settings(aws_region=region)
    if not settings.local_mode:
        try:
            settings.load_from_ssm()
        except Exception:
            # If SSM is unreachable (local dev, tests), fall back to env vars only
            pass
    return settings
