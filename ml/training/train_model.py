#!/usr/bin/env python3
# FILE: cloudsentinel-zero-trust/ml/training/train_model.py
"""CloudSentinel ML Training Pipeline.

Pipeline:
  1. Load data/combined_labeled.parquet
  2. Split: train on baseline (10,000 normal) only (unsupervised IF)
  3. Build sklearn Pipeline: StandardScaler → IsolationForest
  4. Calibrate threshold on anomalous set to maximize F1 with FPR < 5%
  5. Serialize: pipeline + threshold + metadata → joblib
  6. Upload to S3 and register version in SSM

Hyperparameters:
  n_estimators=200, contamination=0.1, max_samples='auto', random_state=42
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.1,
    "max_samples": "auto",
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline and combined datasets.

    Returns:
        (df_baseline, df_combined)
    """
    data_dir = Path(data_dir)
    baseline_path = data_dir / "baseline_events.parquet"
    combined_path = data_dir / "combined_labeled.parquet"

    for path in [baseline_path, combined_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {path}. "
                "Run generate_synthetic_data.py first."
            )

    df_baseline = pd.read_parquet(baseline_path, engine="fastparquet")
    df_combined = pd.read_parquet(combined_path, engine="fastparquet")
    return df_baseline, df_combined


def build_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Build sklearn Pipeline: StandardScaler → IsolationForest."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=p["n_estimators"],
            contamination=p["contamination"],
            max_samples=p["max_samples"],
            random_state=p["random_state"],
            n_jobs=p["n_jobs"],
        )),
    ])


def calibrate_threshold(
    pipeline: Pipeline,
    X_anomalous: np.ndarray,
    X_baseline_sample: np.ndarray,
    max_fpr: float = 0.05,
) -> tuple[float, dict[str, float]]:
    """Find decision threshold that maximizes F1 with FPR constraint.

    Uses the anomalous set (true positives) and a sample of baseline
    (true negatives) to sweep over candidate thresholds.

    Returns:
        (best_threshold_on_normalized_scale, metrics_dict)
    """
    # Get raw decision_function scores
    scores_anom = pipeline.decision_function(X_anomalous)
    scores_norm = pipeline.decision_function(X_baseline_sample)

    # Combine scores and labels (1 = anomaly, 0 = normal)
    all_scores = np.concatenate([scores_norm, scores_anom])
    all_labels = np.concatenate([
        np.zeros(len(scores_norm), dtype=int),
        np.ones(len(scores_anom), dtype=int),
    ])

    # Sweep thresholds on raw score scale
    # Lower raw score → more anomalous in IsolationForest
    percentiles = np.linspace(1, 99, 500)
    thresholds = np.percentile(all_scores, percentiles)

    best_f1 = -1.0
    best_thresh = 0.0
    best_metrics: dict[str, float] = {}

    for thresh in thresholds:
        preds = (all_scores < thresh).astype(int)  # below threshold → anomaly
        fp = np.sum((preds == 1) & (all_labels == 0))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if fpr > max_fpr:
            continue

        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            tp = np.sum((preds == 1) & (all_labels == 1))
            fn = np.sum((preds == 0) & (all_labels == 1))
            best_metrics = {
                "f1_score": round(float(f1), 4),
                "precision": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4),
                "recall": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
                "fpr": round(float(fpr), 4),
                "threshold_raw": round(float(thresh), 6),
            }

    if best_f1 < 0:
        raise RuntimeError(
            f"Could not find threshold with FPR < {max_fpr}. "
            "Consider relaxing constraint or tuning hyperparameters."
        )

    # Convert raw threshold to normalized [0-100] scale for runtime use
    # Same logic as AnomalyDetector._normalize_score
    clamped = max(-0.5, min(0.5, best_thresh))
    normalized_threshold = int(round((0.5 - clamped) * 100))
    normalized_threshold = max(0, min(100, normalized_threshold))
    best_metrics["threshold_normalized"] = float(normalized_threshold)

    return float(normalized_threshold), best_metrics


def compute_auc(
    pipeline: Pipeline,
    X_normal: np.ndarray,
    X_anomalous: np.ndarray,
) -> float:
    """Compute AUC-ROC using decision_function scores."""
    scores_norm = pipeline.decision_function(X_normal)
    scores_anom = pipeline.decision_function(X_anomalous)
    all_scores = np.concatenate([scores_norm, scores_anom])
    all_labels = np.concatenate([
        np.zeros(len(scores_norm)),
        np.ones(len(scores_anom)),
    ])
    # Negate: lower decision_function → more anomalous → higher score
    return float(roc_auc_score(all_labels, -all_scores))


def serialize_model(
    pipeline: Pipeline,
    threshold: float,
    metrics: dict[str, Any],
    version: str,
    output_dir: str | Path,
    params: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Serialize model and metadata to disk.

    Returns:
        (model_path, metadata_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"cloudsentinel_model_{version}.joblib"
    metadata_path = output_dir / f"metadata_{version}.json"

    # Bundle everything needed at inference time
    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_names": FEATURE_NAMES,
        "version": version,
    }
    joblib.dump(artifact, model_path)

    # ── LOCAL_MODE: copy to LocalS3Client path ──
    # LocalS3Client maps: {base_dir}/{bucket}/{key}
    # base_dir=ml/data (default), bucket=local, key=models/isolation_forest/model.joblib
    # → ml/data/local/models/isolation_forest/model.joblib
    # output_dir defaults to ml/data, so we resolve relative to it:
    local_s3_path = Path(output_dir) / "local" / "models" / "isolation_forest" / "model.joblib"
    local_s3_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(model_path, local_s3_path)
    print(f"  [LOCAL] Model copied to {local_s3_path}")

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
        "hyperparameters": params or DEFAULT_PARAMS,
        "threshold_normalized": threshold,
        "metrics": metrics,
        "framework": "scikit-learn",
        "model_type": "IsolationForest",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    return model_path, metadata_path


def upload_to_s3(
    model_path: Path,
    metadata_path: Path,
    bucket: str,
    version: str,
) -> None:
    """Upload model and metadata to S3."""
    import boto3

    s3 = boto3.client("s3")
    s3_prefix = f"models/{version}"

    s3.upload_file(str(model_path), bucket, f"{s3_prefix}/model.joblib")
    print(f"  ✅ Uploaded s3://{bucket}/{s3_prefix}/model.joblib")

    s3.upload_file(str(metadata_path), bucket, f"{s3_prefix}/metadata.json")
    print(f"  ✅ Uploaded s3://{bucket}/{s3_prefix}/metadata.json")

    # Also upload as 'latest' for convenient access
    s3.upload_file(
        str(model_path), bucket, "models/isolation_forest/model.joblib"
    )
    print(f"  ✅ Updated s3://{bucket}/models/isolation_forest/model.joblib (latest)")


def update_ssm_version(version: str, region: str = "us-east-1") -> None:
    """Update SSM Parameter with latest model version."""
    import boto3

    ssm = boto3.client("ssm", region_name=region)
    ssm.put_parameter(
        Name="/cloudsentinel/model/latest-version",
        Value=version,
        Type="String",
        Overwrite=True,
    )
    print(f"  ✅ SSM /cloudsentinel/model/latest-version → {version}")


def publish_cloudwatch_metrics(metrics: dict[str, float], version: str) -> None:
    """Publish training metrics to CloudWatch."""
    import boto3

    cw = boto3.client("cloudwatch")
    namespace = "CloudSentinel/ML"
    dimensions = [{"Name": "ModelVersion", "Value": version}]

    metric_data = []
    for name in ["f1_score", "precision", "recall", "fpr"]:
        if name in metrics:
            metric_data.append({
                "MetricName": name.replace("_", "-").title().replace("-", ""),
                "Dimensions": dimensions,
                "Value": metrics[name],
                "Unit": "None",
            })

    if metric_data:
        cw.put_metric_data(Namespace=namespace, MetricData=metric_data)
        print(f"  ✅ Published {len(metric_data)} metrics to CloudWatch")


def train(
    data_dir: str = "data",
    output_dir: str = "model",
    version: str | None = None,
    params: dict[str, Any] | None = None,
    upload: bool = False,
    bucket: str = "",
) -> dict[str, Any]:
    """Execute the full training pipeline.

    Args:
        data_dir: Directory containing Parquet datasets.
        output_dir: Directory for serialized model artifacts.
        version: Model version string (default: v{timestamp}).
        params: Override hyperparameters.
        upload: Whether to upload to S3 and update SSM.
        bucket: S3 bucket for model storage.

    Returns:
        dict with model path, metrics, threshold, version.
    """
    if version is None:
        version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    effective_params = {**DEFAULT_PARAMS, **(params or {})}

    print("=" * 70)
    print("CloudSentinel — ML Training Pipeline")
    print("=" * 70)
    print(f"  Version:        {version}")
    print(f"  Parameters:     {effective_params}")
    print(f"  Data directory:  {data_dir}")
    print(f"  Output:          {output_dir}")
    print()

    # ── 1. Load Data ──
    print("[1/6] Loading datasets...")
    df_baseline, df_combined = load_data(data_dir)

    X_train = df_baseline[FEATURE_NAMES].values  # 10k normal
    df_anomalous = df_combined[df_combined["is_anomaly"] == True]  # noqa: E712
    X_anomalous = df_anomalous[FEATURE_NAMES].values  # 1k anomalous

    print(f"  Training samples (baseline):  {X_train.shape[0]:,}")
    print(f"  Anomalous samples (calib):    {X_anomalous.shape[0]:,}")

    # ── 2. Build & Train ──
    print("\n[2/6] Training IsolationForest pipeline...")
    pipeline = build_pipeline(effective_params)
    pipeline.fit(X_train)
    print("  ✅ Pipeline trained successfully")

    # ── 3. Calibrate Threshold ──
    print("\n[3/6] Calibrating anomaly threshold (FPR < 5%)...")
    threshold, calib_metrics = calibrate_threshold(
        pipeline, X_anomalous, X_train[:2000]
    )
    print(f"  Threshold (normalized 0-100): {threshold}")
    print(f"  F1-score:      {calib_metrics['f1_score']:.4f}")
    print(f"  Precision:     {calib_metrics['precision']:.4f}")
    print(f"  Recall:        {calib_metrics['recall']:.4f}")
    print(f"  FPR:           {calib_metrics['fpr']:.4f}")

    # ── 4. Compute AUC ──
    print("\n[4/6] Computing AUC-ROC...")
    auc = compute_auc(pipeline, X_train[:2000], X_anomalous)
    calib_metrics["auc_roc"] = round(auc, 4)
    print(f"  AUC-ROC: {auc:.4f}")

    # ── 5. Serialize ──
    print("\n[5/6] Serializing model artifacts...")
    model_path, metadata_path = serialize_model(
        pipeline, threshold, calib_metrics, version, output_dir, effective_params
    )
    print(f"  Model:    {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Metadata: {metadata_path}")

    # ── 6. Upload (optional) ──
    if upload:
        print("\n[6/6] Uploading to S3 and updating SSM...")
        if not bucket:
            raise ValueError("--bucket is required when --upload is set")
        upload_to_s3(model_path, metadata_path, bucket, version)
        update_ssm_version(version)
        publish_cloudwatch_metrics(calib_metrics, version)
    else:
        print("\n[6/6] Skipping S3 upload (use --upload to enable)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"  Version:    {version}")
    print(f"  Threshold:  {threshold}")
    print(f"  AUC-ROC:    {auc:.4f}")
    print(f"  F1-score:   {calib_metrics['f1_score']:.4f}")
    print(f"  FPR:        {calib_metrics['fpr']:.4f}")
    auc_status = "✅ PASS" if auc >= 0.85 else "❌ BELOW TARGET"
    fpr_status = "✅ PASS" if calib_metrics["fpr"] < 0.05 else "❌ ABOVE LIMIT"
    print(f"  AUC ≥ 0.85: {auc_status}")
    print(f"  FPR < 5%:   {fpr_status}")
    print("=" * 70)

    return {
        "version": version,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "threshold": threshold,
        "metrics": calib_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CloudSentinel Isolation Forest model"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing Parquet training data",
    )
    parser.add_argument(
        "--output-dir", type=str, default="model",
        help="Directory for serialized model output",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Model version (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200,
        help="IsolationForest n_estimators",
    )
    parser.add_argument(
        "--contamination", type=float, default=0.1,
        help="IsolationForest contamination parameter",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload model to S3 and update SSM",
    )
    parser.add_argument(
        "--bucket", type=str, default="",
        help="S3 bucket for model artifacts (required with --upload)",
    )
    args = parser.parse_args()

    params = {
        **DEFAULT_PARAMS,
        "n_estimators": args.n_estimators,
        "contamination": args.contamination,
    }

    result = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        version=args.version,
        params=params,
        upload=args.upload,
        bucket=args.bucket,
    )

    if result["metrics"].get("fpr", 1.0) >= 0.05:
        print("\n⚠️  WARNING: FPR ≥ 5% — model may not meet production gate")
        sys.exit(1)


if __name__ == "__main__":
    main()
