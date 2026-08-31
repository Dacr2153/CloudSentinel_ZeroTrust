#!/usr/bin/env python3
# FILE: cloudsentinel-zero-trust/ml/training/model_registry.py
"""S3-based model registry for CloudSentinel ML models.

Provides versioning, metadata tracking, rollback, and comparison
capabilities. All model artifacts live in S3 with a canonical structure:

    s3://{bucket}/models/{version}/model.joblib
    s3://{bucket}/models/{version}/metadata.json
    s3://{bucket}/models/isolation_forest/model.joblib   ← latest symlink

SSM Parameter /cloudsentinel/model/latest-version tracks the active version.

Methods:
    register(model_path, metrics, version)  → Upload + register
    get_latest()                            → Current production version info
    rollback(version)                       → Revert to a previous version
    list_versions()                         → All registered versions
    compare_versions(v1, v2)                → Side-by-side metrics table
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


SSM_VERSION_KEY = "/cloudsentinel/model/latest-version"
SSM_THRESHOLD_KEY = "/cloudsentinel/model/anomaly-threshold"
MODELS_PREFIX = "models"
LATEST_KEY = f"{MODELS_PREFIX}/isolation_forest/model.joblib"


class ModelRegistry:
    """S3-backed model registry with version management.

    Args:
        bucket: S3 bucket name for model artifacts.
        region: AWS region.
    """

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        self._bucket = bucket
        self._region = region
        self._s3 = boto3.client("s3", region_name=region)
        self._ssm = boto3.client("ssm", region_name=region)


    def register(
        self,
        model_path: str | Path,
        metadata_path: str | Path | None = None,
        metrics: dict[str, Any] | None = None,
        version: str | None = None,
        set_as_latest: bool = True,
    ) -> dict[str, str]:
        """Upload and register a new model version.

        Args:
            model_path: Local path to .joblib model artifact.
            metadata_path: Optional local path to metadata.json.
            metrics: Metrics dict (used if metadata_path is not provided).
            version: Version string (required).
            set_as_latest: Whether to update SSM and the 'latest' pointer.

        Returns:
            dict with s3_model_key, s3_metadata_key, version.
        """
        if not version:
            version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        s3_model_key = f"{MODELS_PREFIX}/{version}/model.joblib"
        s3_metadata_key = f"{MODELS_PREFIX}/{version}/metadata.json"

        # Upload model
        self._s3.upload_file(str(model_path), self._bucket, s3_model_key)
        print(f"  ✅ Uploaded s3://{self._bucket}/{s3_model_key}")

        # Upload or generate metadata
        if metadata_path and Path(metadata_path).exists():
            self._s3.upload_file(str(metadata_path), self._bucket, s3_metadata_key)
        else:
            metadata = {
                "version": version,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics or {},
            }
            self._s3.put_object(
                Bucket=self._bucket,
                Key=s3_metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType="application/json",
            )
        print(f"  ✅ Uploaded s3://{self._bucket}/{s3_metadata_key}")

        if set_as_latest:
            self._set_latest(version, model_path)

        return {
            "version": version,
            "s3_model_key": s3_model_key,
            "s3_metadata_key": s3_metadata_key,
        }

    def _set_latest(self, version: str, model_path: Path) -> None:
        """Update the 'latest' pointer in S3 and SSM."""
        # Copy model to latest path
        self._s3.upload_file(str(model_path), self._bucket, LATEST_KEY)
        print(f"  ✅ Updated s3://{self._bucket}/{LATEST_KEY} → {version}")

        # Update SSM
        self._ssm.put_parameter(
            Name=SSM_VERSION_KEY,
            Value=version,
            Type="String",
            Overwrite=True,
        )
        print(f"  ✅ SSM {SSM_VERSION_KEY} → {version}")


    def get_latest(self) -> dict[str, Any]:
        """Get the current production model version and metadata.

        Returns:
            dict with version, metadata, s3_model_uri.
        """
        try:
            resp = self._ssm.get_parameter(Name=SSM_VERSION_KEY)
            version = resp["Parameter"].get("Value", "")
        except ClientError:
            raise RuntimeError(
                f"No model version found in SSM at {SSM_VERSION_KEY}. "
                "Register a model first."
            )

        metadata = self._get_metadata(version)

        return {
            "version": version,
            "s3_model_uri": f"s3://{self._bucket}/{MODELS_PREFIX}/{version}/model.joblib",
            "metadata": metadata,
        }

    def _get_metadata(self, version: str) -> dict[str, Any]:
        """Download metadata.json for a specific version."""
        key = f"{MODELS_PREFIX}/{version}/metadata.json"
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except ClientError:
            return {"version": version, "error": "metadata not found"}


    def list_versions(self) -> list[dict[str, Any]]:
        """List all registered model versions with metadata.

        Returns:
            List of dicts with version, registered_at, metrics summary.
        """
        paginator = self._s3.get_paginator("list_objects_v2")
        versions: list[dict[str, Any]] = []

        for page in paginator.paginate(
            Bucket=self._bucket, Prefix=f"{MODELS_PREFIX}/v", Delimiter="/"
        ):
            for prefix_info in page.get("CommonPrefixes", []):
                prefix = prefix_info.get("Prefix", "")  # models/v20240101/
                version = prefix.rstrip("/").split("/")[-1]
                metadata = self._get_metadata(version)
                versions.append({
                    "version": version,
                    "registered_at": metadata.get("registered_at", metadata.get("created_at", "unknown")),
                    "metrics": metadata.get("metrics", {}),
                })

        # Sort by version descending
        versions.sort(key=lambda v: v["version"], reverse=True)
        return versions


    def rollback(self, version: str) -> dict[str, str]:
        """Roll back to a previous model version.

        Copies the specified version's model to the 'latest' location
        and updates SSM.

        Args:
            version: Target version to restore.

        Returns:
            dict with version, status.
        """
        # Verify the version exists
        source_key = f"{MODELS_PREFIX}/{version}/model.joblib"
        try:
            self._s3.head_object(Bucket=self._bucket, Key=source_key)
        except ClientError:
            raise ValueError(
                f"Version '{version}' not found at s3://{self._bucket}/{source_key}"
            )

        # Copy to latest
        self._s3.copy_object(
            Bucket=self._bucket,
            CopySource={"Bucket": self._bucket, "Key": source_key},
            Key=LATEST_KEY,
        )
        print(f"  ✅ Rolled back s3://{self._bucket}/{LATEST_KEY} → {version}")

        # Update SSM
        self._ssm.put_parameter(
            Name=SSM_VERSION_KEY,
            Value=version,
            Type="String",
            Overwrite=True,
        )
        print(f"  ✅ SSM {SSM_VERSION_KEY} → {version}")

        # Update threshold from metadata if available
        metadata = self._get_metadata(version)
        threshold = metadata.get("threshold_normalized") or metadata.get("metrics", {}).get("threshold_normalized")
        if threshold is not None:
            self._ssm.put_parameter(
                Name=SSM_THRESHOLD_KEY,
                Value=str(threshold),
                Type="String",
                Overwrite=True,
            )
            print(f"  ✅ SSM {SSM_THRESHOLD_KEY} → {threshold}")

        return {"version": version, "status": "rolled_back"}


    def compare_versions(self, v1: str, v2: str) -> dict[str, Any]:
        """Compare two model versions side-by-side.

        Returns:
            dict with comparison table data.
        """
        meta1 = self._get_metadata(v1)
        meta2 = self._get_metadata(v2)

        metrics1 = meta1.get("metrics", {})
        metrics2 = meta2.get("metrics", {})

        all_keys = sorted(set(list(metrics1.keys()) + list(metrics2.keys())))

        comparison: list[dict[str, Any]] = []
        for key in all_keys:
            val1 = metrics1.get(key)
            val2 = metrics2.get(key)
            delta = None
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                delta = round(val2 - val1, 4)

            comparison.append({
                "metric": key,
                v1: val1,
                v2: val2,
                "delta": delta,
            })

        # Print comparison table
        print(f"\n{'Metric':<30}  {v1:>12}  {v2:>12}  {'Delta':>10}")
        print("-" * 70)
        for row in comparison:
            v1_str = f"{row[v1]:.4f}" if isinstance(row[v1], float) else str(row[v1] or "—")
            v2_str = f"{row[v2]:.4f}" if isinstance(row[v2], float) else str(row[v2] or "—")
            d_str = f"{row['delta']:+.4f}" if row["delta"] is not None else "—"
            print(f"  {row['metric']:<28}  {v1_str:>12}  {v2_str:>12}  {d_str:>10}")

        return {
            "v1": v1,
            "v2": v2,
            "meta_v1": meta1,
            "meta_v2": meta2,
            "comparison": comparison,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CloudSentinel ML Model Registry"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register
    reg = subparsers.add_parser("register", help="Register a new model version")
    reg.add_argument("--model", type=str, required=True, help="Path to .joblib model")
    reg.add_argument("--metadata", type=str, default=None, help="Path to metadata.json")
    reg.add_argument("--version", type=str, required=True, help="Version string")
    reg.add_argument("--bucket", type=str, required=True, help="S3 bucket")
    reg.add_argument("--region", type=str, default="us-east-1")

    # latest
    lat = subparsers.add_parser("latest", help="Show current production model")
    lat.add_argument("--bucket", type=str, required=True)
    lat.add_argument("--region", type=str, default="us-east-1")

    # list
    ls = subparsers.add_parser("list", help="List all registered versions")
    ls.add_argument("--bucket", type=str, required=True)
    ls.add_argument("--region", type=str, default="us-east-1")

    # rollback
    rb = subparsers.add_parser("rollback", help="Rollback to a previous version")
    rb.add_argument("--version", type=str, required=True, help="Target version")
    rb.add_argument("--bucket", type=str, required=True)
    rb.add_argument("--region", type=str, default="us-east-1")

    # compare
    cmp = subparsers.add_parser("compare", help="Compare two model versions")
    cmp.add_argument("--v1", type=str, required=True, help="First version")
    cmp.add_argument("--v2", type=str, required=True, help="Second version")
    cmp.add_argument("--bucket", type=str, required=True)
    cmp.add_argument("--region", type=str, default="us-east-1")

    args = parser.parse_args()
    registry = ModelRegistry(bucket=args.bucket, region=args.region)

    if args.command == "register":
        result = registry.register(
            model_path=args.model,
            metadata_path=args.metadata,
            version=args.version,
        )
        print(f"\n✅ Registered {result['version']}")

    elif args.command == "latest":
        info = registry.get_latest()
        print(json.dumps(info, indent=2, default=str))

    elif args.command == "list":
        versions = registry.list_versions()
        print(f"\nRegistered model versions ({len(versions)}):")
        for v in versions:
            metrics = v.get("metrics", {})
            f1 = metrics.get("f1_score", "—")
            auc_val = metrics.get("auc_roc", "—")
            print(f"  {v['version']:20s}  F1={f1}  AUC={auc_val}  ({v['registered_at']})")

    elif args.command == "rollback":
        result = registry.rollback(args.version)
        print(f"\n✅ Rolled back to {result['version']}")

    elif args.command == "compare":
        registry.compare_versions(args.v1, args.v2)


if __name__ == "__main__":
    main()
