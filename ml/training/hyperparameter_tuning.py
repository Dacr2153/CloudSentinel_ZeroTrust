#!/usr/bin/env python3
# FILE: cloudsentinel-zero-trust/ml/training/hyperparameter_tuning.py
"""Hyperparameter tuning for CloudSentinel Isolation Forest.

Grid search over:
  - n_estimators:   [100, 200, 300]
  - contamination:  [0.05, 0.10, 0.15]
  - max_features:   [5, 7, 10]

Total combinations: 27

Optimization metric: F1-score on anomalous set with FPR < 5% constraint.
Parallelization: n_jobs=-1 per IsolationForest fit.

Output:
  - tuning_results.csv    — all 27 configs with metrics
  - best_params.json      — best configuration + metrics
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score
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

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "contamination": [0.05, 0.10, 0.15],
    "max_features": [5, 7, 10],
}

MAX_FPR = 0.05


def _find_best_threshold(
    scores_normal: np.ndarray,
    scores_anomalous: np.ndarray,
    max_fpr: float = MAX_FPR,
) -> tuple[float | None, dict[str, float]]:
    """Sweep raw thresholds to find best F1 with FPR constraint.

    Returns:
        (best_raw_threshold, metrics) or (None, {}) if no valid threshold
    """
    all_scores = np.concatenate([scores_normal, scores_anomalous])
    all_labels = np.concatenate([
        np.zeros(len(scores_normal), dtype=int),
        np.ones(len(scores_anomalous), dtype=int),
    ])

    thresholds = np.percentile(all_scores, np.linspace(1, 99, 300))
    best_f1 = -1.0
    best_thresh = None
    best_metrics: dict[str, float] = {}

    for thresh in thresholds:
        preds = (all_scores < thresh).astype(int)
        fp = np.sum((preds == 1) & (all_labels == 0))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if fpr > max_fpr:
            continue

        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            tp = np.sum((preds == 1) & (all_labels == 1))
            fn = np.sum((preds == 0) & (all_labels == 1))
            best_metrics = {
                "f1_score": round(float(f1), 4),
                "precision": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4),
                "recall": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
                "fpr": round(float(fpr), 4),
            }

    return best_thresh, best_metrics


def _compute_auc(
    scores_normal: np.ndarray, scores_anomalous: np.ndarray
) -> float:
    """AUC-ROC from raw decision_function scores."""
    all_labels = np.concatenate([
        np.zeros(len(scores_normal)),
        np.ones(len(scores_anomalous)),
    ])
    all_scores = np.concatenate([scores_normal, scores_anomalous])
    return float(roc_auc_score(all_labels, -all_scores))


def run_tuning(
    data_dir: str = "data",
    output_dir: str = "tuning",
    param_grid: dict[str, list] | None = None,
) -> dict[str, Any]:
    """Execute grid search over hyperparameter space.

    Args:
        data_dir: Directory containing Parquet datasets.
        output_dir: Directory for tuning output files.
        param_grid: Override parameter grid (default: PARAM_GRID).

    Returns:
        dict with best_params, best_metrics, results_path.
    """
    data_dir_path = Path(data_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    grid = param_grid or PARAM_GRID

    # ── Load data ──
    print("=" * 70)
    print("CloudSentinel — Hyperparameter Tuning")
    print("=" * 70)

    baseline_path = data_dir_path / "baseline_events.parquet"
    combined_path = data_dir_path / "combined_labeled.parquet"

    df_baseline = pd.read_parquet(baseline_path)
    df_combined = pd.read_parquet(combined_path)
    df_anomalous = df_combined[df_combined["is_anomaly"] == True]  # noqa: E712

    X_train = df_baseline[FEATURE_NAMES].values
    X_anomalous = df_anomalous[FEATURE_NAMES].values
    X_normal_eval = X_train[:2000]  # Subset for evaluation speed

    print(f"  Training samples:   {X_train.shape[0]:,}")
    print(f"  Anomalous samples:  {X_anomalous.shape[0]:,}")

    # ── Generate combinations ──
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"  Grid size:          {total} combinations")
    print(f"  Parameters:         {dict(zip(keys, [len(v) for v in values]))}")
    print()

    # ── Run grid search ──
    results: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_result: dict[str, Any] = {}

    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  [{idx:2d}/{total}] {label}", end="  ", flush=True)

        t0 = time.time()

        # Select features if max_features < 10
        n_features = params.get("max_features", 10)
        X_train_sub = X_train[:, :n_features]
        X_anom_sub = X_anomalous[:, :n_features]
        X_norm_sub = X_normal_eval[:, :n_features]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("isolation_forest", IsolationForest(
                n_estimators=params["n_estimators"],
                contamination=params["contamination"],
                max_samples="auto",
                random_state=42,
                n_jobs=-1,
            )),
        ])

        pipe.fit(X_train_sub)

        scores_norm = pipe.decision_function(X_norm_sub)
        scores_anom = pipe.decision_function(X_anom_sub)

        thresh, metrics = _find_best_threshold(scores_norm, scores_anom)
        auc_val = _compute_auc(scores_norm, scores_anom)

        elapsed = time.time() - t0

        row = {
            **params,
            "threshold_raw": thresh,
            "auc_roc": round(auc_val, 4),
            "elapsed_seconds": round(elapsed, 2),
            **metrics,
        }
        results.append(row)

        f1 = metrics.get("f1_score", 0.0)
        status = "✅" if f1 > 0 else "⚠️ no valid threshold"
        print(f"F1={f1:.4f}  AUC={auc_val:.4f}  [{elapsed:.1f}s] {status}")

        if f1 > best_f1:
            best_f1 = f1
            best_result = row.copy()

    # ── Write results CSV ──
    csv_path = output_dir_path / "tuning_results.csv"
    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Write best params JSON ──
    json_path = output_dir_path / "best_params.json"
    best_output = {
        "best_params": {
            "n_estimators": best_result.get("n_estimators"),
            "contamination": best_result.get("contamination"),
            "max_features": best_result.get("max_features"),
            "max_samples": "auto",
            "random_state": 42,
        },
        "best_metrics": {
            "f1_score": best_result.get("f1_score"),
            "precision": best_result.get("precision"),
            "recall": best_result.get("recall"),
            "fpr": best_result.get("fpr"),
            "auc_roc": best_result.get("auc_roc"),
        },
        "threshold_raw": best_result.get("threshold_raw"),
        "tuning_date": datetime.now(timezone.utc).isoformat(),
        "total_combinations": total,
        "total_valid": sum(1 for r in results if r.get("f1_score", 0) > 0),
    }
    json_path.write_text(json.dumps(best_output, indent=2))

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Tuning Complete")
    print("=" * 70)
    print(f"  Best F1-score:   {best_f1:.4f}")
    print("  Best params:")
    for k in ["n_estimators", "contamination", "max_features"]:
        print(f"    {k}: {best_result.get(k)}")
    print(f"  AUC-ROC:         {best_result.get('auc_roc', 'N/A')}")
    print(f"  FPR:             {best_result.get('fpr', 'N/A')}")
    print(f"  Results CSV:     {csv_path}")
    print(f"  Best params:     {json_path}")
    print("=" * 70)

    return {
        "best_params": best_output["best_params"],
        "best_metrics": best_output["best_metrics"],
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "all_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for CloudSentinel IsolationForest"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing Parquet datasets",
    )
    parser.add_argument(
        "--output-dir", type=str, default="tuning",
        help="Directory for tuning results",
    )
    args = parser.parse_args()

    run_tuning(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
