#!/usr/bin/env python3
# FILE: cloudsentinel-zero-trust/ml/training/evaluate_model.py
"""Rigorous evaluation of CloudSentinel Isolation Forest model.

Generates a comprehensive HTML report with:
  - Confusion matrix (heatmap)
  - ROC curve + AUC score
  - Precision-Recall curve + AP score
  - Score distribution overlay (normal vs anomalous)
  - FPR and FNR per attack type
  - Feature importance (permutation-based proxy)
  - Version comparison table (if previous models exist)

FPR Gate: exits with code 1 if FPR ≥ 5% (blocks CI pipeline).
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


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

ATTACK_TYPES = ["credential_stuffing", "privilege_escalation", "data_exfiltration"]
MAX_FPR = 0.05


def _fig_to_base64(fig: Any) -> str:
    """Convert matplotlib figure to base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Render confusion matrix as base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ylabel="True Label", xlabel="Predicted Label",
        title="Confusion Matrix",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _plot_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[str, float]:
    """Render ROC curve as base64 PNG, return (b64, auc_score)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr_arr, tpr_arr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, color="#2196F3", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.axvline(x=0.05, color="#F44336", lw=1, linestyle=":", label="FPR=5% limit")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right")
    plt.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64, float(roc_auc)


def _plot_pr_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[str, float]:
    """Render Precision-Recall curve as base64 PNG, return (b64, ap_score)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#4CAF50", lw=2, label=f"AP = {ap:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curve", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower left")
    plt.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64, float(ap)


def _plot_score_distribution(
    scores_normal: np.ndarray, scores_anom: np.ndarray
) -> str:
    """Render score distribution histogram overlay as base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(scores_normal.min(), scores_anom.min()),
        max(scores_normal.max(), scores_anom.max()),
        80,
    )
    ax.hist(scores_normal, bins=bins, alpha=0.6, color="#4CAF50", label="Normal", density=True)
    ax.hist(scores_anom, bins=bins, alpha=0.6, color="#F44336", label="Anomalous", density=True)
    ax.set(xlabel="Anomaly Score (negated decision_function)",
           ylabel="Density", title="Score Distribution: Normal vs Anomalous")
    ax.legend()
    plt.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _plot_feature_importance(importance: dict[str, float]) -> str:
    """Render feature importance bar chart as base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(importance.keys())
    values = list(importance.values())
    sorted_idx = np.argsort(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [names[i] for i in sorted_idx],
        [values[i] for i in sorted_idx],
        color="#2196F3",
    )
    ax.set(xlabel="Importance (AUC delta)", title="Feature Importance (Permutation)")
    plt.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def load_model(model_path: str | Path) -> dict[str, Any]:
    """Load serialized model artifact."""
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact
    # Fallback: raw pipeline without wrapper
    return {"pipeline": artifact, "threshold": 65, "feature_names": FEATURE_NAMES}


def evaluate_per_attack_type(
    pipeline: Any,
    df_anomalous: pd.DataFrame,
    threshold_raw: float,
) -> list[dict[str, Any]]:
    """Compute metrics broken down by attack type."""
    results = []
    for attack_type in ATTACK_TYPES:
        df_sub = df_anomalous[df_anomalous["attack_type"] == attack_type]
        if df_sub.empty:
            continue

        X = df_sub[FEATURE_NAMES].values
        scores_raw = pipeline.decision_function(X)
        preds = (scores_raw < threshold_raw).astype(int)

        tp = int(preds.sum())
        fn = len(preds) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        results.append({
            "attack_type": attack_type,
            "total": len(df_sub),
            "detected": tp,
            "missed": fn,
            "recall": round(recall, 4),
            "fnr": round(fnr, 4),
        })

    return results


def compute_permutation_importance(
    pipeline: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
) -> dict[str, float]:
    """Proxy permutation importance: measures AUC drop when shuffling each feature."""
    rng = np.random.default_rng(42)
    scores_baseline = -pipeline.decision_function(X)
    auc_baseline = float(
        auc(*roc_curve(y, scores_baseline)[:2])
    ) if len(np.unique(y)) > 1 else 0.5

    importance: dict[str, float] = {}
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        auc_drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, feat_idx])
            scores_perm = -pipeline.decision_function(X_perm)
            try:
                fpr_arr, tpr_arr, _ = roc_curve(y, scores_perm)
                auc_perm = float(auc(fpr_arr, tpr_arr))
            except ValueError:
                auc_perm = 0.5
            auc_drops.append(auc_baseline - auc_perm)
        importance[feat_name] = round(float(np.mean(auc_drops)), 4)

    return importance


def generate_html_report(
    version: str,
    confusion_img: str,
    roc_img: str,
    roc_auc: float,
    pr_img: str,
    ap_score: float,
    dist_img: str,
    importance_img: str,
    importance: dict[str, float],
    per_attack: list[dict],
    overall_metrics: dict[str, float],
) -> str:
    """Generate a self-contained HTML evaluation report."""
    attack_rows = ""
    for a in per_attack:
        attack_rows += f"""
        <tr>
            <td>{a['attack_type']}</td>
            <td>{a['total']}</td>
            <td>{a['detected']}</td>
            <td>{a['missed']}</td>
            <td>{a['recall']:.2%}</td>
            <td>{a['fnr']:.2%}</td>
        </tr>"""

    fpr_color = "#4CAF50" if overall_metrics["fpr"] < MAX_FPR else "#F44336"
    auc_color = "#4CAF50" if roc_auc >= 0.85 else "#F44336"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CloudSentinel Model Evaluation — {version}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #f5f5f5; color: #333; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #1a237e; margin-bottom: 5px; }}
        h2 {{ color: #283593; margin: 30px 0 15px; border-bottom: 2px solid #3f51b5; padding-bottom: 8px; }}
        .meta {{ color: #666; margin-bottom: 30px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .kpi {{ background: white; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .kpi-value {{ font-size: 2em; font-weight: bold; }}
        .kpi-label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .chart-grid img {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
        th {{ background: #3f51b5; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f5f5f5; }}
        .pass {{ color: #4CAF50; font-weight: bold; }}
        .fail {{ color: #F44336; font-weight: bold; }}
        .footer {{ text-align: center; color: #999; margin-top: 40px; padding: 20px; }}
        @media (max-width: 768px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
<div class="container">
    <h1>🛡️ CloudSentinel Model Evaluation Report</h1>
    <p class="meta">Version: <strong>{version}</strong> &mdash; Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

    <h2>Key Performance Indicators</h2>
    <div class="kpi-grid">
        <div class="kpi">
            <div class="kpi-value" style="color: {auc_color}">{roc_auc:.4f}</div>
            <div class="kpi-label">AUC-ROC (target ≥ 0.85)</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">{overall_metrics['f1_score']:.4f}</div>
            <div class="kpi-label">F1-Score</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">{overall_metrics['precision']:.4f}</div>
            <div class="kpi-label">Precision</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">{overall_metrics['recall']:.4f}</div>
            <div class="kpi-label">Recall</div>
        </div>
        <div class="kpi">
            <div class="kpi-value" style="color: {fpr_color}">{overall_metrics['fpr']:.2%}</div>
            <div class="kpi-label">FPR (limit &lt; 5%)</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">{ap_score:.4f}</div>
            <div class="kpi-label">Average Precision</div>
        </div>
    </div>

    <h2>Confusion Matrix</h2>
    <div style="text-align:center">
        <img src="data:image/png;base64,{confusion_img}" alt="Confusion Matrix" style="max-width:500px">
    </div>

    <h2>ROC &amp; Precision-Recall Curves</h2>
    <div class="chart-grid">
        <img src="data:image/png;base64,{roc_img}" alt="ROC Curve">
        <img src="data:image/png;base64,{pr_img}" alt="PR Curve">
    </div>

    <h2>Score Distribution</h2>
    <div style="text-align:center">
        <img src="data:image/png;base64,{dist_img}" alt="Score Distribution" style="max-width:700px">
    </div>

    <h2>Detection by Attack Type</h2>
    <table>
        <thead>
            <tr><th>Attack Type</th><th>Total</th><th>Detected</th><th>Missed</th><th>Recall</th><th>FNR</th></tr>
        </thead>
        <tbody>{attack_rows}</tbody>
    </table>

    <h2>Feature Importance (Permutation)</h2>
    <div style="text-align:center">
        <img src="data:image/png;base64,{importance_img}" alt="Feature Importance" style="max-width:700px">
    </div>

    <h2>CI/CD Gate Check</h2>
    <table>
        <thead><tr><th>Check</th><th>Target</th><th>Actual</th><th>Status</th></tr></thead>
        <tbody>
            <tr>
                <td>AUC-ROC</td><td>≥ 0.85</td><td>{roc_auc:.4f}</td>
                <td class="{'pass' if roc_auc >= 0.85 else 'fail'}">{'✅ PASS' if roc_auc >= 0.85 else '❌ FAIL'}</td>
            </tr>
            <tr>
                <td>FPR</td><td>&lt; 5%</td><td>{overall_metrics['fpr']:.2%}</td>
                <td class="{'pass' if overall_metrics['fpr'] < MAX_FPR else 'fail'}">{'✅ PASS' if overall_metrics['fpr'] < MAX_FPR else '❌ FAIL'}</td>
            </tr>
        </tbody>
    </table>

    <div class="footer">
        CloudSentinel Zero-Trust SIEM &mdash; Evaluation Report &mdash; {version}
    </div>
</div>
</body>
</html>"""


def evaluate(
    model_path: str,
    data_dir: str = "data",
    output_dir: str = "reports",
    version: str | None = None,
) -> dict[str, Any]:
    """Execute full evaluation pipeline.

    Args:
        model_path: Path to .joblib model artifact.
        data_dir: Directory containing combined_labeled.parquet.
        output_dir: Directory for the HTML report.
        version: Version label for the report.

    Returns:
        dict with metrics, report_path, and pass/fail status.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CloudSentinel — Model Evaluation")
    print("=" * 70)

    # ── Load model ──
    print("[1/7] Loading model...")
    artifact = load_model(model_path)
    pipeline = artifact["pipeline"]
    threshold_raw = artifact.get("threshold_raw", None)
    version = version or artifact.get("version") or "unknown"
    assert isinstance(version, str)
    print(f"  Version: {version}")

    # ── Load data ──
    print("[2/7] Loading evaluation dataset...")
    combined_path = Path(data_dir) / "combined_labeled.parquet"
    df = pd.read_parquet(combined_path)

    # Stratified 20% test split (deterministic)
    rng = np.random.default_rng(42)
    test_idx = []
    for label in [True, False]:
        indices = df.index[df["is_anomaly"] == label].tolist()
        n_test = max(1, int(len(indices) * 0.2))
        chosen = rng.choice(indices, size=n_test, replace=False)
        test_idx.extend(chosen.tolist())

    df_test = df.loc[test_idx].copy()
    X_test = df_test[FEATURE_NAMES].values
    y_test: np.ndarray = np.asarray(df_test["is_anomaly"].astype(int).values)
    print(f"  Test set: {len(df_test)} samples ({y_test.sum()} anomalous, {(1-y_test).sum()} normal)")

    # ── Score ──
    print("[3/7] Scoring test set...")
    raw_scores = pipeline.decision_function(X_test)
    negated_scores = -raw_scores  # higher = more anomalous for sklearn metrics

    # Determine threshold on raw scale
    if threshold_raw is None:
        # Use the pipeline's internal decision from contamination
        preds_internal = pipeline.predict(X_test)
        y_pred = (preds_internal == -1).astype(int)
    else:
        y_pred = (raw_scores < threshold_raw).astype(int)

    # ── Overall metrics ──
    print("[4/7] Computing overall metrics...")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    overall = {
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4),
        "recall": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    print(f"  F1={overall['f1_score']}, Prec={overall['precision']}, "
          f"Rec={overall['recall']}, FPR={overall['fpr']:.4f}")

    # ── Per-attack type ──
    print("[5/7] Per-attack-type analysis...")
    df_anom_test: pd.DataFrame = df_test[df_test["is_anomaly"] == True]  # noqa: E712

    # We need a raw threshold for per-type; use pipeline's internal
    if threshold_raw is not None:
        per_attack = evaluate_per_attack_type(pipeline, df_anom_test, threshold_raw)
    else:
        per_attack = []
        for attack_type in ATTACK_TYPES:
            df_sub = df_anom_test[df_anom_test["attack_type"] == attack_type]
            if df_sub.empty:
                continue
            X_sub = df_sub[FEATURE_NAMES].values
            preds_sub = pipeline.predict(X_sub)
            detected = int((preds_sub == -1).sum())
            per_attack.append({
                "attack_type": attack_type,
                "total": len(df_sub),
                "detected": detected,
                "missed": len(df_sub) - detected,
                "recall": round(detected / len(df_sub), 4),
                "fnr": round(1 - detected / len(df_sub), 4),
            })

    for a in per_attack:
        print(f"    {a['attack_type']}: {a['recall']:.2%} recall "
              f"({a['detected']}/{a['total']})")

    # ── Feature importance ──
    print("[6/7] Computing feature importance (permutation)...")
    importance = compute_permutation_importance(pipeline, X_test, y_test, n_repeats=5)

    # ── Generate plots & report ──
    print("[7/7] Generating HTML report...")
    confusion_img = _plot_confusion_matrix(y_test, y_pred)
    roc_img, roc_auc = _plot_roc_curve(y_test, negated_scores)
    pr_img, ap = _plot_pr_curve(y_test, negated_scores)

    # Score distribution
    mask_normal = y_test == 0
    mask_anom = y_test == 1
    dist_img = _plot_score_distribution(negated_scores[mask_normal], negated_scores[mask_anom])
    importance_img = _plot_feature_importance(importance)

    html = generate_html_report(
        version=version,
        confusion_img=confusion_img,
        roc_img=roc_img,
        roc_auc=roc_auc,
        pr_img=pr_img,
        ap_score=ap,
        dist_img=dist_img,
        importance_img=importance_img,
        importance=importance,
        per_attack=per_attack,
        overall_metrics=overall,
    )

    report_path = output_dir_path / f"evaluation_report_{version}.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\n  Report: {report_path}")

    # ── CI Gate ──
    fpr_pass = overall["fpr"] < MAX_FPR
    auc_pass = roc_auc >= 0.85

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"  AUC-ROC: {roc_auc:.4f}  {'✅ PASS' if auc_pass else '❌ FAIL (< 0.85)'}")
    print(f"  FPR:     {overall['fpr']:.4f}  {'✅ PASS' if fpr_pass else '❌ FAIL (≥ 5%)'}")
    print(f"  F1:      {overall['f1_score']:.4f}")
    print("=" * 70)

    return {
        "version": version,
        "report_path": str(report_path),
        "roc_auc": roc_auc,
        "ap_score": ap,
        "metrics": overall,
        "per_attack": per_attack,
        "importance": importance,
        "fpr_pass": fpr_pass,
        "auc_pass": auc_pass,
        "all_pass": fpr_pass and auc_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CloudSentinel ML model and generate HTML report"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to .joblib model artifact",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing combined_labeled.parquet",
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports",
        help="Directory for the HTML report",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Version label (default: from model artifact)",
    )
    args = parser.parse_args()

    result = evaluate(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        version=args.version,
    )

    if not result["all_pass"]:
        print("\n❌ EVALUATION GATE FAILED — model does not meet production criteria")
        sys.exit(1)
    else:
        print("\n✅ All gates passed — model is production-ready")


if __name__ == "__main__":
    main()
