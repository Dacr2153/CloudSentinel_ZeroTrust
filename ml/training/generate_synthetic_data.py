#!/usr/bin/env python3
# FILE: cloudsentinel-zero-trust/ml/training/generate_synthetic_data.py
"""Synthetic data generator for CloudSentinel ML training.

Produces a labeled dataset with 10,000 baseline (normal) events and
1,000 anomalous events across 3 attack categories:
  - Credential Stuffing      (300 events)
  - Privilege Escalation      (350 events)
  - Data Exfiltration         (350 events)

Each event is represented as a 10-dimensional feature vector matching
the FeatureEngineer output format:
  0: hour_of_day_normalized      [0, 1]
  1: day_of_week_normalized      [0, 1]
  2: is_weekend                  {0, 1}
  3: source_geo_risk_score       [0, 1]
  4: api_call_risk_score         [0, 1]
  5: is_root_user                {0, 1}
  6: is_cross_region             {0, 1}
  7: failed_auth_velocity        [0, 1]
  8: api_entropy_1h              [0, 1]
  9: privilege_escalation_score  [0, 1]

Output (Parquet):
  data/baseline_events.parquet
  data/anomalous_events.parquet
  data/combined_labeled.parquet  (column 'is_anomaly': bool, 'attack_type': str)

Seed: 42 (fully reproducible)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
NUM_BASELINE = 10_000
NUM_CREDENTIAL_STUFFING = 300
NUM_PRIVILEGE_ESCALATION = 350
NUM_DATA_EXFILTRATION = 350
NUM_ANOMALOUS = NUM_CREDENTIAL_STUFFING + NUM_PRIVILEGE_ESCALATION + NUM_DATA_EXFILTRATION

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

# 20 fictional users with individual behavioural profiles
USERS: list[dict] = [
    {"name": "alice.johnson", "dept": "engineering", "peak_hour": 10, "region_pref": 0.05},
    {"name": "bob.smith", "dept": "devops", "peak_hour": 9, "region_pref": 0.10},
    {"name": "carol.martinez", "dept": "data-science", "peak_hour": 11, "region_pref": 0.02},
    {"name": "dave.lee", "dept": "security", "peak_hour": 10, "region_pref": 0.08},
    {"name": "eve.wang", "dept": "platform", "peak_hour": 14, "region_pref": 0.15},
    {"name": "frank.taylor", "dept": "backend", "peak_hour": 10, "region_pref": 0.03},
    {"name": "grace.kim", "dept": "frontend", "peak_hour": 15, "region_pref": 0.05},
    {"name": "hank.brown", "dept": "qa", "peak_hour": 9, "region_pref": 0.02},
    {"name": "irene.garcia", "dept": "ml-ops", "peak_hour": 11, "region_pref": 0.12},
    {"name": "jack.wilson", "dept": "infra", "peak_hour": 10, "region_pref": 0.06},
    {"name": "karen.chen", "dept": "engineering", "peak_hour": 14, "region_pref": 0.04},
    {"name": "leo.nguyen", "dept": "devops", "peak_hour": 9, "region_pref": 0.08},
    {"name": "mary.davis", "dept": "compliance", "peak_hour": 10, "region_pref": 0.01},
    {"name": "noah.patel", "dept": "sre", "peak_hour": 11, "region_pref": 0.07},
    {"name": "olivia.jones", "dept": "engineering", "peak_hour": 15, "region_pref": 0.05},
    {"name": "paul.thomas", "dept": "backend", "peak_hour": 10, "region_pref": 0.03},
    {"name": "quinn.white", "dept": "data-science", "peak_hour": 13, "region_pref": 0.09},
    {"name": "rachel.moore", "dept": "security", "peak_hour": 10, "region_pref": 0.04},
    {"name": "sam.clark", "dept": "platform", "peak_hour": 14, "region_pref": 0.11},
    {"name": "tina.hall", "dept": "qa", "peak_hour": 9, "region_pref": 0.02},
]


def _generate_baseline(rng: np.random.Generator) -> np.ndarray:
    """Generate 10,000 normal events as a (10000, 10) matrix.

    Distribution design:
    - Hours: bimodal Gaussian peaks at 10AM and 3PM (working hours)
    - Days: weekdays dominant (90%), small weekend presence (10%)
    - Geo risk: very low (mu=0.08, sigma=0.05)
    - API risk: low-moderate (power-law, most reads)
    - Root: extremely rare (0.2%)
    - Cross-region: 5-15% per user
    - Failed auth: near zero (mu=0.02)
    - API entropy: moderate variety (mu=0.35)
    - Privilege escalation: near zero (mu=0.01)
    """
    n = NUM_BASELINE
    data = np.zeros((n, 10), dtype=np.float64)

    # Assign each event to a random user
    user_indices = rng.integers(0, len(USERS), size=n)

    for i in range(n):
        user = USERS[user_indices[i]]

        # Feature 0: hour_of_day_normalized
        # Bimodal: peak_hour and peak_hour+5, with sigma=2h
        if rng.random() < 0.6:
            hour = rng.normal(user["peak_hour"], 2.0)
        else:
            hour = rng.normal(user["peak_hour"] + 5, 2.0)
        # Rare off-hours access (3% chance)
        if rng.random() < 0.03:
            hour = rng.uniform(0, 24)
        hour = np.clip(hour, 0, 23)
        data[i, 0] = hour / 23.0

        # Feature 1: day_of_week_normalized
        if rng.random() < 0.90:
            dow = rng.integers(0, 5)  # Mon-Fri
        else:
            dow = rng.integers(5, 7)  # Sat-Sun
        data[i, 1] = dow / 6.0

        # Feature 2: is_weekend
        data[i, 2] = 1.0 if dow >= 5 else 0.0

        # Feature 3: source_geo_risk_score — low, mostly US
        data[i, 3] = np.clip(rng.normal(0.08, 0.05), 0, 1)

        # Feature 4: api_call_risk_score — power-law (most reads are low risk)
        data[i, 4] = np.clip(rng.exponential(0.12), 0, 1)

        # Feature 5: is_root_user — very rare
        data[i, 5] = 1.0 if rng.random() < 0.002 else 0.0

        # Feature 6: is_cross_region
        data[i, 6] = 1.0 if rng.random() < user["region_pref"] else 0.0

        # Feature 7: failed_auth_velocity
        data[i, 7] = np.clip(rng.exponential(0.02), 0, 1)

        # Feature 8: api_entropy_1h — moderate
        data[i, 8] = np.clip(rng.normal(0.35, 0.12), 0, 1)

        # Feature 9: privilege_escalation_score
        data[i, 9] = np.clip(rng.exponential(0.01), 0, 1)

    return data


def _generate_credential_stuffing(rng: np.random.Generator) -> np.ndarray:
    """Credential Stuffing (300 events).

    Characteristics:
    - Multiple distinct source IPs → high geo risk (rotating IPs from many countries)
    - Mostly ConsoleLogin failures → high failed_auth_velocity
    - Unusual geolocation → high geo risk
    - Off-hours (night) → hour near 0 or 23
    - API risk moderate (ConsoleLogin ~ 0.5)
    """
    n = NUM_CREDENTIAL_STUFFING
    data = np.zeros((n, 10), dtype=np.float64)

    for i in range(n):
        # Nighttime / off-hours
        hour = rng.choice([rng.normal(2, 1.5), rng.normal(22, 1.5)])
        data[i, 0] = np.clip(hour, 0, 23) / 23.0

        # Any day, slightly more weekend
        dow = rng.integers(0, 7) if rng.random() < 0.6 else rng.integers(5, 7)
        data[i, 1] = dow / 6.0
        data[i, 2] = 1.0 if dow >= 5 else 0.0

        # High geo risk — rotated IPs from risky countries
        data[i, 3] = np.clip(rng.normal(0.7, 0.15), 0.3, 1.0)

        # ConsoleLogin risk ~ 0.5
        data[i, 4] = np.clip(rng.normal(0.5, 0.1), 0.2, 0.8)

        # Not root typically
        data[i, 5] = 1.0 if rng.random() < 0.05 else 0.0

        # Often cross-region
        data[i, 6] = 1.0 if rng.random() < 0.85 else 0.0

        # Very high failed auth velocity
        data[i, 7] = np.clip(rng.normal(0.85, 0.1), 0.5, 1.0)

        # Moderate-low entropy (same API repeated)
        data[i, 8] = np.clip(rng.normal(0.15, 0.08), 0, 0.5)

        # No privilege escalation
        data[i, 9] = np.clip(rng.exponential(0.02), 0, 0.15)

    return data


def _generate_privilege_escalation(rng: np.random.Generator) -> np.ndarray:
    """Privilege Escalation (350 events).

    Characteristics:
    - IAM sequences: ListUsers → CreateAccessKey → AttachUserPolicy → AssumeRole
    - High API risk (IAM mutations ~0.8-1.0)
    - High privilege_escalation_score
    - Moderate-high API entropy (diverse IAM calls in short time)
    - Off-hours for some, normal hours for insider threat variant
    """
    n = NUM_PRIVILEGE_ESCALATION
    data = np.zeros((n, 10), dtype=np.float64)

    for i in range(n):
        # ~60% off-hours, ~40% insider during work hours
        if rng.random() < 0.6:
            hour = rng.choice([rng.normal(3, 2), rng.normal(23, 1)])
        else:
            hour = rng.normal(12, 3)
        data[i, 0] = np.clip(hour, 0, 23) / 23.0

        dow = rng.integers(0, 7)
        data[i, 1] = dow / 6.0
        data[i, 2] = 1.0 if dow >= 5 else 0.0

        # Moderate geo risk (some come from known locations)
        data[i, 3] = np.clip(rng.normal(0.35, 0.2), 0, 1.0)

        # High API risk (IAM mutations)
        data[i, 4] = np.clip(rng.normal(0.85, 0.1), 0.5, 1.0)

        # Sometimes root
        data[i, 5] = 1.0 if rng.random() < 0.15 else 0.0

        # Cross-region moderate
        data[i, 6] = 1.0 if rng.random() < 0.4 else 0.0

        # Some failed attempts in the sequence
        data[i, 7] = np.clip(rng.normal(0.3, 0.15), 0, 0.8)

        # High entropy (many different IAM APIs in short window)
        data[i, 8] = np.clip(rng.normal(0.75, 0.1), 0.4, 1.0)

        # Very high privilege escalation score
        data[i, 9] = np.clip(rng.normal(0.85, 0.1), 0.5, 1.0)

    return data


def _generate_data_exfiltration(rng: np.random.Generator) -> np.ndarray:
    """Data Exfiltration (350 events).

    Characteristics:
    - ListBuckets → GetObject masivo (>100 files/min)
    - Off-hours predominantly
    - New/unusual IP → higher geo risk
    - High API risk for S3 bulk reads
    - Moderate API entropy (S3 read + list patterns)
    """
    n = NUM_DATA_EXFILTRATION
    data = np.zeros((n, 10), dtype=np.float64)

    for i in range(n):
        # Mostly off-hours
        if rng.random() < 0.75:
            hour = rng.choice([rng.normal(1, 1.5), rng.normal(23, 1)])
        else:
            hour = rng.normal(11, 2)
        data[i, 0] = np.clip(hour, 0, 23) / 23.0

        dow = rng.integers(0, 7)
        data[i, 1] = dow / 6.0
        data[i, 2] = 1.0 if dow >= 5 else 0.0

        # Elevated geo risk — new IP
        data[i, 3] = np.clip(rng.normal(0.5, 0.2), 0.1, 1.0)

        # Moderate-high API risk (S3 read/list actions)
        data[i, 4] = np.clip(rng.normal(0.6, 0.15), 0.3, 0.9)

        # Rarely root
        data[i, 5] = 1.0 if rng.random() < 0.03 else 0.0

        # Sometimes cross-region
        data[i, 6] = 1.0 if rng.random() < 0.5 else 0.0

        # Low failed auth (successful access)
        data[i, 7] = np.clip(rng.exponential(0.05), 0, 0.3)

        # Moderate entropy (ListBuckets + GetObject in volume)
        data[i, 8] = np.clip(rng.normal(0.55, 0.15), 0.2, 0.9)

        # Low privilege escalation (already has access)
        data[i, 9] = np.clip(rng.exponential(0.05), 0, 0.3)

    return data


def generate_dataset(output_dir: str | Path = "data") -> dict[str, Path]:
    """Generate all datasets and write to Parquet files.

    Returns:
        dict mapping dataset name to output file Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # ── Generate feature matrices ──
    baseline = _generate_baseline(rng)
    cred_stuff = _generate_credential_stuffing(rng)
    priv_esc = _generate_privilege_escalation(rng)
    data_exfil = _generate_data_exfiltration(rng)

    # ── Build DataFrames ──
    df_baseline = pd.DataFrame(baseline, columns=FEATURE_NAMES)
    df_baseline["is_anomaly"] = False
    df_baseline["attack_type"] = "normal"
    df_baseline["user"] = [
        USERS[rng.integers(0, len(USERS))]["name"] for _ in range(NUM_BASELINE)
    ]

    df_cred = pd.DataFrame(cred_stuff, columns=FEATURE_NAMES)
    df_cred["is_anomaly"] = True
    df_cred["attack_type"] = "credential_stuffing"
    df_cred["user"] = [
        USERS[rng.integers(0, len(USERS))]["name"] for _ in range(NUM_CREDENTIAL_STUFFING)
    ]

    df_priv = pd.DataFrame(priv_esc, columns=FEATURE_NAMES)
    df_priv["is_anomaly"] = True
    df_priv["attack_type"] = "privilege_escalation"
    df_priv["user"] = [
        USERS[rng.integers(0, len(USERS))]["name"] for _ in range(NUM_PRIVILEGE_ESCALATION)
    ]

    df_exfil = pd.DataFrame(data_exfil, columns=FEATURE_NAMES)
    df_exfil["is_anomaly"] = True
    df_exfil["attack_type"] = "data_exfiltration"
    df_exfil["user"] = [
        USERS[rng.integers(0, len(USERS))]["name"] for _ in range(NUM_DATA_EXFILTRATION)
    ]

    df_anomalous = pd.concat([df_cred, df_priv, df_exfil], ignore_index=True)
    df_combined = pd.concat([df_baseline, df_anomalous], ignore_index=True)

    # Shuffle combined set
    df_combined = df_combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ── Write Parquet ──
    paths: dict[str, Path] = {}

    path_baseline = output_dir / "baseline_events.parquet"
    df_baseline.to_parquet(path_baseline, index=False, engine="fastparquet")
    paths["baseline"] = path_baseline

    path_anomalous = output_dir / "anomalous_events.parquet"
    df_anomalous.to_parquet(path_anomalous, index=False, engine="fastparquet")
    paths["anomalous"] = path_anomalous

    path_combined = output_dir / "combined_labeled.parquet"
    df_combined.to_parquet(path_combined, index=False, engine="fastparquet")
    paths["combined"] = path_combined

    # ── Summary ──
    print("=" * 70)
    print("CloudSentinel — Synthetic Data Generation Complete")
    print("=" * 70)
    print(f"  Baseline events:         {len(df_baseline):>7,}")
    print(f"  Anomalous events:        {len(df_anomalous):>7,}")
    print(f"    - Credential Stuffing: {NUM_CREDENTIAL_STUFFING:>7,}")
    print(f"    - Privilege Escalation:{NUM_PRIVILEGE_ESCALATION:>7,}")
    print(f"    - Data Exfiltration:   {NUM_DATA_EXFILTRATION:>7,}")
    print(f"  Combined total:          {len(df_combined):>7,}")
    print("-" * 70)
    print(f"  Feature dimensions:      {len(FEATURE_NAMES)}")
    print(f"  Random seed:             {SEED}")
    print(f"  Output directory:        {output_dir.resolve()}")
    print("-" * 70)

    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name:20s}  →  {path}  ({size_kb:.1f} KB)")

    print("=" * 70)

    # ── Feature distribution summary ──
    print("\nFeature Distribution Summary (baseline):")
    print(df_baseline[FEATURE_NAMES].describe().round(4).to_string())
    print("\nFeature Distribution Summary (anomalous):")
    print(df_anomalous[FEATURE_NAMES].describe().round(4).to_string())

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CloudSentinel training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for Parquet files (default: data/)",
    )
    args = parser.parse_args()
    generate_dataset(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
