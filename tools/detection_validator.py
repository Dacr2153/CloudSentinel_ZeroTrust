#!/usr/bin/env python3
# FILE: cloudsentinel-zero-trust/tools/detection_validator.py
"""Post-attack detection validator for CloudSentinel.

Validates that the CloudSentinel pipeline detected attack scenarios
within the MTTD target and with acceptable FPR.

Workflow:
  1. Load attack_timeline.json from the attack simulator
  2. Wait for CloudTrail delivery + pipeline processing
  3. Query OpenSearch for alerts generated after the attack
  4. Calculate MTTD per scenario (alert_timestamp - attack_timestamp)
  5. Calculate FPR from a clean baseline window
  6. Generate validation_report.json
  7. Exit code 0 if all gates pass, 1 if any fail

Usage:
  python detection_validator.py \
    --attack-timeline tools/attack_timeline.json \
    --opensearch-endpoint http://10.0.1.x:9200 \
    --wait-seconds 180
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error
import urllib.parse


MTTD_TARGET_SECONDS = 120
FPR_TARGET = 0.05
ALERTS_INDEX = "cloudsentinel-alerts-*"
EVENTS_INDEX = "cloudsentinel-events-*"
DEFAULT_WAIT = 180  # seconds


class OpenSearchQuery:
    """Minimal OpenSearch HTTP client for validation queries."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint.rstrip("/")

    def search(self, index: str, body: dict) -> dict:
        """Execute a search query and return the response."""
        url = f"{self._endpoint}/{index}/_search"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(
                f"OpenSearch query failed ({e.code}): {body_text}"
            )
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot connect to OpenSearch: {e}")

    def count(self, index: str, body: dict) -> int:
        """Execute a count query and return the total."""
        url = f"{self._endpoint}/{index}/_count"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("count", 0)
        except (urllib.error.HTTPError, urllib.error.URLError):
            return 0

    def health(self) -> str:
        """Check cluster health status."""
        url = f"{self._endpoint}/_cluster/health"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("status", "unknown")
        except Exception:
            return "unreachable"


def load_timeline(path: str | Path) -> list[dict[str, Any]]:
    """Load attack_timeline.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Attack timeline not found: {p}")
    return json.loads(p.read_text())


def get_scenario_start_times(
    timeline: list[dict],
) -> dict[int, str]:
    """Get the earliest timestamp per scenario."""
    starts: dict[int, str] = {}
    for event in timeline:
        scenario = event["scenario"]
        ts = event["timestamp"]
        if scenario not in starts or ts < starts[scenario]:
            starts[scenario] = ts
    return starts


def get_expected_rules(timeline: list[dict]) -> dict[int, set[str]]:
    """Get expected rules per scenario."""
    rules: dict[int, set[str]] = {}
    for event in timeline:
        scenario = event["scenario"]
        rule = event.get("expected_rule", "")
        if rule:
            rules.setdefault(scenario, set()).add(rule)
    return rules


def calculate_mttd(
    os_client: OpenSearchQuery,
    scenario: int,
    attack_start: str,
    expected_rules: set[str],
) -> dict[str, Any]:
    """Calculate MTTD for a single scenario.

    Queries OpenSearch for alerts with matching rules that were generated
    after the attack start time.

    Returns:
        dict with mttd_seconds, target, pass, detected_rules, alert_count.
    """
    # Build query: alerts after attack start with expected rules
    should_clauses = [
        {"term": {"cloudsentinel.rules_matched": rule}}
        for rule in expected_rules
    ]

    query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": attack_start}}},
                ],
                "should": should_clauses,
                "minimum_should_match": 1 if should_clauses else 0,
            }
        },
        "sort": [{"@timestamp": "asc"}],
        "size": 100,
    }

    # Also search for ML anomaly alerts
    ml_query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": attack_start}}},
                    {"term": {"cloudsentinel.is_anomaly": True}},
                    {"range": {"cloudsentinel.anomaly_score": {"gte": 65}}},
                ]
            }
        },
        "sort": [{"@timestamp": "asc"}],
        "size": 10,
    }

    # Execute queries
    rule_results = os_client.search(ALERTS_INDEX, query)
    ml_results = os_client.search(EVENTS_INDEX, ml_query)

    # Find earliest detection
    earliest_detection: str | None = None
    detected_rules: set[str] = set()
    alert_count = 0

    rule_hits = rule_results.get("hits", {}).get("hits", [])
    alert_count += len(rule_hits)
    for hit in rule_hits:
        src = hit.get("_source", {})
        ts = src.get("@timestamp", "")
        rules = src.get("cloudsentinel", {}).get("rules_matched", [])
        if isinstance(rules, str):
            rules = [rules]
        detected_rules.update(rules)
        if ts and (earliest_detection is None or ts < earliest_detection):
            earliest_detection = ts

    ml_hits = ml_results.get("hits", {}).get("hits", [])
    for hit in ml_hits:
        src = hit.get("_source", {})
        ts = src.get("@timestamp", "")
        if ts and (earliest_detection is None or ts < earliest_detection):
            earliest_detection = ts
        alert_count += 1

    # Calculate MTTD
    if earliest_detection:
        try:
            attack_dt = datetime.fromisoformat(attack_start.replace("Z", "+00:00"))
            detect_dt = datetime.fromisoformat(earliest_detection.replace("Z", "+00:00"))
            mttd = (detect_dt - attack_dt).total_seconds()
        except (ValueError, TypeError):
            mttd = -1
    else:
        mttd = -1  # Not detected

    is_pass = 0 < mttd <= MTTD_TARGET_SECONDS

    return {
        "scenario": scenario,
        "attack_start": attack_start,
        "first_detection": earliest_detection,
        "mttd_seconds": round(mttd, 1) if mttd > 0 else None,
        "target_seconds": MTTD_TARGET_SECONDS,
        "pass": is_pass,
        "detected_rules": sorted(detected_rules),
        "expected_rules": sorted(expected_rules),
        "alert_count": alert_count,
    }


def calculate_fpr(
    os_client: OpenSearchQuery,
    baseline_start: str,
    baseline_end: str,
) -> dict[str, Any]:
    """Calculate False Positive Rate during a clean baseline window.

    Counts alerts and total events in the given time window.
    FPR = alerts_in_baseline / total_events_in_baseline

    Returns:
        dict with fpr, total_events, false_alerts, target, pass.
    """
    # Count total events in baseline window
    events_query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": baseline_start,
                    "lte": baseline_end,
                }
            }
        }
    }
    total_events = os_client.count(EVENTS_INDEX, events_query)

    # Count alerts (false positives) in baseline window
    alerts_query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": baseline_start, "lte": baseline_end}}},
                ]
            }
        }
    }
    false_alerts = os_client.count(ALERTS_INDEX, alerts_query)

    fpr = false_alerts / total_events if total_events > 0 else 0.0
    is_pass = fpr < FPR_TARGET

    return {
        "fpr": round(fpr, 4),
        "total_events": total_events,
        "false_alerts": false_alerts,
        "target": FPR_TARGET,
        "pass": is_pass,
    }


def validate(
    timeline_path: str,
    opensearch_endpoint: str,
    wait_seconds: int = DEFAULT_WAIT,
    baseline_hours: int = 1,
    output_dir: str = "tools",
) -> dict[str, Any]:
    """Execute full post-attack validation.

    Args:
        timeline_path: Path to attack_timeline.json.
        opensearch_endpoint: OpenSearch HTTP endpoint.
        wait_seconds: Seconds to wait for CloudTrail delivery + processing.
        baseline_hours: Hours of clean baseline for FPR calculation.
        output_dir: Directory for validation_report.json.

    Returns:
        dict with mttd_results, fpr, overall_pass.
    """
    print("=" * 60)
    print("  CloudSentinel — Detection Validator")
    print("=" * 60)

    # ── Load timeline ──
    print("\n[1/5] Loading attack timeline...")
    timeline = load_timeline(timeline_path)
    scenario_starts = get_scenario_start_times(timeline)
    expected_rules = get_expected_rules(timeline)

    print(f"  Scenarios:      {sorted(scenario_starts.keys())}")
    print(f"  Total events:   {len(timeline)}")
    for s, ts in sorted(scenario_starts.items()):
        rules = expected_rules.get(s, set())
        print(f"    Scenario {s}: started={ts}, expected_rules={sorted(rules)}")

    # ── Check OpenSearch health ──
    print("\n[2/5] Checking OpenSearch connectivity...")
    os_client = OpenSearchQuery(opensearch_endpoint)
    health = os_client.health()
    print(f"  Cluster health: {health}")
    if health == "unreachable":
        print("  ❌ Cannot reach OpenSearch. Aborting.")
        return {"error": "OpenSearch unreachable", "overall_pass": False}

    # ── Wait for pipeline processing ──
    print(f"\n[3/5] Waiting {wait_seconds}s for CloudTrail delivery + pipeline...")
    for remaining in range(wait_seconds, 0, -30):
        print(f"  ⏳ {remaining}s remaining...")
        time.sleep(min(30, remaining))

    # ── Calculate MTTD per scenario ──
    print("\n[4/5] Calculating MTTD per scenario...")
    mttd_results: dict[str, dict] = {}

    for scenario_num in sorted(scenario_starts.keys()):
        attack_start = scenario_starts[scenario_num]
        rules = expected_rules.get(scenario_num, set())
        result = calculate_mttd(os_client, scenario_num, attack_start, rules)
        mttd_results[f"scenario_{scenario_num}"] = result

        mttd_str = f"{result['mttd_seconds']}s" if result["mttd_seconds"] else "NOT DETECTED"
        status = "✅ PASS" if result["pass"] else "❌ FAIL"
        print(f"  Scenario {scenario_num}: MTTD = {mttd_str} "
              f"(target: <{MTTD_TARGET_SECONDS}s) — {status}")
        print(f"    Detected rules: {result['detected_rules']}")
        print(f"    Alert count: {result['alert_count']}")

    # ── Calculate FPR ──
    print("\n[5/5] Calculating False Positive Rate...")
    # Baseline window: 1 hour before the earliest attack
    earliest_attack = min(scenario_starts.values())
    baseline_end_dt = datetime.fromisoformat(earliest_attack.replace("Z", "+00:00"))
    baseline_start_dt = baseline_end_dt - timedelta(hours=baseline_hours)

    fpr_result = calculate_fpr(
        os_client,
        baseline_start_dt.isoformat(),
        baseline_end_dt.isoformat(),
    )

    fpr_status = "✅ PASS" if fpr_result["pass"] else "❌ FAIL"
    print(f"  FPR: {fpr_result['fpr']:.2%} (target: <{FPR_TARGET:.0%}) — {fpr_status}")
    print(f"  Baseline events: {fpr_result['total_events']}, "
          f"False alerts: {fpr_result['false_alerts']}")

    # ── Determine overall pass ──
    all_mttd_pass = all(
        r["pass"] for r in mttd_results.values()
    )
    overall_pass = all_mttd_pass and fpr_result["pass"]

    # ── Generate report ──
    report = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "opensearch_endpoint": opensearch_endpoint,
        "mttd_results": mttd_results,
        "fpr": fpr_result["fpr"],
        "fpr_target": FPR_TARGET,
        "fpr_pass": fpr_result["pass"],
        "fpr_details": fpr_result,
        "all_mttd_pass": all_mttd_pass,
        "overall_pass": overall_pass,
    }

    output_path = Path(output_dir) / "validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  📁 Report saved to {output_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Validation Summary")
    print("=" * 60)
    for key, result in mttd_results.items():
        mttd_str = f"{result['mttd_seconds']}s" if result["mttd_seconds"] else "N/A"
        status = "✅" if result["pass"] else "❌"
        print(f"  {status} {key}: MTTD = {mttd_str} (target: <{MTTD_TARGET_SECONDS}s)")
    print(f"  {'✅' if fpr_result['pass'] else '❌'} FPR: {fpr_result['fpr']:.1%} "
          f"(target: <{FPR_TARGET:.0%})")
    overall_icon = "✅" if overall_pass else "❌"
    print(f"  {overall_icon} OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CloudSentinel Post-Attack Detection Validator"
    )
    parser.add_argument(
        "--attack-timeline",
        type=str,
        default="tools/attack_timeline.json",
        help="Path to attack_timeline.json from attack_simulator",
    )
    parser.add_argument(
        "--opensearch-endpoint",
        type=str,
        required=True,
        help="OpenSearch HTTP endpoint (e.g. http://10.0.1.x:9200)",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=DEFAULT_WAIT,
        help=f"Seconds to wait for CloudTrail + pipeline (default: {DEFAULT_WAIT})",
    )
    parser.add_argument(
        "--baseline-hours",
        type=int,
        default=1,
        help="Hours of clean baseline for FPR calculation (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tools",
        help="Directory for validation_report.json (default: tools/)",
    )

    args = parser.parse_args()

    report = validate(
        timeline_path=args.attack_timeline,
        opensearch_endpoint=args.opensearch_endpoint,
        wait_seconds=args.wait_seconds,
        baseline_hours=args.baseline_hours,
        output_dir=args.output_dir,
    )

    if not report.get("overall_pass", False):
        print("\n❌ VALIDATION GATE FAILED — detection targets not met")
        sys.exit(1)
    else:
        print("\n✅ All detection gates passed")


if __name__ == "__main__":
    main()
