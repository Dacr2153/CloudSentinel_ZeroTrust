#!/usr/bin/env python3
# FILE: cloudsentinel-zero-trust/tools/attack_simulator.py
"""CloudSentinel Autonomous Attack Simulator.

Executes 3 controlled attack scenarios using boto3 to generate real
CloudTrail events for detection testing. No Pacu dependency required.

Scenarios:
  1. S3 Bucket Enumeration + Data Exfiltration
  2. IAM Privilege Escalation
  3. Credential Stuffing (failed auth flood)

Features:
  - Generates attack_timeline.json for post-attack validation
  - Respects AWS rate limits with configurable delays
  - dry_run mode: validates permissions without executing actions
  - Cleanup: reverts all changes (least damage principle)

Usage:
  python attack_simulator.py --scenario all
  python attack_simulator.py --scenario 1 --dry-run
  python attack_simulator.py --cleanup
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


SIMULATOR_PREFIX = "cloudsentinel-sim"
ATTACK_USER = f"{SIMULATOR_PREFIX}-attacker"
ATTACK_ROLE = f"{SIMULATOR_PREFIX}-escalation-role"
ATTACK_POLICY_NAME = f"{SIMULATOR_PREFIX}-test-policy"
ATTACK_BUCKET_PREFIX = f"{SIMULATOR_PREFIX}-test"
MTTD_TARGET_SECONDS = 120
DEFAULT_DELAY = 0.5  # seconds between API calls


class AttackTimeline:
    """Records attack actions with timestamps for MTTD validation."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def record(
        self,
        scenario: int,
        action: str,
        details: str = "",
        expected_rule: str = "",
    ) -> None:
        self._events.append({
            "scenario": scenario,
            "action": action,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expected_detection_by": (
                datetime.now(timezone.utc).timestamp() + MTTD_TARGET_SECONDS
            ),
            "expected_rule": expected_rule,
        })

    def save(self, path: str | Path = "attack_timeline.json") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self._events, indent=2))
        print(f"\n  📁 Timeline saved to {output}")
        return output

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events.copy()


def scenario_1_s3_exfiltration(
    timeline: AttackTimeline,
    delay: float = DEFAULT_DELAY,
    dry_run: bool = False,
) -> list[str]:
    """S3 Bucket Enumeration + Data Exfiltration.

    Steps:
      1. ListBuckets — enumerate all buckets
      2. GetBucketAcl — check for misconfigurations
      3. ListObjectsV2 — enumerate contents
      4. GetObject — bulk download (simulated)

    Returns:
        List of resources created (for cleanup).
    """
    print("\n" + "=" * 60)
    print("  SCENARIO 1: S3 Bucket Enumeration + Data Exfiltration")
    print("=" * 60)

    s3 = boto3.client("s3")
    created_resources: list[str] = []

    # ── Create test bucket with sample objects ──
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    test_bucket = f"{ATTACK_BUCKET_PREFIX}-{account_id[:8]}"

    if not dry_run:
        try:
            s3.create_bucket(Bucket=test_bucket)
            created_resources.append(f"s3://{test_bucket}")
            print(f"  [SETUP] Created test bucket: {test_bucket}")

            # Upload dummy sensitive files
            for i in range(20):
                s3.put_object(
                    Bucket=test_bucket,
                    Key=f"sensitive-data/file-{i:04d}.csv",
                    Body=f"dummy-data-row-{i}",
                )
            print("  [SETUP] Uploaded 20 test objects")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "BucketAlreadyOwnedByYou":
                print(f"  ⚠️  Bucket creation failed: {e}")
    else:
        print("  [DRY RUN] Would create test bucket and objects")

    time.sleep(delay)

    # ── Step 1: ListBuckets ──
    print("\n  [ATTACK] Step 1: Enumerating all S3 buckets...")
    timeline.record(1, "ListBuckets", "Enumerate all buckets", "RULE-006")
    if not dry_run:
        buckets = s3.list_buckets().get("Buckets", [])
        print(f"  → Found {len(buckets)} buckets")
    else:
        print("  [DRY RUN] Would call ListBuckets")
    time.sleep(delay)

    # ── Step 2: GetBucketAcl ──
    print("  [ATTACK] Step 2: Checking bucket ACLs...")
    if not dry_run:
        for bucket in buckets[:5]:  # Limit to 5 to avoid rate limits
            try:
                s3.get_bucket_acl(Bucket=bucket.get("Name", ""))
                timeline.record(1, "GetBucketAcl", f"bucket={bucket.get('Name', '')}")
            except ClientError:
                pass
            time.sleep(delay)
    else:
        print("  [DRY RUN] Would call GetBucketAcl on up to 5 buckets")

    # ── Step 3: ListObjectsV2 ──
    print("  [ATTACK] Step 3: Listing objects in target bucket...")
    timeline.record(1, "ListObjectsV2", f"bucket={test_bucket}")
    if not dry_run:
        try:
            objects = s3.list_objects_v2(Bucket=test_bucket).get("Contents", [])
            print(f"  → Found {len(objects)} objects in {test_bucket}")
        except ClientError as e:
            print(f"  ⚠️  ListObjects failed: {e}")
            objects = []
    else:
        print("  [DRY RUN] Would call ListObjectsV2")
        objects = []
    time.sleep(delay)

    # ── Step 4: Mass GetObject (trigger RULE-006) ──
    print("  [ATTACK] Step 4: Mass downloading objects (>50/min trigger)...")
    if not dry_run:
        for i, obj in enumerate(objects):
            try:
                s3.get_object(Bucket=test_bucket, Key=obj.get("Key", ""))
                timeline.record(1, "GetObject", f"key={obj.get('Key', '')}", "RULE-006")
            except ClientError:
                pass
            # Minimal delay to hit >50/min rate
            time.sleep(0.1)
            if (i + 1) % 10 == 0:
                print(f"    → Downloaded {i + 1}/{len(objects)} objects")
    else:
        print("  [DRY RUN] Would call GetObject on all objects rapidly")

    print("\n  ✅ Scenario 1 complete")
    return created_resources


def scenario_2_iam_privesc(
    timeline: AttackTimeline,
    delay: float = DEFAULT_DELAY,
    dry_run: bool = False,
) -> list[str]:
    """IAM Privilege Escalation.

    Steps:
      1. ListUsers — enumerate IAM users
      2. ListUserPolicies / ListAttachedUserPolicies — find targets
      3. CreateAccessKey — backdoor a user
      4. AttachUserPolicy — escalate privileges
      5. AssumeRole — attempt role escalation

    Returns:
        List of resources created (for cleanup).
    """
    print("\n" + "=" * 60)
    print("  SCENARIO 2: IAM Privilege Escalation")
    print("=" * 60)

    iam = boto3.client("iam")
    sts = boto3.client("sts")
    created_resources: list[str] = []

    # ── Setup: Create simulation user ──
    if not dry_run:
        try:
            iam.create_user(UserName=ATTACK_USER)
            created_resources.append(f"iam:user:{ATTACK_USER}")
            print(f"  [SETUP] Created simulation user: {ATTACK_USER}")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "EntityAlreadyExists":
                print(f"  ⚠️  User creation failed: {e}")
    else:
        print(f"  [DRY RUN] Would create user {ATTACK_USER}")

    time.sleep(delay)

    # ── Step 1: ListUsers ──
    print("\n  [ATTACK] Step 1: Enumerating IAM users...")
    timeline.record(2, "ListUsers", "IAM user enumeration", "RULE-002")
    if not dry_run:
        users = iam.list_users().get("Users", [])
        print(f"  → Found {len(users)} IAM users")
    else:
        print("  [DRY RUN] Would call ListUsers")
    time.sleep(delay)

    # ── Step 2: Enumerate policies ──
    print("  [ATTACK] Step 2: Enumerating user policies...")
    if not dry_run:
        for user in users[:3]:
            try:
                iam.list_user_policies(UserName=user["UserName"])
                iam.list_attached_user_policies(UserName=user["UserName"])
                timeline.record(2, "ListUserPolicies", f"user={user['UserName']}")
            except ClientError:
                pass
            time.sleep(delay)
    else:
        print("  [DRY RUN] Would enumerate policies for up to 3 users")

    # ── Step 3: CreateAccessKey (RULE-004 trigger) ──
    print("  [ATTACK] Step 3: Creating backdoor access key...")
    timeline.record(2, "CreateAccessKey", f"user={ATTACK_USER}", "RULE-004")
    key_id = None
    if not dry_run:
        try:
            resp = iam.create_access_key(UserName=ATTACK_USER)
            key_id = resp["AccessKey"]["AccessKeyId"]
            created_resources.append(f"iam:access-key:{ATTACK_USER}:{key_id}")
            print(f"  → Created access key: {key_id[:8]}...")
        except ClientError as e:
            print(f"  ⚠️  CreateAccessKey failed: {e}")
    else:
        print("  [DRY RUN] Would create access key")
    time.sleep(delay)

    # ── Step 4: AttachUserPolicy (RULE-004 trigger within 60s of Step 3) ──
    print("  [ATTACK] Step 4: Attaching admin policy (privilege escalation)...")
    timeline.record(
        2, "AttachUserPolicy",
        f"user={ATTACK_USER}, policy=ReadOnlyAccess",
        "RULE-004",
    )
    if not dry_run:
        try:
            iam.attach_user_policy(
                UserName=ATTACK_USER,
                PolicyArn="arn:aws:iam::aws:policy/ReadOnlyAccess",
            )
            created_resources.append(
                f"iam:attached-policy:{ATTACK_USER}:arn:aws:iam::aws:policy/ReadOnlyAccess"
            )
            print("  → Attached ReadOnlyAccess policy")
        except ClientError as e:
            print(f"  ⚠️  AttachUserPolicy failed: {e}")
    else:
        print("  [DRY RUN] Would attach policy")
    time.sleep(delay)

    # ── Step 5: AssumeRole rapid succession (RULE-003 trigger) ──
    print("  [ATTACK] Step 5: Rapid AssumeRole attempts (>10 in 5 min)...")
    account_id = sts.get_caller_identity()["Account"]
    for i in range(12):
        role_name = f"{ATTACK_ROLE}-{i}"
        timeline.record(2, "AssumeRole", f"role={role_name}", "RULE-003")
        if not dry_run:
            try:
                sts.assume_role(
                    RoleArn=f"arn:aws:iam::{account_id}:role/{role_name}",
                    RoleSessionName=f"sim-session-{i}",
                )
            except ClientError:
                pass  # Expected: role doesn't exist
        time.sleep(0.2)  # Rapid succession
        if (i + 1) % 4 == 0:
            print(f"    → AssumeRole attempt {i + 1}/12")

    print("\n  ✅ Scenario 2 complete")
    return created_resources


def scenario_3_credential_stuffing(
    timeline: AttackTimeline,
    delay: float = DEFAULT_DELAY,
    dry_run: bool = False,
    count: int = 50,
) -> list[str]:
    """Credential Stuffing + CloudTrail Tampering.

    Steps:
      1. Flood of failed AssumeRole calls (simulates brute-force)
      2. StopLogging attempt (defense evasion)

    Returns:
        List of resources created (for cleanup).
    """
    print("\n" + "=" * 60)
    print("  SCENARIO 3: Credential Stuffing + Console Takeover")
    print("=" * 60)

    sts = boto3.client("sts")
    cloudtrail = boto3.client("cloudtrail")
    created_resources: list[str] = []

    target_users = [
        "admin", "root", "deploy", "ci-cd", "terraform",
        "ops-user", "dev-admin", "backup-user", "audit-user", "lambda-exec",
    ]

    # ── Step 1: Credential stuffing flood ──
    print(f"\n  [ATTACK] Step 1: Generating {count} failed auth events...")
    account_id = sts.get_caller_identity()["Account"]

    for i in range(count):
        user = target_users[i % len(target_users)]
        timeline.record(
            3,
            "AssumeRole (DENIED)",
            f"target_user={user}, attempt={i+1}",
            "RULE-001",
        )

        if not dry_run:
            try:
                sts.assume_role(
                    RoleArn=f"arn:aws:iam::{account_id}:role/nonexistent-{user}",
                    RoleSessionName=f"brute-force-{i}",
                    DurationSeconds=900,
                )
            except ClientError:
                pass  # Expected: access denied
        time.sleep(delay * 0.4)  # Fast to simulate brute force

        if (i + 1) % 10 == 0:
            print(f"    → Generated {i + 1}/{count} failed auth events")

    # ── Step 2: CloudTrail tampering attempt (RULE-007) ──
    print("\n  [ATTACK] Step 2: Attempting CloudTrail tampering...")
    timeline.record(
        3, "StopLogging", "Attempt to disable CloudTrail", "RULE-007"
    )

    if not dry_run:
        # List trails to find our trail
        trails = cloudtrail.describe_trails().get("trailList", [])
        sim_trail = None
        for trail in trails:
            if "cloudsentinel" in trail.get("Name", "").lower():
                sim_trail = trail["Name"]
                break

        if sim_trail:
            # NOTE: We attempt StopLogging but expect it to be denied by IAM policy
            # The attempt itself generates the CloudTrail event we want to detect
            try:
                cloudtrail.stop_logging(Name=sim_trail)
                # If it succeeds, immediately re-enable!
                cloudtrail.start_logging(Name=sim_trail)
                created_resources.append(f"cloudtrail:stopped:{sim_trail}")
                print(f"  ⚠️  StopLogging succeeded (re-enabled immediately): {sim_trail}")
            except ClientError as e:
                print(f"  → StopLogging denied (expected): {e.response.get('Error', {}).get('Code', 'Unknown')}")
        else:
            print("  → No CloudSentinel trail found (testing with generic call)")
            try:
                cloudtrail.stop_logging(Name="nonexistent-trail")
            except ClientError:
                pass
    else:
        print("  [DRY RUN] Would attempt StopLogging on CloudSentinel trail")

    print("\n  ✅ Scenario 3 complete")
    return created_resources


def cleanup(resources: list[str] | None = None) -> None:
    """Revert all changes made during attack simulation.

    If no resources list is provided, attempts generic cleanup of
    known simulator resources.
    """
    print("\n" + "=" * 60)
    print("  CLEANUP: Reverting attack simulation changes")
    print("=" * 60)

    iam = boto3.client("iam")
    s3 = boto3.client("s3")
    cloudtrail = boto3.client("cloudtrail")

    # ── Clean IAM resources ──
    print("\n  [IAM] Cleaning up simulation user...")
    try:
        # Delete access keys
        keys = iam.list_access_keys(UserName=ATTACK_USER).get(
            "AccessKeyMetadata", []
        )
        for key in keys:
            iam.delete_access_key(
                UserName=ATTACK_USER, AccessKeyId=key["AccessKeyId"]
            )
            print(f"    Deleted access key: {key['AccessKeyId'][:8]}...")

        # Detach policies
        policies = iam.list_attached_user_policies(UserName=ATTACK_USER).get(
            "AttachedPolicies", []
        )
        for pol in policies:
            iam.detach_user_policy(
                UserName=ATTACK_USER, PolicyArn=pol["PolicyArn"]
            )
            print(f"    Detached policy: {pol['PolicyName']}")

        # Delete inline policies
        inline = iam.list_user_policies(UserName=ATTACK_USER).get(
            "PolicyNames", []
        )
        for pname in inline:
            iam.delete_user_policy(UserName=ATTACK_USER, PolicyName=pname)

        # Delete user
        iam.delete_user(UserName=ATTACK_USER)
        print(f"    Deleted user: {ATTACK_USER}")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "NoSuchEntity":
            print(f"    ⚠️  IAM cleanup error: {e}")

    # ── Clean S3 test bucket ──
    print("\n  [S3] Cleaning up test bucket...")
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    test_bucket = f"{ATTACK_BUCKET_PREFIX}-{account_id[:8]}"
    try:
        # Delete all objects
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=test_bucket):
            objects = page.get("Contents", [])
            if objects:
                s3.delete_objects(
                    Bucket=test_bucket,
                    Delete={"Objects": [{"Key": o.get("Key", "")} for o in objects]},
                )
        s3.delete_bucket(Bucket=test_bucket)
        print(f"    Deleted bucket: {test_bucket}")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "NoSuchBucket":
            print(f"    ⚠️  S3 cleanup error: {e}")

    # ── Ensure CloudTrail is running ──
    print("\n  [CloudTrail] Ensuring logging is active...")
    trails = cloudtrail.describe_trails().get("trailList", [])
    for trail in trails:
        if "cloudsentinel" in trail.get("Name", "").lower():
            try:
                status = cloudtrail.get_trail_status(Name=trail["Name"])
                if not status.get("IsLogging", True):
                    cloudtrail.start_logging(Name=trail["Name"])
                    print(f"    Re-enabled logging: {trail['Name']}")
                else:
                    print(f"    Logging active: {trail['Name']}")
            except ClientError:
                pass

    print("\n  ✅ Cleanup complete")


def run_simulation(
    scenarios: list[int] | None = None,
    delay: float = DEFAULT_DELAY,
    dry_run: bool = False,
    output_dir: str = "tools",
    cred_stuff_count: int = 50,
) -> dict[str, Any]:
    """Execute selected attack scenarios.

    Args:
        scenarios: List of scenario numbers to run (default: all [1,2,3]).
        delay: Delay between API calls in seconds.
        dry_run: If True, validate permissions without executing.
        output_dir: Directory for attack_timeline.json.
        cred_stuff_count: Number of failed auth events for scenario 3.

    Returns:
        dict with timeline, resources_created.
    """
    if scenarios is None:
        scenarios = [1, 2, 3]

    timeline = AttackTimeline()
    all_resources: list[str] = []

    print("=" * 60)
    print("  CloudSentinel — Attack Simulator")
    print("=" * 60)
    print(f"  Scenarios:  {scenarios}")
    print(f"  Delay:      {delay}s")
    print(f"  Dry run:    {dry_run}")
    print(f"  Output:     {output_dir}/")

    # Verify AWS credentials
    try:
        identity = boto3.client("sts").get_caller_identity()
        print(f"  Account:    {identity['Account']}")
        print(f"  Caller:     {identity['Arn']}")
    except ClientError as e:
        print(f"\n  ❌ AWS credential error: {e}")
        return {"error": str(e)}

    start_time = datetime.now(timezone.utc)

    if 1 in scenarios:
        resources = scenario_1_s3_exfiltration(timeline, delay, dry_run)
        all_resources.extend(resources)

    if 2 in scenarios:
        resources = scenario_2_iam_privesc(timeline, delay, dry_run)
        all_resources.extend(resources)

    if 3 in scenarios:
        resources = scenario_3_credential_stuffing(
            timeline, delay, dry_run, cred_stuff_count
        )
        all_resources.extend(resources)

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Save timeline
    timeline_path = timeline.save(Path(output_dir) / "attack_timeline.json")

    # Summary
    print("\n" + "=" * 60)
    print("  Simulation Summary")
    print("=" * 60)
    print(f"  Scenarios executed:   {len(scenarios)}")
    print(f"  Total events logged:  {len(timeline.events)}")
    print(f"  Resources created:    {len(all_resources)}")
    print(f"  Elapsed time:         {elapsed:.1f}s")
    print(f"  Timeline:             {timeline_path}")

    if not dry_run and all_resources:
        print("\n  ⚠️  Run with --cleanup to revert all changes")

    return {
        "scenarios": scenarios,
        "timeline_path": str(timeline_path),
        "events_count": len(timeline.events),
        "resources_created": all_resources,
        "elapsed_seconds": round(elapsed, 1),
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CloudSentinel Attack Simulator — Red Team automation"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario to run: 1, 2, 3, or 'all' (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between API calls in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate permissions without executing actions",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Revert all changes from previous simulation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tools",
        help="Directory for output files (default: tools/)",
    )
    parser.add_argument(
        "--cred-stuff-count",
        type=int,
        default=50,
        help="Number of failed auth events for scenario 3 (default: 50)",
    )

    args = parser.parse_args()

    if args.cleanup:
        cleanup()
        return

    if args.scenario == "all":
        scenarios = [1, 2, 3]
    else:
        scenarios = [int(s.strip()) for s in args.scenario.split(",")]

    run_simulation(
        scenarios=scenarios,
        delay=args.delay,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        cred_stuff_count=args.cred_stuff_count,
    )


if __name__ == "__main__":
    main()
