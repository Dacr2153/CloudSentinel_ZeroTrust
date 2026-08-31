# FILE: cloudsentinel-zero-trust/src/integrations/sns_client.py
"""SNS client factory with LOCAL_MODE stdout/file fallback.

In LOCAL_MODE, alert messages are:
  1. Printed to stdout with Rich formatting (visible in demo)
  2. Appended as JSONL to tools/alerts/alerts.jsonl
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

from src.utils.config import get_settings
from src.utils.logger import CloudSentinelLogger

logger = CloudSentinelLogger(service="sns_client")

_client: Any = None


def get_sns_client() -> Any:
    """Return a singleton SNS client (LocalSNSClient in LOCAL_MODE)."""
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    if settings.local_mode:
        _client = LocalSNSClient(alerts_dir=settings.local_alerts_dir)
        logger.info("LOCAL_MODE: SNS client using stdout + %s", settings.local_alerts_dir)
    else:
        _client = boto3.client("sns", region_name=settings.aws_region)
        logger.info("SNS client initialized for region %s", settings.aws_region)
    return _client


class LocalSNSClient:
    """Stdout + JSONL-file SNS mock for LOCAL_MODE.

    publish() prints the alert to terminal and appends to alerts.jsonl.
    Mimics enough of the boto3 SNS API that existing code works unchanged.
    """

    def __init__(self, alerts_dir: str = "tools/alerts") -> None:
        self._alerts_path = Path(alerts_dir)
        self._alerts_path.mkdir(parents=True, exist_ok=True)
        self._alerts_file = self._alerts_path / "alerts.jsonl"

    def publish(
        self,
        TopicArn: str = "",
        Subject: str = "",
        Message: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Write alert to file and print a summary to stdout."""
        ts = datetime.now(timezone.utc).isoformat()
        message_id = f"local-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

        # ── Persist to JSONL ──
        record = {
            "message_id": message_id,
            "timestamp": ts,
            "topic_arn": TopicArn,
            "subject": Subject,
            "message": Message,
        }
        with open(self._alerts_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # ── Print to stdout (rich if available, plain fallback) ──
        self._print_alert(Subject, Message, ts, message_id)

        return {"MessageId": message_id, "ResponseMetadata": {"HTTPStatusCode": 200}}

    @staticmethod
    def _print_alert(subject: str, message: str, ts: str, msg_id: str) -> None:
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            console.print(
                Panel(
                    f"[bold red]{subject}[/bold red]\n\n{message[:500]}...",
                    title=f"[yellow]CLOUDSENTINEL ALERT[/yellow] [dim]{msg_id[:12]}[/dim]",
                    subtitle=f"[dim]{ts}[/dim]",
                    border_style="red",
                    expand=False,
                )
            )
        except ImportError:
            # Fallback without rich
            separator = "=" * 60
            print(f"\n{separator}")
            print(f"  CLOUDSENTINEL ALERT  [{ts}]")
            print(separator)
            print(f"  Subject: {subject}")
            print(f"  {message[:300]}")
            print(separator + "\n")

