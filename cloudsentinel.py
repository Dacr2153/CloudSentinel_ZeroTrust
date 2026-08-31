#!/usr/bin/env python3
"""CloudSentinel Zero-Trust — single entry point.

Interactive menu-driven CLI and non-interactive batch mode for the
CloudSentinel anomaly detection pipeline.

Usage:
    python cloudsentinel.py                       # interactive menu
    python cloudsentinel.py --no-menu --train     # non-interactive train
    python cloudsentinel.py --no-menu --run-pipeline
    python cloudsentinel.py --no-menu --run-tests
    python cloudsentinel.py --no-menu --evaluate

Environment:
    export $(grep -v '^#' .env.local | xargs)
    python cloudsentinel.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap: ensure project root is in sys.path ──────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── LOCAL_MODE defaults (set before any src/ import) ───────────────
os.environ.setdefault("CLOUDSENTINEL_LOCAL_MODE", "true")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "local")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "local")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("CLOUDSENTINEL_AWS_REGION", "us-east-1")
os.environ.setdefault("CLOUDSENTINEL_MODEL_BUCKET", "local")
os.environ.setdefault("CLOUDSENTINEL_OPENSEARCH_ENDPOINT", "http://localhost:9200")
os.environ.setdefault(
    "CLOUDSENTINEL_SNS_TOPIC_ARN",
    "arn:aws:sns:local:000000000000:cloudsentinel-alerts",
)

from rich.align import Align  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.prompt import Prompt  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

console = Console()

# ── Paths ───────────────────────────────────────────────────────────
MODEL_PATH = (
    PROJECT_ROOT / "ml" / "data" / "local" / "models" / "isolation_forest" / "model.joblib"
)
EVENTS_DIR = PROJECT_ROOT / "ml" / "data" / "cloudtrail_samples"
ALERTS_FILE = PROJECT_ROOT / "tools" / "alerts" / "alerts.jsonl"

# Python interpreter: prefer the venv that ships with the project
_venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable


# ── Banner ──────────────────────────────────────────────────────────

BANNER = r"""
   ██████╗██╗      ██████╗ ██╗   ██╗██████╗
  ██╔════╝██║     ██╔═══██╗██║   ██║██╔══██╗
  ██║     ██║     ██║   ██║██║   ██║██║  ██║
  ██║     ██║     ██║   ██║██║   ██║██║  ██║
  ╚██████╗███████╗╚██████╔╝╚██████╔╝██████╔╝
   ╚═════╝╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝

  ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗
  ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║
  ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║
  ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║
  ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
  ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
"""


def _print_banner() -> None:
    console.print(
        Panel(
            Align.center(Text(BANNER, style="bold cyan")),
            subtitle="[dim]Zero-Trust CloudTrail Anomaly Detection[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print(
        Align.center(
            "[dim]MITRE ATT&CK  ·  Isolation Forest ML  ·  8 Deterministic Rules[/dim]"
        )
    )
    console.print()


# ── Status helpers ───────────────────────────────────────────────────

def _status_table() -> Table:
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
    )
    table.add_column("Key", style="dim")
    table.add_column("Value")

    # Model status
    if MODEL_PATH.exists():
        size_kb = MODEL_PATH.stat().st_size // 1024
        mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, tz=timezone.utc)
        model_str = f"[green]✓ Loaded[/green]  [dim]({size_kb} KB · {mtime.strftime('%Y-%m-%d %H:%M')} UTC)[/dim]"
    else:
        model_str = "[red]✗ Not trained[/red]  [dim](run option 1)[/dim]"
    table.add_row("ML Model", model_str)

    # Events
    event_files = list(EVENTS_DIR.glob("*.json")) + list(EVENTS_DIR.glob("*.json.gz"))
    if event_files:
        events_str = f"[green]✓ {len(event_files)} file(s)[/green]  [dim]({EVENTS_DIR})[/dim]"
    else:
        events_str = "[yellow]⚠ No events[/yellow]  [dim](export from AWS CloudTrail)[/dim]"
    table.add_row("CloudTrail Events", events_str)

    # Recent alerts
    if ALERTS_FILE.exists():
        count = sum(1 for _ in open(ALERTS_FILE, encoding="utf-8"))
        alerts_str = f"[yellow]⚠ {count} alert(s)[/yellow]  [dim]({ALERTS_FILE})[/dim]"
    else:
        alerts_str = "[green]✓ No alerts on file[/green]"
    table.add_row("Alerts", alerts_str)

    # Mode
    mode = os.environ.get("CLOUDSENTINEL_LOCAL_MODE", "false").lower()
    mode_str = "[cyan]LOCAL (no AWS required)[/cyan]" if mode == "true" else "[yellow]AWS[/yellow]"
    table.add_row("Mode", mode_str)

    return table


def _print_status() -> None:
    console.print(
        Panel(
            _status_table(),
            title="[bold]System Status[/bold]",
            border_style="dim",
        )
    )
    console.print()


# ── Menu ─────────────────────────────────────────────────────────────

_MENU_ITEMS = [
    ("1", "Setup & Train ML Model",
     "Generate labeled training data and train the Isolation Forest anomaly detector."),
    ("2", "Analyze CloudTrail Events",
     "Run the full detection pipeline on real CloudTrail JSON files you provide."),
    ("3", "View Recent Alerts",
     "Display alerts detected during the last pipeline run."),
    ("4", "Evaluate Model Performance",
     "Compute ROC-AUC, precision/recall, and confusion matrix on test data."),
    ("5", "Run Test Suite",
     "Execute all 59 unit tests to verify pipeline integrity."),
    ("6", "System Status",
     "Refresh and display current model, events, and alert status."),
    ("0", "Exit", "Quit CloudSentinel."),
]


def _print_menu() -> None:
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("Opt", style="bold cyan", no_wrap=True, width=4)
    table.add_column("Action", style="bold white", min_width=28)
    table.add_column("Description", style="dim")

    for opt, action, desc in _MENU_ITEMS:
        style = "dim" if opt == "0" else ""
        table.add_row(
            f"[{style}][{opt}][/{style}]" if style else f"[{opt}]",
            f"[{style}]{action}[/{style}]" if style else action,
            f"[{style}]{desc}[/{style}]" if style else desc,
        )

    console.print(
        Panel(
            table,
            title="[bold]Main Menu[/bold]",
            border_style="cyan",
        )
    )


def _prompt() -> str:
    return Prompt.ask(
        "\n[bold cyan]Select option[/bold cyan]",
        choices=[item[0] for item in _MENU_ITEMS],
        show_choices=False,
    )


# ── Action handlers ───────────────────────────────────────────────────

def _run(cmd: list[str], env: dict | None = None) -> int:
    """Run a subprocess, streaming output to terminal. Returns exit code."""
    merged = {**os.environ, **(env or {})}
    proc = subprocess.run(cmd, env=merged)
    return proc.returncode


def action_train() -> None:
    console.print(Rule("[bold cyan]Setup & Train ML Model[/bold cyan]"))
    console.print()
    console.print("  [dim]Step 1/2:[/dim] Generating synthetic labeled training data...")
    console.print(
        "  [dim]Note:[/dim] Synthetic data is used [italic]only[/italic] for training "
        "(standard ML practice).\n"
        "  The pipeline detects anomalies in [bold]real[/bold] CloudTrail events you provide.\n"
    )

    rc = _run([PYTHON, "ml/training/generate_synthetic_data.py", "--output-dir", "ml/data"])
    if rc != 0:
        console.print(f"\n  [red]✗ Data generation failed (exit {rc})[/red]")
        return

    console.print("\n  [dim]Step 2/2:[/dim] Training Isolation Forest model...\n")
    rc = _run([
        PYTHON, "ml/training/train_model.py",
        "--data-dir", "ml/data",
        "--output-dir", "ml/data",
    ])
    if rc != 0:
        console.print(f"\n  [red]✗ Training failed (exit {rc})[/red]")
        return

    console.print()
    if MODEL_PATH.exists():
        size_kb = MODEL_PATH.stat().st_size // 1024
        console.print(
            Panel(
                f"[green]Model trained successfully[/green]\n\n"
                f"  Path: [cyan]{MODEL_PATH}[/cyan]\n"
                f"  Size: {size_kb} KB",
                border_style="green",
            )
        )
    else:
        console.print("[yellow]⚠ Training finished but model file not found at expected path.[/yellow]")


def action_analyze() -> None:
    console.print(Rule("[bold cyan]Analyze CloudTrail Events[/bold cyan]"))
    console.print()

    event_files = list(EVENTS_DIR.glob("*.json")) + list(EVENTS_DIR.glob("*.json.gz"))
    if not event_files:
        console.print(
            Panel(
                "[yellow]No CloudTrail event files found.[/yellow]\n\n"
                f"Place real CloudTrail exports (JSON or .json.gz) in:\n"
                f"  [cyan]{EVENTS_DIR}[/cyan]\n\n"
                "How to export from AWS:\n"
                "  • AWS Console → CloudTrail → Event history → [bold]Download JSON[/bold]\n"
                "  • Or download .json.gz files from your S3 CloudTrail bucket",
                title="[yellow]No Events Available[/yellow]",
                border_style="yellow",
            )
        )
        return

    if not MODEL_PATH.exists():
        console.print(
            Panel(
                "[red]ML model not found.[/red]\n\n"
                "Run [bold]option 1[/bold] (Setup & Train) first.",
                border_style="red",
            )
        )
        return

    console.print(
        f"  Found [cyan]{len(event_files)}[/cyan] event file(s) in [dim]{EVENTS_DIR}[/dim]\n"
    )
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Import here so env vars are already set before src/ is imported
    from src.pipeline.runner import LocalPipelineRunner  # noqa: E402
    from src.utils.config import get_settings  # noqa: E402

    get_settings.cache_clear()

    try:
        runner = LocalPipelineRunner(
            events_dir=str(EVENTS_DIR),
            opensearch_enabled=False,
        )
    except Exception as exc:
        console.print(f"[red]✗ Failed to initialise pipeline: {exc}[/red]")
        return

    status_row = Table.grid(padding=(0, 1))
    status_row.add_column()
    status_row.add_column()
    status_row.add_row(
        "  ML detector:",
        "[green]ready[/green]" if runner.model_ready else "[yellow]disabled (model not loaded)[/yellow]",
    )
    console.print(status_row)
    console.print()

    with console.status("[cyan]Running detection pipeline…[/cyan]"):
        try:
            stats = runner.run()
        except Exception as exc:
            console.print(f"[red]✗ Pipeline error: {exc}[/red]")
            return

    # Summary table
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("Files processed", str(stats.get("files", 0)))
    summary.add_row("Events extracted", str(stats.get("events_total", 0)))
    summary.add_row("Events normalized", str(stats.get("events_normalized", 0)))
    summary.add_row("Anomalies (ML)", str(stats.get("anomalies_detected", 0)))
    summary.add_row("Rule triggers", str(stats.get("rules_triggered", 0)))
    alerts_n = stats.get("alerts_dispatched", 0)
    alert_color = "yellow" if alerts_n > 0 else "green"
    summary.add_row(
        "Alerts dispatched",
        f"[{alert_color}]{alerts_n}[/{alert_color}]",
    )
    summary.add_row("Errors", str(stats.get("errors", 0)))
    elapsed = stats.get("elapsed_seconds", 0)
    summary.add_row("Elapsed", f"{elapsed:.2f}s")

    console.print(Panel(summary, title="[bold]Pipeline Summary[/bold]", border_style="cyan"))

    if alerts_n > 0:
        console.print(
            f"\n  [yellow]⚠  {alerts_n} alert(s) written to:[/yellow] [cyan]{ALERTS_FILE}[/cyan]\n"
            "   Run [bold]option 3[/bold] to review them."
        )
    else:
        console.print("\n  [green]✓ Pipeline complete — no alerts triggered.[/green]")


def action_alerts() -> None:
    console.print(Rule("[bold cyan]Recent Alerts[/bold cyan]"))
    console.print()

    if not ALERTS_FILE.exists() or ALERTS_FILE.stat().st_size == 0:
        console.print("  [green]No alerts on record. Pipeline has not detected any threats.[/green]")
        return

    alerts = []
    with open(ALERTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not alerts:
        console.print("  [green]Alert file is empty.[/green]")
        return

    console.print(f"  [yellow]{len(alerts)} alert(s)[/yellow] found:\n")

    table = Table(
        title=f"Alerts — {ALERTS_FILE}",
        border_style="yellow",
        show_lines=True,
    )
    table.add_column("Timestamp", style="dim", no_wrap=True)
    table.add_column("Subject", style="bold yellow")
    table.add_column("Message (excerpt)", style="white")

    for alert in alerts[-20:]:  # Show last 20
        ts = alert.get("timestamp", "")[:19]
        subject = alert.get("subject", "(no subject)")
        message = alert.get("message", "")
        # Try to parse message JSON for cleaner display
        try:
            msg_data = json.loads(message)
            excerpt = (
                f"rule={msg_data.get('rule_id', '?')}  "
                f"severity={msg_data.get('severity', '?')}  "
                f"actor={msg_data.get('actor', {}).get('user', '?')}"
            )
        except (json.JSONDecodeError, AttributeError):
            excerpt = str(message)[:120]

        table.add_row(ts, subject, excerpt)

    if len(alerts) > 20:
        console.print(f"  [dim](showing last 20 of {len(alerts)} alerts)[/dim]\n")

    console.print(table)
    console.print()

    clear = Prompt.ask(
        "\n  [dim]Clear alerts file?[/dim]",
        choices=["y", "n"],
        default="n",
    )
    if clear == "y":
        ALERTS_FILE.write_text("")
        console.print("  [green]✓ Alerts cleared.[/green]")


def action_evaluate() -> None:
    console.print(Rule("[bold cyan]Model Performance Evaluation[/bold cyan]"))
    console.print()

    if not MODEL_PATH.exists():
        console.print(
            Panel(
                "[red]ML model not found.[/red]\n\nRun [bold]option 1[/bold] first.",
                border_style="red",
            )
        )
        return

    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)

    rc = _run([
        PYTHON, "ml/training/evaluate_model.py",
        "--model-path", str(MODEL_PATH),
        "--data-dir", "ml/data",
        "--output-dir", str(output_dir),
    ])

    if rc == 0:
        console.print(f"\n  [green]✓ Evaluation complete.[/green] Reports saved to [cyan]{output_dir}[/cyan]")
    else:
        console.print(f"\n  [red]✗ Evaluation failed (exit {rc})[/red]")


def action_tests() -> None:
    console.print(Rule("[bold cyan]Test Suite[/bold cyan]"))
    console.print()

    rc = _run([
        PYTHON, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "--no-header",
    ])

    console.print()
    if rc == 0:
        console.print("[green]✓ All tests passed.[/green]")
    else:
        console.print(f"[red]✗ Tests failed (exit {rc})[/red]")


def action_status() -> None:
    console.print(Rule("[bold cyan]System Status[/bold cyan]"))
    console.print()
    _print_status()


# ── Main loop ────────────────────────────────────────────────────────

_DISPATCH = {
    "1": action_train,
    "2": action_analyze,
    "3": action_alerts,
    "4": action_evaluate,
    "5": action_tests,
    "6": action_status,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CloudSentinel Zero-Trust — Interactive CLI",
    )
    parser.add_argument("--no-menu", action="store_true", help="Non-interactive mode")
    parser.add_argument("--train", action="store_true", help="(--no-menu) Train model")
    parser.add_argument("--run-pipeline", action="store_true", help="(--no-menu) Run pipeline")
    parser.add_argument("--run-tests", action="store_true", help="(--no-menu) Run tests")
    parser.add_argument("--evaluate", action="store_true", help="(--no-menu) Evaluate model")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Non-interactive mode ────────────────────────────────────────
    if args.no_menu:
        if args.train:
            action_train()
        elif args.run_pipeline:
            action_analyze()
        elif args.run_tests:
            action_tests()
        elif args.evaluate:
            action_evaluate()
        else:
            console.print("[red]--no-menu requires --train | --run-pipeline | --run-tests | --evaluate[/red]")
            sys.exit(1)
        return

    # ── Interactive mode ────────────────────────────────────────────
    console.clear()
    _print_banner()
    _print_status()

    while True:
        _print_menu()

        try:
            choice = _prompt()
        except (KeyboardInterrupt, EOFError):
            console.print("\n\n  [dim]Interrupted. Goodbye.[/dim]\n")
            break

        if choice == "0":
            console.print("\n  [dim]Goodbye.[/dim]\n")
            break

        console.print()
        handler = _DISPATCH.get(choice)
        if handler:
            handler()

        console.print()
        Prompt.ask("  [dim]Press Enter to return to menu[/dim]", default="")
        console.clear()
        _print_banner()
        _print_status()


if __name__ == "__main__":
    main()
