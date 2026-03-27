"""CLI helpers for the top-level cluster_topics module."""

from __future__ import annotations

import argparse
from datetime import date
from typing import Any, Callable


def parse_cluster_topics_args() -> argparse.Namespace:
    """Parse CLI args for cluster_topics.py."""
    parser = argparse.ArgumentParser(description="Run topic clustering for one day.")
    parser.add_argument(
        "--date",
        default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        help="Skip LLM labeling (useful for profiling / CI dry runs)",
    )
    return parser.parse_args()


def print_cluster_summary(summary: dict[str, Any]) -> None:
    """Print the standard cluster_topics completion summary."""
    print("[cluster_topics] Run complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def run_cluster_topics_cli(
    run_fn: Callable[..., dict[str, Any]],
    aborted_exc: type[Exception],
) -> int:
    """Execute the cluster_topics CLI flow using injected pipeline functions."""
    args = parse_cluster_topics_args()
    target = date.fromisoformat(args.date) if args.date else date.today()

    try:
        summary = run_fn(target_date=target, skip_labeling=args.skip_labeling)
    except aborted_exc as exc:
        print(f"[cluster_topics] ABORTED: {exc}", flush=True)
        return 2

    print_cluster_summary(summary)
    return 0
