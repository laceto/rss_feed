"""Reusable backfill orchestration utilities.

Extracted from backfill.py so the backfill flow can be imported and reused
without going through the CLI entrypoint.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

from cluster_topics import ClusteringAborted, DuplicateDateError, run as cluster_run
from constants import BRIEFINGS_DIR, TOPIC_TRENDS_FILE

from .briefings import build_briefing, save_briefing


def trading_dates(start: date, end: date) -> list[date]:
    """Return weekdays between start and end inclusive."""
    out = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            out.append(current)
        current += timedelta(days=1)
    return out


def clustered_dates(trends_path=TOPIC_TRENDS_FILE) -> set[str]:
    """Return the set of dates already present in topic_trends.tsv."""
    if not trends_path.exists():
        return set()
    df = pd.read_csv(trends_path, sep="\t")
    return set(df["date"].astype(str).unique())


def briefing_dates(briefings_dir=BRIEFINGS_DIR) -> set[str]:
    """Return the set of dates that already have a saved briefing JSON."""
    if not briefings_dir.exists():
        return set()
    return {path.stem for path in briefings_dir.glob("*.json")}


def phase1_cluster(dates: list[date], sleep_s: float) -> dict[str, int]:
    """Run cluster_topics.run() for every date not already in topic_trends.tsv."""
    done = clustered_dates()
    stats = {"skipped": 0, "clustered": 0, "aborted": 0, "errors": 0}
    total = len(dates)

    for i, run_date in enumerate(dates, 1):
        date_str = str(run_date)
        if date_str in done:
            print(f"[{i:>3}/{total}] {date_str}  SKIP")
            stats["skipped"] += 1
            continue

        print(f"[{i:>3}/{total}] {date_str}  clustering ...", flush=True)
        try:
            cluster_run(target_date=run_date, skip_labeling=True)
            stats["clustered"] += 1
            print(f"[{i:>3}/{total}] {date_str}  OK")
        except ClusteringAborted:
            stats["aborted"] += 1
            print(f"[{i:>3}/{total}] {date_str}  ABORTED (degenerate / no articles)")
        except DuplicateDateError:
            stats["skipped"] += 1
            print(f"[{i:>3}/{total}] {date_str}  SKIP (duplicate date in trends)")
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            print(f"[{i:>3}/{total}] {date_str}  ERROR {exc}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    return stats


def phase2_briefing(dates: list[date], use_rag: bool, sleep_s: float) -> dict[str, int]:
    """Run build_briefing() for every date not already saved in data/briefings/."""
    done = briefing_dates()
    stats = {"skipped": 0, "generated": 0, "no_spikes": 0, "errors": 0}
    total = len(dates)

    for i, run_date in enumerate(dates, 1):
        date_str = str(run_date)
        if date_str in done:
            print(f"[{i:>3}/{total}] {date_str}  SKIP")
            stats["skipped"] += 1
            continue

        print(f"[{i:>3}/{total}] {date_str}  briefing ...", flush=True)
        try:
            briefing = build_briefing(run_date, top_n=5, use_rag=use_rag)
            if briefing.get("n_spikes", 0) > 0:
                save_briefing(briefing)
                stats["generated"] += 1
                print(f"[{i:>3}/{total}] {date_str}  OK")
            else:
                stats["no_spikes"] += 1
                print(f"[{i:>3}/{total}] {date_str}  OK (no spikes)")
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            print(f"[{i:>3}/{total}] {date_str}  ERROR {exc}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    return stats
