"""
backfill.py

Two-phase back-fill:
  Phase 1 — cluster_topics for every missing date (cheap, no LLM labeling)
  Phase 2 — daily_briefing for every date (RAG optional)

Usage:
    python backfill.py                          # Sep 2025 -> today, both phases
    python backfill.py --start 2025-08-01       # custom start date
    python backfill.py --phase1-only            # clustering only, no briefings
    python backfill.py --phase2-only            # briefings only (trends must exist)
    python backfill.py --no-rag                 # briefings without RAG summaries
    python backfill.py --sleep 2                # seconds between dates (rate limiting)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
TRENDS_FILE  = PROJECT_ROOT / "data" / "topic_trends.tsv"

DEFAULT_START = date(2025, 9, 1)


def _trading_dates(start: date, end: date) -> list[date]:
    """Return weekdays between start and end inclusive."""
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:   # Mon-Fri
            out.append(d)
        d += timedelta(days=1)
    return out


def _clustered_dates() -> set[str]:
    """Return the set of dates already in topic_trends.tsv."""
    if not TRENDS_FILE.exists():
        return set()
    df = pd.read_csv(TRENDS_FILE, sep="\t")
    return set(df["date"].astype(str).unique())


def _briefing_dates() -> set[str]:
    """Return dates that already have a briefing JSON."""
    briefings_dir = PROJECT_ROOT / "data" / "briefings"
    if not briefings_dir.exists():
        return set()
    return {p.stem for p in briefings_dir.glob("*.json")}


def _run(cmd: list[str]) -> int:
    """Run a subprocess and return its exit code."""
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


# -- Phase 1 ------------------------------------------------------------------

def phase1_cluster(dates: list[date], sleep_s: float) -> dict:
    """Run cluster_topics.py for every date not already in topic_trends.tsv.

    Exit code 2 = ClusteringAborted (weekend / sparse day) -> logged, not fatal.
    """
    done  = _clustered_dates()
    stats = {"skipped": 0, "clustered": 0, "aborted": 0, "errors": 0}
    total = len(dates)

    for i, d in enumerate(dates, 1):
        ds = str(d)
        if ds in done:
            print(f"[{i:>3}/{total}] {ds}  SKIP")
            stats["skipped"] += 1
            continue

        print(f"[{i:>3}/{total}] {ds}  clustering ...", flush=True)
        rc = _run([sys.executable, "cluster_topics.py", "--date", ds, "--skip-labeling"])

        if rc == 0:
            stats["clustered"] += 1
            print(f"[{i:>3}/{total}] {ds}  OK")
        elif rc == 2:
            stats["aborted"] += 1
            print(f"[{i:>3}/{total}] {ds}  ABORTED (degenerate / no articles)")
        else:
            stats["errors"] += 1
            print(f"[{i:>3}/{total}] {ds}  ERROR exit={rc}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    return stats


# -- Phase 2 ------------------------------------------------------------------

def phase2_briefing(dates: list[date], use_rag: bool, sleep_s: float) -> dict:
    """Run daily_briefing.py for every date not already in data/briefings/."""
    done  = _briefing_dates()
    stats = {"skipped": 0, "generated": 0, "no_spikes": 0, "errors": 0}
    total = len(dates)

    for i, d in enumerate(dates, 1):
        ds = str(d)
        if ds in done:
            print(f"[{i:>3}/{total}] {ds}  SKIP")
            stats["skipped"] += 1
            continue

        print(f"[{i:>3}/{total}] {ds}  briefing ...", flush=True)
        cmd = [sys.executable, "daily_briefing.py", "--date", ds, "--save"]
        if not use_rag:
            cmd.append("--no-rag")

        rc = _run(cmd)

        if rc == 0:
            if (PROJECT_ROOT / "data" / "briefings" / f"{ds}.json").exists():
                stats["generated"] += 1
                print(f"[{i:>3}/{total}] {ds}  OK")
            else:
                stats["no_spikes"] += 1
                print(f"[{i:>3}/{total}] {ds}  OK (no spikes)")
        else:
            stats["errors"] += 1
            print(f"[{i:>3}/{total}] {ds}  ERROR exit={rc}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    return stats


# -- CLI ----------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Back-fill topic clusters and briefings")
    p.add_argument("--start",       default=str(DEFAULT_START),
                   help="Start date YYYY-MM-DD (default: 2025-09-01)")
    p.add_argument("--end",         default=str(date.today()),
                   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--phase1-only", action="store_true",
                   help="Run clustering only, skip briefings")
    p.add_argument("--phase2-only", action="store_true",
                   help="Run briefings only (topic_trends must already exist)")
    p.add_argument("--no-rag",      action="store_true",
                   help="Briefings without RAG summaries (no API calls)")
    p.add_argument("--sleep",       type=float, default=1.0,
                   help="Seconds to sleep between dates (default: 1)")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse()
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    dates = _trading_dates(start, end)

    print(f"Back-fill  : {start} -> {end}  ({len(dates)} trading days)")
    print(f"Phase1-only: {args.phase1_only}")
    print(f"Phase2-only: {args.phase2_only}")
    print(f"RAG enabled: {not args.no_rag}")
    print(f"Sleep      : {args.sleep}s between dates\n")

    if not args.phase2_only:
        s1 = phase1_cluster(dates, args.sleep)
        print(f"\nPhase 1 done: {s1}\n")

    if not args.phase1_only:
        s2 = phase2_briefing(dates, use_rag=not args.no_rag, sleep_s=args.sleep)
        print(f"\nPhase 2 done: {s2}\n")
