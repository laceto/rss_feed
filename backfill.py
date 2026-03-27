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
import json
import time
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv

from cluster_topics import ClusteringAborted, DuplicateDateError, run as cluster_run
from constants import BRIEFINGS_DIR, TOPIC_TRENDS_FILE
from daily_briefing import build_briefing

load_dotenv()

# Sep 2025 = first date with sector results in this deployment
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
    if not TOPIC_TRENDS_FILE.exists():
        return set()
    df = pd.read_csv(TOPIC_TRENDS_FILE, sep="\t")
    return set(df["date"].astype(str).unique())


def _briefing_dates() -> set[str]:
    """Return dates that already have a briefing JSON."""
    if not BRIEFINGS_DIR.exists():
        return set()
    return {p.stem for p in BRIEFINGS_DIR.glob("*.json")}


# -- Phase 1 ------------------------------------------------------------------

def phase1_cluster(dates: list[date], sleep_s: float) -> dict:
    """Run cluster_topics.run() for every date not already in topic_trends.tsv.

    ClusteringAborted (degenerate / no articles) is non-fatal — counted as
    'aborted'. DuplicateDateError means the date is already present but was
    not caught by _clustered_dates() — counted as 'skipped'.
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
        try:
            cluster_run(target_date=d, skip_labeling=True)
            stats["clustered"] += 1
            print(f"[{i:>3}/{total}] {ds}  OK")
        except ClusteringAborted:
            stats["aborted"] += 1
            print(f"[{i:>3}/{total}] {ds}  ABORTED (degenerate / no articles)")
        except DuplicateDateError:
            stats["skipped"] += 1
            print(f"[{i:>3}/{total}] {ds}  SKIP (duplicate date in trends)")
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            print(f"[{i:>3}/{total}] {ds}  ERROR {exc}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    return stats


# -- Phase 2 ------------------------------------------------------------------

def phase2_briefing(dates: list[date], use_rag: bool, sleep_s: float) -> dict:
    """Run build_briefing() for every date not already in data/briefings/.

    Writes BRIEFINGS_DIR/{date}.json when spikes are found. Dates with no
    spikes are counted separately — the file is intentionally not written so
    the sentinel stays clear for a future re-run.

    NOTE: build_briefing() calls sys.exit(1) if TOPIC_TRENDS_FILE is absent.
          SystemExit is BaseException, not Exception, so it propagates through
          the per-date except clause and aborts the loop — the correct behaviour
          since all remaining dates would fail the same way.
    """
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
        try:
            briefing = build_briefing(d, top_n=5, use_rag=use_rag)
            if briefing.get("n_spikes", 0) > 0:
                out = BRIEFINGS_DIR / f"{ds}.json"
                BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps(briefing, indent=2, default=str), encoding="utf-8"
                )
                stats["generated"] += 1
                print(f"[{i:>3}/{total}] {ds}  OK")
            else:
                stats["no_spikes"] += 1
                print(f"[{i:>3}/{total}] {ds}  OK (no spikes)")
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            print(f"[{i:>3}/{total}] {ds}  ERROR {exc}")

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
