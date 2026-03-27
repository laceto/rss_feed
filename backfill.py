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
from datetime import date

from dotenv import load_dotenv

from pipeline.backfill import phase1_cluster, phase2_briefing, trading_dates

load_dotenv()

# Sep 2025 = first date with sector results in this deployment
DEFAULT_START = date(2025, 9, 1)


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
    dates = trading_dates(start, end)

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
