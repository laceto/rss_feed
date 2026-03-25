# Back-fill Daily Briefings — Sep 2025 → Today

Goal: generate `data/briefings/{date}.json` for every trading day from
September 2025 through today by reconstructing topic cluster history and
running `daily_briefing.py` for each date.

---

## Overview

The back-fill has two sequential phases:

```
Phase 1 — Cluster back-fill (no API cost)
  cluster_topics.py --date {d} --skip-labeling
  → populates data/topic_trends.tsv  (append-only)
  → updates  data/topic_centroids.json

Phase 2 — Briefing back-fill (OpenAI API)
  daily_briefing.py --date {d} --save [--no-rag]
  → writes data/briefings/{date}.json
```

Both phases must process dates in **strict ascending order**.
Running out of order corrupts `topic_centroids.json` and breaks
topic continuity for all subsequent dates.

---

## Prerequisites

### 1. Verify vectorstore coverage

September dates use a 45-day rolling window → need articles back to
August 2025 in FAISS.

```python
import pandas as pd
df = pd.read_csv("data/vectorstore/feeds_registry.tsv", sep="\t")
print(df["date"].min(), df["date"].max(), len(df))
```

**If `min` is later than 2025-08-01**, run `embed_feeds.py` first to
back-fill the index before proceeding.

### 2. Check existing topic_trends coverage

```python
import pandas as pd
from pathlib import Path

p = Path("data/topic_trends.tsv")
if p.exists():
    df = pd.read_csv(p, sep="\t")
    print(f"Dates already in topic_trends: {sorted(df['date'].unique())}")
else:
    print("topic_trends.tsv not found — full back-fill needed")
```

---

## Implementation: `backfill.py`

Place at project root. Handles both phases, resumes safely after any crash.

```python
"""
backfill.py

Two-phase back-fill:
  Phase 1 — cluster_topics for every missing date (cheap, no LLM labeling)
  Phase 2 — daily_briefing for every date (RAG optional)

Usage:
    python backfill.py                          # Sep 2025 → today, both phases
    python backfill.py --start 2025-11-01       # custom start date
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
        if d.weekday() < 5:   # Mon–Fri
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


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def phase1_cluster(dates: list[date], sleep_s: float) -> dict:
    """Run cluster_topics.py for every date not already in topic_trends.tsv.

    Exit code 2 = ClusteringAborted (weekend / sparse day) → logged, not fatal.
    """
    done   = _clustered_dates()
    stats  = {"skipped": 0, "clustered": 0, "aborted": 0, "errors": 0}

    for d in dates:
        ds = str(d)
        if ds in done:
            print(f"[phase1] {ds}  SKIP (already clustered)")
            stats["skipped"] += 1
            continue

        print(f"[phase1] {ds}  clustering ...")
        rc = _run([sys.executable, "cluster_topics.py", "--date", ds, "--skip-labeling"])

        if rc == 0:
            stats["clustered"] += 1
            print(f"[phase1] {ds}  OK")
        elif rc == 2:
            stats["aborted"] += 1
            print(f"[phase1] {ds}  ABORTED (degenerate / no articles — skipped)")
        else:
            stats["errors"] += 1
            print(f"[phase1] {ds}  ERROR exit={rc}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    return stats


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def phase2_briefing(dates: list[date], use_rag: bool, sleep_s: float) -> dict:
    """Run daily_briefing.py for every date not already in data/briefings/."""
    done  = _briefing_dates()
    stats = {"skipped": 0, "generated": 0, "no_spikes": 0, "errors": 0}

    for d in dates:
        ds = str(d)
        if ds in done:
            print(f"[phase2] {ds}  SKIP (briefing exists)")
            stats["skipped"] += 1
            continue

        print(f"[phase2] {ds}  briefing ...")
        cmd = [sys.executable, "daily_briefing.py", "--date", ds, "--save"]
        if not use_rag:
            cmd.append("--no-rag")

        rc = _run(cmd)

        if rc == 0:
            # Check whether a file was actually written (no spikes = no file)
            if (PROJECT_ROOT / "data" / "briefings" / f"{ds}.json").exists():
                stats["generated"] += 1
                print(f"[phase2] {ds}  OK")
            else:
                stats["no_spikes"] += 1
                print(f"[phase2] {ds}  OK (no spikes — file not written)")
        else:
            stats["errors"] += 1
            print(f"[phase2] {ds}  ERROR exit={rc}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Back-fill topic clusters and briefings")
    p.add_argument("--start",        default=str(DEFAULT_START),
                   help="Start date YYYY-MM-DD (default: 2025-09-01)")
    p.add_argument("--end",          default=str(date.today()),
                   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--phase1-only",  action="store_true",
                   help="Run clustering only, skip briefings")
    p.add_argument("--phase2-only",  action="store_true",
                   help="Run briefings only (topic_trends must already exist)")
    p.add_argument("--no-rag",       action="store_true",
                   help="Briefings without RAG summaries (no API calls)")
    p.add_argument("--sleep",        type=float, default=1.0,
                   help="Seconds to sleep between dates (default: 1)")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse()
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    dates = _trading_dates(start, end)

    print(f"Back-fill: {start} → {end}  ({len(dates)} trading days)")
    print(f"Phase1-only : {args.phase1_only}")
    print(f"Phase2-only : {args.phase2_only}")
    print(f"RAG enabled : {not args.no_rag}")
    print(f"Sleep       : {args.sleep}s between dates\n")

    if not args.phase2_only:
        s1 = phase1_cluster(dates, args.sleep)
        print(f"\nPhase 1 summary: {s1}\n")

    if not args.phase1_only:
        s2 = phase2_briefing(dates, use_rag=not args.no_rag, sleep_s=args.sleep)
        print(f"\nPhase 2 summary: {s2}\n")
```

---

## Recommended Run Order

```bash
# Step 1 — verify vectorstore reach
python -c "
import pandas as pd
df = pd.read_csv('data/vectorstore/feeds_registry.tsv', sep='\t')
print(df['date'].min(), df['date'].max(), len(df))
"

# Step 2 — cluster back-fill only (no API cost, ~20 min)
python backfill.py --phase1-only

# Step 3 — inspect topic_trends.tsv
python -c "
import pandas as pd
df = pd.read_csv('data/topic_trends.tsv', sep='\t')
print(f'{len(df)} rows, {df[\"date\"].nunique()} dates')
print(df.groupby(\"date\")[\"topic_id\"].count().describe())
"

# Step 4 — briefings without RAG (instant, zero cost)
python backfill.py --phase2-only --no-rag

# Step 5 — optionally add RAG summaries for dates that had spikes
python backfill.py --phase2-only   # re-runs skipped for existing files
```

---

## Resuming After a Crash

Both phases are **idempotent**:

- Phase 1 skips dates already in `topic_trends.tsv`.
- Phase 2 skips dates that already have `data/briefings/{date}.json`.

Just re-run the same command — it picks up exactly where it left off.

---

## Expected Output

| Metric | Approximate value |
|---|---|
| Trading days Sep 2025 → Mar 2026 | ~135 |
| Dates with articles (weekdays) | ~130 |
| Dates with `ClusteringAborted` (holidays/gaps) | 0–10 |
| Rows in `topic_trends.tsv` after back-fill | ~2,500 |
| Briefings with ≥1 spike (after 7-day warm-up) | ~120 |
| OpenAI API calls (RAG, 5 spikes/day) | ~600 |
| Estimated API cost (`gpt-4o-mini`, strategy=expand) | < $0.10 |
| Phase 1 wall-clock time | ~20 min |
| Phase 2 wall-clock time (with RAG) | ~30–60 min |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Vectorstore missing Aug 2025 articles → sparse windows | Run `embed_feeds.py` first; check registry min date |
| Out-of-order dates corrupt centroids | `backfill.py` always sorts dates ascending |
| `ClusteringAborted` on holiday/gap date | Caught, logged, loop continues |
| RAG summaries blend old + new articles (no date filter) | Use `--no-rag` for back-fill; RAG is best for live dates |
| Re-running overwrites good centroid history | Phase 1 skips already-clustered dates — centroids never re-written for done dates |

---

## Key Open Question

Does `data/vectorstore/feeds_registry.tsv` contain articles dated before
2025-08-01? Run the verification command in Prerequisites before starting.
If not, `embed_feeds.py` must back-fill FAISS first — everything else
depends on it.
