"""
create_batch_briefings.py

Build OpenAI Batch API tasks for RAG narrative generation across all dates
that have topic spike data but no saved briefing yet.

Architecture:
  1. Load FAISS + BM25 once locally (no API calls for this step)
  2. For each unprocessed date, detect spiking topics via get_emerging_topics()
  3. For each spike, run hybrid retrieval locally → embed context into the task prompt
  4. Submit all tasks in one batch job via kitai.batch.submit_batch_job()
  5. Persist batch ID + spike metadata sidecar for the collection step

custom_id convention:  "briefing-YYYY-MM-DD-{topic_id[:8]}"
                        → retrieve_batch_briefings.py routes by this key

Sidecar file (data/pending_briefings_meta.json):
  Maps each custom_id → {date, topic_id, label, spike_ratio, article_count}
  so the collection step can reconstruct the full briefing without re-running
  get_emerging_topics().

Invariants:
  - Never writes to topic_trends.tsv or any pipeline state file.
  - Skips dates that already have data/briefings/{date}.json.
  - Retrieval uses strategy="none" (no query-translation API calls here).
  - Fails fast if topic_trends.tsv is absent.

Usage:
    python create_batch_briefings.py             # all unprocessed dates
    python create_batch_briefings.py --top 3     # top N spikes per date
    python create_batch_briefings.py --start 2025-10-01 --end 2025-12-31
    python create_batch_briefings.py --dry-run   # show counts, no submission

Failure modes:
  - OPENAI_API_KEY missing: fails on kitai.batch.submit_batch_job()
  - FAISS missing: fails on get_resources() with FileNotFoundError
  - topic_trends.tsv absent: exits with instructions
  - All dates already processed: exits cleanly with no submission
"""

from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from pipeline.briefing_batch_submit import TOP_N_DEFAULT, run_briefing_batch_submission

# ── Configuration ─────────────────────────────────────────────────────────────

# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit RAG briefing batch to OpenAI")
    p.add_argument("--start",   default=None, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end",     default=None, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--top",     type=int, default=TOP_N_DEFAULT,
                   help=f"Max spikes per date (default: {TOP_N_DEFAULT})")
    p.add_argument("--dry-run", action="store_true",
                   help="Show counts only — no FAISS load, no submission")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print("=== Briefing Batch Submission ===")
    try:
        run_briefing_batch_submission(
            start=args.start,
            end=args.end,
            top_n=args.top,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
