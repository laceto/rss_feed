"""
retrieve_batch_briefings.py

Collect completed OpenAI Batch API results for daily RAG briefings.

Architecture:
  1. Read batch ID from data/pending_briefings_batch.txt
  2. Check batch status via kitai.batch.check_batch_job()
  3. Download results via kitai.batch.download_batch_results()
  4. Parse each item: extract LLM answer + use pre-retrieved sources from sidecar
  5. For each spike, run sector cross-check locally (no API calls)
  6. Group spikes by date → write data/briefings/{date}.json

custom_id convention (set by create_batch_briefings.py):
  "briefing-YYYY-MM-DD-{topic_id[:8]}"  →  data/briefings/YYYY-MM-DD.json

Exit codes (mirrors sector batch pattern for CI compatibility):
  0  — success (all items collected)
  1  — hard failure (batch failed/expired, or file missing)
  2  — not ready yet (batch still in progress); safe to retry later

Debugging:
  - data/pending_briefings_batch.txt  — active batch ID
  - data/pending_briefings_meta.json  — spike metadata keyed by custom_id
  - data/batch_tasks_briefings.jsonl  — raw tasks submitted (debug copy)
  - Per-item failures are logged but do not abort collection
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from constants import (
    BATCH_FILE_BRIEFINGS,
    BRIEFINGS_DIR,
    BRIEFINGS_BATCH_META_FILE,
    PENDING_BRIEFINGS_BATCH_FILE,
)
from daily_briefing import _sector_crosscheck
from kitai.batch import check_batch_job, download_batch_results

# ── Batch status ──────────────────────────────────────────────────────────────

def read_pending_batch_id() -> str:
    """Read the active batch job ID; fail fast if the sentinel file is absent."""
    if not PENDING_BRIEFINGS_BATCH_FILE.exists():
        print(
            f"[error] No pending batch found at {PENDING_BRIEFINGS_BATCH_FILE}.\n"
            "Run create_batch_briefings.py first to submit a batch job."
        )
        sys.exit(1)
    return PENDING_BRIEFINGS_BATCH_FILE.read_text(encoding="utf-8").strip()


def read_spike_metadata() -> dict:
    """Load the sidecar metadata file written by create_batch_briefings.py.

    Returns {custom_id: {date, topic_id, label, spike_ratio, article_count, sources}}.
    Exits with error if the file is missing (create script was not run or was cleared).
    """
    if not BRIEFINGS_BATCH_META_FILE.exists():
        print(
            f"[error] Metadata sidecar not found at {BRIEFINGS_BATCH_META_FILE}.\n"
            "Re-run create_batch_briefings.py to regenerate it."
        )
        sys.exit(1)
    return json.loads(BRIEFINGS_BATCH_META_FILE.read_text(encoding="utf-8"))


def check_status(client: OpenAI, batch_id: str) -> None:
    """Check batch status; exit 2 if still in progress, exit 1 if terminal failure."""
    status = check_batch_job(client, batch_id)
    counts = status.get("request_counts", {})

    print(
        f"Batch {batch_id} -> status: {status['status']} | "
        f"completed: {counts.get('completed', '?')} | "
        f"failed: {counts.get('failed', '?')}"
    )

    _TERMINAL = {"completed", "failed", "expired", "cancelled"}
    if status["status"] not in _TERMINAL:
        print("Batch is still in progress. Re-run this script once it completes.")
        sys.exit(2)  # exit 2 = not ready; retry is safe

    if status["status"] != "completed":
        print(
            f"[error] Batch ended with status '{status['status']}'. "
            "No results to collect. Check the OpenAI dashboard for details."
        )
        sys.exit(1)


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_custom_id(custom_id: str) -> tuple[str, str] | None:
    """Extract (date, topic_id_prefix) from 'briefing-YYYY-MM-DD-{tid[:8]}'.

    Returns None if the format does not match (defensive guard).
    """
    parts = custom_id.split("-", maxsplit=2)
    # Expected: ["briefing", "YYYY", "MM-DD-{tid[:8]}"]
    # After split(maxsplit=2): ["briefing", "YYYY", "MM-DD-xxxxxxxx"]
    if len(parts) != 3 or parts[0] != "briefing":
        return None
    # Reconstruct date: parts[1] is "YYYY", parts[2] starts with "MM-DD-"
    rest = parts[2]  # "MM-DD-xxxxxxxx"
    dash_pos = rest.rfind("-")
    if dash_pos == -1:
        return None
    date_str = f"{parts[1]}-{rest[:dash_pos]}"   # "YYYY-MM-DD"
    tid_prefix = rest[dash_pos + 1:]               # "xxxxxxxx" (first 8 chars of UUID)
    return date_str, tid_prefix


def collect_results(
    items:    list[dict],
    metadata: dict,
) -> tuple[dict[str, list[dict]], int, int]:
    """Parse batch results and group assembled spikes by date.

    For each item:
      - Extracts LLM answer from the response body
      - Looks up pre-retrieved sources from the sidecar metadata
      - Runs sector cross-check locally
      - Groups into {date: [spike_dict, ...]}

    Returns:
        spikes_by_date — {date: [assembled spike dict]}
        ok_count       — items successfully parsed
        failed_count   — items with errors or parse failures
    """
    spikes_by_date: dict[str, list[dict]] = defaultdict(list)
    ok_count    = 0
    failed_count = 0

    for item in items:
        custom_id = item.get("custom_id", "unknown")

        # ── Look up metadata ──────────────────────────────────────────────
        meta = metadata.get(custom_id)
        if meta is None:
            print(f"[warn] No metadata for custom_id '{custom_id}' — skipping.")
            failed_count += 1
            continue

        date_str = meta["date"]

        # ── Check for item-level API error ────────────────────────────────
        if item.get("error"):
            print(f"[fail] {date_str}/{meta['label']!r}: batch item error -> {item['error']}")
            failed_count += 1
            continue

        response = item.get("response")
        if response is None or response.get("status_code") != 200:
            code = response.get("status_code") if response else "null"
            print(f"[fail] {date_str}/{meta['label']!r}: HTTP {code}")
            failed_count += 1
            continue

        # ── Extract LLM answer ────────────────────────────────────────────
        try:
            answer = response["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            print(f"[fail] {date_str}/{meta['label']!r}: unexpected response structure -> {exc}")
            failed_count += 1
            continue

        # ── Sector cross-check (local, no API call) ───────────────────────
        sectors = _sector_crosscheck(meta["label"])

        spikes_by_date[date_str].append({
            "topic_id":      meta["topic_id"],
            "label":         meta["label"],
            "spike_ratio":   meta["spike_ratio"],
            "article_count": meta["article_count"],
            "rag_answer":    answer,
            "rag_sources":   meta["sources"][:4],  # cap at 4 like daily_briefing.py
            "sectors":       sectors,
        })
        print(f"[ok]   {date_str} / {meta['label']!r}")
        ok_count += 1

    return dict(spikes_by_date), ok_count, failed_count


# ── Writing briefing JSONs ────────────────────────────────────────────────────

def save_briefings(spikes_by_date: dict[str, list[dict]]) -> int:
    """Write or merge data/briefings/{date}.json for each date.

    If a briefing JSON already exists for a date (e.g. from a partial run),
    it is overwritten with the freshly collected spikes.

    Returns the number of files written.
    """
    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    for date_str, spikes in sorted(spikes_by_date.items()):
        briefing = {
            "date":     date_str,
            "n_spikes": len(spikes),
            "spikes":   spikes,
        }
        out_path = BRIEFINGS_DIR / f"{date_str}.json"
        out_path.write_text(json.dumps(briefing, indent=2, default=str), encoding="utf-8")
        print(f"  Wrote {out_path}  ({len(spikes)} spike(s))")
        written += 1

    return written


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Briefing Batch Collection ===")

    # 1. Read sentinel files
    batch_id = read_pending_batch_id()
    metadata = read_spike_metadata()
    print(f"Pending batch : {batch_id}")
    print(f"Spike metadata: {len(metadata)} tasks loaded from {BRIEFINGS_BATCH_META_FILE}")

    # 2. Check status — exits with 2 if still in progress
    client = OpenAI()
    check_status(client, batch_id)

    # 3. Download results
    print("Downloading batch results...")
    items = download_batch_results(client, batch_id)
    print(f"Downloaded {len(items)} item(s).")

    # 4. Parse and group by date
    spikes_by_date, ok_count, failed_count = collect_results(items, metadata)

    # 5. Write briefing JSONs
    print(f"\nWriting briefing files...")
    written = save_briefings(spikes_by_date)

    # 6. Summary
    print(
        f"\nCollection complete: {ok_count} ok, {failed_count} failed, "
        f"{written} briefing file(s) written."
    )

    # 7. Clear sentinel files only on full success
    if failed_count == 0:
        PENDING_BRIEFINGS_BATCH_FILE.unlink()
        BRIEFINGS_BATCH_META_FILE.unlink()
        print(f"Cleared sentinel files.")
    else:
        print(
            f"[warn] {failed_count} item(s) failed — keeping sentinel files for inspection.\n"
            "Manually delete them after investigating the failures above."
        )


if __name__ == "__main__":
    main()
