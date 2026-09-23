"""
retrieve_batch_tracks.py
Collect a completed track-metadata batch and merge it into
data/track_metadata.jsonl (+ regenerated data/track_metadata.csv).

Architecture:
  1. Read batch ID + sidecar written by create_batch_tracks.py
  2. check_batch_job -> exit 2 while in progress, exit 1 on failed/expired/cancelled
  3. download_batch_results -> parse_result_item per item (per-item failures logged)
  4. merge_records (first-write-wins) -> write_records (atomic)
  5. Clear sentinels

Exit codes (same contract as the sector/briefing retrieve scripts, for CI retry loops):
  0 — collected (possibly with per-item failures; see log)
  1 — no pending batch, or batch ended failed/expired/cancelled
  2 — batch still in progress; safe to retry later

Sentinel policy: sentinels are cleared after collection even when some items
failed, and after a terminal batch failure. Failed tracks are absent from
track_metadata.jsonl, so the next create_batch_tracks.py run resubmits them.

Debugging:
  - Each failed item prints "[fail] <custom_id> (<artist> - <title>): <reason>"
  - data/batch_tasks_tracks.jsonl holds the exact request that was sent
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kitai.batch import check_batch_job, download_batch_results

from pipeline import track_metadata as tm
from pipeline.constants import PENDING_TRACKS_BATCH_FILE, TRACKS_BATCH_META_FILE


def _clear_sentinels() -> None:
    PENDING_TRACKS_BATCH_FILE.unlink(missing_ok=True)
    TRACKS_BATCH_META_FILE.unlink(missing_ok=True)
    print("Cleared pending sentinel files.")


def _load_sidecar() -> tuple[str, str | None, dict[str, tm.TrackInput]]:
    """(model, output_tag, inputs). The sidecar is the single source of truth for
    which store a batch belongs to; older sidecars without output_tag -> main store."""
    meta = json.loads(TRACKS_BATCH_META_FILE.read_text(encoding="utf-8"))
    inputs = {
        cid: tm.TrackInput(v["track_id"], v["artist"], v["title"], v["play_count"])
        for cid, v in meta["tracks"].items()
    }
    return meta.get("model", tm.TRACK_METADATA_MODEL), meta.get("output_tag"), inputs


def main(client=None) -> int:
    print("=== Track Metadata Batch Collection ===")

    if not PENDING_TRACKS_BATCH_FILE.exists() or not TRACKS_BATCH_META_FILE.exists():
        print(f"[error] Missing {PENDING_TRACKS_BATCH_FILE} or {TRACKS_BATCH_META_FILE}.\n"
              "Run create_batch_tracks.py first.")
        return 1

    batch_id = PENDING_TRACKS_BATCH_FILE.read_text(encoding="utf-8").strip()
    model, output_tag, inputs = _load_sidecar()
    store, store_csv = tm.store_paths(output_tag)
    print(f"Pending batch: {batch_id} ({len(inputs)} tracks, model {model}) -> {store}")

    if client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI()

    status = check_batch_job(client, batch_id)
    counts = status["counts"]
    print(f"Status: {status['status']} | total: {counts['total']} | "
          f"completed: {counts['completed']} | failed: {counts['failed']}")

    if not status["is_terminal"]:
        print("Batch still in progress. Re-run later.")
        return 2
    if not status["is_complete"]:
        print(f"[error] Batch ended with status '{status['status']}'. No results. "
              "Tracks will be resubmitted on the next create run.")
        _clear_sentinels()
        return 1

    items = download_batch_results(client, batch_id)
    print(f"Downloaded {len(items)} item(s).")

    records, failed = [], 0
    for item in items:
        try:
            records.append(tm.parse_result_item(item, inputs, batch_id=batch_id, model=model))
        except tm.TrackResultError as exc:
            cid = item.get("custom_id", "<missing>")
            t = inputs.get(cid)
            label = f"{t.artist} - {t.title}" if t else "?"
            print(f"[fail] {cid} ({label}): {exc}")
            failed += 1

    missing = len(inputs) - len(items)
    existing = tm.load_records(store)
    merged, added, dupes = tm.merge_records(existing, records)
    tm.write_records(merged, store, store_csv)

    identified = sum(1 for r in records if r["metadata"]["identified"])
    print(f"\nCollected: {len(records)} ok ({identified} identified), {failed} failed, "
          f"{missing} missing from output, {dupes} duplicate(s) skipped.")
    print(f"Added {added} record(s) -> {store} (total {len(merged)}); CSV view -> {store_csv}")
    if failed or missing:
        print(f"[warn] {failed + missing} track(s) not collected; they will be resubmitted next run.")

    _clear_sentinels()
    return 0


if __name__ == "__main__":
    sys.exit(main())
