"""
enrich_tracks_direct.py
Synchronous alternative to the track-metadata Batch API pipeline: calls
/v1/chat/completions directly, one request per track, and merges results into
the same data/track_metadata.jsonl store (+ CSV view).

Use it for quick tests / small top-ups (results in seconds, full price). Use
batch/create_batch_tracks.py + retrieve_batch_tracks.py for bulk runs (50% cheaper,
up to 24h latency).

Single source of truth: request bodies come from pipeline.track_metadata.build_task
and responses are validated by pipeline.track_metadata.parse_result_item — the
completion is wrapped in the Batch API result-item shape so both paths share
one parser. Records carry batch_id="direct".

Architecture:
  1. Load tracks; drop already-enriched ones and (default) ones in the pending batch
  2. Call chat.completions.create(**task["body"]) with a small thread pool
  3. Parse every response; per-track failures are logged, not raised
  4. merge_records (first-write-wins) + write_records (atomic)

Usage:
    python enrich/enrich_tracks_direct.py --limit 10
    python enrich/enrich_tracks_direct.py --include-pending   # also redo tracks in the pending batch
    python enrich/enrich_tracks_direct.py --refresh --limit 20  # re-ask + overwrite existing records
    python enrich/enrich_tracks_direct.py --model gpt-5 --output-tag gpt-5 --limit 20  # side-by-side run
    python enrich/enrich_tracks_direct.py --workers 8 --model gpt-4.1

Exit codes: 0 = done (possibly partial; see log) or nothing to do,
            1 = every attempted track failed (auth / model / schema problem).

Failure modes:
  - OPENAI_API_KEY missing: every call fails -> exit 1 with the error per track
  - HTTP 429: the OpenAI client retries (max_retries=2); lower --workers if persistent
  - Crash mid-run: nothing is written (records are written once at the end);
    re-run is safe because nothing was recorded
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import track_metadata as tm
from pipeline.openai_results import call_chat_completions
from pipeline.constants import TRACKS_BATCH_META_FILE, TRACKS_INPUT_FILE

DIRECT_BATCH_ID = "direct"
DEFAULT_WORKERS = 4


def _pending_batch_ids() -> set[str]:
    if not TRACKS_BATCH_META_FILE.exists():
        return set()
    meta = json.loads(TRACKS_BATCH_META_FILE.read_text(encoding="utf-8"))
    return {v["track_id"] for v in meta["tracks"].values()}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich tracks via direct OpenAI calls (no Batch API)")
    p.add_argument("--limit", type=int, default=None, help="Max tracks this run (default: all remaining)")
    p.add_argument("--model", default=tm.TRACK_METADATA_MODEL,
                   help=f"OpenAI chat model (default: {tm.TRACK_METADATA_MODEL})")
    p.add_argument("--input", type=Path, default=TRACKS_INPUT_FILE,
                   help=f"Track list CSV (default: {TRACKS_INPUT_FILE})")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Parallel requests (default: {DEFAULT_WORKERS})")
    p.add_argument("--include-pending", action="store_true",
                   help="Also process tracks already submitted in the pending batch")
    p.add_argument("--output-tag", default=None,
                   help="Write to a separate store data/track_metadata_<tag>.{jsonl,csv} "
                        "(e.g. a candidate model) instead of the default store")
    p.add_argument("--refresh", action="store_true",
                   help="Re-ask for tracks already in the store and overwrite them on success")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, client=None) -> int:
    args = _parse_args(argv)
    print("=== Track Metadata Direct Enrichment ===")

    jsonl_path, csv_path = tm.store_paths(args.output_tag)
    tracks = tm.load_tracks(args.input)
    existing = tm.load_records(jsonl_path)
    excluded = set(existing)
    pending = set() if args.include_pending else _pending_batch_ids()
    excluded |= pending
    if args.refresh:
        todo = tm.select_tracks_to_submit(
            [t for t in tracks if t.track_id not in pending], set(), limit=args.limit, refresh=True)
    else:
        todo = tm.select_tracks_to_submit(tracks, excluded, limit=args.limit)

    print(f"Output store           : {jsonl_path}")
    print(f"Unique tracks          : {len(tracks)}")
    print(f"Already enriched       : {len(set(existing) & {t.track_id for t in tracks})}")
    print(f"Skipped (pending batch): {len(pending)}")
    print(f"To process this run    : {len(todo)} (model {args.model}, {args.workers} workers"
          f"{', REFRESH: overwrite on success' if args.refresh else ''})")
    if not todo:
        print("Nothing to do.")
        return 0

    if client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI()

    bodies = {tm.custom_id_for(t): tm.build_task(t, model=args.model)["body"] for t in todo}
    items = call_chat_completions(client, bodies, workers=args.workers)

    inputs = {tm.custom_id_for(t): t for t in todo}
    records, failed = [], 0
    for item in items:
        t = inputs[item["custom_id"]]
        try:
            rec = tm.parse_result_item(item, inputs, batch_id=DIRECT_BATCH_ID, model=args.model)
            records.append(rec)
            m = rec["metadata"]
            print(f"[ok]   {t.artist} - {t.title} -> {', '.join(m['artists'])} - {m['title']} "
                  f"| {m['record_label']} {m['release_year']} | {m['primary_genre']} | {m['confidence']} "
                  f"| basis: {m['description_basis']}")
            if m["review_blurb"]:
                print(f"       \"{m['review_blurb']}\"")
        except tm.TrackResultError as exc:
            print(f"[fail] {item['custom_id']} ({t.artist} - {t.title}): {exc}")
            failed += 1

    # refresh: a failed re-ask keeps the old record (only successes overwrite)
    merged, added, dupes = tm.merge_records(existing, records, overwrite=args.refresh)
    if added:
        tm.write_records(merged, jsonl_path, csv_path)

    print(f"\nDone: {len(records)} ok, {failed} failed, {dupes} duplicate(s) skipped. "
          f"Added {added} -> {jsonl_path} (total {len(merged)}).")
    return 1 if not records else 0


if __name__ == "__main__":
    sys.exit(main())
