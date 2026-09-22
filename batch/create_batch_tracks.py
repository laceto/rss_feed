"""
create_batch_tracks.py
Submit an OpenAI Batch API job that enriches data/unique_tracks.csv with
structured track + artist metadata (label, year, genre, BPM, mood, bio, ...).

Architecture:
  1. Refuse if a tracks batch is already pending (one in flight at a time)
  2. Load unique tracks from TRACKS_INPUT_FILE; skip both-unknown rows
  3. Drop tracks whose track_id is already in TRACK_METADATA_FILE (idempotent)
  4. Build one strict-JSON-schema chat task per track (pipeline.track_metadata)
  5. Submit via kitai.batch.submit_batch_job
  6. Persist batch ID + sidecar {custom_id -> input track} for the retrieve step

Usage:
    python batch/create_batch_tracks.py               # all un-enriched tracks
    python batch/create_batch_tracks.py --limit 20    # smoke-test on 20 tracks
    python batch/create_batch_tracks.py --dry-run     # write debug JSONL only

Exit codes: 0 = submitted or nothing to do, 1 = a batch is already pending.

Debugging:
  - data/batch_tasks_tracks.jsonl  — exact tasks built (written on every run, incl. --dry-run)
  - data/pending_tracks_batch.txt  — active batch ID
  - data/pending_tracks_meta.json  — {batch_id, model, tracks: {custom_id: input}}

Failure modes:
  - OPENAI_API_KEY missing / invalid: openai.AuthenticationError from submit
  - HTTP 400 on every item: strict schema rejected -> see pipeline/openai_schema.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kitai.batch import submit_batch_job

from pipeline import track_metadata as tm
from pipeline.constants import (
    BATCH_FILE_TRACKS,
    PENDING_TRACKS_BATCH_FILE,
    TRACK_METADATA_FILE,
    TRACKS_BATCH_META_FILE,
    TRACKS_INPUT_FILE,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit track-metadata batch to OpenAI")
    p.add_argument("--limit", type=int, default=None,
                   help="Max tracks to submit this run (default: all remaining)")
    p.add_argument("--model", default=tm.TRACK_METADATA_MODEL,
                   help=f"OpenAI chat model (default: {tm.TRACK_METADATA_MODEL})")
    p.add_argument("--input", type=Path, default=TRACKS_INPUT_FILE,
                   help=f"Track list CSV (default: {TRACKS_INPUT_FILE})")
    p.add_argument("--dry-run", action="store_true",
                   help="Build tasks and write debug JSONL; do not submit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, client=None) -> int:
    args = _parse_args(argv)
    print("=== Track Metadata Batch Submission ===")

    if PENDING_TRACKS_BATCH_FILE.exists():
        pending = PENDING_TRACKS_BATCH_FILE.read_text(encoding="utf-8").strip()
        print(f"[error] Batch {pending} is still pending ({PENDING_TRACKS_BATCH_FILE}).\n"
              "Run retrieve_batch_tracks.py first.")
        return 1

    tracks = tm.load_tracks(args.input)
    done_ids = set(tm.load_records(TRACK_METADATA_FILE))
    todo = tm.select_tracks_to_submit(tracks, done_ids, limit=args.limit)

    print(f"Unique tracks in {args.input} : {len(tracks)}")
    print(f"Already enriched               : {len(done_ids & {t.track_id for t in tracks})}")
    print(f"To submit this run             : {len(todo)}")

    if not todo:
        print("Nothing to submit.")
        return 0

    tasks = [tm.build_task(t, model=args.model) for t in todo]

    BATCH_FILE_TRACKS.parent.mkdir(parents=True, exist_ok=True)
    with BATCH_FILE_TRACKS.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"Debug JSONL saved -> {BATCH_FILE_TRACKS}")

    if args.dry_run:
        print("--dry-run: not submitting.")
        return 0

    if client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI()

    print(f"Submitting {len(tasks)} task(s) with model {args.model} ...")
    batch_id = submit_batch_job(
        client, tasks, endpoint="/v1/chat/completions",
        metadata={"description": "track_metadata"},
    )
    print(f"Batch submitted: {batch_id}")

    PENDING_TRACKS_BATCH_FILE.write_text(batch_id, encoding="utf-8")
    TRACKS_BATCH_META_FILE.write_text(json.dumps({
        "batch_id": batch_id,
        "model": args.model,
        "tracks": {
            tm.custom_id_for(t): {"track_id": t.track_id, "artist": t.artist,
                                  "title": t.title, "play_count": t.play_count}
            for t in todo
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Batch ID saved  -> {PENDING_TRACKS_BATCH_FILE}")
    print(f"Sidecar saved   -> {TRACKS_BATCH_META_FILE}")
    print("Run retrieve_batch_tracks.py once the batch completes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
