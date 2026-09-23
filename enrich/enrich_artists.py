"""
enrich_artists.py
Build one profile per unique artist found in a track store, via direct OpenAI
chat completions (no Batch API: a library has a few hundred artists at most).

Reads cleaned names from data/track_metadata[_<tracks-tag>].jsonl, so run it
AFTER the track enrichment. Writes data/artist_profiles[_<output-tag>].{jsonl,csv}.

Usage:
    python enrich/enrich_artists.py --tracks-tag gpt-5 --output-tag gpt-5 --model gpt-5
    python enrich/enrich_artists.py --limit 10           # top-10 artists by track count
    python enrich/enrich_artists.py --refresh            # re-ask + overwrite on success

Exit codes: 0 = done (possibly partial; see log) or nothing to do,
            1 = track store missing, or every attempted artist failed.

Debugging: each artist prints "[ok] name | known/unknown | labels" or
"[fail] custom_id (name): reason". Nothing is written until all calls return.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import artist_profiles as ap
from pipeline import track_metadata as tm
from pipeline.openai_results import ResultError, call_chat_completions

DIRECT_SOURCE = "direct"


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Enrich unique artists from a track store")
    p.add_argument("--tracks-tag", default=None, help="Read data/track_metadata_<tag>.jsonl (default: main store)")
    p.add_argument("--output-tag", default=None, help="Write data/artist_profiles_<tag>.* (default: main store)")
    p.add_argument("--model", default=tm.TRACK_METADATA_MODEL,
                   help=f"OpenAI chat model (default: {tm.TRACK_METADATA_MODEL})")
    p.add_argument("--limit", type=int, default=None, help="Max artists this run (highest track count first)")
    p.add_argument("--workers", type=int, default=4, help="Parallel requests (default: 4)")
    p.add_argument("--refresh", action="store_true", help="Re-ask existing artists; overwrite on success")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, client=None) -> int:
    args = _parse_args(argv)
    print("=== Artist Profile Enrichment ===")

    tracks_path, _ = tm.store_paths(args.tracks_tag)
    out_jsonl, out_csv = ap.store_paths(args.output_tag)
    if not tracks_path.exists():
        print(f"[error] track store not found: {tracks_path} (run track enrichment first)")
        return 1

    artists = ap.collect_artists(tm.load_records(tracks_path))
    existing = ap.load_records(out_jsonl)
    todo = [a for a in artists if args.refresh or a.artist_id not in existing]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Track store       : {tracks_path}")
    print(f"Output store      : {out_jsonl}")
    print(f"Unique artists    : {len(artists)}")
    print(f"Already profiled  : {len(set(existing) & {a.artist_id for a in artists})}")
    print(f"To process        : {len(todo)} (model {args.model}{', REFRESH' if args.refresh else ''})")
    if not todo:
        print("Nothing to do.")
        return 0

    if client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI()

    inputs = {ap.custom_id_for(a): a for a in todo}
    bodies = {cid: ap.build_artist_task(a, model=args.model)["body"] for cid, a in inputs.items()}
    items = call_chat_completions(client, bodies, workers=args.workers)

    records, failed = [], 0
    for item in items:
        a = inputs[item["custom_id"]]
        try:
            rec = ap.parse_artist_result(item, inputs, source=DIRECT_SOURCE, model=args.model)
            records.append(rec)
            p = rec["profile"]
            print(f"[ok]   {a.name} ({a.track_count} tracks) | {'known' if p['known'] else 'unknown'} "
                  f"| {'; '.join(p['associated_labels'][:3])}")
        except ResultError as exc:
            print(f"[fail] {item['custom_id']} ({a.name}): {exc}")
            failed += 1

    merged, added, dupes = ap.merge_records(existing, records, overwrite=args.refresh)
    if added:
        ap.write_records(merged, out_jsonl, out_csv)
    known = sum(r["profile"]["known"] for r in records)
    print(f"\nDone: {len(records)} ok ({known} known), {failed} failed. "
          f"Wrote {added} -> {out_jsonl} (total {len(merged)}).")
    return 1 if not records else 0


if __name__ == "__main__":
    sys.exit(main())
