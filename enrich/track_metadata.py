"""
track_metadata.py

Look up structured metadata (artist, title, mix, label, year, genre, ...) for
tracks in data/unique_tracks.csv using webai.TrackResearcher (Tavily search +
OpenAI structured output).

Input  : UNIQUE_TRACKS_FILE   (columns: artist, title, count)
Output : TRACK_METADATA_FILE  (one row per input track; input_artist/input_title
         identify the source row, remaining columns come from webai.TrackInfo)

Idempotent: rows already present in the output with a non-error status are
skipped; errored rows are retried and replaced.

Usage:
    PYTHONPATH=. python enrich/track_metadata.py                     # first 10 rows
    PYTHONPATH=. python enrich/track_metadata.py --limit 25 --offset 10
    PYTHONPATH=. python enrich/track_metadata.py --dry-run           # no API calls

Env:
    TAVILY_API_KEY  (required)
    OPENAI_API_KEY  (required)
    OPENAI_MODEL    (optional, default gpt-4o-mini)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.constants import (  # noqa: E402
    TRACK_METADATA_FILE,
    TRACK_METADATA_LIMIT,
    UNIQUE_TRACKS_FILE,
)

logger = logging.getLogger(__name__)

KEY_COLS = ["input_artist", "input_title"]
LIST_SEP = "; "


def load_tracks(path: Path, limit: int, offset: int = 0) -> pd.DataFrame:
    """
    Return rows [offset, offset + limit) of the unique-tracks CSV.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if the ``artist`` / ``title`` columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Track list not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"artist", "title"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df.iloc[offset : offset + limit].reset_index(drop=True)


def _flatten(info) -> dict:
    """TrackInfo -> flat dict; list fields joined so they fit in one CSV cell."""
    record = info.model_dump()
    for key, value in record.items():
        if isinstance(value, list):
            record[key] = LIST_SEP.join(str(v) for v in value)
    return record


def enrich_tracks(
    tracks: pd.DataFrame, researcher, existing: pd.DataFrame | None
) -> pd.DataFrame:
    """
    Research every track not already enriched and merge with *existing*.

    Rows in *existing* with ``status == "error"`` are retried and replaced.

    Returns:
        Combined DataFrame (existing kept rows first, then new rows).
    """
    if existing is None or existing.empty:
        kept = pd.DataFrame(columns=KEY_COLS)
    else:
        kept = existing[existing["status"] != "error"]
    done = set(zip(kept["input_artist"], kept["input_title"]))

    new_rows: list[dict] = []
    for artist, title in zip(tracks["artist"], tracks["title"]):
        if (artist, title) in done:
            print(f"  skip (already enriched) -> {artist} | {title}")
            continue
        info = researcher.research_track(artist, title)
        print(f"  {info.status:<9} -> {artist} | {title}")
        new_rows.append({"input_artist": artist, "input_title": title, **_flatten(info)})
        done.add((artist, title))

    frames = [f for f in (kept, pd.DataFrame(new_rows)) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=KEY_COLS)
    out = pd.concat(frames, ignore_index=True)
    if "release_year" in out:
        # keep years as integers (2004, not 2004.0) despite missing values
        out["release_year"] = pd.to_numeric(out["release_year"], errors="coerce").astype("Int64")
    return out


def save_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to *path* via a .tmp file + os.replace()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _build_researcher():
    from langchain_openai import ChatOpenAI
    from webai import TrackResearcher

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return TrackResearcher(model=model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=UNIQUE_TRACKS_FILE)
    parser.add_argument("--output", type=Path, default=TRACK_METADATA_FILE)
    parser.add_argument("--limit", type=int, default=TRACK_METADATA_LIMIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print inputs, no API calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tracks = load_tracks(args.input, limit=args.limit, offset=args.offset)
    print(f"Loaded {len(tracks)} tracks from {args.input} (offset={args.offset})")

    if args.dry_run:
        for artist, title in zip(tracks["artist"], tracks["title"]):
            print(f"  {artist} | {title}")
        return 0

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    existing = pd.read_csv(args.output, dtype=str, keep_default_na=False) if args.output.exists() else None
    result = enrich_tracks(tracks, _build_researcher(), existing)
    save_atomic(result, args.output)
    print(f"Wrote {len(result)} rows -> {args.output}")

    n_err = int((result["status"] == "error").sum()) if "status" in result else 0
    if n_err:
        print(f"WARNING: {n_err} track(s) failed; they will be retried on the next run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
