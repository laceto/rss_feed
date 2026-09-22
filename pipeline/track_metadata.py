"""
track_metadata.py
Pure logic for enriching a DJ track list with structured metadata via the
OpenAI Batch API. No network calls, no module-level side effects: the
batch/create_batch_tracks.py and batch/retrieve_batch_tracks.py CLIs own I/O
with OpenAI (through kitai.batch).

Data flow:
  data/unique_tracks.csv (artist,title,count)
    -> load_tracks()             TrackInput per unique normalised (artist, title)
    -> select_tracks_to_submit() drop tracks already in data/track_metadata.jsonl
    -> build_task()              one /v1/chat/completions task per track
    ... OpenAI Batch API ...
    -> parse_result_item()       validated record dict (raises TrackResultError)
    -> merge_records() + write_records()

Invariants:
  - track_id = sha1(normalised "artist|title")[:12]; stable across runs, so it
    is the idempotency key. custom_id = "track-{track_id}".
  - Rows whose artist AND title are both unknown placeholders are skipped —
    there is nothing for the model to identify.
  - Raw artist/title strings are sent to the model unchanged; cleaning messy
    tags (artist in title field, vinyl side prefixes, mojibake) is the model's
    job and is reported back in `parsing_notes`.
  - merge_records() is first-write-wins: an existing track_id is never
    overwritten. Delete its line from the JSONL to force re-enrichment.
  - write_records() writes atomically (.tmp + os.replace) and regenerates the
    CSV view from the full record set.

Failure modes (all raised as TrackResultError by parse_result_item):
  - unknown custom_id (sidecar mismatch), item-level error, non-200 HTTP,
    model refusal, truncated output (finish_reason=length), JSON/schema mismatch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field, ValidationError

from .openai_schema import make_openai_strict

# ── Configuration ─────────────────────────────────────────────────────────────

# gpt-4.1-mini: materially better recall of underground electronic releases
# than nano, still cheap at batch pricing (~650 short requests).
TRACK_METADATA_MODEL = "gpt-4.1-mini"
CUSTOM_ID_PREFIX = "track-"
UNKNOWN_TOKENS = frozenset({"", "(unknown)", "unknown"})


# ── Structured output schema ──────────────────────────────────────────────────

Genre = Literal[
    "minimal techno", "techno", "tech house", "house", "deep house",
    "minimal house", "microhouse", "progressive house", "acid house",
    "electro", "other",
]


class ArtistInfo(BaseModel):
    """Background on one primary artist of the track."""

    name: str = Field(..., description="Canonical artist / act name.")
    real_name: str | None = Field(..., description="Legal name if publicly known, else null.")
    aliases: list[str] = Field(..., description="Other names the artist releases under.")
    country: str | None = Field(..., description="Country of origin, e.g. 'Germany'.")
    city: str | None = Field(..., description="Home city / scene, e.g. 'Berlin'.")
    active_since: int | None = Field(..., description="Year of first release or career start.")
    associated_labels: list[str] = Field(..., description="Labels the artist is known for (own or frequent).")
    associated_acts: list[str] = Field(..., description="Groups, duos or frequent collaborators.")
    short_bio: str | None = Field(..., description="1-2 factual sentences; null if unknown.")


class TrackMetadata(BaseModel):
    """Structured metadata for a single track in a minimal / tech house / house library."""

    identified: bool = Field(..., description="True only if you recognise this specific track.")
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in the identification and in the facts below.")
    artists: list[str] = Field(..., description="Cleaned primary artist name(s), canonical spelling.")
    title: str | None = Field(..., description="Cleaned track title without mix name.")
    mix_name: str | None = Field(..., description="Version, e.g. 'Original Mix', 'Konrad Black Remix'.")
    remixers: list[str] = Field(..., description="Remixer(s) if this is a remix/edit.")
    featured_artists: list[str] = Field(..., description="Featured / vocal artists.")
    record_label: str | None = Field(..., description="Label of the original release.")
    catalog_number: str | None = Field(..., description="Catalog number, e.g. 'PLAYHOUSE 133'.")
    release_title: str | None = Field(..., description="EP / album / compilation the track appeared on.")
    release_type: Literal["single", "EP", "album", "compilation", "unknown"] = Field(
        ..., description="Type of the original release.")
    release_year: int | None = Field(..., description="Year of original release.")
    formats: list[Literal["vinyl", "digital", "CD", "cassette"]] = Field(
        ..., description="Known release formats.")
    primary_genre: Genre = Field(..., description="Single best-fit genre.")
    subgenres: list[str] = Field(..., description="Finer style tags, e.g. 'Romanian minimal', 'loop techno'.")
    bpm_estimate: int | None = Field(..., description="Typical tempo in BPM; null if no reasonable basis.")
    musical_key: str | None = Field(..., description="Key (e.g. 'A minor' or Camelot '8A') if known.")
    energy: Literal["low", "medium", "high"] | None = Field(..., description="Dance-floor energy.")
    mood_tags: list[str] = Field(..., description="Mood descriptors, e.g. 'hypnotic', 'dark', 'groovy'.")
    dj_set_role: Literal["warm-up", "peak-time", "closing", "after-hours", "any"] | None = Field(
        ..., description="Where the track typically fits in a DJ set.")
    instrumentation: list[str] = Field(
        ..., description="Salient sound elements, e.g. '303 acid line', 'vocal sample', 'organ stabs'.")
    similar_artists: list[str] = Field(..., description="Up to 5 stylistically similar artists.")
    description: str | None = Field(..., description="1-3 factual sentences about the track.")
    artist_details: list[ArtistInfo] = Field(..., description="One entry per name in `artists`.")
    parsing_notes: str | None = Field(
        ..., description="How the raw tag was interpreted (e.g. 'artist was in title field; removed side A1').")


STRICT_SCHEMA: dict = make_openai_strict(TrackMetadata.model_json_schema())

SYSTEM_PROMPT = """\
You are a music librarian and DJ specialised in minimal techno, tech house and house
(roughly 1990-present: labels like Perlon, Minus, Cocoon, Poker Flat, Playhouse,
Get Physical, Desolat, Cadenza, Kompakt, Trax, Relief, Defected).

You receive ONE track as JSON {"artist": ..., "title": ...} taken from messy DJ
software / ID3 tags. Typical problems you must handle:
- artist is "(unknown)", "VA" or empty and the real "Artist - Title" is in the title field
- vinyl side prefixes ("A1", "B2"), catalog numbers, file names with underscores,
  bitrate suffixes ("-320"), truncated titles, dots instead of spaces
- mojibake / broken encoding (e.g. "ÃÂme" is the artist "Âme")

RULES:
- Output ONLY JSON matching the schema.
- Never invent facts. If you are not reasonably sure of a field, use null
  (or [] for lists). Prefer null over a guess for label, catalog number, year, key.
- Set identified=false and confidence="low" when you do not recognise the specific track;
  still return the cleaned artist/title you can infer from the raw tag.
- artist_details must contain one entry per name in `artists`.
- bpm_estimate may be a genre-typical estimate only when identified=true.
- Explain in parsing_notes how you interpreted the raw tag when it needed cleaning.
"""


# ── Inputs ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrackInput:
    track_id: str
    artist: str
    title: str
    play_count: int


class TrackResultError(Exception):
    """A single batch result item could not be turned into a valid record."""


def _normalise(value: str) -> str:
    return " ".join(value.split()).lower()


def make_track_id(artist: str, title: str) -> str:
    """Stable 12-hex-char id from case/whitespace-normalised artist + title."""
    key = f"{_normalise(artist)}|{_normalise(title)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def custom_id_for(track: TrackInput) -> str:
    return f"{CUSTOM_ID_PREFIX}{track.track_id}"


def _is_unknown(value: str) -> bool:
    return _normalise(value) in UNKNOWN_TOKENS


def load_tracks(path: Path) -> list[TrackInput]:
    """Read artist,title,count CSV -> unique TrackInputs in file order.

    Duplicate normalised rows are merged (play counts summed, first spelling kept).

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: CSV lacks an 'artist' or 'title' column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Track list not found: {path}")

    tracks: dict[str, TrackInput] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"artist", "title"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)} (need artist,title)")

        for row in reader:
            artist = (row.get("artist") or "").strip()
            title = (row.get("title") or "").strip()
            if _is_unknown(artist) and _is_unknown(title):
                continue
            raw_count = (row.get("count") or "").strip()
            count = int(raw_count) if raw_count.isdigit() else 1

            tid = make_track_id(artist, title)
            if tid in tracks:
                prev = tracks[tid]
                tracks[tid] = TrackInput(tid, prev.artist, prev.title, prev.play_count + count)
            else:
                tracks[tid] = TrackInput(tid, artist, title, count)

    return list(tracks.values())


def select_tracks_to_submit(
    tracks: list[TrackInput], done_ids: set[str], limit: int | None = None,
) -> list[TrackInput]:
    """Tracks not yet enriched, capped at `limit` (None or 0 = no cap)."""
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    todo = [t for t in tracks if t.track_id not in done_ids]
    return todo[:limit] if limit else todo


# ── Batch tasks ───────────────────────────────────────────────────────────────

def build_task(track: TrackInput, model: str = TRACK_METADATA_MODEL) -> dict:
    """One OpenAI Batch API /v1/chat/completions task with strict structured output."""
    return {
        "custom_id": custom_id_for(track),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "track_metadata",
                    "schema": STRICT_SCHEMA,
                    "strict": True,
                },
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"artist": track.artist, "title": track.title}, ensure_ascii=False)},
            ],
        },
    }


# ── Results ───────────────────────────────────────────────────────────────────

def parse_result_item(
    item: dict, inputs: dict[str, TrackInput], batch_id: str, model: str,
) -> dict:
    """Validate one raw batch result item and return the output record.

    Record shape: {track_id, input_artist, input_title, play_count, batch_id,
    model, metadata: TrackMetadata-dict}.

    Raises:
        TrackResultError: see module docstring for the cases.
    """
    custom_id = item.get("custom_id", "<missing>")
    track = inputs.get(custom_id)
    if track is None:
        raise TrackResultError(f"unknown custom_id {custom_id!r} (not in sidecar metadata)")

    if item.get("error"):
        raise TrackResultError(f"batch item error: {item['error']}")

    response = item.get("response") or {}
    status = response.get("status_code")
    if status != 200:
        raise TrackResultError(f"HTTP {status}")

    try:
        choice = response["body"]["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TrackResultError(f"unexpected response structure: {exc!r}") from exc

    if message.get("refusal"):
        raise TrackResultError(f"model refused: {message['refusal']}")
    if choice.get("finish_reason") == "length":
        raise TrackResultError("output truncated (finish_reason=length)")

    try:
        metadata = TrackMetadata.model_validate_json(message.get("content") or "")
    except ValidationError as exc:
        raise TrackResultError(f"response does not match schema: {exc.errors()[:3]}") from exc

    return {
        "track_id": track.track_id,
        "input_artist": track.artist,
        "input_title": track.title,
        "play_count": track.play_count,
        "batch_id": batch_id,
        "model": model,
        "metadata": metadata.model_dump(),
    }


def load_records(path: Path) -> dict[str, dict]:
    """Read the JSONL store -> {track_id: record}. Missing file -> {}."""
    path = Path(path)
    if not path.exists():
        return {}
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[rec["track_id"]] = rec
    return records


def merge_records(
    existing: dict[str, dict], new: Iterable[dict],
) -> tuple[dict[str, dict], int, int]:
    """First-write-wins merge. Returns (merged, added_count, duplicate_count)."""
    merged = dict(existing)
    added = dupes = 0
    for rec in new:
        if rec["track_id"] in merged:
            dupes += 1
            continue
        merged[rec["track_id"]] = rec
        added += 1
    return merged, added, dupes


_RECORD_COLUMNS = ["track_id", "input_artist", "input_title", "play_count", "batch_id", "model"]
_METADATA_COLUMNS = list(TrackMetadata.model_fields)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return "; ".join(value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # per-file: .jsonl and .csv must not share one
    with tmp.open("w", encoding="utf-8", newline="") as f:
        write(f)
    os.replace(tmp, path)


def write_records(records: dict[str, dict], jsonl_path: Path, csv_path: Path) -> None:
    """Atomically write the JSONL store (sorted by track_id) and its flattened CSV view."""
    ordered = [records[k] for k in sorted(records)]

    def write_jsonl(f):
        for rec in ordered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_csv(f):
        writer = csv.writer(f)
        writer.writerow(_RECORD_COLUMNS + _METADATA_COLUMNS)
        for rec in ordered:
            meta = rec.get("metadata") or {}
            writer.writerow([_cell(rec.get(c)) for c in _RECORD_COLUMNS]
                            + [_cell(meta.get(c)) for c in _METADATA_COLUMNS])

    _atomic_write(Path(jsonl_path), write_jsonl)
    _atomic_write(Path(csv_path), write_csv)
