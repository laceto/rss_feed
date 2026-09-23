"""
artist_profiles.py
One structured profile per unique artist, shared by all their tracks. Pure logic:
no network calls, no module-level side effects (enrich/enrich_artists.py owns I/O).

Why a separate stage: asking for artist background inside every track request
repeats the same question per track (cost) and lets the answers drift between
tracks of the same artist (consistency). Here each artist is asked exactly once.

Data flow:
  track store (data/track_metadata[_<tag>].jsonl) -- cleaned names, not raw tags
    -> collect_artists()     unique names from artists + remixers + featured_artists,
                              with track_count and up to MAX_EXAMPLE_TITLES titles
    -> build_artist_task()   one chat request per artist (titles disambiguate
                              homonyms, e.g. 'Luciano' the DJ vs the singer)
    -> parse_artist_result() validated + enforce_no_inference()
    -> data/artist_profiles[_<tag>].{jsonl,csv}

Invariants:
  - artist_id = sha1(normalised name)[:12]; join key back to tracks is the
    normalised artist name (make_artist_id(name) for any name in a track record).
  - No inference: known=False -> only `name` survives, short_bio = UNKNOWN_ARTIST_NOTE.
  - Placeholder names (VA, unknown, ...) never become artists.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .constants import ARTIST_PROFILES_CSV, ARTIST_PROFILES_FILE
from .openai_results import ResultError, validate_result_item
from .openai_schema import make_openai_strict
from .track_metadata import (
    TRACK_METADATA_MODEL,
    atomic_write,
    cell,
    is_reasoning_model,
    load_records as _load_records,
    merge_records as _merge_records,
    tagged_paths,
)

CUSTOM_ID_PREFIX = "artist-"
MAX_EXAMPLE_TITLES = 5
PLACEHOLDER_NAMES = frozenset({"", "va", "various", "various artists", "(unknown)", "unknown"})
UNKNOWN_ARTIST_NOTE = "Artist not known: no reliable information available."


class ArtistProfile(BaseModel):
    """Background on one artist / act."""

    known: bool = Field(..., description="True only if you actually know this artist.")
    confidence: Literal["high", "medium", "low"] = Field(..., description="Confidence in the facts below.")
    name: str = Field(..., description="Canonical artist / act name.")
    real_name: str | None = Field(..., description="Legal name if publicly known, else null.")
    aliases: list[str] = Field(..., description="Other names the artist releases under.")
    country: str | None = Field(..., description="Country of origin, e.g. 'Germany'.")
    city: str | None = Field(..., description="Home city / scene, e.g. 'Berlin'.")
    active_since: int | None = Field(..., description="Year of first release or career start.")
    associated_labels: list[str] = Field(..., description="Labels the artist is known for (own or frequent).")
    associated_acts: list[str] = Field(..., description="Groups, duos or frequent collaborators.")
    styles: list[str] = Field(..., description="Styles the artist is known for, e.g. 'minimal techno'.")
    notable_releases: list[str] = Field(..., description="Up to 5 well-known tracks or releases.")
    short_bio: str | None = Field(..., description="1-3 factual sentences.")


STRICT_SCHEMA: dict = make_openai_strict(ArtistProfile.model_json_schema())

SYSTEM_PROMPT = """\
You are a music librarian specialised in minimal techno, tech house and house
(roughly 1990-present).

You receive ONE artist as JSON {"artist": ..., "example_tracks_in_library": [...]}.
The example tracks come from a DJ's minimal / tech house / house library; use them
only to tell apart artists with the same name (e.g. the Chilean-Swiss DJ Luciano vs
the Italian singer) -- they are not facts to repeat back.

RULES:
- Output ONLY JSON matching the schema.
- Do not infer. Use only what you actually know about THIS artist. If unsure of a
  field, use null (or [] for lists). Prefer null over a guess for real name, city, year.
- If you do not know the artist: known=false, confidence="low", keep the name,
  every other field null / [], and short_bio "Artist not known."
"""


@dataclass(frozen=True)
class ArtistInput:
    artist_id: str
    name: str
    track_count: int
    example_titles: list[str] = field(default_factory=list)


def _normalise(value: str) -> str:
    return " ".join(value.split()).lower()


def make_artist_id(name: str) -> str:
    return hashlib.sha1(_normalise(name).encode("utf-8")).hexdigest()[:12]


def custom_id_for(artist: ArtistInput) -> str:
    return f"{CUSTOM_ID_PREFIX}{artist.artist_id}"


def store_paths(tag: str | None) -> tuple[Path, Path]:
    return tagged_paths(ARTIST_PROFILES_FILE, ARTIST_PROFILES_CSV, tag)


def collect_artists(track_records: dict[str, dict]) -> list[ArtistInput]:
    """Unique artists across artists/remixers/featured_artists of all track records.

    First spelling wins for the display name. Sorted by track_count desc, then name,
    so `--limit N` picks the artists that matter most for the library.
    """
    names: dict[str, str] = {}
    counts: dict[str, int] = {}
    titles: dict[str, list[str]] = {}
    for rec in track_records.values():
        meta = rec.get("metadata") or {}
        people = (meta.get("artists") or []) + (meta.get("remixers") or []) + (meta.get("featured_artists") or [])
        seen_here: set[str] = set()
        for name in people:
            name = (name or "").strip()
            if _normalise(name) in PLACEHOLDER_NAMES:
                continue
            aid = make_artist_id(name)
            if aid in seen_here:
                continue
            seen_here.add(aid)
            names.setdefault(aid, name)
            counts[aid] = counts.get(aid, 0) + 1
            title = meta.get("title") or rec.get("input_title")
            bucket = titles.setdefault(aid, [])
            if title and title not in bucket and len(bucket) < MAX_EXAMPLE_TITLES:
                bucket.append(title)
    artists = [ArtistInput(aid, names[aid], counts[aid], titles.get(aid, [])) for aid in names]
    return sorted(artists, key=lambda a: (-a.track_count, a.name.lower()))


def build_artist_task(artist: ArtistInput, model: str = TRACK_METADATA_MODEL) -> dict:
    body = {
        "model": model,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "artist_profile", "schema": STRICT_SCHEMA, "strict": True},
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"artist": artist.name, "example_tracks_in_library": artist.example_titles},
                ensure_ascii=False)},
        ],
    }
    if not is_reasoning_model(model):
        body["temperature"] = 0
    return {"custom_id": custom_id_for(artist), "method": "POST",
            "url": "/v1/chat/completions", "body": body}


_CLEARED_NULL = ("real_name", "country", "city", "active_since")
_CLEARED_LIST = ("aliases", "associated_labels", "associated_acts", "styles", "notable_releases")


def enforce_no_inference(profile: ArtistProfile) -> ArtistProfile:
    """known=False -> only the name survives; the bio says the artist is not known."""
    if profile.known:
        return profile
    update: dict = {f: None for f in _CLEARED_NULL}
    update.update({f: [] for f in _CLEARED_LIST})
    update.update(confidence="low", short_bio=UNKNOWN_ARTIST_NOTE)
    return profile.model_copy(update=update)


def parse_artist_result(item: dict, inputs: dict[str, ArtistInput], source: str, model: str) -> dict:
    """Validated record {artist_id, name, track_count, source, model, profile}.

    Raises:
        ResultError: unknown custom_id, or any validate_result_item() failure.
    """
    cid = item.get("custom_id", "<missing>")
    artist = inputs.get(cid)
    if artist is None:
        raise ResultError(f"unknown custom_id {cid!r}")
    profile = enforce_no_inference(validate_result_item(item, ArtistProfile))
    return {"artist_id": artist.artist_id, "name": artist.name, "track_count": artist.track_count,
            "source": source, "model": model, "profile": profile.model_dump()}


def load_records(path: Path) -> dict[str, dict]:
    return _load_records(path, key="artist_id")


def merge_records(existing: dict[str, dict], new, overwrite: bool = False):
    return _merge_records(existing, new, overwrite=overwrite, key="artist_id")


_RECORD_COLUMNS = ["artist_id", "name", "track_count", "source", "model"]
_PROFILE_COLUMNS = [f for f in ArtistProfile.model_fields if f != "name"]


def write_records(records: dict[str, dict], jsonl_path: Path, csv_path: Path) -> None:
    """Atomically write the JSONL store (sorted by artist_id) and its flat CSV view."""
    ordered = [records[k] for k in sorted(records)]

    def write_jsonl(f):
        for rec in ordered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_csv(f):
        w = csv.writer(f)
        w.writerow(_RECORD_COLUMNS + _PROFILE_COLUMNS)
        for rec in ordered:
            prof = rec.get("profile") or {}
            w.writerow([cell(rec.get(c)) for c in _RECORD_COLUMNS]
                       + [cell(prof.get(c)) for c in _PROFILE_COLUMNS])

    atomic_write(Path(jsonl_path), write_jsonl)
    atomic_write(Path(csv_path), write_csv)
