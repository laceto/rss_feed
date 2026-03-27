"""OpenAI Batch API — sector analysis task building and submission.

Extracted from create_batch_files_v2.py.

Includes the SectorAnalysis / MultiSectorAnalysis Pydantic models,
the make_openai_strict schema helper, and functions to build and submit
a daily sector-analysis batch job.

Module-level path constants and the OpenAI client have been converted to
explicit parameters so callers can inject their own paths and credentials.

Submission uses kitai.batch.submit_batch_job — the JSONL file upload and
batch creation steps are handled internally by kitai.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from kitai.batch import submit_batch_job
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic.json_schema import model_json_schema

from constants import SectorName

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_CHARS = 60_000  # ~15k tokens safety cap per date chunk

SYSTEM_PROMPT = """\
You are Ava, a sharp trader assistant. Analyze the FULL news feed and extract ALL relevant sectors.

RULES:
- Output ONLY valid JSON matching the schema provided.
- If no sector can be identified, return: {"sectors": []}
- Each sector must use a name from the enum exactly as written.
- Fill ALL required fields for each sector.
- No duplicate sectors. Maximum 8 sectors.
- No extra text outside JSON.
"""


# ── Pydantic models ─────────────────────────────────────────────────────────────


class SectorAnalysis(BaseModel):
    """Structured extraction for a single sector from the day's news."""

    entities: list[str] = Field(
        default_factory=list,
        description="Named companies, organizations, or notable figures mentioned.",
    )
    sector: SectorName = Field(..., description="Sector name from the fixed taxonomy.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ..., description="Overall market sentiment for this sector."
    )
    news_category: Literal[
        "earnings", "M&A", "regulation", "macro",
        "appointments", "products", "markets", "other",
    ] = Field(..., description="Primary type of news event driving this sector.")
    extraction_status: Literal["ok", "partial"] = Field(
        default="ok",
        description="'ok' if full extraction succeeded; 'partial' if evidence was thin.",
    )


class MultiSectorAnalysis(BaseModel):
    """Analysis of all relevant sectors found in a single day's news feed."""

    sectors: list[SectorAnalysis] = Field(
        ...,
        description=(
            "List of ALL relevant sectors (1-8 max). "
            "Only include sectors with clear evidence in the news."
        ),
    )


# ── Schema helpers ──────────────────────────────────────────────────────────────


def make_openai_strict(schema: dict) -> dict:
    """Recursively add additionalProperties:false and required[] for OpenAI strict mode.

    OpenAI strict JSON schema requires every object node to have:
      - additionalProperties: false
      - required: [all property names]
    Without this the Batch API returns a 400 error when strict:true is set.

    This replaces the private openai.lib._pydantic.to_strict_json_schema call.
    """
    schema = dict(schema)

    if "$defs" in schema:
        schema["$defs"] = {k: make_openai_strict(v) for k, v in schema["$defs"].items()}

    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
        schema["properties"] = {
            k: make_openai_strict(v) for k, v in schema["properties"].items()
        }

    if "items" in schema:
        schema["items"] = make_openai_strict(schema["items"])

    return schema


# Pre-computed strict schema — used by build_batch_tasks().
STRICT_SCHEMA: dict = make_openai_strict(model_json_schema(MultiSectorAnalysis))


# ── Data preparation ───────────────────────────────────────────────────────────


def build_daily_contents(df: pd.DataFrame, results_dir: Path) -> dict[str, str]:
    """Return {date: joined_text} for each unprocessed publication date.

    Each item is formatted as 'YYYY-MM-DD: Title: Description' for LLM context.
    Truncates to MAX_CHARS to prevent token-limit errors on large date groups.
    Skips dates that already have a result file in results_dir (incremental sentinel).

    Args:
        df:          DataFrame with columns pubDate, title, description.
        results_dir: Directory where per-date JSON results are written; used
                     to detect already-processed dates.

    Returns:
        Mapping of date string -> joined article text for all unprocessed dates.
    """
    daily_contents: dict[str, str] = {}

    for date, group in df.groupby("pubDate"):
        result_path = results_dir / f"{date}.json"
        if result_path.exists():
            print(f"[skip] {date} already processed -> {result_path}")  # NOTE: uses print()
            continue

        lines = (
            group["pubDate"]
            + ": "
            + group["title"].fillna("")
            + ": "
            + group["description"].fillna("")
        )
        joined = ". ".join(lines)

        if len(joined) > MAX_CHARS:
            print(f"[warn] {date}: truncating {len(joined):,} -> {MAX_CHARS:,} chars")  # NOTE: uses print()
            joined = joined[:MAX_CHARS]

        daily_contents[date] = joined

    return daily_contents


def build_batch_tasks(daily_contents: dict[str, str]) -> list[dict]:
    """Build one OpenAI Batch API task per date.

    custom_id format: 'sector-YYYY-MM-DD' — used by the collection step to
    map results back to their date.

    Args:
        daily_contents: {date: joined_article_text} from build_daily_contents().

    Returns:
        List of batch task dicts ready to be written to a JSONL file.
    """
    tasks = []
    for date, content in sorted(daily_contents.items()):
        task = {
            "custom_id": f"sector-{date}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-nano",
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "multi_sector_analysis",
                        "schema": STRICT_SCHEMA,
                        "strict": True,
                    },
                },
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            },
        }
        tasks.append(task)
    return tasks


# ── Submission ──────────────────────────────────────────────────────────────────


def submit_batch(
    tasks: list[dict],
    client: OpenAI,
    batch_file: Path | None = None,
) -> str:
    """Submit a sector-analysis batch job via kitai.batch; return the batch job ID.

    Optionally writes a local JSONL copy of the tasks to batch_file before
    submission — useful as a committed audit artifact (data/batch_tasks_sector.jsonl).
    The actual upload and batch creation are handled by kitai.batch.submit_batch_job.

    Args:
        tasks:      List of task dicts from build_batch_tasks().
        client:     Authenticated OpenAI client instance.
        batch_file: Optional path for a local JSONL debug copy of the tasks.
                    No file is written when None.

    Returns:
        The batch job ID string (e.g. "batch_abc123").

    Failure modes:
        - OpenAI auth error: missing or invalid OPENAI_API_KEY.
        - Schema rejection (400): strict schema rules not satisfied.
        - Quota error: daily/monthly token or request limits hit.

    NOTE: signature changed from submit_batch(tasks, batch_file, client) —
          batch_file is now optional and follows client in the argument list.
    """
    if batch_file is not None:
        batch_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {len(tasks)} task(s) to {batch_file}")  # NOTE: uses print()
        with batch_file.open("w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")

    print(f"Submitting {len(tasks)} task(s) to OpenAI Batch API...")  # NOTE: uses print()
    batch_id = submit_batch_job(client, tasks, endpoint="/v1/chat/completions")
    print(f"Batch job created: {batch_id}")  # NOTE: uses print()
    return batch_id


def persist_batch_id(batch_id: str, pending_file: Path) -> None:
    """Save the batch job ID to a sentinel file for the collection step.

    Args:
        batch_id:     The batch job ID string from submit_batch().
        pending_file: Path to write the ID (e.g. data/pending_sector_batch.txt).
    """
    pending_file.parent.mkdir(parents=True, exist_ok=True)
    pending_file.write_text(batch_id)
    print(f"Batch ID saved -> {pending_file}")  # NOTE: uses print()
