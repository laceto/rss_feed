"""
create_batch_files_v2.py
Submits a daily batch job to OpenAI for sector-level news analysis.

Architecture:
- Reads ONLY top-level output/feeds*.txt files (output/enriched/ is excluded by design)
- Groups news by publication date; skips dates already in data/sector_results/
- Submits one batch task per unprocessed date to the OpenAI Batch API
- Persists the batch job ID to data/pending_sector_batch.txt for the retrieval step

Invariants:
- glob("feeds*.txt") never recurses into subdirectories
- Each date produces exactly one batch task
- MAX_CHARS cap prevents token-limit rejections from very large date groups
"""

from pathlib import Path
import json
import sys

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.json_schema import model_json_schema
from typing import Literal
from openai import OpenAI

from constants import (
    SectorName,
    RAW_FEED_DIR,
    SECTOR_RESULTS_DIR as RESULTS_DIR,
    BATCH_FILE,
    PENDING_BATCH_FILE,
)

load_dotenv()
client = OpenAI()

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_CHARS = 60_000  # ~15k tokens safety cap per date chunk


# ── Pydantic Models ─────────────────────────────────────────────────────────────

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
            "List of ALL relevant sectors (1–8 max). "
            "Only include sectors with clear evidence in the news."
        ),
    )


# ── Schema helpers ──────────────────────────────────────────────────────────────

def _make_openai_strict(schema: dict) -> dict:
    """Recursively add additionalProperties:false and required[] for OpenAI strict mode.

    OpenAI strict JSON schema requires every object node to have:
    - additionalProperties: false
    - required: [all property names]
    Without this the Batch API returns a 400 error when strict:true is set.
    This replaces the private openai.lib._pydantic.to_strict_json_schema call.
    """
    schema = dict(schema)

    if "$defs" in schema:
        schema["$defs"] = {k: _make_openai_strict(v) for k, v in schema["$defs"].items()}

    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
        schema["properties"] = {
            k: _make_openai_strict(v) for k, v in schema["properties"].items()
        }

    if "items" in schema:
        schema["items"] = _make_openai_strict(schema["items"])

    return schema


STRICT_SCHEMA = _make_openai_strict(model_json_schema(MultiSectorAnalysis))


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_raw_feeds() -> pd.DataFrame:
    """Load all top-level feed files into a single deduplicated DataFrame.

    Uses Path.glob('feeds*.txt') which matches only the top-level output/ directory,
    deliberately excluding output/enriched/ (which uses a different schema).
    Deduplicates on description before returning to avoid sending duplicates to the LLM.
    """
    feed_files = sorted(RAW_FEED_DIR.glob("feeds*.txt"))
    if not feed_files:
        print(f"[error] No feed files found in {RAW_FEED_DIR}. Exiting.")
        sys.exit(0)

    raw_dfs = [pd.read_csv(f, sep="\t") for f in feed_files]
    combined = pd.concat(raw_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["description"])
    combined["pubDate"] = pd.to_datetime(combined["pubDate"]).dt.strftime("%Y-%m-%d")
    print(f"Loaded {len(combined)} unique feed items from {len(feed_files)} file(s).")
    return combined


def build_daily_contents(df: pd.DataFrame) -> dict[str, str]:
    """Return {date: joined_text} for each unprocessed publication date.

    Each item is formatted as 'YYYY-MM-DD: Title: Description' for LLM context.
    Truncates to MAX_CHARS to prevent token-limit errors on large date groups.
    Skips dates that already have a result file in RESULTS_DIR (incremental sentinel).
    """
    daily_contents: dict[str, str] = {}

    for date, group in df.groupby("pubDate"):
        result_path = RESULTS_DIR / f"{date}.json"
        if result_path.exists():
            print(f"[skip] {date} already processed → {result_path}")
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
            print(f"[warn] {date}: truncating {len(joined):,} → {MAX_CHARS:,} chars")
            joined = joined[:MAX_CHARS]

        daily_contents[date] = joined

    return daily_contents


# ── Batch task building ──────────────────────────────────────────────────────────

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


def build_batch_tasks(daily_contents: dict[str, str]) -> list[dict]:
    """Build one OpenAI Batch API task per date.

    custom_id format: 'sector-YYYY-MM-DD' — used by the collection step to
    map results back to their date and write data/sector_results/{date}.json.
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

def submit_batch(tasks: list[dict]) -> str:
    """Write tasks to JSONL, upload to OpenAI, create batch job; return batch job ID.

    Failure modes:
    - OpenAI auth error: missing or invalid OPENAI_API_KEY in environment
    - Schema rejection (400): strict schema rules not satisfied — check _make_openai_strict
    - Quota error: daily/monthly token or request limits hit
    """
    BATCH_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(tasks)} task(s) to {BATCH_FILE}")
    with BATCH_FILE.open("w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    print("Uploading batch file to OpenAI...")
    with BATCH_FILE.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"Uploaded: {uploaded.id}")

    print("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch job created: {batch_job.id} (status: {batch_job.status})")
    return batch_job.id


def persist_batch_id(batch_id: str) -> None:
    """Save the batch job ID so the collection step can retrieve results later."""
    PENDING_BATCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    PENDING_BATCH_FILE.write_text(batch_id)
    print(f"Batch ID saved → {PENDING_BATCH_FILE}")


# ── Entry point ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Sector Batch Submission ===")

    # 1. Load and deduplicate raw feeds (top-level only)
    combined_df = load_raw_feeds()

    # 2. Build per-date content strings; skip already-processed dates
    daily_contents = build_daily_contents(combined_df)
    if not daily_contents:
        print("All dates already processed. Nothing to submit.")
        return

    print(f"Dates to process: {sorted(daily_contents)}")

    # 3. Build batch tasks
    tasks = build_batch_tasks(daily_contents)

    # 4. Submit to OpenAI Batch API
    batch_id = submit_batch(tasks)

    # 5. Persist batch ID for the collection step
    persist_batch_id(batch_id)

    print(f"\nDone. Submitted {len(tasks)} task(s) under batch {batch_id}.")
    print("Run collect_batch_results.py once the batch completes (~minutes to hours).")


if __name__ == "__main__":
    main()
