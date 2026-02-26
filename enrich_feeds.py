"""
enrich_feeds.py — Enrich CNBC RSS feed files using direct async LangChain calls.

Architecture
------------
    output/feeds{date}.txt          (raw, immutable — written by download.R)
            |
            |  async LLM extraction  (this script)
            v
    output/enriched/feeds{date}.txt (guid + enriched columns only)

Join enriched back to raw:
    raw_df.merge(enriched_df, on="guid", how="left")
    # rows not yet enriched, or deduped-out rows, surface as NaN — expected

Flow
----
    1. SCAN        — detect dates not yet in output/enriched/
    2. LOAD + DEDUP — read rows; deduplicate by normalized description
    3. ENRICH      — concurrent async LLM calls, bounded by MAX_CONCURRENCY
    4. WRITE       — one TSV per date; sentinel file prevents re-processing

Deduplication
-------------
    Rows whose normalized (lowercased, stripped) description matches a
    previously-seen description are skipped. First occurrence wins (by
    file date order, then row order within file). Deduplicated rows are
    NOT written to the enriched output — they appear as NaN in the
    raw+enriched join, which is correct (they are not unique stories).

Incremental safety
------------------
    - Enriched file presence is the sentinel: if output/enriched/feeds{date}.txt
      exists, that date is skipped on the next run.
    - Dates where every extraction failed are NOT written — retried next run.
    - Partial successes ARE written (failed rows carry extraction_status='failed').
    - Raw files are never modified.

Usage
-----
    python enrich_feeds.py
"""

import asyncio
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — single place to change paths, model, and concurrency
# ---------------------------------------------------------------------------

RAW_DIR = Path("output")
ENRICHED_DIR = Path("output/enriched")
DATE_PATTERN = re.compile(r"^feeds(\d{4}-\d{2}-\d{2})\.txt$")
MODEL = "gpt-4.1-nano"
MAX_CONCURRENCY = 20  # max simultaneous in-flight LLM calls

# ---------------------------------------------------------------------------
# Enrichment schema
#
# Single source of truth for extracted fields.
# All downstream code (writing, joining, analysis) derives from this model.
# ---------------------------------------------------------------------------

class FeedEnrichment(BaseModel):
    """Structured metadata extracted from one CNBC news item (title + description)."""

    entities: list[str] = Field(
        default_factory=list,
        description=(
            "Named companies or organisations mentioned "
            "(e.g. ['IBM', 'Federal Reserve', 'NVIDIA']). "
            "Empty list if none are clearly identified."
        ),
    )
    sector: Literal[
        "Technology", "Finance", "Energy", "Healthcare",
        "Consumer", "Industrial", "Policy", "Other",
    ] = Field(..., description="Primary market sector the news concerns.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description=(
            "Market sentiment implied by the news from an equity investor's perspective. "
            "'positive' = likely good for prices, 'negative' = likely bad, 'neutral' = informational."
        ),
    )
    news_category: Literal[
        "earnings", "M&A", "regulation", "macro",
        "appointments", "products", "markets", "other",
    ] = Field(..., description="Category of the news event.")


# ---------------------------------------------------------------------------
# LangChain extraction chain
#
# Chain: ChatPromptTemplate | ChatOpenAI | PydanticOutputParser
# ---------------------------------------------------------------------------

def build_chain():
    """Return the extraction chain. Call once per run; reuse across all invocations."""
    parser = PydanticOutputParser(pydantic_object=FeedEnrichment)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a financial news analyst. "
                "Extract structured metadata from the provided news headline and description. "
                "Be precise. When uncertain about a field, pick the closest valid option."
            ),
        ),
        ("user", "News item:\n{text}\n\n{format_instructions}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    llm = ChatOpenAI(model=MODEL, temperature=0)
    return prompt | llm | parser


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_new_feed_files() -> list[tuple[str, Path]]:
    """
    Return (date_str, path) for each raw feed file not yet enriched, sorted by date.

    A file is considered enriched if output/enriched/feeds{date}.txt exists.
    Files without a YYYY-MM-DD suffix (e.g. feeds.txt) are skipped.
    """
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(RAW_DIR.glob("feeds*.txt")):
        match = DATE_PATTERN.match(path.name)
        if not match:
            continue
        date = match.group(1)
        if not (ENRICHED_DIR / path.name).exists():
            results.append((date, path))
    return results


# ---------------------------------------------------------------------------
# Row loading with deduplication
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Canonical form used for deduplication comparison."""
    return text.lower().strip()


def load_rows(
    files: list[tuple[str, Path]],
) -> tuple[list[dict], list[dict]]:
    """
    Read feed files and return parallel lists ready for LLM processing.

    Deduplication:
        Rows whose normalized description matches an already-seen description
        are silently dropped. First occurrence (by date order, then row order)
        is kept. Deduped rows will appear as NaN in the raw+enriched join.

    Returns:
        inputs    — [{"text": str}, ...]              (prompt template variables)
        index_map — [{"guid": str, "date": str}, ...]  (result reassembly key)

    Invariant: len(inputs) == len(index_map), same positional order.
    """
    inputs: list[dict] = []
    index_map: list[dict] = []
    seen: set[str] = set()  # normalized descriptions seen so far
    n_total = n_dupes = n_invalid = 0

    for date, path in files:
        try:
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        except Exception as exc:
            log.warning("Skipping %s — could not read: %s", path, exc)
            continue

        for _, row in df.iterrows():
            n_total += 1
            title = row.get("title", "").strip()
            description = row.get("description", "").strip()
            guid = row.get("guid", "").strip()

            # Skip rows with missing identity or no content
            if not guid or not (title or description):
                n_invalid += 1
                continue

            # Deduplicate on normalized description
            key = _normalize(description)
            if key in seen:
                n_dupes += 1
                continue
            seen.add(key)

            text = f"{title}. {description}".strip(". ")
            inputs.append({"text": text})
            index_map.append({"guid": guid, "date": date})

    log.info(
        "Loaded %d rows from %d file(s) → %d unique  |  %d duplicates removed  |  %d invalid skipped",
        n_total, len(files), len(inputs), n_dupes, n_invalid,
    )
    return inputs, index_map


# ---------------------------------------------------------------------------
# Async enrichment
#
# Uses asyncio.gather(return_exceptions=True) so individual row failures
# are captured as Exception instances rather than aborting the entire run.
# A semaphore caps concurrent in-flight LLM calls at MAX_CONCURRENCY.
# ---------------------------------------------------------------------------

async def enrich_batch(
    chain,
    inputs: list[dict],
) -> list[FeedEnrichment | Exception]:
    """
    Run LLM extraction concurrently over all inputs.

    Returns a list positionally aligned with inputs. Each element is either
    a FeedEnrichment (success) or an Exception (failure for that row).
    """
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    n_done = 0

    async def invoke_one(inp: dict) -> FeedEnrichment:
        nonlocal n_done
        async with sem:
            result = await chain.ainvoke(inp)
        n_done += 1
        if n_done % 100 == 0 or n_done == len(inputs):
            log.info("Progress: %d / %d rows enriched", n_done, len(inputs))
        return result

    log.info("Enriching %d rows (max_concurrency=%d)…", len(inputs), MAX_CONCURRENCY)
    results = await asyncio.gather(
        *[invoke_one(inp) for inp in inputs],
        return_exceptions=True,
    )
    return list(results)


# ---------------------------------------------------------------------------
# Result writing
# ---------------------------------------------------------------------------

def write_enriched_files(
    results: list[FeedEnrichment | Exception],
    index_map: list[dict],
) -> None:
    """
    Group results by date and write one enriched TSV per date.

    - Failed rows (Exception) are written with extraction_status='failed'
      and empty fields, and logged at WARNING with guid + date.
    - A date is NOT written if every row failed — it will be retried on
      the next run since no sentinel file is created.
    """
    rows_by_date: dict[str, list[dict]] = defaultdict(list)

    for i, item in enumerate(results):
        meta = index_map[i]
        date, guid = meta["date"], meta["guid"]

        if isinstance(item, Exception):
            log.warning(
                "Extraction failed  guid=%-12s  date=%s  error=%s",
                guid, date, item,
            )
            row = {
                "guid": guid,
                "entities": "",
                "sector": "",
                "sentiment": "",
                "news_category": "",
                "extraction_status": "failed",
            }
        else:
            enrichment: FeedEnrichment = item
            row = {
                "guid": guid,
                "entities": "|".join(enrichment.entities),
                "sector": enrichment.sector,
                "sentiment": enrichment.sentiment,
                "news_category": enrichment.news_category,
                "extraction_status": "ok",
            }

        rows_by_date[date].append(row)

    written = skipped = 0
    for date, rows in rows_by_date.items():
        df = pd.DataFrame(rows)
        ok_count = (df["extraction_status"] == "ok").sum()
        total = len(df)

        if ok_count == 0:
            log.warning(
                "All %d rows failed for date=%s — skipping write (will retry next run)",
                total, date,
            )
            skipped += 1
            continue

        out_path = ENRICHED_DIR / f"feeds{date}.txt"
        df.to_csv(out_path, sep="\t", index=False)
        log.info("Wrote %s  (%d / %d ok)", out_path, ok_count, total)
        written += 1

    log.info("Done: %d date(s) written, %d date(s) skipped.", written, skipped)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    load_dotenv()

    new_files = find_new_feed_files()
    if not new_files:
        log.info("No new feed files to process.")
        return

    log.info("Unprocessed dates: %s", [d for d, _ in new_files])

    inputs, index_map = load_rows(new_files)
    if not inputs:
        log.warning("No valid rows found in new files.")
        return

    chain = build_chain()
    results = await enrich_batch(chain, inputs)
    write_enriched_files(results, index_map)


if __name__ == "__main__":
    asyncio.run(main())
