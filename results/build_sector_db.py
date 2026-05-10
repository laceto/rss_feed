"""
build_sector_db.py
Reads all data/sector_results/{date}.json files and builds a SQLite database
at data/sector_results.db.

Usage (CLI):
    python build_sector_db.py

Can also be imported:
    from build_sector_db import build

Invariants:
- Full rebuild every run — no incremental sync complexity.
- Atomic write: build into a .tmp file, then os.replace() to the final path.
  The final path is either fully updated or untouched on crash.
- date is always taken from the filename stem, not from the JSON body.
- Malformed JSON files are logged to stderr and skipped; the build continues.

Schema
------
sector_analyses (id, date, sector, sentiment, sentiment_score,
                 news_category, extraction_status, batch_id)
sector_entities  (id, analysis_id FK, entity)
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

from constants import SECTOR_DB_FILE, SECTOR_RESULTS_DIR, SENTIMENT_SCORE

# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS sector_analyses (
    id                INTEGER PRIMARY KEY,
    date              TEXT    NOT NULL,
    sector            TEXT    NOT NULL,
    sentiment         TEXT    NOT NULL,
    sentiment_score   INTEGER NOT NULL,
    news_category     TEXT    NOT NULL,
    extraction_status TEXT    NOT NULL,
    batch_id          TEXT
);

CREATE TABLE IF NOT EXISTS sector_entities (
    id          INTEGER PRIMARY KEY,
    analysis_id INTEGER NOT NULL REFERENCES sector_analyses(id) ON DELETE CASCADE,
    entity      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sa_date        ON sector_analyses(date);
CREATE INDEX IF NOT EXISTS idx_sa_sector      ON sector_analyses(sector);
CREATE INDEX IF NOT EXISTS idx_sa_date_sector ON sector_analyses(date, sector);
CREATE INDEX IF NOT EXISTS idx_se_entity_lower ON sector_entities(lower(entity));
CREATE INDEX IF NOT EXISTS idx_se_analysis_id  ON sector_entities(analysis_id);
"""

# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict | None:
    """
    Read and parse one sector-results JSON file.

    Returns the parsed dict on success, or None on any parse/IO error.
    Errors are logged to stderr so the caller can continue with other files.

    Expected shape: {"date": "...", "batch_id": "...", "sectors": [...]}
    """
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        # Accept both {"sectors": [...]} wrapper and bare list shapes.
        if isinstance(data, list):
            return {"sectors": data, "batch_id": None}
        return data
    except Exception as exc:  # noqa: BLE001
        print(f"[build_sector_db] SKIP {path.name}: {exc}", file=sys.stderr)
        return None


def _insert_date(
    conn: sqlite3.Connection,
    date: str,
    data: dict,
) -> int:
    """
    Insert all SectorAnalysis records for one date into the open connection.

    Returns the number of sector_analyses rows inserted.
    """
    batch_id = data.get("batch_id")
    sectors = data.get("sectors") or []
    inserted = 0

    for sector_obj in sectors:
        sentiment = sector_obj.get("sentiment", "neutral")
        score = SENTIMENT_SCORE.get(sentiment, 0)

        cursor = conn.execute(
            """
            INSERT INTO sector_analyses
                (date, sector, sentiment, sentiment_score,
                 news_category, extraction_status, batch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                sector_obj.get("sector", ""),
                sentiment,
                score,
                sector_obj.get("news_category", ""),
                sector_obj.get("extraction_status", ""),
                batch_id,
            ),
        )
        analysis_id = cursor.lastrowid
        inserted += 1

        for entity in sector_obj.get("entities") or []:
            conn.execute(
                "INSERT INTO sector_entities (analysis_id, entity) VALUES (?, ?)",
                (analysis_id, entity),
            )

    return inserted


# ── Public API ────────────────────────────────────────────────────────────────


def build(
    db_path: Path = SECTOR_DB_FILE,
    results_dir: Path = SECTOR_RESULTS_DIR,
) -> int:
    """
    Full rebuild of the SQLite database from all *.json files in results_dir.

    Writes atomically: builds into db_path.tmp, then os.replace() to db_path.
    Returns the total number of sector_analyses rows inserted.
    """
    json_files = sorted(results_dir.glob("*.json"))

    tmp_path = db_path.with_suffix(".db.tmp")
    # Remove stale tmp if a previous run crashed mid-write.
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(tmp_path)
    conn.executescript(_DDL)

    total = 0
    for path in json_files:
        data = _load_json(path)
        if data is None:
            continue
        date = path.stem  # filename without .json — authoritative date
        total += _insert_date(conn, date, data)

    conn.commit()
    conn.close()

    # Atomic replace: final path is either fully updated or the old version.
    os.replace(tmp_path, db_path)

    return total


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    json_count = sum(1 for _ in SECTOR_RESULTS_DIR.glob("*.json"))
    if json_count == 0:
        print(
            f"[build_sector_db] No JSON files found in {SECTOR_RESULTS_DIR}. "
            "Nothing to do.",
            file=sys.stderr,
        )
        return

    total = build()
    date_count = json_count
    print(
        f"Built {SECTOR_DB_FILE} — {total} rows across {date_count} dates."
    )


if __name__ == "__main__":
    main()
