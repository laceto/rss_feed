"""Sector result file reading and SQLite database building.

Extracted from:
  - read_sector_results.py (load_sector_results, build_sector_dataframe)
  - build_sector_db.py     (load_sector_json, insert_sector_date, build_sector_db)

Invariants:
  - date is always taken from the filename stem, not from the JSON body.
  - Malformed JSON files are logged to stderr and skipped; processing continues.
  - build_sector_db writes atomically via .db.tmp + os.replace().
  - load_sector_results flattens entities list to pipe-separated string for TSV storage.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from constants import SENTIMENT_SCORE

# ── DDL ───────────────────────────────────────────────────────────────────────

_SECTOR_DB_DDL = """
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

CREATE INDEX IF NOT EXISTS idx_sa_date         ON sector_analyses(date);
CREATE INDEX IF NOT EXISTS idx_sa_sector       ON sector_analyses(sector);
CREATE INDEX IF NOT EXISTS idx_sa_date_sector  ON sector_analyses(date, sector);
CREATE INDEX IF NOT EXISTS idx_se_entity_lower ON sector_entities(lower(entity));
CREATE INDEX IF NOT EXISTS idx_se_analysis_id  ON sector_entities(analysis_id);
"""


# ── SQLite helpers ────────────────────────────────────────────────────────────


def load_sector_json(path: Path) -> dict | None:
    """Read and parse one sector-results JSON file.

    Returns the parsed dict on success, or None on any parse/IO error.
    Errors are logged to stderr so the caller can continue with other files.

    Expected shape: {"date": "...", "batch_id": "...", "sectors": [...]}
    Also accepts a bare list shape, which is normalised to {"sectors": [...], "batch_id": None}.
    """
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        # Accept both {"sectors": [...]} wrapper and bare list shapes.
        if isinstance(data, list):
            return {"sectors": data, "batch_id": None}
        return data
    except Exception as exc:  # noqa: BLE001
        print(f"[sector_io] SKIP {path.name}: {exc}", file=sys.stderr)
        return None


def insert_sector_date(
    conn: sqlite3.Connection,
    date: str,
    data: dict,
) -> int:
    """Insert all SectorAnalysis records for one date into the open connection.

    Args:
        conn: Open SQLite connection (caller manages commit/close).
        date: YYYY-MM-DD date string (authoritative — from filename, not JSON body).
        data: Parsed sector JSON dict with "sectors" and optional "batch_id" keys.

    Returns:
        Number of sector_analyses rows inserted.
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


def build_sector_db(db_path: Path, results_dir: Path) -> int:
    """Full rebuild of the SQLite database from all *.json files in results_dir.

    Writes atomically: builds into db_path.tmp, then os.replace() to db_path.
    The final path is either fully updated or untouched on crash.

    Args:
        db_path:     Destination SQLite file (e.g. data/sector_results.db).
        results_dir: Directory containing per-date JSON result files.

    Returns:
        Total number of sector_analyses rows inserted.
    """
    json_files = sorted(results_dir.glob("*.json"))

    tmp_path = db_path.with_suffix(".db.tmp")
    # Remove stale tmp if a previous run crashed mid-write.
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(tmp_path)
    conn.executescript(_SECTOR_DB_DDL)

    total = 0
    for path in json_files:
        data = load_sector_json(path)
        if data is None:
            continue
        date = path.stem  # filename without .json — authoritative date
        total += insert_sector_date(conn, date, data)

    conn.commit()
    conn.close()

    # Atomic replace: final path is either fully updated or the old version.
    os.replace(tmp_path, db_path)

    return total


# ── Sector result file reading ─────────────────────────────────────────────────


def load_sector_results(results_dir: Path) -> list[dict]:
    """Read all per-date JSON files from results_dir and flatten into rows.

    Each row is one sector entry with an added 'date' field.
    Skips files that are missing, empty, or malformed — logs a warning per file.

    Entities list is flattened to a pipe-separated string for TSV storage.
    To reconstruct the list: row["entities"].split("|")

    Args:
        results_dir: Directory containing per-date JSON result files.

    Returns:
        List of flat dicts, one per sector entry.
    """
    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        print(f"[warn] No result files found in {results_dir}.")  # NOTE: uses print()
        return []

    rows = []
    for path in result_files:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] Skipping {path.name}: {exc}")  # NOTE: uses print()
            continue

        date = data.get("date", path.stem)
        sectors = data.get("sectors", [])

        if not sectors:
            print(f"[info] {date}: no sectors extracted (empty result).")  # NOTE: uses print()
            continue

        for sector in sectors:
            row = {"date": date, **sector}
            # Flatten entities list -> pipe-separated string for TSV storage
            if isinstance(row.get("entities"), list):
                row["entities"] = "|".join(row["entities"])
            rows.append(row)

    return rows


def build_sector_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert flat sector rows to a DataFrame with stable column order, sorted by date.

    Args:
        rows: List of flat sector dicts from load_sector_results().

    Returns:
        DataFrame with columns: date, sector, entities, sentiment,
        news_category, extraction_status (plus any extra columns), sorted by date.
    """
    if not rows:
        return pd.DataFrame(columns=[
            "date", "sector", "entities", "sentiment", "news_category", "extraction_status",
        ])

    df = pd.DataFrame(rows)
    base_cols = ["date", "sector", "entities", "sentiment", "news_category", "extraction_status"]
    ordered = [c for c in base_cols if c in df.columns]
    extra = [c for c in df.columns if c not in base_cols]
    return df[ordered + extra].sort_values("date").reset_index(drop=True)
