"""High-level sector database build orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sector_io import build_sector_db


def run_sector_db_build(db_path: Path, results_dir: Path) -> dict[str, Any]:
    """Build the sector SQLite database and return a small execution summary."""
    json_count = sum(1 for _ in results_dir.glob("*.json"))
    if json_count == 0:
        return {
            "db_path": db_path,
            "results_dir": results_dir,
            "json_count": 0,
            "row_count": 0,
        }

    row_count = build_sector_db(db_path, results_dir)
    return {
        "db_path": db_path,
        "results_dir": results_dir,
        "json_count": json_count,
        "row_count": row_count,
    }
