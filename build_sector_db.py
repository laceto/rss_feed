"""
build_sector_db.py

CLI wrapper for pipeline.sector_io.build_sector_db.

Reads all data/sector_results/{date}.json files and builds a SQLite
database at data/sector_results.db.

Usage (CLI):
    python build_sector_db.py

To use programmatically, import from the package:
    from pipeline.sector_io import build_sector_db
"""

import sys

from constants import SECTOR_DB_FILE, SECTOR_RESULTS_DIR
from pipeline.sector_io import build_sector_db


def main() -> None:
    json_count = sum(1 for _ in SECTOR_RESULTS_DIR.glob("*.json"))
    if json_count == 0:
        print(
            f"[build_sector_db] No JSON files found in {SECTOR_RESULTS_DIR}. "
            "Nothing to do.",
            file=sys.stderr,
        )
        return

    total = build_sector_db(SECTOR_DB_FILE, SECTOR_RESULTS_DIR)
    print(f"Built {SECTOR_DB_FILE} — {total} rows across {json_count} dates.")


if __name__ == "__main__":
    main()
