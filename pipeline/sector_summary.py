"""High-level sector summary build orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sector_io import build_sector_dataframe, load_sector_results

RESULTS_DIR = Path("data") / "sector_results"
OUTPUT_FILE = Path("data") / "sector_summary.tsv"


def run_sector_summary_build(
    results_dir: Path = RESULTS_DIR,
    output_file: Path = OUTPUT_FILE,
) -> dict[str, Any]:
    """Build sector_summary.tsv from per-date sector result files."""
    print("=== Build Sector Summary ===")

    rows = load_sector_results(results_dir)
    if not rows:
        print("Nothing to summarize.")
        return {
            "row_count": 0,
            "date_count": 0,
            "output_file": output_file,
            "empty": True,
        }

    df = build_sector_dataframe(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep="\t", index=False)

    print(f"Wrote {len(df)} row(s) across {df['date'].nunique()} date(s) -> {output_file}")
    print(df.to_string(max_rows=20))
    return {
        "row_count": len(df),
        "date_count": int(df["date"].nunique()),
        "output_file": output_file,
        "empty": False,
    }
