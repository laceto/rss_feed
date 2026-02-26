"""
read_sector_results.py
Reads all per-date sector result files and builds a consolidated flat DataFrame.

Runs after retrieve_batch_file_results.py, which writes:
    data/sector_results/{date}.json

Input:  data/sector_results/*.json
Output: data/sector_summary.tsv  (one row per date × sector, sorted by date)

Schema:
    date | sector | entities | sentiment | news_category | extraction_status

Entities list is stored as a pipe-separated string for TSV compatibility.
To reconstruct the list: row["entities"].split("|")

Debugging:
- Missing dates: check data/sector_results/ for gaps
- Malformed files: warnings are printed per file; check data/batch_output_sector.jsonl
  for the raw OpenAI response
"""

from pathlib import Path
import json
import sys

import pandas as pd

RESULTS_DIR = Path("data") / "sector_results"
OUTPUT_FILE = Path("data") / "sector_summary.tsv"


def load_sector_results() -> list[dict]:
    """Read all per-date JSON files from RESULTS_DIR and flatten into rows.

    Each row is one sector entry with an added 'date' field.
    Skips files that are missing, empty, or malformed — logs a warning per file.
    """
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        print(f"[warn] No result files found in {RESULTS_DIR}.")
        return []

    rows = []
    for path in result_files:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] Skipping {path.name}: {exc}")
            continue

        date = data.get("date", path.stem)
        sectors = data.get("sectors", [])

        if not sectors:
            print(f"[info] {date}: no sectors extracted (empty result).")
            continue

        for sector in sectors:
            row = {"date": date, **sector}
            # Flatten entities list → pipe-separated string for TSV storage
            if isinstance(row.get("entities"), list):
                row["entities"] = "|".join(row["entities"])
            rows.append(row)

    return rows


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert flat rows to a DataFrame with a stable column order, sorted by date."""
    if not rows:
        return pd.DataFrame(columns=[
            "date", "sector", "entities", "sentiment", "news_category", "extraction_status",
        ])

    df = pd.DataFrame(rows)
    base_cols = ["date", "sector", "entities", "sentiment", "news_category", "extraction_status"]
    ordered = [c for c in base_cols if c in df.columns]
    extra = [c for c in df.columns if c not in base_cols]
    return df[ordered + extra].sort_values("date").reset_index(drop=True)


def main() -> None:
    print("=== Build Sector Summary ===")

    rows = load_sector_results()
    if not rows:
        print("Nothing to summarize.")
        sys.exit(0)

    df = build_dataframe(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)

    print(f"Wrote {len(df)} row(s) across {df['date'].nunique()} date(s) → {OUTPUT_FILE}")
    print(df.to_string(max_rows=20))


if __name__ == "__main__":
    main()
