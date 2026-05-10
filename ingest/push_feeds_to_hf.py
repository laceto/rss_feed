"""
push_feeds_to_hf.py

Cold-start push: reads all output/feeds*.txt files, deduplicates on guid,
and publishes the full history to the Hugging Face Dataset repo as Parquet.

Run once. After this, only push_new_feeds_to_hf.py is needed for daily updates.

Usage:
    python push_feeds_to_hf.py             # push all feeds
    python push_feeds_to_hf.py --dry-run   # show row count only, no push

Invariants:
    - Dedup key is guid — stable RSS identifier.
    - source_file column is added for traceability (e.g. "feeds2025-10-15.txt").
    - pubDate is normalised to YYYY-MM-DD string.
    - Never modifies local feed files.

Failure modes:
    - HF_TOKEN missing: fails with KeyError — add to .env.
    - HUGGINGFACE_REPO missing: fails with KeyError — add to .env.
    - No feed files found in output/: exits with message.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

RAW_FEED_DIR = Path("output")
HF_REPO      = os.environ["HUGGINGFACE_REPO"]
HF_TOKEN     = os.environ["HF_TOKEN"]


def load_all_feeds() -> pd.DataFrame:
    """Load every output/feeds*.txt into one deduplicated DataFrame."""
    files = sorted(RAW_FEED_DIR.glob("feeds*.txt"))
    if not files:
        raise FileNotFoundError(f"No feed files found in {RAW_FEED_DIR}/")

    dfs = []
    for f in files:
        df = pd.read_csv(f, sep="\t")
        df["source_file"] = f.name
        df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce").dt.strftime("%Y-%m-%d")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["guid"])
    print(f"Loaded {before:,} rows from {len(files)} file(s), "
          f"{len(combined):,} after guid dedup.")
    return combined


def push(df: pd.DataFrame) -> None:
    """Convert DataFrame to HF Dataset and push to hub."""
    ds = Dataset.from_pandas(df, preserve_index=False)
    print(f"Pushing {len(ds):,} rows to {HF_REPO} ...")
    ds.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"Done. Dataset: https://huggingface.co/datasets/{HF_REPO}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cold-start push of all feeds to HF Datasets")
    p.add_argument("--dry-run", action="store_true",
                   help="Show row count only — no push")
    args = p.parse_args()

    df = load_all_feeds()
    valid_dates = df["pubDate"].dropna()
    print(f"Date range: {valid_dates.min()} -> {valid_dates.max()}")
    print(f"Columns: {list(df.columns)}")

    if args.dry_run:
        print("\n--dry-run: no push made.")
        return

    push(df)


if __name__ == "__main__":
    main()
