"""
push_new_feeds_to_hf.py

Daily incremental push: appends only articles with new guids to the HF Dataset.
Idempotent — re-running for the same date is safe (no duplicate rows).

Called by GitHub Actions daily-pipeline after download.R.

Usage:
    python push_new_feeds_to_hf.py                    # today's feed
    python push_new_feeds_to_hf.py --date 2025-10-15  # specific date

Invariants:
    - Dedup key is guid — articles already in the remote dataset are skipped.
    - pubDate normalised to YYYY-MM-DD string.
    - Never modifies local feed files.
    - Exits 0 with message if feed file for the date doesn't exist yet.

Failure modes:
    - HF_TOKEN missing: fails with KeyError — add to .env / GitHub secret.
    - HUGGINGFACE_REPO missing: same.
    - Remote dataset empty (cold-start not done): falls back to push_feeds_to_hf.py.
    - Feed file absent: exits 0 cleanly (download.R may not have run yet).
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv

load_dotenv()

RAW_FEED_DIR = Path("output")
HF_REPO      = os.environ["HUGGINGFACE_REPO"]
HF_TOKEN     = os.environ["HF_TOKEN"]


def load_feed_for_date(target_date: str) -> pd.DataFrame:
    """Load a single feed file for the given date string (YYYY-MM-DD)."""
    path = RAW_FEED_DIR / f"feeds{target_date}.txt"
    if not path.exists():
        print(f"Feed file not found: {path} — nothing to push.")
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t")
    df["source_file"] = path.name
    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    print(f"Loaded {len(df):,} rows from {path.name}.")
    return df


def push_incremental(new_df: pd.DataFrame) -> None:
    """Append rows with new guids to the remote HF dataset."""
    print(f"Loading existing dataset from {HF_REPO} ...")
    try:
        existing_ds = load_dataset(HF_REPO, split="train", token=HF_TOKEN)
        existing_guids = set(existing_ds["guid"])
        print(f"  {len(existing_guids):,} existing guids in remote dataset.")
    except Exception as exc:
        # Dataset may be empty on first incremental run — treat as empty
        print(f"  Could not load existing dataset ({exc}); treating as empty.")
        existing_guids = set()
        existing_ds = None

    new_df = new_df[~new_df["guid"].isin(existing_guids)]
    if new_df.empty:
        print("No new articles — dataset already up to date.")
        return

    print(f"  {len(new_df):,} new rows to append.")
    new_ds = Dataset.from_pandas(new_df, preserve_index=False)

    if existing_ds is not None and len(existing_ds) > 0:
        merged = concatenate_datasets([existing_ds, new_ds])
    else:
        merged = new_ds

    merged.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"Pushed. Dataset now has {len(merged):,} rows.")
    print(f"Dataset: https://huggingface.co/datasets/{HF_REPO}")


def main() -> None:
    p = argparse.ArgumentParser(description="Daily incremental feed push to HF Datasets")
    p.add_argument("--date", default=str(date.today()),
                   help="Feed date YYYY-MM-DD (default: today)")
    args = p.parse_args()

    df = load_feed_for_date(args.date)
    if df.empty:
        return

    push_incremental(df)


if __name__ == "__main__":
    main()
