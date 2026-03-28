"""
push_new_analysis_to_hf.py

Daily incremental push: appends only new rows to the three HF analysis datasets.
Idempotent — re-running for the same date is safe (composite key dedup).

Called by GitHub Actions collect-sector-results workflow.

Datasets updated:
  {HF_USERNAME}/sector-analysis   ← data/sector_summary.tsv   (dedup: date+sector)
  {HF_USERNAME}/topic-trends      ← data/topic_trends.tsv      (dedup: date+topic_id)
  {HF_USERNAME}/entity-sentiment  ← data/entity_sentiment_ts.tsv (dedup: date+entity+sector)

Usage:
    python push_new_analysis_to_hf.py   # push today's new rows across all three datasets

Invariants:
    - Never modifies local data files.
    - Rows already present in the remote dataset (matched on composite key) are skipped.
    - If a remote dataset is empty (cold-start not done), falls back to full push.

Failure modes:
    - HF_TOKEN missing: fails with KeyError — add to .env / GitHub secret.
    - HUGGINGFACE_REPO missing: same.
    - Source TSV absent: exits with FileNotFoundError.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_REPO     = os.environ["HUGGINGFACE_REPO"]
HF_USERNAME = HF_REPO.split("/")[0]

SECTOR_SUMMARY_FILE   = Path("data") / "sector_summary.tsv"
TOPIC_TRENDS_FILE     = Path("data") / "topic_trends.tsv"
ENTITY_SENTIMENT_FILE = Path("data") / "entity_sentiment_ts.tsv"

# Each entry: (dataset-name, local-file, composite-dedup-key-columns)
DATASETS: list[tuple[str, Path, list[str]]] = [
    ("sector-analysis",  SECTOR_SUMMARY_FILE,   ["date", "sector"]),
    ("topic-trends",     TOPIC_TRENDS_FILE,      ["date", "topic_id"]),
    ("entity-sentiment", ENTITY_SENTIMENT_FILE,  ["date", "entity", "sector"]),
]


def _composite_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def push_incremental(repo_id: str, local_path: Path, key_cols: list[str]) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Source file not found: {local_path}")

    local_df = pd.read_csv(local_path, sep="\t")
    print(f"  Local rows: {len(local_df):,}")

    print(f"  Loading existing dataset from {repo_id} ...")
    try:
        existing_ds = load_dataset(repo_id, split="train", token=HF_TOKEN)
        existing_df = existing_ds.to_pandas()
        existing_keys = set(_composite_key(existing_df, key_cols))
        print(f"  Remote rows: {len(existing_df):,} | unique keys: {len(existing_keys):,}")
    except Exception as exc:
        print(f"  Could not load remote dataset ({exc}); treating as empty.")
        existing_ds   = None
        existing_keys = set()

    new_df = local_df[~_composite_key(local_df, key_cols).isin(existing_keys)]
    if new_df.empty:
        print("  No new rows — dataset already up to date.")
        return

    print(f"  {len(new_df):,} new rows to append.")
    new_ds = Dataset.from_pandas(new_df, preserve_index=False)

    if existing_ds is not None and len(existing_ds) > 0:
        # Cast to match existing schema — newer PyArrow uses large_string
        # but the remote dataset has string; concatenate_datasets rejects mismatch.
        new_ds = new_ds.cast(existing_ds.features)
        merged = concatenate_datasets([existing_ds, new_ds])
    else:
        merged = new_ds

    merged.push_to_hub(repo_id, token=HF_TOKEN)
    print(f"  Pushed. Dataset now has {len(merged):,} rows.")
    print(f"  https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    for name, path, key_cols in DATASETS:
        repo_id = f"{HF_USERNAME}/{name}"
        print(f"\n--- {name} (dedup: {'+'.join(key_cols)}) ---")
        push_incremental(repo_id, path, key_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
