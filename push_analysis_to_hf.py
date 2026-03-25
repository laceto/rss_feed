"""
push_analysis_to_hf.py

Cold-start push: creates three HF Dataset repos and uploads all existing
analysis data in one shot.

Datasets created:
  {HF_USERNAME}/sector-analysis   ← data/sector_summary.tsv
  {HF_USERNAME}/topic-trends      ← data/topic_trends.tsv
  {HF_USERNAME}/entity-sentiment  ← data/entity_sentiment_ts.tsv

Run once. After this, only push_new_analysis_to_hf.py is needed for daily updates.

Usage:
    python push_analysis_to_hf.py             # create repos + push all data
    python push_analysis_to_hf.py --dry-run   # show row counts only, no push

Invariants:
    - HF_USERNAME is derived from HUGGINGFACE_REPO (everything before the first "/").
    - Dedup keys: sector_summary → date+sector; topic_trends → date+topic_id;
      entity_sentiment → date+entity+sector.
    - Never modifies local data files.

Failure modes:
    - HF_TOKEN missing: fails with KeyError — add to .env.
    - HUGGINGFACE_REPO missing: fails with KeyError — add to .env.
    - Source TSV absent: exits with FileNotFoundError.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_REPO     = os.environ["HUGGINGFACE_REPO"]
HF_USERNAME = HF_REPO.split("/")[0]

SECTOR_SUMMARY_FILE   = Path("data") / "sector_summary.tsv"
TOPIC_TRENDS_FILE     = Path("data") / "topic_trends.tsv"
ENTITY_SENTIMENT_FILE = Path("data") / "entity_sentiment_ts.tsv"

DATASETS = {
    "sector-analysis":  SECTOR_SUMMARY_FILE,
    "topic-trends":     TOPIC_TRENDS_FILE,
    "entity-sentiment": ENTITY_SENTIMENT_FILE,
}


def _create_repo(api: HfApi, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    print(f"  Repo ready: https://huggingface.co/datasets/{repo_id}")


def _load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    print(f"  Loaded {len(df):,} rows from {path.name}")
    return df


def _push(df: pd.DataFrame, repo_id: str) -> None:
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(repo_id, token=HF_TOKEN)
    print(f"  Pushed {len(ds):,} rows -> https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cold-start push of analysis data to HF Datasets")
    p.add_argument("--dry-run", action="store_true", help="Show row counts only — no push")
    args = p.parse_args()

    api = HfApi(token=HF_TOKEN)

    for name, path in DATASETS.items():
        repo_id = f"{HF_USERNAME}/{name}"
        print(f"\n--- {name} ---")

        df = _load_tsv(path)

        if args.dry_run:
            print(f"  --dry-run: would push {len(df):,} rows to {repo_id}")
            continue

        _create_repo(api, repo_id)
        _push(df, repo_id)

    if args.dry_run:
        print("\n--dry-run: no push made.")
    else:
        print("\nAll datasets pushed.")


if __name__ == "__main__":
    main()
