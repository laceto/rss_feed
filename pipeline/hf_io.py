"""HuggingFace dataset I/O utilities.

Extracted from:
  - push_feeds_to_hf.py      (load_feeds_from_files, push_feeds_to_hub)
  - create_batch_files_v2.py (load_feeds_from_hf)
  - push_analysis_to_hf.py   (create_hf_dataset_repo, load_tsv, push_df_to_hub)

All module-level path constants and env reads have been replaced with
explicit parameters so callers can inject their own paths and credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


# ── Feed file loading ──────────────────────────────────────────────────────────


def load_feeds_from_files(raw_feed_dir: Path) -> pd.DataFrame:
    """Load every feeds*.txt from raw_feed_dir into one deduplicated DataFrame.

    Dedup key is guid — stable RSS identifier.
    source_file column is added for traceability (e.g. "feeds2025-10-15.txt").
    pubDate is normalised to YYYY-MM-DD string.

    Raises:
        FileNotFoundError: if no feeds*.txt files exist in raw_feed_dir.
    """
    files = sorted(raw_feed_dir.glob("feeds*.txt"))
    if not files:
        raise FileNotFoundError(f"No feed files found in {raw_feed_dir}/")

    dfs = []
    for f in files:
        df = pd.read_csv(f, sep="\t")
        df["source_file"] = f.name
        df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce").dt.strftime("%Y-%m-%d")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["guid"])
    print(  # NOTE: uses print()
        f"Loaded {before:,} rows from {len(files)} file(s), "
        f"{len(combined):,} after guid dedup."
    )
    return combined


def push_feeds_to_hub(df: pd.DataFrame, repo: str, token: str) -> None:
    """Convert a feeds DataFrame to a HF Dataset and push to hub.

    Args:
        df:    DataFrame of feed articles.
        repo:  HuggingFace repo ID (e.g. "username/feeds").
        token: HF_TOKEN with write access.
    """
    ds = Dataset.from_pandas(df, preserve_index=False)
    print(f"Pushing {len(ds):,} rows to {repo} ...")  # NOTE: uses print()
    ds.push_to_hub(repo, token=token)
    print(f"Done. Dataset: https://huggingface.co/datasets/{repo}")  # NOTE: uses print()


# ── Feed loading from HF ───────────────────────────────────────────────────────


def load_feeds_from_hf(hf_repo: str, hf_token: str | None = None) -> pd.DataFrame:
    """Load all feed articles from a Hugging Face Dataset (split="train").

    Deduplicates on description before returning to avoid sending duplicates
    to the LLM. pubDate is normalised to YYYY-MM-DD.

    Args:
        hf_repo:  HuggingFace dataset repo ID (e.g. "username/feeds").
        hf_token: Optional HF token (required for private repos).

    Failure modes:
        - Network unavailable: load_dataset raises ConnectTimeout.
        - Empty dataset: prints error and calls sys.exit(0).

    NOTE: calls sys.exit(0) when the dataset is empty.
    """
    print(f"Loading feeds from HF dataset {hf_repo} ...")  # NOTE: uses print()
    ds = load_dataset(hf_repo, split="train", token=hf_token)
    if len(ds) == 0:
        print("[error] HF dataset is empty. Run push_feeds_to_hf.py first.")  # NOTE: uses print()
        sys.exit(0)  # NOTE: uses sys.exit(0) — empty dataset, nothing to do

    combined = ds.to_pandas()
    combined = combined.drop_duplicates(subset=["description"])
    combined["pubDate"] = (
        pd.to_datetime(combined["pubDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    )
    print(f"Loaded {len(combined):,} unique feed items from {hf_repo}.")  # NOTE: uses print()
    return combined


# ── Analysis dataset I/O ───────────────────────────────────────────────────────


def create_hf_dataset_repo(api: HfApi, repo_id: str) -> None:
    """Create or ensure a public HuggingFace dataset repo exists.

    Idempotent: exist_ok=True means re-running the same repo_id is safe.
    """
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    print(f"  Repo ready: https://huggingface.co/datasets/{repo_id}")  # NOTE: uses print()


def load_tsv(path: Path) -> pd.DataFrame:
    """Load a TSV file and return as a DataFrame.

    Raises:
        FileNotFoundError: if path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    print(f"  Loaded {len(df):,} rows from {path.name}")  # NOTE: uses print()
    return df


def push_df_to_hub(df: pd.DataFrame, repo_id: str, token: str) -> None:
    """Push an arbitrary DataFrame as a HuggingFace Dataset to the hub.

    Args:
        df:      DataFrame to push (preserve_index=False).
        repo_id: HuggingFace repo ID (e.g. "username/sector-analysis").
        token:   HF_TOKEN with write access.
    """
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(repo_id, token=token)
    print(f"  Pushed {len(ds):,} rows -> https://huggingface.co/datasets/{repo_id}")  # NOTE: uses print()


# ── Incremental daily push ──────────────────────────────────────────────────────


def load_feed_for_date(target_date: str, raw_feed_dir: Path) -> pd.DataFrame:
    """Load a single feed file for the given date string (YYYY-MM-DD).

    Returns an empty DataFrame if the file does not exist (e.g. download.R has
    not run yet for that date), rather than raising.  Callers should check
    df.empty before proceeding.

    Args:
        target_date:  Date string in YYYY-MM-DD format.
        raw_feed_dir: Directory containing feeds{date}.txt files (e.g. Path("output")).
    """
    path = raw_feed_dir / f"feeds{target_date}.txt"
    if not path.exists():
        print(f"Feed file not found: {path} - nothing to push.")  # NOTE: uses print()
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t")
    df["source_file"] = path.name
    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    print(f"Loaded {len(df):,} rows from {path.name}.")  # NOTE: uses print()
    return df


def push_incremental(new_df: pd.DataFrame, hf_repo: str, hf_token: str) -> None:
    """Append rows with new guids to the remote HF dataset.

    Idempotent — if all guids in new_df already exist in the remote dataset,
    prints a message and returns without modifying the dataset.

    Args:
        new_df:   DataFrame of new feed articles (must have a 'guid' column).
        hf_repo:  HuggingFace repo ID (e.g. "username/feeds").
        hf_token: HF_TOKEN with write access.

    Failure modes:
        - Remote dataset empty (cold-start not done): falls back gracefully
          by treating existing_guids as empty and pushing new_df in full.
        - All guids already present: prints message and returns early.
    """
    print(f"Loading existing dataset from {hf_repo} ...")  # NOTE: uses print()
    try:
        existing_ds = load_dataset(hf_repo, split="train", token=hf_token)
        existing_guids = set(existing_ds["guid"])
        print(f"  {len(existing_guids):,} existing guids in remote dataset.")  # NOTE: uses print()
    except Exception as exc:
        # Dataset may be empty on first incremental run — treat as empty
        print(f"  Could not load existing dataset ({exc}); treating as empty.")  # NOTE: uses print()
        existing_guids = set()
        existing_ds = None

    new_df = new_df[~new_df["guid"].isin(existing_guids)]
    if new_df.empty:
        print("No new articles - dataset already up to date.")  # NOTE: uses print()
        return

    print(f"  {len(new_df):,} new rows to append.")  # NOTE: uses print()
    new_ds = Dataset.from_pandas(new_df, preserve_index=False)

    if existing_ds is not None and len(existing_ds) > 0:
        from datasets import concatenate_datasets
        merged = concatenate_datasets([existing_ds, new_ds])
    else:
        merged = new_ds

    merged.push_to_hub(hf_repo, token=hf_token)
    print(f"Pushed. Dataset now has {len(merged):,} rows.")  # NOTE: uses print()
    print(f"Dataset: https://huggingface.co/datasets/{hf_repo}")  # NOTE: uses print()
