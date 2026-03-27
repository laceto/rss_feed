# Plan: Publish CNBC Feed Data to Hugging Face Datasets

Migrate the daily RSS feed files from `output/feeds{date}.txt` (local repo)
to a public Hugging Face Dataset so any Python project can consume the data
with a single import, without cloning this repo.

---

## Goal

```
Current:  output/feeds{date}.txt  committed to rss_feed (mixed with analysis code)
Target:   hf://datasets/{your-hf-username}/cnbc-feeds  (standalone, daily-updated)
```

Consumers get the full feed history with one line:

```python
from datasets import load_dataset
ds = load_dataset("your-username/cnbc-feeds", split="train")
```

---

## Dataset Schema

Each row is one RSS article. Columns mirror the existing TSV schema exactly
so downstream scripts (`create_batch_files_v2.py`, `embed_feeds.py`) need
zero changes if they switch from reading local TSVs to loading from HF.

| Column | Type | Notes |
|---|---|---|
| `title` | string | Article headline |
| `description` | string | Article body/summary |
| `link` | string | Canonical URL |
| `guid` | string | **Dedup key** — stable RSS identifier |
| `type` | string | RSS item type |
| `id` | string | Internal RSS ID |
| `sponsored` | bool | True if sponsored content |
| `pubDate` | string | Publication date `YYYY-MM-DD` |
| `source_file` | string | Origin filename e.g. `feeds2025-10-15.txt` |

Total size estimate: ~8,500 rows now, growing ~150/day. At ~300 bytes/row
≈ 2.5 MB Parquet (compressed) — trivial for HF free tier.

---

## Prerequisites

```bash
pip install datasets huggingface_hub
```

Create a HF account at huggingface.co, then generate a write token:
  Settings → Access Tokens → New token (role: write)

Add it to `.env`:

```
HF_TOKEN=hf_...
HUGGINGFACE_REPO=your-username/cnbc-feeds
```

And to GitHub Actions secrets: `HF_TOKEN`, `HUGGINGFACE_REPO`.

---

## Step 1 — Create the Dataset Repo (once)

```python
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id=os.environ["HUGGINGFACE_REPO"],
    repo_type="dataset",
    private=False,   # set True if you want a private dataset
    exist_ok=True,
)
print("Dataset repo ready.")
```

---

## Step 2 — Cold-Start Push (`push_feeds_to_hf.py`)

Reads all existing `output/feeds*.txt` files, deduplicates on `guid`,
and pushes the full history as a Parquet dataset in one shot.

```python
"""
push_feeds_to_hf.py

Cold-start: push all historical feed files to the HF Dataset repo.
Run once. After this, only the daily incremental push is needed.

Usage:
    python push_feeds_to_hf.py             # push all feeds
    python push_feeds_to_hf.py --dry-run   # show row count only
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
HF_REPO      = os.environ["HUGGINGFACE_REPO"]   # "username/cnbc-feeds"
HF_TOKEN     = os.environ["HF_TOKEN"]


def load_all_feeds() -> pd.DataFrame:
    files = sorted(RAW_FEED_DIR.glob("feeds*.txt"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep="\t")
        df["source_file"] = f.name
        df["pubDate"] = pd.to_datetime(df["pubDate"]).dt.strftime("%Y-%m-%d")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["guid"])
    print(f"Loaded {before} rows, {len(combined)} after guid dedup.")
    return combined


def push(df: pd.DataFrame) -> None:
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"Pushed {len(ds)} rows to {HF_REPO}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    df = load_all_feeds()
    if args.dry_run:
        print("--dry-run: no push.")
    else:
        push(df)
```

---

## Step 3 — Daily Incremental Push (`push_new_feeds_to_hf.py`)

Runs after `download.R` in the daily pipeline. Downloads the current HF
dataset, appends only rows with new `guid` values, pushes back.
Idempotent: re-running for the same date is safe.

```python
"""
push_new_feeds_to_hf.py

Incremental update: append today's new feed articles to the HF Dataset.
Skips guids already present in the remote dataset.

Usage:
    python push_new_feeds_to_hf.py                        # today's feed
    python push_new_feeds_to_hf.py --date 2025-10-15      # specific date
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from dotenv import load_dotenv

load_dotenv()

RAW_FEED_DIR = Path("output")
HF_REPO      = os.environ["HUGGINGFACE_REPO"]
HF_TOKEN     = os.environ["HF_TOKEN"]


def load_new_feed(target_date: str) -> pd.DataFrame:
    path = RAW_FEED_DIR / f"feeds{target_date}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Feed file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    df["source_file"] = path.name
    df["pubDate"] = pd.to_datetime(df["pubDate"]).dt.strftime("%Y-%m-%d")
    return df


def push_incremental(new_df: pd.DataFrame) -> None:
    # Load existing dataset to get known guids
    print(f"Loading existing dataset from {HF_REPO}...")
    existing_ds = load_dataset(HF_REPO, split="train")
    existing_guids = set(existing_ds["guid"])
    print(f"  {len(existing_guids)} existing guids.")

    # Filter to genuinely new rows only
    new_df = new_df[~new_df["guid"].isin(existing_guids)]
    if new_df.empty:
        print("No new articles — dataset already up to date.")
        return

    print(f"  {len(new_df)} new rows to append.")
    new_ds  = Dataset.from_pandas(new_df, preserve_index=False)
    merged  = concatenate_datasets([existing_ds, new_ds])
    merged.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"Pushed. Dataset now has {len(merged)} rows.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=str(date.today()))
    args = p.parse_args()

    df = load_new_feed(args.date)
    push_incremental(df)
```

---

## Step 4 — Wire into GitHub Actions

Add a step in the `daily-pipeline` workflow **after** `download.R` and
**before** `create_batch_files_v2.py`:

```yaml
- name: Push new feeds to Hugging Face
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
    HUGGINGFACE_REPO: ${{ secrets.HUGGINGFACE_REPO }}
  run: |
    pip install datasets huggingface_hub
    python push_new_feeds_to_hf.py --date ${{ env.TODAY }}
```

The `TODAY` env var should already be set by the existing workflow
(`TODAY: $(date +%Y-%m-%d)`).

---

## Step 5 — Consumer API

Any Python project on any machine can now read the data:

```python
# ── Load full history ──────────────────────────────────────────────
from datasets import load_dataset

ds = load_dataset("your-username/cnbc-feeds", split="train")
df = ds.to_pandas()
print(df.shape)           # (8500+, 9)
print(df["pubDate"].max()) # most recent date

# ── Filter by date ─────────────────────────────────────────────────
ds_oct = ds.filter(lambda row: row["pubDate"].startswith("2025-10"))
print(len(ds_oct))

# ── Streaming (large datasets, no full download) ───────────────────
ds_stream = load_dataset("your-username/cnbc-feeds", split="train", streaming=True)
for row in ds_stream.take(5):
    print(row["title"])

# ── Load as pandas directly ────────────────────────────────────────
import pandas as pd
from datasets import load_dataset
df = load_dataset("your-username/cnbc-feeds", split="train").to_pandas()
```

No API key, no repo clone, no file path dependency.

---

## Step 6 — Update rss_feed Pipeline (optional, later)

The existing pipeline reads from `output/feeds*.txt` locally. Once the HF
dataset is established and stable, `create_batch_files_v2.py` can optionally
fall back to it if local files are absent:

```python
# In create_batch_files_v2.py — future optional change
def load_raw_feeds() -> pd.DataFrame:
    local_files = sorted(RAW_FEED_DIR.glob("feeds*.txt"))
    if local_files:
        # current behaviour — fast, no network call
        ...
    else:
        # fallback: load from HF (e.g. in a clean CI environment)
        from datasets import load_dataset
        return load_dataset(os.environ["HUGGINGFACE_REPO"], split="train").to_pandas()
```

This is a **future** step — do not change the primary pipeline until the HF
dataset has been live and validated for at least one week.

---

## Run Order

```bash
# 1. Install deps (once)
pip install datasets huggingface_hub

# 2. Add to .env
echo "HF_TOKEN=hf_..." >> .env
echo "HUGGINGFACE_REPO=your-username/cnbc-feeds" >> .env

# 3. Create HF repo (once)
python -c "
from huggingface_hub import HfApi
from dotenv import load_dotenv; import os; load_dotenv()
HfApi(token=os.environ['HF_TOKEN']).create_repo(
    os.environ['HUGGINGFACE_REPO'], repo_type='dataset', exist_ok=True)
print('done')
"

# 4. Cold-start push (once, ~1 min for 8500 rows)
python push_feeds_to_hf.py

# 5. Verify
python -c "
from datasets import load_dataset
ds = load_dataset('your-username/cnbc-feeds', split='train')
print(len(ds), 'rows | latest:', max(ds['pubDate']))
"

# 6. Wire daily incremental push into GitHub Actions (see Step 4)
```

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| HF API rate limit on daily push (~150 rows) | Tiny payload; free tier handles this easily |
| `load_dataset` re-downloads full dataset on every incremental push | Acceptable at current size; switch to Parquet shard-per-day approach if dataset exceeds 1 GB |
| Private dataset requires HF_TOKEN on all consumer machines | Keep public for now; switch to private + token if data becomes sensitive |
| HF outage blocks daily pipeline | `push_new_feeds_to_hf.py` is a non-blocking step; main pipeline continues if it fails |
| Duplicate guids from re-running the same date | Dedup on `guid` before every push — idempotent by design |

---

## Key Open Question

Should `output/feeds*.txt` files be **removed from this repo** once the HF
dataset is the source of truth, or kept as a local cache for performance?
Keeping them avoids a network call in CI but doubles storage. Decide after
the HF dataset has been live for one week.
