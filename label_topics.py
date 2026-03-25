"""
label_topics.py

Assign LLM labels to all unlabeled topic_ids in topic_trends.tsv without
re-clustering. Uses the existing data/topic_clusters/{date}.json files
(written by cluster_topics.py) to fetch article titles per topic.

Flow:
  1. Load topic_trends.tsv — find all unique topic_ids
  2. Load topic_labels.json — already-labeled ids are skipped (cache-first)
  3. For each unlabeled topic_id, collect up to 15 article titles from the
     most recent topic_clusters/{date}.json that contains it
  4. Call get_label (gpt-4o-mini) — result saved to cache immediately
  5. Rewrite topic_trends.tsv with labels populated
  6. Save updated topic_labels.json

Usage:
    python label_topics.py           # label all unlabeled topics
    python label_topics.py --dry-run # show counts, no API calls
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT  = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from cluster_topics import get_label, load_label_cache, save_label_cache
from constants import TOPIC_CLUSTERS_DIR, TOPIC_LABELS_FILE, TOPIC_TRENDS_FILE


def _collect_titles(topic_id: str, clusters_dir: Path, n: int = 15) -> list[str]:
    """Return up to n article titles for topic_id from the most recent cluster file."""
    # Walk files in descending date order — use the most recent appearance
    for path in sorted(clusters_dir.glob("*.json"), reverse=True):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        titles = [
            r["title"] for r in rows
            if r.get("topic_id") == topic_id and r.get("title")
        ]
        if titles:
            return titles[:n]
    return []


def label_topics(dry_run: bool = False) -> None:
    trends_path  = Path(TOPIC_TRENDS_FILE)
    labels_path  = Path(TOPIC_LABELS_FILE)
    clusters_dir = Path(TOPIC_CLUSTERS_DIR)

    if not trends_path.exists():
        print("ERROR: topic_trends.tsv not found. Run backfill.py --phase1-only first.")
        sys.exit(1)

    df    = pd.read_csv(trends_path, sep="\t")
    cache = load_label_cache(labels_path)

    all_ids    = df["topic_id"].unique().tolist()
    unlabeled  = [tid for tid in all_ids if not cache.get(tid)]

    print(f"Total unique topic_ids : {len(all_ids)}")
    print(f"Already labeled        : {len(all_ids) - len(unlabeled)}")
    print(f"Need labeling          : {len(unlabeled)}")

    if dry_run:
        print("\n--dry-run: no API calls made.")
        return

    if not unlabeled:
        print("All topics already labeled.")
    else:
        for i, tid in enumerate(unlabeled, 1):
            titles = _collect_titles(tid, clusters_dir)
            if not titles:
                print(f"  [{i:>4}/{len(unlabeled)}] {tid[:8]}...  no articles found, skipping")
                continue

            label = get_label(tid, cache, titles)
            print(f"  [{i:>4}/{len(unlabeled)}] {tid[:8]}...  -> {label!r}")

            # Save after every label so a crash loses at most one
            save_label_cache(cache, labels_path)

        print(f"\nLabeled {len(unlabeled)} topics. Cache saved to {labels_path}")

    # ── Update topic_trends.tsv with labels ──────────────────────────────────
    before_labeled = df["topic_label"].notna().sum()
    df["topic_label"] = df["topic_id"].map(cache)
    after_labeled  = df["topic_label"].notna().sum()

    df.to_csv(trends_path, sep="\t", index=False)
    print(f"\ntopic_trends.tsv updated:")
    print(f"  Rows with label before : {before_labeled}")
    print(f"  Rows with label after  : {after_labeled}")
    print(f"  Total rows             : {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label unlabeled topics in topic_trends.tsv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts only, no API calls")
    args = parser.parse_args()
    label_topics(dry_run=args.dry_run)
