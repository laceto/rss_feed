"""
label_topics.py

Assign LLM labels to all unlabeled topic_ids in topic_trends.tsv without
re-clustering. Uses kitai.batch to submit all labeling calls in a single
OpenAI Batch API job (50% cheaper, async).

Flow:
  1. Load topic_trends.tsv — find all unique topic_ids
  2. Load topic_labels.json — already-labeled ids are skipped (cache-first)
  3. For each unlabeled topic_id, collect up to 15 article titles from the
     most recent topic_clusters/{date}.json that contains it
  4. Build one batch task per topic using the same prompt as _label_via_llm
  5. Submit via kitai.batch.submit_batch_job → poll until complete
  6. Parse results → update label cache + rewrite topic_trends.tsv

Usage:
    python label_topics.py           # label all unlabeled topics
    python label_topics.py --dry-run # show counts, no API calls

custom_id convention: "label-{topic_id}"

Invariants:
    - Idempotent: already-labeled topics are never re-submitted.
    - On partial failure (some batch items error), successfully labeled
      topics are still saved before raising.
    - topic_labels.json is written atomically via save_label_cache.
    - topic_trends.tsv is rewritten in-place (same schema).

Failure modes:
    - topic_trends.tsv absent: exits with error message.
    - No topic_clusters/{date}.json files: topics get no titles → skipped.
    - Batch item error: logged, topic left unlabeled for next run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kitai.batch import submit_batch_job, poll_until_complete, download_batch_results
from pipeline.cluster_topics import load_label_cache, save_label_cache
from pipeline.constants import TOPIC_CLUSTERS_DIR, TOPIC_LABELS_FILE, TOPIC_TRENDS_FILE

_MODEL      = "gpt-4o-mini"
_MAX_TOKENS = 20
_TEMP       = 0.2
_N_TITLES   = 15


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_titles(topic_id: str, clusters_dir: Path, n: int = _N_TITLES) -> list[str]:
    """Return up to n article titles for topic_id from the most recent cluster file."""
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


def _build_prompt(titles: list[str]) -> str:
    headlines = "\n".join(f"- {t}" for t in titles)
    return (
        "These are news headlines from the same topic cluster:\n"
        f"{headlines}\n\n"
        "Give a short topic label of 3-5 words. "
        "Be specific (e.g. 'Fed rate pause bets', not 'economic news'). "
        "Return only the label, no explanation."
    )


def _build_tasks(unlabeled: list[str], clusters_dir: Path) -> tuple[list[dict], list[str]]:
    """Build batch tasks for all unlabeled topic_ids.

    Returns (tasks, skipped_ids) where skipped_ids had no article titles.
    """
    tasks: list[dict] = []
    skipped: list[str] = []

    for tid in unlabeled:
        titles = _collect_titles(tid, clusters_dir)
        if not titles:
            skipped.append(tid)
            continue
        tasks.append({
            "custom_id": f"label-{tid}",
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":       _MODEL,
                "messages":    [{"role": "user", "content": _build_prompt(titles)}],
                "max_tokens":  _MAX_TOKENS,
                "temperature": _TEMP,
            },
        })

    return tasks, skipped


def _parse_results(results: list[dict]) -> tuple[dict[str, str], int]:
    """Extract topic_id → label from batch results.

    Returns (labels_dict, n_errors).
    custom_id format: "label-{topic_id}"
    """
    labels: dict[str, str] = {}
    n_errors = 0

    for item in results:
        cid = item.get("custom_id", "")
        if not cid.startswith("label-"):
            continue
        topic_id = cid[len("label-"):]

        if item.get("error"):
            print(f"  [batch error] {topic_id[:8]}... : {item['error']}")
            n_errors += 1
            continue

        try:
            label = (
                item["response"]["body"]["choices"][0]["message"]["content"].strip()
            )
            labels[topic_id] = label
        except (KeyError, IndexError) as exc:
            print(f"  [parse error] {topic_id[:8]}... : {exc}")
            n_errors += 1

    return labels, n_errors


# ── Main ──────────────────────────────────────────────────────────────────────

def label_topics(dry_run: bool = False) -> None:
    trends_path  = Path(TOPIC_TRENDS_FILE)
    labels_path  = Path(TOPIC_LABELS_FILE)
    clusters_dir = Path(TOPIC_CLUSTERS_DIR)

    if not trends_path.exists():
        print("ERROR: topic_trends.tsv not found. Run backfill.py --phase1-only first.")
        sys.exit(1)

    df    = pd.read_csv(trends_path, sep="\t")
    cache = load_label_cache(labels_path)

    all_ids   = df["topic_id"].unique().tolist()
    unlabeled = [tid for tid in all_ids if not cache.get(tid)]

    print(f"Total unique topic_ids : {len(all_ids)}")
    print(f"Already labeled        : {len(all_ids) - len(unlabeled)}")
    print(f"Need labeling          : {len(unlabeled)}")

    if dry_run:
        print("\n--dry-run: no API calls made.")
        return

    if not unlabeled:
        print("All topics already labeled.")
    else:
        tasks, skipped = _build_tasks(unlabeled, clusters_dir)

        if skipped:
            print(f"\nSkipped {len(skipped)} topic(s) with no article titles.")

        if not tasks:
            print("No tasks to submit.")
        else:
            print(f"\nSubmitting {len(tasks)} tasks to OpenAI Batch API ...")
            client = OpenAI()
            job_id = submit_batch_job(client, tasks)
            print(f"Batch job submitted: {job_id}")
            print("Waiting for completion (polls every 30s) ...")

            statuses = poll_until_complete(client, [job_id], poll_interval_seconds=30)
            if statuses[job_id]["status"] != "completed":
                print(
                    f"ERROR: batch job ended with status '{statuses[job_id]['status']}'. "
                    "Check OpenAI dashboard."
                )
                sys.exit(1)

            results = download_batch_results(client, job_id)
            new_labels, n_errors = _parse_results(results)

            cache.update(new_labels)
            save_label_cache(cache, labels_path)

            print(f"\nLabeled  : {len(new_labels)}")
            print(f"Errors   : {n_errors}")
            print(f"Skipped  : {len(skipped)} (no articles)")
            print(f"Cache saved to {labels_path}")

    # ── Update topic_trends.tsv with labels ──────────────────────────────────
    before = df["topic_label"].notna().sum()
    df["topic_label"] = df["topic_id"].map(cache)
    after  = df["topic_label"].notna().sum()

    df.to_csv(trends_path, sep="\t", index=False)
    print(f"\ntopic_trends.tsv updated:")
    print(f"  Rows with label before : {before}")
    print(f"  Rows with label after  : {after}")
    print(f"  Total rows             : {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label unlabeled topics via OpenAI Batch API")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts only, no API calls")
    args = parser.parse_args()
    label_topics(dry_run=args.dry_run)
