"""
create_batch_briefings.py

Build OpenAI Batch API tasks for RAG narrative generation across all dates
that have topic spike data but no saved briefing yet.

Architecture:
  1. Load FAISS + BM25 once locally (no API calls for this step)
  2. For each unprocessed date, detect spiking topics via get_emerging_topics()
  3. For each spike, run hybrid retrieval locally → embed context into the task prompt
  4. Submit all tasks in one batch job via kitai.batch.submit_batch_job()
  5. Persist batch ID + spike metadata sidecar for the collection step

custom_id convention:  "briefing-YYYY-MM-DD-{topic_id[:8]}"
                        → retrieve_batch_briefings.py routes by this key

Sidecar file (data/pending_briefings_meta.json):
  Maps each custom_id → {date, topic_id, label, spike_ratio, article_count}
  so the collection step can reconstruct the full briefing without re-running
  get_emerging_topics().

Invariants:
  - Never writes to topic_trends.tsv or any pipeline state file.
  - Skips dates that already have data/briefings/{date}.json.
  - Retrieval uses strategy="none" (no query-translation API calls here).
  - Fails fast if topic_trends.tsv is absent.

Usage:
    python create_batch_briefings.py             # all unprocessed dates
    python create_batch_briefings.py --top 3     # top N spikes per date
    python create_batch_briefings.py --start 2025-10-01 --end 2025-12-31
    python create_batch_briefings.py --dry-run   # show counts, no submission

Failure modes:
  - OPENAI_API_KEY missing: fails on kitai.batch.submit_batch_job()
  - FAISS missing: fails on _get_resources() with FileNotFoundError
  - topic_trends.tsv absent: exits with instructions
  - All dates already processed: exits cleanly with no submission
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cluster_topics import get_emerging_topics
from pipeline.constants import (
    BATCH_FILE_BRIEFINGS,
    BRIEFINGS_DIR,
    BRIEFINGS_BATCH_META_FILE,
    PENDING_BRIEFINGS_BATCH_FILE,
    TOPIC_TRENDS_FILE,
)
from pipeline.hybrid_rag import (
    K_BM25,
    K_SEMANTIC,
    WEIGHTS_SPARSE,
    CHAT_MODEL,
    _get_resources,
)
from kitai.batch import submit_batch_job
from kitai.retriever import (
    create_BM25retriever_from_docs,
    create_hybrid_retriever,
    create_retriever,
    reorder_docs,
)

# ── Configuration ─────────────────────────────────────────────────────────────

TOP_N_DEFAULT      = 5      # max spikes per date
MAX_CONTEXT_CHARS  = 8_000  # safety cap on embedded context per task


# ── Retrieval helper ──────────────────────────────────────────────────────────

def _retrieve_docs_for_label(label: str, res: dict) -> list[dict]:
    """Run hybrid FAISS+BM25 retrieval for a topic label; return source dicts.

    Uses strategy='none' — no query-translation API calls.  The LLM answer
    is deferred to the batch job; only retrieval runs locally here.

    Args:
        label: Human-readable topic label used as the retrieval query.
        res:   Resource dict from hybrid_rag._get_resources().

    Returns:
        List of source dicts with keys: title, date, link, snippet, guid.
        Empty list if retrieval fails.
    """
    try:
        bm25_ret   = create_BM25retriever_from_docs(docs=res["corpus"], k=K_BM25)
        vector_ret = create_retriever(
            vs=res["vs"],
            search_type="similarity",
            search_kwargs={"k": K_SEMANTIC},
        )
        hybrid = create_hybrid_retriever(
            sparse_retriever=bm25_ret,
            semantic_retriever=vector_ret,
            weights_sparse=WEIGHTS_SPARSE,
        )
        raw_docs = hybrid.invoke(label)
        ordered  = reorder_docs(raw_docs)
        return [
            {
                "title":   doc.metadata.get("title", ""),
                "date":    doc.metadata.get("date", ""),
                "link":    doc.metadata.get("link", ""),
                "snippet": doc.page_content[:200].replace("\n", " "),
                "guid":    doc.metadata.get("guid", ""),
            }
            for doc in ordered
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] retrieval failed for {label!r}: {exc}")
        return []


def _build_prompt(label: str, sources: list[dict]) -> str:
    """Format retrieved sources into a system+user message pair for the batch task."""
    query = (
        f"What is happening with {label}? "
        f"Summarise the key developments and market implications in 3-4 sentences."
    )
    # Truncate joined context to avoid token-limit rejections
    context_parts = [
        f"{s['date']}: {s['title']}: {s['snippet']}"
        for s in sources
        if s.get("title")
    ]
    context = "\n\n".join(context_parts)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    return {
        "system": (
            "You are a financial news analyst. "
            "Use the provided news excerpts to answer the question. "
            "If the context lacks enough information, say so briefly."
        ),
        "user": f"Context:\n{context}\n\nQuestion: {query}",
        "query": query,  # kept for metadata; not sent to OpenAI
    }


# ── Batch task builder ────────────────────────────────────────────────────────

def _custom_id(d: str, topic_id: str) -> str:
    return f"briefing-{d}-{topic_id[:8]}"


def build_tasks(
    trends_df: pd.DataFrame,
    dates: list[str],
    top_n: int,
    res: dict,
) -> tuple[list[dict], dict]:
    """Build batch tasks and spike metadata for all unprocessed dates.

    Returns:
        tasks    — list of OpenAI Batch API task dicts
        metadata — {custom_id: {date, topic_id, label, spike_ratio,
                                article_count, sources}} for the sidecar file
    """
    tasks:    list[dict] = []
    metadata: dict       = {}

    for d in sorted(dates):
        run_date = datetime.strptime(d, "%Y-%m-%d").date()
        spikes   = get_emerging_topics(run_date, trends_df)
        if not spikes:
            print(f"  {d}: no spikes detected — skipping")
            continue

        spikes = spikes[:top_n]
        print(f"  {d}: {len(spikes)} spike(s)")

        for spike in spikes:
            tid   = spike["topic_id"]
            label = (spike["label"] if isinstance(spike["label"], str) else "") or ""
            if not label:
                print(f"    [{tid[:8]}] no label — skipping (run label_topics.py first)")
                continue

            # Run retrieval locally — no API calls
            sources = _retrieve_docs_for_label(label, res)
            if not sources:
                print(f"    [{tid[:8]}] {label!r}: no sources retrieved — skipping")
                continue

            prompt = _build_prompt(label, sources)
            cid    = _custom_id(d, tid)

            tasks.append({
                "custom_id": cid,
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":       CHAT_MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user",   "content": prompt["user"]},
                    ],
                },
            })
            metadata[cid] = {
                "date":          d,
                "topic_id":      tid,
                "label":         label,
                "spike_ratio":   spike["spike_ratio"],
                "article_count": spike["article_count"],
                "sources":       sources,  # pre-retrieved; stored for collect step
            }
            print(f"    [{tid[:8]}] {label!r} — {len(sources)} sources")

    return tasks, metadata


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit RAG briefing batch to OpenAI")
    p.add_argument("--start",   default=None, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end",     default=None, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--top",     type=int, default=TOP_N_DEFAULT,
                   help=f"Max spikes per date (default: {TOP_N_DEFAULT})")
    p.add_argument("--dry-run", action="store_true",
                   help="Show counts only — no FAISS load, no submission")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print("=== Briefing Batch Submission ===")

    # ── Load trends ───────────────────────────────────────────────────────────
    trends_path = Path(TOPIC_TRENDS_FILE)
    if not trends_path.exists():
        print(f"ERROR: {trends_path} not found. Run cluster_topics.py first.")
        sys.exit(1)

    trends_df = pd.read_csv(trends_path, sep="\t")
    all_dates = sorted(trends_df["date"].unique().tolist())

    # ── Apply date filters ────────────────────────────────────────────────────
    if args.start:
        all_dates = [d for d in all_dates if d >= args.start]
    if args.end:
        all_dates = [d for d in all_dates if d <= args.end]

    # ── Skip dates with existing briefings ───────────────────────────────────
    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    unprocessed = [
        d for d in all_dates
        if not (BRIEFINGS_DIR / f"{d}.json").exists()
    ]

    print(f"Dates in trends  : {len(all_dates)}")
    print(f"Already briefed  : {len(all_dates) - len(unprocessed)}")
    print(f"To process       : {len(unprocessed)}")

    if not unprocessed:
        print("All dates already have briefings. Nothing to submit.")
        return

    if args.dry_run:
        print("\n--dry-run: no FAISS load, no API submission.")
        return

    # ── Load FAISS + BM25 once ───────────────────────────────────────────────
    print("\nLoading FAISS + BM25 resources (one-time init)...")
    res = _get_resources()
    print(f"Resources ready: {len(res['corpus'])} documents in corpus.")

    # ── Build tasks ───────────────────────────────────────────────────────────
    print(f"\nBuilding batch tasks (top {args.top} spikes per date)...")
    tasks, metadata = build_tasks(trends_df, unprocessed, args.top, res)

    if not tasks:
        print("\nNo tasks built — all spikes were skipped (no labels / no sources).")
        print("Run label_topics.py to label any unlabeled topics, then retry.")
        return

    print(f"\nTotal tasks: {len(tasks)} across {len(set(m['date'] for m in metadata.values()))} date(s)")

    # ── Write debug JSONL ─────────────────────────────────────────────────────
    BATCH_FILE_BRIEFINGS.parent.mkdir(parents=True, exist_ok=True)
    with BATCH_FILE_BRIEFINGS.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"Debug JSONL saved -> {BATCH_FILE_BRIEFINGS}")

    # ── Submit via kitai.batch ────────────────────────────────────────────────
    client   = res["openai_client"]
    print("Submitting batch to OpenAI...")
    batch_id = submit_batch_job(client, tasks)
    print(f"Batch submitted: {batch_id}")

    # ── Persist sentinel files ────────────────────────────────────────────────
    PENDING_BRIEFINGS_BATCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    PENDING_BRIEFINGS_BATCH_FILE.write_text(batch_id, encoding="utf-8")
    print(f"Batch ID saved  -> {PENDING_BRIEFINGS_BATCH_FILE}")

    BRIEFINGS_BATCH_META_FILE.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Spike metadata  -> {BRIEFINGS_BATCH_META_FILE}")

    print(f"\nDone. {len(tasks)} tasks submitted under batch {batch_id}.")
    print("Run retrieve_batch_briefings.py once the batch completes.")


if __name__ == "__main__":
    main()
