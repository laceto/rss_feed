"""Briefing batch task building and submission utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cluster_topics import get_emerging_topics
from constants import (
    BATCH_FILE_BRIEFINGS,
    BRIEFINGS_BATCH_META_FILE,
    BRIEFINGS_DIR,
    PENDING_BRIEFINGS_BATCH_FILE,
    TOPIC_TRENDS_FILE,
)
from hybrid_rag import (
    CHAT_MODEL,
    K_BM25,
    K_SEMANTIC,
    WEIGHTS_SPARSE,
    get_resources,
)
from kitai.batch import submit_batch_job
from kitai.retriever import (
    create_BM25retriever_from_docs,
    create_hybrid_retriever,
    create_retriever,
    reorder_docs,
)

TOP_N_DEFAULT = 5
MAX_CONTEXT_CHARS = 8_000


def retrieve_docs_for_label(label: str, res: dict) -> list[dict]:
    """Run hybrid retrieval for a topic label and return compact source dicts."""
    try:
        bm25_ret = create_BM25retriever_from_docs(docs=res["corpus"], k=K_BM25)
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
        ordered = reorder_docs(raw_docs)
        return [
            {
                "title": doc.metadata.get("title", ""),
                "date": doc.metadata.get("date", ""),
                "link": doc.metadata.get("link", ""),
                "snippet": doc.page_content[:200].replace("\n", " "),
                "guid": doc.metadata.get("guid", ""),
            }
            for doc in ordered
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] retrieval failed for {label!r}: {exc}")
        return []


def build_prompt(label: str, sources: list[dict]) -> dict[str, str]:
    """Format retrieved sources into the messages used by the batch task."""
    query = (
        f"What is happening with {label}? "
        f"Summarise the key developments and market implications in 3-4 sentences."
    )
    context_parts = [
        f"{source['date']}: {source['title']}: {source['snippet']}"
        for source in sources
        if source.get("title")
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
        "query": query,
    }


def build_briefing_custom_id(date_str: str, topic_id: str) -> str:
    """Return the batch custom_id for a briefing task."""
    return f"briefing-{date_str}-{topic_id[:8]}"


def build_briefing_batch_tasks(
    trends_df: pd.DataFrame,
    dates: list[str],
    top_n: int,
    res: dict,
) -> tuple[list[dict], dict[str, dict]]:
    """Build batch tasks and sidecar metadata for the given briefing dates."""
    tasks: list[dict] = []
    metadata: dict[str, dict] = {}

    for date_str in sorted(dates):
        run_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        spikes = get_emerging_topics(run_date, trends_df)
        if not spikes:
            print(f"  {date_str}: no spikes detected — skipping")
            continue

        spikes = spikes[:top_n]
        print(f"  {date_str}: {len(spikes)} spike(s)")

        for spike in spikes:
            topic_id = spike["topic_id"]
            label = (spike["label"] if isinstance(spike["label"], str) else "") or ""
            if not label:
                print(f"    [{topic_id[:8]}] no label — skipping (run label_topics.py first)")
                continue

            sources = retrieve_docs_for_label(label, res)
            if not sources:
                print(f"    [{topic_id[:8]}] {label!r}: no sources retrieved — skipping")
                continue

            prompt = build_prompt(label, sources)
            custom_id = build_briefing_custom_id(date_str, topic_id)

            tasks.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": CHAT_MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                },
            })
            metadata[custom_id] = {
                "date": date_str,
                "topic_id": topic_id,
                "label": label,
                "spike_ratio": spike["spike_ratio"],
                "article_count": spike["article_count"],
                "sources": sources,
            }
            print(f"    [{topic_id[:8]}] {label!r} — {len(sources)} sources")

    return tasks, metadata


def run_briefing_batch_submission(
    start: str | None = None,
    end: str | None = None,
    top_n: int = TOP_N_DEFAULT,
    dry_run: bool = False,
    trends_path: Path = TOPIC_TRENDS_FILE,
    briefings_dir: Path = BRIEFINGS_DIR,
    batch_file: Path = BATCH_FILE_BRIEFINGS,
    pending_batch_file: Path = PENDING_BRIEFINGS_BATCH_FILE,
    metadata_file: Path = BRIEFINGS_BATCH_META_FILE,
) -> dict[str, Any]:
    """Run the end-to-end briefing batch submission flow."""
    trends_path = Path(trends_path)
    if not trends_path.exists():
        raise FileNotFoundError(f"{trends_path} not found. Run cluster_topics.py first.")

    trends_df = pd.read_csv(trends_path, sep="\t")
    all_dates = sorted(trends_df["date"].unique().tolist())

    if start:
        all_dates = [date_str for date_str in all_dates if date_str >= start]
    if end:
        all_dates = [date_str for date_str in all_dates if date_str <= end]

    briefings_dir.mkdir(parents=True, exist_ok=True)
    unprocessed = [
        date_str for date_str in all_dates
        if not (briefings_dir / f"{date_str}.json").exists()
    ]

    print(f"Dates in trends  : {len(all_dates)}")
    print(f"Already briefed  : {len(all_dates) - len(unprocessed)}")
    print(f"To process       : {len(unprocessed)}")

    summary: dict[str, Any] = {
        "all_dates": len(all_dates),
        "already_briefed": len(all_dates) - len(unprocessed),
        "to_process": len(unprocessed),
        "top_n": top_n,
        "dry_run": dry_run,
    }

    if not unprocessed:
        print("All dates already have briefings. Nothing to submit.")
        summary["status"] = "noop"
        return summary

    if dry_run:
        print("\n--dry-run: no FAISS load, no API submission.")
        summary["status"] = "dry_run"
        return summary

    print("\nLoading FAISS + BM25 resources (one-time init)...")
    res = get_resources()
    print(f"Resources ready: {len(res['corpus'])} documents in corpus.")

    print(f"\nBuilding batch tasks (top {top_n} spikes per date)...")
    tasks, metadata = build_briefing_batch_tasks(trends_df, unprocessed, top_n, res)

    if not tasks:
        print("\nNo tasks built — all spikes were skipped (no labels / no sources).")
        print("Run label_topics.py to label any unlabeled topics, then retry.")
        summary["status"] = "noop_no_tasks"
        summary["task_count"] = 0
        return summary

    task_dates = len({item["date"] for item in metadata.values()})
    print(f"\nTotal tasks: {len(tasks)} across {task_dates} date(s)")

    batch_file.parent.mkdir(parents=True, exist_ok=True)
    with batch_file.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")
    print(f"Debug JSONL saved -> {batch_file}")

    print("Submitting batch to OpenAI...")
    batch_id = submit_batch_job(res["openai_client"], tasks)
    print(f"Batch submitted: {batch_id}")

    pending_batch_file.parent.mkdir(parents=True, exist_ok=True)
    pending_batch_file.write_text(batch_id, encoding="utf-8")
    print(f"Batch ID saved  -> {pending_batch_file}")

    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Spike metadata  -> {metadata_file}")

    print(f"\nDone. {len(tasks)} tasks submitted under batch {batch_id}.")
    print("Run retrieve_batch_briefings.py once the batch completes.")

    summary.update({
        "status": "submitted",
        "task_count": len(tasks),
        "task_dates": task_dates,
        "batch_id": batch_id,
    })
    return summary
