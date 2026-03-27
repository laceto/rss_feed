"""
embed_feeds.py
Builds and incrementally updates a FAISS vectorstore from CNBC feed articles
using the OpenAI Batch API via kitai.

Behaviour:
- Cold start (no existing vectorstore): embeds ALL feed articles and creates
  the store from scratch.
- Incremental (store already exists): embeds only articles whose guid is not
  yet in the registry, then appends them to the existing store via
  FAISS.add_embeddings (no rebuild required).
- No-op: exits 0 cleanly if no new articles are found.

Run:
    python embed_feeds.py

Environment:
    OPENAI_API_KEY  — required (loaded from .env)
    LOG_LEVEL       — optional, default INFO (set to DEBUG for batch poll detail)

Invariants:
- metadata["id"] is a monotonic integer, never reused across runs.
- feeds_registry.tsv is saved AFTER store.save_local() — a write failure
  leaves the registry consistent with the previous successful state.
- The registry's guid column is the single source of truth for what is in
  the FAISS store. FAISS contents ≡ guids in feeds_registry.tsv.
- Deduplication key: guid (stable RSS-feed identifier per article).

Failure modes:
- Batch API error/expiry  → RuntimeError; registry not written; next run retries.
- Partial batch (some items fail) → dropped silently; retried next run.
- FAISS save fails        → registry not written; consistent state preserved.
- All embeddings fail     → exits 1 without touching store or registry.
"""

import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from constants import FEEDS_REGISTRY_FILE, RAW_FEED_DIR, VECTORSTORE_DIR
from pipeline.vectorstore_io import (
    EMBED_MODEL,
    align_pairs_to_docs,
    assign_ids,
    build_documents,
    find_new_articles,
    init_vectorstore,
    load_feed_articles,
    load_registry,
    run_embedding_batch,
    save_registry,
    update_vectorstore,
)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBED_DIMENSIONS = 1536   # must match EMBED_MODEL; used by FAISS + OpenAIEmbeddings


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    client           = OpenAI()
    embeddings_model = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS)

    # 1. Load existing state
    registry = load_registry(FEEDS_REGISTRY_FILE)

    # 2. Discover new articles
    try:
        all_df = load_feed_articles(RAW_FEED_DIR)
    except FileNotFoundError as exc:
        log.error("%s. Exiting.", exc)
        sys.exit(0)

    new_df = find_new_articles(all_df, registry)
    if new_df.empty:
        log.info("No new articles to embed. Exiting.")
        sys.exit(0)

    new_df = assign_ids(new_df, registry)

    # 3. Build LangChain Documents
    docs = build_documents(new_df)

    # 4. Embed via OpenAI Batch API
    pairs = run_embedding_batch(docs, client)

    # 5. Align embeddings to doc order; drop any API-level failures
    aligned_pairs, aligned_docs = align_pairs_to_docs(pairs, docs)

    if not aligned_docs:
        log.error("All embeddings failed — nothing to add to the store. Exiting.")
        sys.exit(1)

    # 6. Init or incrementally update the vectorstore
    if VECTORSTORE_DIR.exists():
        store = update_vectorstore(aligned_pairs, aligned_docs, embeddings_model, VECTORSTORE_DIR)
    else:
        store = init_vectorstore(aligned_docs, aligned_pairs, embeddings_model)

    # 7. Persist the store BEFORE writing the registry.
    #    If save_local fails, the registry stays at its previous consistent state
    #    and the next run will re-embed the same articles.
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTORSTORE_DIR))
    log.info("Vectorstore saved to %s.", VECTORSTORE_DIR)

    # 8. Append new rows to the registry (atomic write via save_registry)
    new_rows = pd.DataFrame({
        "id":    [doc.metadata["id"]    for doc in aligned_docs],
        "date":  [doc.metadata["date"]  for doc in aligned_docs],
        "title": [doc.metadata["title"] for doc in aligned_docs],
        "link":  [doc.metadata["link"]  for doc in aligned_docs],
        "guid":  [doc.metadata["guid"]  for doc in aligned_docs],
    })
    updated_registry = pd.concat([registry, new_rows], ignore_index=True)
    save_registry(updated_registry, FEEDS_REGISTRY_FILE)

    log.info(
        "[done] Embedded %d new articles. Total in registry: %d.",
        len(aligned_docs),
        len(updated_registry),
    )


if __name__ == "__main__":
    main()
