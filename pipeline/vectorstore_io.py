"""Vectorstore build and incremental-update utilities for the FAISS feed index.

Extracted from embed_feeds.py.  All functions take explicit path parameters —
no module-level path constants are read inside function bodies, so every
function is testable without touching the filesystem.

Invariants (preserved from embed_feeds.py):
- metadata["id"] is a monotonic integer, never reused across runs.
- registry is written AFTER store.save_local() — a save failure leaves the
  registry consistent with the previous successful state.
- The registry's guid column is the single source of truth for FAISS contents.
- Deduplication key: guid (stable RSS-feed identifier per article).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from kitai.batch import (
    build_embedding_tasks,
    download_batch_results,
    parse_embedding_results,
    poll_until_complete,
    submit_batch_job,
)
from kitai.index import create_vectorstore

log = logging.getLogger(__name__)

# ── Module-level defaults (injectable as parameters) ──────────────────────────

EMBED_MODEL   = "text-embedding-3-small"
POLL_INTERVAL = 30   # seconds between batch status polls

REGISTRY_COLUMNS = ["id", "date", "title", "link", "guid"]


# ── Registry ──────────────────────────────────────────────────────────────────

def load_registry(registry_file: Path) -> pd.DataFrame:
    """Load the article registry, or return an empty frame if it does not exist.

    The registry is the ground truth for which articles are in the FAISS store.

    Args:
        registry_file: Path to feeds_registry.tsv.

    Returns:
        DataFrame with columns: id (int), date (str), title (str),
        link (str), guid (str).
    """
    if not registry_file.exists():
        log.info("Registry not found — starting fresh.")
        return pd.DataFrame(columns=REGISTRY_COLUMNS).astype({"id": int})

    df = pd.read_csv(registry_file, sep="\t", dtype={"id": int})
    log.info("Loaded registry: %d articles.", len(df))
    return df


def save_registry(registry: pd.DataFrame, registry_file: Path) -> None:
    """Write the registry as a TSV file using an atomic tmp → rename write.

    Must be called AFTER store.save_local() so that a store write failure
    leaves the registry at its previous consistent state.

    Args:
        registry:      DataFrame to write.
        registry_file: Destination path (feeds_registry.tsv).
    """
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = registry_file.with_suffix(".tmp")
    registry.to_csv(tmp_path, sep="\t", index=False)
    tmp_path.replace(registry_file)
    log.info("Registry saved: %d articles total.", len(registry))


# ── Feed loading ───────────────────────────────────────────────────────────────

def load_feed_articles(raw_feed_dir: Path) -> pd.DataFrame:
    """Load all feeds*.txt from raw_feed_dir, dedup on guid, add date column.

    Delegates glob + concat + dedup to pipeline.hf_io.load_feeds_from_files.
    Adds a 'date' column (YYYY-MM-DD) aliased from the normalised pubDate.

    Args:
        raw_feed_dir: Directory containing feeds*.txt files (e.g. output/).

    Returns:
        DataFrame with columns including: guid, title, description, link,
        pubDate (normalised YYYY-MM-DD string), date (alias of pubDate).

    Raises:
        FileNotFoundError: if no feeds*.txt files exist in raw_feed_dir.
    """
    from .hf_io import load_feeds_from_files
    df = load_feeds_from_files(raw_feed_dir)
    # hf_io normalises pubDate to YYYY-MM-DD in-place; alias as 'date'
    # so downstream functions can use row['date'] consistently.
    df["date"] = df["pubDate"]
    log.info("Loaded %d unique feed articles.", len(df))
    return df


# ── Dedup and ID assignment ────────────────────────────────────────────────────

def find_new_articles(all_df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose guid is not yet recorded in the registry.

    Args:
        all_df:   Full feed DataFrame.
        registry: Existing registry DataFrame.

    Returns:
        Subset of all_df with unembedded articles; empty DataFrame if none.
    """
    known_guids = set(registry["guid"]) if not registry.empty else set()
    new_df = all_df[~all_df["guid"].isin(known_guids)].copy()

    log.info(
        "%d total articles | %d already embedded | %d new",
        len(all_df),
        len(known_guids),
        len(new_df),
    )
    return new_df


def assign_ids(new_df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Assign globally unique monotonic integer IDs to new articles.

    Continues from max(registry.id) + 1, or starts at 0 on the first run.
    IDs are never reused.

    Args:
        new_df:   Articles to assign IDs to.
        registry: Existing registry (used to determine next_id).

    Returns:
        Copy of new_df with an 'id' column added.
    """
    next_id = int(registry["id"].max()) + 1 if not registry.empty else 0
    new_df = new_df.copy()
    new_df["id"] = range(next_id, next_id + len(new_df))
    log.info("Assigned IDs %d–%d.", next_id, next_id + len(new_df) - 1)
    return new_df


# ── Document building ──────────────────────────────────────────────────────────

def build_documents(new_df: pd.DataFrame) -> list[Document]:
    """Convert feed rows to LangChain Documents ready for batch embedding.

    Content format: "{date}: {title}: {description}"
    — matches the pattern used by create_batch_files_v2.py for consistency.

    Each document carries:
        metadata["id"]    (int)  — docstore key in the FAISS store
        metadata["date"]  (str)  — YYYY-MM-DD
        metadata["title"] (str)  — article headline
        metadata["link"]  (str)  — article URL
        metadata["guid"]  (str)  — stable RSS article identifier

    Args:
        new_df: DataFrame with id, date, title, description, link, guid columns.

    Returns:
        List of LangChain Document objects.
    """
    docs = []
    for row in new_df.itertuples(index=False):
        content = (
            f"{row.date}: "
            f"{str(row.title).strip()}: "
            f"{str(row.description).strip()}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "id":    int(row.id),
                "date":  str(row.date),
                "title": str(row.title),
                "link":  str(row.link),
                "guid":  str(row.guid),
            },
        ))

    log.info("Built %d documents for embedding.", len(docs))
    return docs


# ── Embedding batch ────────────────────────────────────────────────────────────

def run_embedding_batch(
    docs: list[Document],
    client: OpenAI,
    embed_model: str = EMBED_MODEL,
    poll_interval: int = POLL_INTERVAL,
) -> list[tuple[str, list[float]]]:
    """Submit docs to the OpenAI Batch API and return (custom_id, embedding) pairs.

    Uses kitai.batch throughout:
        build_embedding_tasks → submit_batch_job → poll_until_complete
        → download_batch_results → parse_embedding_results

    The custom_id is doc.metadata["id"] directly (raw value, no prefix).

    Args:
        docs:          Documents to embed.
        client:        OpenAI client instance.
        embed_model:   Embedding model name (default: text-embedding-3-small).
        poll_interval: Seconds between batch status polls (default: 30).

    Returns:
        List of (custom_id, embedding_vector) pairs.

    Raises:
        RuntimeError: If the batch job does not reach 'completed' status.
    """
    tasks  = build_embedding_tasks(docs, model=embed_model)
    job_id = submit_batch_job(client, tasks)
    log.info("Submitted embedding batch: %s (%d tasks)", job_id, len(tasks))

    statuses = poll_until_complete(client, [job_id], poll_interval_seconds=poll_interval)
    if statuses[job_id]["status"] != "completed":
        raise RuntimeError(
            f"Embedding batch {job_id} ended with status '{statuses[job_id]['status']}'. "
            "Check the OpenAI dashboard for details."
        )

    results = download_batch_results(client, job_id)
    pairs   = parse_embedding_results(results)

    log.info("Parsed %d embeddings from batch %s.", len(pairs), job_id)
    return pairs


# ── Embedding alignment ────────────────────────────────────────────────────────

def align_pairs_to_docs(
    pairs: list[tuple[str, list[float]]],
    docs: list[Document],
) -> tuple[list[tuple[str, list[float]]], list[Document]]:
    """Re-align (custom_id, embedding) pairs to match the docs list order.

    Drops any doc whose embedding is missing (API-level failure for that item).
    Those articles will be retried on the next run since they won't be in the
    registry.

    Args:
        pairs: List of (custom_id, embedding) from parse_embedding_results.
               custom_id = raw doc.metadata["id"] (integer as string).
        docs:  Original document list in submission order.

    Returns:
        aligned_text_emb_pairs: List[(page_content, embedding)] — input for
            FAISS.add_embeddings or create_vectorstore.
        aligned_docs: Corresponding Document list (same order, failures excluded).
    """
    emb_by_id = {int(cid): emb for cid, emb in pairs}

    aligned_pairs: list[tuple[str, list[float]]] = []
    aligned_docs:  list[Document] = []
    dropped = 0

    for doc in docs:
        doc_id = doc.metadata["id"]
        if doc_id in emb_by_id:
            aligned_pairs.append((doc.page_content, emb_by_id[doc_id]))
            aligned_docs.append(doc)
        else:
            log.warning("No embedding returned for doc id=%d — will retry next run.", doc_id)
            dropped += 1

    if dropped:
        log.warning("%d document(s) excluded due to missing embeddings.", dropped)

    log.info("Aligned %d document-embedding pairs.", len(aligned_docs))
    return aligned_pairs, aligned_docs


# ── Vectorstore init / update ─────────────────────────────────────────────────

def init_vectorstore(
    docs: list[Document],
    text_emb_pairs: list[tuple[str, list[float]]],
    embeddings_model: OpenAIEmbeddings,
) -> FAISS:
    """Build a new FAISS store from scratch (cold start path).

    Uses kitai.index.create_vectorstore which requires:
    - len(docs) == len(text_emb_pairs)
    - every doc.metadata["id"] is unique

    The embeddings_model is stored as the query encoder for future
    similarity_search("query string") calls. It is NOT called here.

    Args:
        docs:             Documents (same order as text_emb_pairs).
        text_emb_pairs:   (page_content, embedding_vector) pairs.
        embeddings_model: Query encoder stored on the FAISS store.

    Returns:
        Newly created FAISS store.
    """
    embeddings_ndarray = np.array(
        [emb for _, emb in text_emb_pairs], dtype=np.float32
    )
    store = create_vectorstore(docs, embeddings_ndarray, embeddings_model)
    log.info("Initialised new vectorstore with %d vectors.", store.index.ntotal)
    return store


def update_vectorstore(
    text_emb_pairs: list[tuple[str, list[float]]],
    aligned_docs: list[Document],
    embeddings_model: OpenAIEmbeddings,
    vectorstore_dir: Path,
) -> FAISS:
    """Load the existing FAISS store and append new embeddings (incremental path).

    FAISS.add_embeddings grows the index in-place — no full rebuild needed.

    Integer metadata["id"] values are passed as docstore keys to stay
    consistent with the integer keys created by kitai.index.create_vectorstore
    at init time.

    Args:
        text_emb_pairs:  (page_content, embedding_vector) pairs for new articles.
        aligned_docs:    Corresponding Document list.
        embeddings_model: Query encoder stored on the FAISS store.
        vectorstore_dir: Directory containing the existing FAISS index.

    Returns:
        Updated FAISS store (modified in-place, returned for chaining).
    """
    store = FAISS.load_local(
        str(vectorstore_dir),
        embeddings_model,
        allow_dangerous_deserialization=True,
    )
    before = store.index.ntotal

    store.add_embeddings(
        text_embeddings=text_emb_pairs,
        metadatas=[doc.metadata for doc in aligned_docs],
        ids=[doc.metadata["id"] for doc in aligned_docs],
    )

    log.info(
        "Updated vectorstore: %d -> %d vectors (+%d).",
        before,
        store.index.ntotal,
        store.index.ntotal - before,
    )
    return store
