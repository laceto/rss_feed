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

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from constants import FEEDS_REGISTRY_FILE, RAW_FEED_DIR, VECTORSTORE_DIR
from kitai.batch import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    build_embedding_tasks,
    download_batch_results,
    parse_embedding_results,
    poll_until_complete,
    submit_batch_job,
)
from kitai.index import create_vectorstore

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBED_MODEL      = DEFAULT_EMBEDDING_MODEL       # "text-embedding-3-small"
EMBED_DIMENSIONS = DEFAULT_EMBEDDING_DIMENSIONS  # 1536
POLL_INTERVAL    = 30.0                          # seconds between batch status polls

REGISTRY_COLUMNS = ["id", "date", "title", "link", "guid"]


# ── Registry ──────────────────────────────────────────────────────────────────

def load_registry() -> pd.DataFrame:
    """Load the article registry, or return an empty frame if it does not exist.

    The registry is the ground truth for which articles are in the FAISS store.

    Returns:
        DataFrame with columns: id (int), date (str), title (str),
        link (str), guid (str).
    """
    if not FEEDS_REGISTRY_FILE.exists():
        log.info("Registry not found — starting fresh.")
        return pd.DataFrame(columns=REGISTRY_COLUMNS).astype({"id": int})

    df = pd.read_csv(FEEDS_REGISTRY_FILE, sep="\t", dtype={"id": int})
    log.info("Loaded registry: %d articles.", len(df))
    return df


def save_registry(registry: pd.DataFrame) -> None:
    """Write the registry as a TSV file.

    Uses an atomic write (tmp → rename) to prevent a half-written file
    if the process is interrupted between write and flush.

    Must be called AFTER store.save_local() so that a store write failure
    leaves the registry at its previous consistent state.
    """
    FEEDS_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = FEEDS_REGISTRY_FILE.with_suffix(".tmp")
    registry.to_csv(tmp_path, sep="\t", index=False)
    tmp_path.replace(FEEDS_REGISTRY_FILE)
    log.info("Registry saved: %d articles total.", len(registry))


# ── Feed loading ──────────────────────────────────────────────────────────────

def load_all_feed_files() -> pd.DataFrame:
    """Load all top-level output/feeds*.txt files into a single DataFrame.

    Top-level only (no recursion into output/enriched/ etc.) — same constraint
    as create_batch_files_v2.py.

    Deduplicates on guid (stable per-article identifier from the RSS feed).
    Adds a 'date' column (YYYY-MM-DD) derived from pubDate.

    Returns:
        DataFrame with columns: link, guid, type, id, sponsored, title,
        description, pubDate, date.

    Exits 0 if no feed files are found.
    """
    feed_files = sorted(RAW_FEED_DIR.glob("feeds*.txt"))
    if not feed_files:
        log.error("No feed files found in %s. Exiting.", RAW_FEED_DIR)
        sys.exit(0)

    raw_dfs = [pd.read_csv(f, sep="\t") for f in feed_files]
    combined = pd.concat(raw_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["guid"])
    combined["date"] = pd.to_datetime(combined["pubDate"]).dt.strftime("%Y-%m-%d")

    log.info(
        "Loaded %d unique articles from %d file(s).",
        len(combined),
        len(feed_files),
    )
    return combined


def find_new_articles(all_df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose guid is not yet recorded in the registry.

    Exits 0 cleanly if there is nothing new to embed.
    """
    known_guids = set(registry["guid"]) if not registry.empty else set()
    new_df = all_df[~all_df["guid"].isin(known_guids)].copy()

    log.info(
        "%d total articles | %d already embedded | %d new",
        len(all_df),
        len(known_guids),
        len(new_df),
    )

    if new_df.empty:
        log.info("No new articles to embed. Exiting.")
        sys.exit(0)

    return new_df


def assign_ids(new_df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Assign globally unique monotonic integer IDs to new articles.

    Continues from max(registry.id) + 1, or starts at 0 on the first run.
    IDs are never reused.
    """
    next_id = int(registry["id"].max()) + 1 if not registry.empty else 0
    new_df = new_df.copy()
    new_df["id"] = range(next_id, next_id + len(new_df))
    log.info("Assigned IDs %d–%d.", next_id, next_id + len(new_df) - 1)
    return new_df


# ── Document building ─────────────────────────────────────────────────────────

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
    """
    docs = []
    for _, row in new_df.iterrows():
        content = (
            f"{row['date']}: "
            f"{str(row['title']).strip()}: "
            f"{str(row['description']).strip()}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "id":    int(row["id"]),
                "date":  str(row["date"]),
                "title": str(row["title"]),
                "link":  str(row["link"]),
                "guid":  str(row["guid"]),
            },
        ))

    log.info("Built %d documents for embedding.", len(docs))
    return docs


# ── Embedding batch ───────────────────────────────────────────────────────────

def run_embedding_batch(
    docs: list[Document],
    client: OpenAI,
) -> list[tuple[str, list[float]]]:
    """Submit docs to the OpenAI Batch API and return (custom_id, embedding) pairs.

    Uses kitai.batch throughout:
        build_embedding_tasks → submit_batch_job → poll_until_complete
        → download_batch_results → parse_embedding_results

    The custom_id format set by kitai.batch is "custom_id_{doc.metadata['id']}".

    Raises:
        RuntimeError: If the batch job does not reach 'completed' status.
    """
    tasks  = build_embedding_tasks(docs, model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS)
    job_id = submit_batch_job(
        client,
        tasks,
        endpoint="/v1/embeddings",
        metadata={"description": "feeds_embed"},
    )
    log.info("Submitted embedding batch: %s (%d tasks)", job_id, len(tasks))

    completed_ids = poll_until_complete(client, [job_id], poll_interval=POLL_INTERVAL)
    if job_id not in completed_ids:
        raise RuntimeError(
            f"Embedding batch {job_id} did not complete successfully. "
            "Check the OpenAI dashboard for details."
        )

    results = download_batch_results(client, job_id)
    pairs   = parse_embedding_results(results)

    log.info("Parsed %d embeddings from batch %s.", len(pairs), job_id)
    return pairs


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
               custom_id format: "custom_id_{int_id}".
        docs:  Original document list in submission order.

    Returns:
        aligned_text_emb_pairs: List[(page_content, embedding)] — input for
            FAISS.add_embeddings or create_vectorstore.
        aligned_docs: Corresponding Document list (same order, failures excluded).
    """
    emb_by_id = {
        int(cid.split("custom_id_")[1]): emb
        for cid, emb in pairs
    }

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
) -> FAISS:
    """Load the existing FAISS store and append new embeddings (incremental path).

    FAISS.add_embeddings grows the index in-place — no full rebuild needed.

    Integer metadata["id"] values are passed as docstore keys to stay
    consistent with the integer keys created by kitai.index.create_vectorstore
    at init time.
    """
    store = FAISS.load_local(
        str(VECTORSTORE_DIR),
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    client           = OpenAI()
    embeddings_model = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS)

    # 1. Load existing state
    registry = load_registry()

    # 2. Discover new articles (exits 0 if none)
    all_df = load_all_feed_files()
    new_df = find_new_articles(all_df, registry)
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
        store = update_vectorstore(aligned_pairs, aligned_docs, embeddings_model)
    else:
        store = init_vectorstore(aligned_docs, aligned_pairs, embeddings_model)

    # 7. Persist the store BEFORE writing the registry
    #    If save_local fails, the registry stays at its previous consistent state
    #    and the next run will re-embed the same articles.
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTORSTORE_DIR))
    log.info("Vectorstore saved to %s.", VECTORSTORE_DIR)

    # 8. Append new rows to the registry (atomic write)
    new_rows = pd.DataFrame({
        "id":    [doc.metadata["id"]    for doc in aligned_docs],
        "date":  [doc.metadata["date"]  for doc in aligned_docs],
        "title": [doc.metadata["title"] for doc in aligned_docs],
        "link":  [doc.metadata["link"]  for doc in aligned_docs],
        "guid":  [doc.metadata["guid"]  for doc in aligned_docs],
    })
    updated_registry = pd.concat([registry, new_rows], ignore_index=True)
    save_registry(updated_registry)

    log.info(
        "[done] Embedded %d new articles. Total in registry: %d.",
        len(aligned_docs),
        len(updated_registry),
    )


if __name__ == "__main__":
    main()
