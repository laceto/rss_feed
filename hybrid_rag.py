"""
hybrid_rag.py
=============
Load an existing FAISS vector store, build a hybrid retriever (BM25 + semantic),
optionally expand the query via kitai.query_translation, and return an
LLM-generated answer.

Architecture
------------
                         user query
                              |
              translate_query() [optional, kitai.query_translation]
                    expand / decompose / step_back
                              |
                    augmented query list
                              |
                   retrieve_for_queries()
                    (one hybrid.invoke per query)
                              |
                   deduplicate by guid (first-seen wins)
                              |
                         ┌─ BM25Retriever          (keyword)
  FAISS .faiss/.pkl      │
         │               └─ VectorStoreRetriever   (semantic)
         │                         │
         └─────────────────────────┤
                                   ▼
                           EnsembleRetriever  (RRF merge)
                                   │
                              reorder_docs()  (LongContextReorder)
                                   │
                             context string
                                   │
                               ChatOpenAI
                                   │
                                answer

Query translation strategies (QUERY_TRANSLATION_STRATEGY):
  "expand"     — paraphrase variants; best for synonym/phrasing coverage (default)
  "decompose"  — sub-questions; best for multi-part queries
  "step_back"  — abstract questions; best when the answer lives in foundational context
  "none"       — skip translation; single query, same as the original behaviour

Invariants
----------
- OPENAI_API_KEY must be present in .env (or the environment).
- The vectorstore folder must contain both the .faiss and .pkl files
  named after FAISS_INDEX_NAME.
- Docs are extracted from the loaded FAISS docstore; the same corpus
  is used for both the vector retriever and BM25 — they are never out of sync.
- Deduplication key is metadata["guid"] — stable per-article identifier.
  First-retrieved occurrence wins; later duplicates are discarded.

Debugging
---------
- Set LOG_LEVEL=DEBUG for verbose retrieval logs (BM25 scores, RRF ranks).
- If FAISS raises a deserialization error, confirm allow_dangerous_deserialization=True
  is acceptable — it is safe here because we control the .pkl file.
- If the answer is poor, try lowering weights_sparse toward 0.0 (more semantic)
  or raising it toward 1.0 (more keyword-driven).
- If query translation produces off-topic augmented queries, switch strategy to
  "expand" or "none", or supply custom few-shot examples via the few_shot_examples
  parameter in translate_query().
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI as _OpenAIClient

from constants import VECTORSTORE_DIR
from kitai.query_translation import decompose_query, expand_query, step_back_query
from kitai.retriever import (
    create_BM25retriever_from_docs,
    create_hybrid_retriever,
    create_retriever,
    reorder_docs,
)

# ---------------------------------------------------------------------------
# Embeddings shim — avoids langchain_openai (incompatible with langchain-core 0.3.x)
# ---------------------------------------------------------------------------

class _OpenAIEmbeddings(Embeddings):
    """Thin wrapper around the openai client satisfying langchain_core.embeddings.Embeddings."""

    def __init__(self, model: str, client: _OpenAIClient) -> None:
        self._model = model
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts, model=self._model, dimensions=EMBED_DIMENSIONS
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Configuration — change these constants to adapt the script
# ---------------------------------------------------------------------------

# VECTORSTORE_DIR is imported from constants.py (data/vectorstore/feeds)
# FAISS.save_local() always writes files named "index.faiss" / "index.pkl"
FAISS_INDEX_NAME = "index"

# Must match the model used at embed time (embed_feeds.py uses kitai defaults:
# DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small", DEFAULT_EMBEDDING_DIMENSIONS = 1536)
EMBEDDING_MODEL  = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536
CHAT_MODEL       = "gpt-4o-mini"

K_SEMANTIC = 6   # docs to retrieve from FAISS per query
K_BM25 = 6       # docs to retrieve from BM25 per query
WEIGHTS_SPARSE = 0.5  # 0.0 = pure semantic, 1.0 = pure BM25

# Query translation strategy applied before retrieval.
# "expand"    — paraphrase variants (default; best for synonym/phrasing coverage)
# "decompose" — sub-questions (best for multi-part queries)
# "step_back" — abstract questions (best when answer is in foundational context)
# "none"      — skip translation; single query
QUERY_TRANSLATION_STRATEGY: str = "expand"
STEP_BACK_NUM: int = 2  # only used when QUERY_TRANSLATION_STRATEGY == "step_back"

QUERY = "lists chronologically news about oil prices since venezuelan maduro leaving"  # <-- change this to test different queries

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_vectorstore(
    folder: Path,
    index_name: str,
    embeddings_model: _OpenAIEmbeddings,
) -> FAISS:
    """Load a FAISS index from disk.

    Raises:
        FileNotFoundError: If .faiss or .pkl files are missing.
    """
    faiss_file = folder / f"{index_name}.faiss"
    pkl_file = folder / f"{index_name}.pkl"

    for f in (faiss_file, pkl_file):
        if not f.exists():
            raise FileNotFoundError(
                f"Expected vectorstore file not found: {f}\n"
                f"Run the embedding pipeline first to create the index."
            )

    logger.info("Loading FAISS index from %s …", folder)
    vs = FAISS.load_local(
        folder_path=str(folder),
        embeddings=embeddings_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,  # safe: we own this .pkl
    )
    logger.info("Loaded %d vectors.", vs.index.ntotal)
    return vs


def extract_docs(vs: FAISS) -> list[Document]:
    """Pull all Documents stored in the FAISS in-memory docstore.

    These are used to build the BM25 index over the same corpus, so
    both retrievers always see identical documents.
    """
    return list(vs.docstore._dict.values())


def build_hybrid_retriever(vs: FAISS, docs: list[Document]):
    """Wire together BM25 + semantic retriever via EnsembleRetriever."""
    bm25 = create_BM25retriever_from_docs(docs=docs, k=K_BM25)
    vector = create_retriever(
        vs=vs,
        search_type="similarity",
        search_kwargs={"k": K_SEMANTIC},
    )
    hybrid = create_hybrid_retriever(
        sparse_retriever=bm25,
        semantic_retriever=vector,
        weights_sparse=WEIGHTS_SPARSE,
    )
    logger.info(
        "Hybrid retriever ready (BM25 k=%d, semantic k=%d, weights=%s).",
        K_BM25,
        K_SEMANTIC,
        hybrid.weights,
    )
    return hybrid


def translate_query(
    model: ChatOpenAI,
    query: str,
    strategy: str,
    *,
    few_shot_examples: list[dict] | None = None,
) -> list[str]:
    """Expand a single user query into an augmented query pool.

    Runs the chosen kitai.query_translation strategy against the query and
    returns a deduplicated flat list: [original, augmented_1, augmented_2, ...].
    The original is always position 0 so the literal user intent is never lost.

    Args:
        model: LangChain ChatOpenAI instance used by the translation chain.
        query: The raw user question.
        strategy: One of "expand", "decompose", "step_back", "none".
        few_shot_examples: Optional domain-specific few-shot override passed
            straight through to the chosen kitai strategy function.

    Returns:
        A deduplicated list of query strings. Length is 1 when strategy="none".

    Raises:
        ValueError: If strategy is not a recognised value.
    """
    if strategy == "none":
        return [query]

    kwargs = {"few_shot_examples": few_shot_examples} if few_shot_examples else {}

    if strategy == "expand":
        raw = expand_query(model, [query], **kwargs)
        augmented = [pq.paraphrased_query for pq in raw[0]]
    elif strategy == "decompose":
        raw = decompose_query(model, [query], **kwargs)
        augmented = [dq.decomposed_query for dq in raw[0]]
    elif strategy == "step_back":
        raw = step_back_query(model, [query], num_queries=STEP_BACK_NUM, **kwargs)
        augmented = [gq.general_query for gq in raw[0]]
    else:
        raise ValueError(
            f"Unknown query translation strategy: {strategy!r}. "
            "Choose from 'expand', 'decompose', 'step_back', 'none'."
        )

    # Original first; deduplicate while preserving order
    seen: set[str] = {query}
    queries = [query]
    for q in augmented:
        if q not in seen:
            queries.append(q)
            seen.add(q)

    logger.info(
        "Query translation [%s]: %d queries from %r",
        strategy, len(queries), query,
    )
    for i, q in enumerate(queries):
        tag = " [original]" if i == 0 else ""
        logger.debug("  query %d: %s%s", i, q, tag)

    return queries


def retrieve_for_queries(
    queries: list[str],
    hybrid,
) -> list[Document]:
    """Run each query through the hybrid retriever; deduplicate by guid.

    Invokes the hybrid retriever once per query string and merges the
    results into a single list. Documents seen in earlier queries (by guid)
    are not added again — first-retrieved occurrence wins, preserving the
    ranking signal from the first (original) query.

    Args:
        queries: Flat list of query strings. First entry should be the
            original user query (from translate_query).
        hybrid: EnsembleRetriever instance (from build_hybrid_retriever).

    Returns:
        Deduplicated list of Document objects, ordered by first-seen retrieval.
    """
    # batch() issues all retriever calls in parallel (langchain Runnable protocol)
    results: list[list[Document]] = hybrid.batch(queries)

    seen_guids: set[str] = set()
    merged: list[Document] = []

    for q, docs in zip(queries, results):
        added = 0
        for doc in docs:
            guid = doc.metadata.get("guid", "")
            if guid not in seen_guids:
                seen_guids.add(guid)
                merged.append(doc)
                added += 1
        logger.debug("Query %r -> %d retrieved, %d new after dedup.", q, len(docs), added)

    logger.info(
        "Retrieved %d unique documents across %d query/queries.",
        len(merged), len(queries),
    )
    return merged


def answer_query(query: str, context_docs: list[Document], client: _OpenAIClient) -> str:
    """Send the reordered context + query to the LLM and return the answer."""
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = (
        f"Use the following news excerpts to answer the question.\n"
        f"If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file or environment."
        )

    # 1. Load vectorstore
    openai_client    = _OpenAIClient(api_key=api_key)
    embeddings_model = _OpenAIEmbeddings(model=EMBEDDING_MODEL, client=openai_client)
    vs = load_vectorstore(VECTORSTORE_DIR, FAISS_INDEX_NAME, embeddings_model)

    # 2. Extract corpus for BM25 (same docs as FAISS — always in sync)
    corpus = extract_docs(vs)
    logger.info("Corpus size: %d documents.", len(corpus))

    # 3. Build hybrid retriever
    hybrid = build_hybrid_retriever(vs, corpus)

    # 4. Translate query into an augmented pool, then retrieve + deduplicate
    chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    queries    = translate_query(chat_model, QUERY, strategy=QUERY_TRANSLATION_STRATEGY)
    merged     = retrieve_for_queries(queries, hybrid)

    # 5. Reorder merged pool for LLM context (LongContextReorder)
    ordered = reorder_docs(merged)
    logger.info("Final context: %d documents after reorder.", len(ordered))

    # 6. Generate answer
    answer = answer_query(QUERY, ordered, openai_client)

    # 7. Print results
    print("\n" + "=" * 70)
    print(f"QUERY     : {QUERY}")
    print(f"STRATEGY  : {QUERY_TRANSLATION_STRATEGY} ({len(queries)} queries)")
    print("=" * 70)
    if len(queries) > 1:
        print("\nAUGMENTED QUERIES:")
        for i, q in enumerate(queries):
            tag = " [original]" if i == 0 else ""
            print(f"  {i}. {q}{tag}")
    print(f"\nSOURCE DOCUMENTS ({len(ordered)}, LongContextReorder applied):")
    for i, doc in enumerate(ordered, 1):
        snippet = doc.page_content[:120].replace("\n", " ")
        meta = {k: v for k, v in doc.metadata.items() if k != "id"}
        print(f"  [{i}] {snippet}...")
        if meta:
            print(f"       metadata: {meta}")
    print("\nANSWER:")
    print(answer)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
