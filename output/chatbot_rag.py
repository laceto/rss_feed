# -*- coding: utf-8 -*-
"""
chatbot_rag.py
==============
Streamlit chatbot that wraps hybrid_rag.py's retrieval and answering pipeline
into an interactive chat interface.

Architecture
------------
                        Streamlit UI
                              |
                    +---------+---------+
                    |                   |
              sidebar controls      chat area
           (strategy, k, weights)  (messages + source docs)
                    |
              +-----+----------------------------------------------+
              |  @st.cache_resource (loaded once)                   |
              |  FAISS vectorstore + corpus docs +                  |
              |  OpenAI client + ChatOpenAI model                   |
              +-----+----------------------------------------------+
                    |
              build_hybrid_retriever(k_bm25, k_semantic, weights)
              (cheap rebuild on sidebar change)
                    |
              per-query flow (on each st.chat_input submit)
                    |
              [spinner] translate_query(strategy) -> list[str]
                    |
              [spinner] retrieve_for_queries(hybrid) -> deduplicated docs
                    |
              [spinner] reorder_docs() -> LongContextReorder
                    |
              [stream] _stream_answer(openai_client) -> Generator[str]
                    |
              append to st.session_state.messages

Session state
-------------
  messages : list[dict]
    role    : "user" | "assistant"
    content : str
    sources : list[dict] | None  -- only on assistant messages
      title   : str
      date    : str
      link    : str
      snippet : str
    queries : list[str] | None   -- augmented queries, only when len > 1

Caching invariants
------------------
- _load_base_resources() is cached on (folder, index_name, embedding_model,
  embed_dimensions, api_key) — api_key is an explicit param so key rotation
  correctly invalidates the cache.
- The hybrid retriever is NOT cached; it is rebuilt every render cycle using
  the live sidebar values (BM25 over ~8k docs takes < 0.5 s).

Streaming invariants
--------------------
- Steps 1-3 (translation, retrieval, reorder) run inside st.spinner because
  they produce no incremental output.
- Step 4 (LLM answer) uses st.write_stream() which renders tokens as they
  arrive and returns the full concatenated string for session state.
- _stream_answer() uses stream=True on the OpenAI chat completions API and
  yields only non-None delta content, skipping role/finish chunks.

Debugging
---------
- Set LOG_LEVEL=DEBUG in the environment for verbose retrieval logs.
- If FAISS raises a deserialization error confirm the .pkl file is present.
- If answers are poor, adjust the strategy or weights in the sidebar.

Run:
    streamlit run chatbot_rag.py
"""

import logging
import os
from pathlib import Path
from typing import Generator

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from openai import OpenAI as _OpenAIClient

# Re-use all business logic from hybrid_rag — single source of truth
from hybrid_rag import (
    CHAT_MODEL,
    EMBED_DIMENSIONS,
    EMBEDDING_MODEL,
    FAISS_INDEX_NAME,
    K_BM25,
    K_SEMANTIC,
    WEIGHTS_SPARSE,
    _OpenAIEmbeddings,
    extract_docs,
    load_vectorstore,
    retrieve_for_queries,
    translate_query,
)
from kitai.retriever import (
    create_BM25retriever_from_docs,
    create_hybrid_retriever,
    create_retriever,
    reorder_docs,
)
from constants import VECTORSTORE_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="News RAG Chatbot",
    page_icon=":newspaper:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    st.error(
        "OPENAI_API_KEY not found. Add it to your .env file or environment and restart.",
        icon="🔑",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Streaming answer generator
# ---------------------------------------------------------------------------

def _stream_answer(
    query: str,
    context_docs: list[Document],
    client: _OpenAIClient,
) -> Generator[str, None, None]:
    """Stream an LLM answer token-by-token.

    Uses the same prompt template as hybrid_rag.answer_query so behaviour
    is identical — only the delivery mechanism differs (streaming vs blocking).

    Args:
        query: The original user question.
        context_docs: Reordered context documents from the RAG pipeline.
        client: Shared OpenAI client (from cached resources).

    Yields:
        Non-empty string tokens as they arrive from the API.

    Failure modes:
        - OpenAI API errors propagate as exceptions; Streamlit will surface them.
        - Empty context_docs yields an answer that says so (LLM instruction).
    """
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = (
        "Use the following news excerpts to answer the question.\n"
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# UI helper — source document expander (avoids duplication between
# history replay and new-message rendering)
# ---------------------------------------------------------------------------

def _render_sources(sources: list[dict]) -> None:
    """Render a collapsible expander listing source documents.

    Args:
        sources: List of dicts with keys title, date, link, snippet.
                 No-op when empty.
    """
    if not sources:
        return
    with st.expander(f"Source documents ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            st.markdown(
                f"**[{i}]** {src['title']}  \n"
                f"*{src['date']}*"
                + (f" · [{src['link']}]({src['link']})" if src["link"] else "")
            )
            st.caption(src["snippet"])
            if i < len(sources):
                st.divider()


# ---------------------------------------------------------------------------
# UI helper — augmented query expander
# ---------------------------------------------------------------------------

def _render_queries(queries: list[str]) -> None:
    """Render a collapsible expander listing augmented queries.

    Only shown when len(queries) > 1 (i.e. translation was applied).

    Args:
        queries: Flat list; queries[0] is always the original user query.
    """
    if not queries or len(queries) <= 1:
        return
    with st.expander(f"Augmented queries ({len(queries)} total)"):
        for i, q in enumerate(queries):
            tag = " *(original)*" if i == 0 else ""
            st.markdown(f"**{i}.** {q}{tag}")


# ---------------------------------------------------------------------------
# Cached base resources — loaded once per Streamlit server process
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vectorstore ...")
def _load_base_resources(
    folder: Path,
    index_name: str,
    embedding_model: str,
    embed_dimensions: int,
    api_key: str,
) -> tuple:
    """Load FAISS store, extract corpus docs, and create shared API clients.

    api_key is an explicit parameter (not a closure) so that key rotation
    correctly invalidates the cache.

    Returns:
        (vectorstore, corpus_docs, openai_client, chat_model)
    """
    openai_client = _OpenAIClient(api_key=api_key)
    embeddings = _OpenAIEmbeddings(model=embedding_model, client=openai_client)
    vs = load_vectorstore(folder, index_name, embeddings)
    corpus = extract_docs(vs)
    logger.info("Corpus loaded: %d documents.", len(corpus))
    chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=api_key)
    return vs, corpus, openai_client, chat_model


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar — retrieval controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Retrieval settings")

    strategy = st.selectbox(
        "Query translation strategy",
        options=["expand", "decompose", "step_back", "none"],
        index=0,
        help=(
            "expand     - paraphrase variants (best for synonym/phrasing coverage)\n"
            "decompose  - sub-questions (best for multi-part queries)\n"
            "step_back  - abstract questions (best for foundational context)\n"
            "none       - single query, no translation"
        ),
    )

    k_semantic = st.slider(
        "Semantic k (FAISS)",
        min_value=1, max_value=20, value=K_SEMANTIC, step=1,
        help="Number of documents retrieved by the FAISS vector retriever per query.",
    )

    k_bm25 = st.slider(
        "BM25 k (keyword)",
        min_value=1, max_value=20, value=K_BM25, step=1,
        help="Number of documents retrieved by the BM25 keyword retriever per query.",
    )

    weights_sparse = st.slider(
        "BM25 weight (0 = pure semantic, 1 = pure keyword)",
        min_value=0.0, max_value=1.0, value=WEIGHTS_SPARSE, step=0.05,
        help="Blending weight for the EnsembleRetriever (RRF merge).",
    )

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        f"Vectorstore: `{VECTORSTORE_DIR}`\n\n"
        f"Embedding model: `{EMBEDDING_MODEL}` ({EMBED_DIMENSIONS}-dim)\n\n"
        f"Chat model: `{CHAT_MODEL}`"
    )

# ---------------------------------------------------------------------------
# Load base resources (cached) + build hybrid retriever
# ---------------------------------------------------------------------------

vs, corpus, openai_client, chat_model = _load_base_resources(
    VECTORSTORE_DIR, FAISS_INDEX_NAME, EMBEDDING_MODEL, EMBED_DIMENSIONS, _api_key
)

# Rebuild hybrid retriever with current sidebar params (fast, < 0.5 s).
# build_hybrid_retriever() in hybrid_rag uses module-level constants, so we
# call the kitai primitives directly to honour the live sidebar values.
_bm25_ret   = create_BM25retriever_from_docs(docs=corpus, k=k_bm25)
_vector_ret = create_retriever(vs=vs, search_type="similarity", search_kwargs={"k": k_semantic})
hybrid = create_hybrid_retriever(
    sparse_retriever=_bm25_ret,
    semantic_retriever=_vector_ret,
    weights_sparse=weights_sparse,
)

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("News RAG Chatbot")
st.caption(
    "Ask questions about the financial news corpus. "
    "Adjust retrieval settings in the sidebar."
)

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            _render_queries(msg.get("queries") or [])
            _render_sources(msg.get("sources") or [])

# ---------------------------------------------------------------------------
# Chat input handler
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about the news..."):
    # --- Display user message ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- Generate assistant response ---
    with st.chat_message("assistant"):
        # Steps 1-3: translation + retrieval + reorder (no incremental output)
        with st.spinner("Retrieving context..."):
            queries = translate_query(chat_model, prompt, strategy=strategy)
            merged  = retrieve_for_queries(queries, hybrid)
            ordered = reorder_docs(merged)
            logger.info(
                "Context ready: %d docs from %d quer%s.",
                len(ordered), len(queries), "y" if len(queries) == 1 else "ies",
            )

        # Step 4: stream LLM answer token-by-token
        answer = st.write_stream(_stream_answer(prompt, ordered, openai_client))

        # Post-answer metadata expanders
        _render_queries(queries)

        sources = [
            {
                "title":   doc.metadata.get("title", "Untitled"),
                "date":    doc.metadata.get("date", ""),
                "link":    doc.metadata.get("link", ""),
                "snippet": doc.page_content[:200].replace("\n", " "),
            }
            for doc in ordered
        ]
        _render_sources(sources)

    # Persist to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "queries": queries,
        "sources": sources,
    })
