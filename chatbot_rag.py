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
                    ┌─────────┴─────────┐
                    │                   │
              sidebar controls      chat area
           (strategy, k, weights)  (messages + source docs)
                    │
              ┌─────┴──────────────────────────────────┐
              │  @st.cache_resource (loaded once)       │
              │  FAISS vectorstore + corpus docs +      │
              │  OpenAI client + ChatOpenAI model       │
              └─────┬──────────────────────────────────┘
                    │
              build_hybrid_retriever(k_bm25, k_semantic, weights)
              (cheap rebuild on sidebar change)
                    │
              per-query flow (on each st.chat_input submit)
                    │
              translate_query(strategy) → list[str]
                    │
              retrieve_for_queries(hybrid) → deduplicated docs
                    │
              reorder_docs() → LongContextReorder
                    │
              answer_query(openai_client) → str
                    │
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
  embed_dimensions) — never invalidated at runtime; a page reload rebuilds.
- build_hybrid_retriever() is NOT cached separately; it is rebuilt whenever
  the sidebar params change (BM25 over ~8 k docs takes < 0.5 s).

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

import streamlit as st
from dotenv import load_dotenv
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
    answer_query,
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
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
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
# Cached base resources — loaded once per Streamlit server process
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vectorstore …")
def _load_base_resources(
    folder: Path,
    index_name: str,
    embedding_model: str,
    embed_dimensions: int,
) -> tuple:
    """Load FAISS store, extract corpus docs, and create shared API clients.

    Cached forever within the Streamlit process — only rebuilt on server
    restart or cache clear.

    Returns:
        (vectorstore, corpus_docs, openai_client, chat_model)
    """
    openai_client = _OpenAIClient(api_key=_api_key)
    embeddings = _OpenAIEmbeddings(model=embedding_model, client=openai_client)
    vs = load_vectorstore(folder, index_name, embeddings)
    corpus = extract_docs(vs)
    logger.info("Corpus loaded: %d documents.", len(corpus))
    chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=_api_key)
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
            "expand     — paraphrase variants (best for synonym/phrasing coverage)\n"
            "decompose  — sub-questions (best for multi-part queries)\n"
            "step_back  — abstract questions (best for foundational context)\n"
            "none       — single query, no translation"
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
    VECTORSTORE_DIR, FAISS_INDEX_NAME, EMBEDDING_MODEL, EMBED_DIMENSIONS
)

# Rebuild hybrid retriever with current sidebar params (fast, < 0.5 s).
# build_hybrid_retriever() uses module-level constants, so we call the
# kitai primitives directly to honour the live sidebar values instead.
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
            # Show augmented queries if translation was applied
            queries = msg.get("queries")
            if queries and len(queries) > 1:
                with st.expander(f"Augmented queries ({len(queries)} total)"):
                    for i, q in enumerate(queries):
                        tag = " *(original)*" if i == 0 else ""
                        st.markdown(f"**{i}.** {q}{tag}")

            # Show source documents
            sources = msg.get("sources")
            if sources:
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
# Chat input handler
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about the news…"):
    # --- Display user message ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- Generate assistant response ---
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer…"):
            # 1. Query translation
            queries = translate_query(chat_model, prompt, strategy=strategy)

            # 2. Hybrid retrieval + dedup
            merged = retrieve_for_queries(queries, hybrid)

            # 3. LongContextReorder
            ordered = reorder_docs(merged)
            logger.info("Final context: %d documents.", len(ordered))

            # 4. LLM answer
            answer = answer_query(prompt, ordered, openai_client)

        st.write(answer)

        # Augmented queries (if any)
        if len(queries) > 1:
            with st.expander(f"Augmented queries ({len(queries)} total)"):
                for i, q in enumerate(queries):
                    tag = " *(original)*" if i == 0 else ""
                    st.markdown(f"**{i}.** {q}{tag}")

        # Source documents
        sources = []
        for doc in ordered:
            meta = doc.metadata
            sources.append({
                "title":   meta.get("title", "Untitled"),
                "date":    meta.get("date", ""),
                "link":    meta.get("link", ""),
                "snippet": doc.page_content[:200].replace("\n", " "),
            })

        if sources:
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

    # Persist to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "queries": queries,
        "sources": sources,
    })
