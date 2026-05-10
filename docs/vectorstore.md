# Feed Vectorstore

FAISS vectorstore of all feed articles, built once and updated daily by CI.

## Files

| Path | Description |
|---|---|
| `data/vectorstore/feeds/index.faiss` | FAISS flat-L2 index (1536-dim, `text-embedding-3-small`) |
| `data/vectorstore/feeds/index.pkl` | LangChain InMemoryDocstore + `index_to_docstore_id` map |
| `data/vectorstore/feeds_registry.tsv` | Ground truth: one row per embedded article (`id, date, title, link, guid`) |

## Key Facts

- **Cold start**: `python embed_feeds.py` once to build from all existing feeds
- **Incremental**: CI runs daily — only new guids (not in registry) are embedded and appended via `FAISS.add_embeddings`. No rebuild.
- **Dedup key**: `guid` (stable RSS identifier). Registry is append-only.
- **ID scheme**: monotonic integers starting at 0, assigned at embed time, never reused.
- **Doc content format**: `"{date}: {title}: {description}"` — matches sector analysis format

## Loading for Search

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from constants import VECTORSTORE_DIR

store = FAISS.load_local(
    str(VECTORSTORE_DIR),
    OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536),
    allow_dangerous_deserialization=True
)
results = store.similarity_search("Fed rate decision", k=5)
```

Or use `hybrid_rag.py` / `chatbot_rag.py` which handle this automatically.
