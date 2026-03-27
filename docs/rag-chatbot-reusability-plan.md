# RAG Chatbot Reusability Plan

**Goal:** A user with a new FAISS vectorstore should be able to spin up a working,
correctly wired Streamlit RAG chatbot in under 5 minutes without touching the
retrieval or streaming machinery.

**Core stack (fixed):** FAISS + BM25 (`kitai.retriever`) + `kitai.query_translation`
+ OpenAI streaming + Streamlit UI.

---

## What Varies Per Chatbot

Only four things differ between chatbots:

| Parameter | Example |
|---|---|
| `folder` | `data/vectorstore/feeds/` |
| `index_name` | `index` |
| `title` | `"Financial News Q&A"` |
| `system_prompt` | optional custom answer style |

Everything else — `@st.cache_resource`, `_stream_answer`, BM25 rebuild, sidebar,
session state, dedup, LongContextReorder — is identical and must not be duplicated.

---

## Hidden Invariants to Preserve

- Cache key must include `(folder, index_name, api_key)` — not just `api_key` — or two
  chatbots in the same process will share the wrong vectorstore.
- `ask()` in `hybrid_rag.py` is the single public retrieval surface. The chatbot is a
  thin UI layer over it, never a fork of the retrieval logic.
- Config validation must happen at startup (`sys.exit(1)` with a clear message if the
  FAISS path is missing), not mid-query.

---

## Recommended Implementation Steps

### Step 1 — Add `ChatbotConfig` dataclass

Add to `hybrid_rag.py` (or a new `chatbot_config.py`):

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ChatbotConfig:
    folder:         Path
    index_name:     str  = "index"
    title:          str  = "RAG Chatbot"
    system_prompt:  str  = (
        "Use the following excerpts to answer the question.\n"
        "If the context doesn't contain enough information, say so."
    )
    # Sidebar defaults
    default_strategy:      str   = "expand"
    default_k_semantic:    int   = 6
    default_k_bm25:        int   = 6
    default_weights_sparse: float = 0.5

    def __post_init__(self):
        self.folder = Path(self.folder)
        faiss = self.folder / f"{self.index_name}.faiss"
        pkl   = self.folder / f"{self.index_name}.pkl"
        if not faiss.exists() or not pkl.exists():
            raise ValueError(
                f"FAISS index not found at {self.folder}/{self.index_name}.*\n"
                "Run embed_feeds.py first."
            )
```

**Deliverable:** typed, validated config with sensible defaults; `ValueError` at
construction time if the path is wrong.

**Test (Red → Green):**
```python
def test_chatbot_config_rejects_missing_path():
    with pytest.raises(ValueError, match="FAISS index not found"):
        ChatbotConfig(folder="/nonexistent", index_name="index")

def test_chatbot_config_accepts_valid_path(tmp_faiss_dir):
    cfg = ChatbotConfig(folder=tmp_faiss_dir, index_name="index")
    assert cfg.title == "RAG Chatbot"
```

---

### Step 2 — Accept `--config` in `chatbot_rag.py`

Refactor `chatbot_rag.py` to parse a TOML config file (stdlib `tomllib` / `tomli`):

```bash
streamlit run chatbot_rag.py -- --config earnings.toml
```

Example `earnings.toml`:
```toml
folder      = "data/vectorstore/earnings"
index_name  = "index"
title       = "Earnings Call Q&A"
system_prompt = "Answer only from earnings call transcripts."
default_k_semantic    = 8
default_weights_sparse = 0.3
```

If `--config` is omitted, fall back to the existing defaults (current behaviour preserved).

**Deliverable:** `streamlit run chatbot_rag.py -- --config X.toml` runs a fully branded
chatbot over a different vectorstore. No other code changes needed for a new chatbot.

---

### Step 3 — Fix the cache key

Current (broken for multi-store):
```python
@st.cache_resource(show_spinner="Loading ...")
def _load_base_resources(folder, index_name, embedding_model, embed_dimensions, api_key):
```

The signature already accepts `folder` + `index_name` — confirm they are passed as
explicit arguments (not captured from a global). Streamlit keys the cache on all
positional args, so this is automatically correct **if** the values are not constants.

**Decision point:** if all chatbots run in separate `streamlit run` processes (different
ports), this is already safe and Step 3 is a no-op. Only relevant if a multi-tenant
single-process design (Alternative E below) is ever pursued.

---

### Step 4 — `new_chatbot.py` scaffold generator *(optional, high leverage)*

```bash
python new_chatbot.py \
  --folder data/vectorstore/earnings \
  --index-name index \
  --title "Earnings Q&A"
```

Emits:
- `earnings_chatbot.toml` — pre-filled config
- `earnings_chatbot.py` — three-line launcher (`import chatbot_rag; chatbot_rag.main(config_path="earnings_chatbot.toml")`)

**Deliverable:** zero manual editing. One command → one runnable chatbot.

---

### Step 5 — Update the `streamlit-rag-chatbot` skill

Add a "Bootstrap a new chatbot" section to
`~/.claude/skills/streamlit-rag-chatbot/SKILL.md` covering:

1. Run `new_chatbot.py` (or copy the config template)
2. Fill in `folder`, `index_name`, `title`
3. `streamlit run chatbot_rag.py -- --config my.toml`

The skill becomes a complete bootstrap guide — a developer reading it can spin up a
new chatbot without opening `chatbot_rag.py`.

---

## Alternatives Considered

| Option | Rationale | When to prefer |
|---|---|---|
| **Config file + template** (chosen) | Zero duplication; bug fixes flow automatically | Default — works for 1–5 chatbots |
| **`RagChatbot` class** | Pythonic; hides all Streamlit boilerplate | If users want to embed the chatbot in a larger app |
| **`app_factory` function** | Testable; no TOML dep | If config file feels heavyweight |
| **Multi-tenant single app** | One process, dropdown selects store | If you have 5+ stores and want one URL |
| **Scaffold generator** | Fastest time-to-running-app | If copy-paste errors are the observed pain point |

---

## Key Open Question

> Is the bottleneck _"too many files to copy"_ (solved by the generator) or
> _"too many lines to understand"_ (solved by the config/class abstraction)?

If users are copy-pasting and making mistakes → **start with Step 2** (config file,
one afternoon, immediate payoff).

If users are confused about what to change → **start with Step 1** (`ChatbotConfig`
dataclass, makes the surface area explicit).
