# Plan: Feed Embeddings with kitai

> **Status: IMPLEMENTED ✓**
> Cold start completed 2026-03-09: 7714 articles embedded (0 failures) in ~25 min.
> Vectorstore at `data/vectorstore/feeds/`. Registry at `data/vectorstore/feeds_registry.tsv`.
> CI workflow at `.github/workflows/embed_feeds.yml` — runs after `collect-sector-results`.

## Overview

A two-part addition to the daily pipeline:

1. **`embed_feeds.py`** — builds and incrementally updates a FAISS vectorstore from all CNBC feed articles using the OpenAI Batch API via `kitai`.
2. **`.github/workflows/embed_feeds.yml`** — CI workflow that runs after `collect-sector-results` and commits the updated store.

---

## New Files

| File | Purpose |
|---|---|
| `embed_feeds.py` | Single script — handles both cold-start build and incremental updates |
| `.github/workflows/embed_feeds.yml` | CI workflow: trigger, embed, commit |
| `data/vectorstore/feeds_registry.tsv` | Ground truth for which articles are in the FAISS store |
| `data/vectorstore/feeds/index.faiss` | FAISS binary index (committed to git) |
| `data/vectorstore/feeds/index.pkl` | LangChain InMemoryDocstore + id map (committed to git) |

### Constants to add to `constants.py`

```python
VECTORSTORE_DIR     = Path("data") / "vectorstore" / "feeds"
FEEDS_REGISTRY_FILE = Path("data") / "vectorstore" / "feeds_registry.tsv"
```

---

## `embed_feeds.py` — Logic Outline

### Module-level constants

```python
from constants import VECTORSTORE_DIR, FEEDS_REGISTRY_FILE

EMBED_MODEL      = "text-embedding-3-large"   # DEFAULT_EMBEDDING_MODEL from kitai.batch
EMBED_DIMENSIONS = 3072                        # DEFAULT_EMBEDDING_DIMENSIONS from kitai.batch
RAW_FEED_DIR     = Path("output")             # same as create_batch_files_v2.py
```

---

### Function signatures and responsibilities

#### `load_registry() -> pd.DataFrame`
- If `FEEDS_REGISTRY_FILE` does not exist → return empty DataFrame with columns `[id, date, title, link, guid]`.
- Otherwise: `pd.read_csv(FEEDS_REGISTRY_FILE, sep="\t", dtype={"id": int})`.

#### `save_registry(registry: pd.DataFrame) -> None`
- Ensure parent directory exists.
- Write with `index=False, sep="\t"`.
- Only place that writes the registry.

#### `load_all_feed_files() -> pd.DataFrame`
- Glob `RAW_FEED_DIR.glob("feeds*.txt")` — **top-level only**, no recursion (same constraint as `create_batch_files_v2.py`).
- Read each with `pd.read_csv(f, sep="\t")`.
- Concatenate + drop duplicates on `"guid"` (stable per-article identifier).
- Add a `date` column: `pd.to_datetime(df["pubDate"]).dt.strftime("%Y-%m-%d")`.
- Return DataFrame with columns: `[title, description, link, guid, pubDate, date]`.
- Exit cleanly (`sys.exit(0)`) if no feed files found.

#### `find_new_articles(all_df, registry) -> pd.DataFrame`
- `known_guids = set(registry["guid"])` (empty set when registry is empty).
- Filter: `new_df = all_df[~all_df["guid"].isin(known_guids)]`.
- Log: `{len(all_df)} total | {len(known_guids)} already embedded | {len(new_df)} new`.
- Return `new_df`. Exit cleanly (`sys.exit(0)`) with "No new articles to embed" if empty.

#### `assign_ids(new_df, registry) -> pd.DataFrame`
- `next_id = int(registry["id"].max()) + 1` if registry non-empty, else `0`.
- Assign contiguous integers: `new_df["id"] = range(next_id, next_id + len(new_df))`.
- IDs are monotonic integers, never reused.

#### `build_documents(new_df) -> list[Document]`
- For each row:
  - `page_content = f"{row['date']}: {row['title']}: {row['description']}"` — matches the pattern in `create_batch_files_v2.py`.
  - `metadata = {"id": int(row["id"]), "date": row["date"], "title": str(row["title"]), "link": str(row["link"]), "guid": str(row["guid"])}`.
- Use a plain loop (not `kitai.transform.df_to_docs`) to retain full control over the `metadata["id"]` assignment from the registry-allocated IDs.

#### `run_embedding_batch(docs, client) -> list[tuple[str, list[float]]]`
- `tasks = build_embedding_tasks(docs, model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS)`
- `job_id = submit_batch_job(client, tasks, endpoint="/v1/embeddings", metadata={"description": "feeds_embed"})`
- Log: `Submitted embedding batch: {job_id}`.
- `completed_ids = poll_until_complete(client, [job_id], poll_interval=30.0)`
- If `job_id` not in `completed_ids` → raise `RuntimeError(f"Batch {job_id} did not complete successfully")`.
- `results = download_batch_results(client, completed_ids[0])`
- `pairs = parse_embedding_results(results)` → `list[(custom_id, embedding)]`
- Log: `Parsed {len(pairs)} embeddings from batch {job_id}`.
- Return `pairs`.

#### `pairs_to_ordered(pairs, docs) -> tuple[list[tuple[str, list[float]]], list[Document]]`
- `custom_id` format from `kitai.batch`: `"custom_id_{doc.metadata['id']}"`.
- Build: `emb_by_id = {int(cid.split("custom_id_")[1]): emb for cid, emb in pairs}`.
- Re-align to `docs` order. Drop any doc whose embedding is missing (batch item failed) — log dropped IDs.
- Return `(aligned_text_emb_pairs, aligned_docs)` where `aligned_text_emb_pairs` is `list[(page_content, embedding_vector)]` for `FAISS.add_embeddings`.

#### `init_vectorstore(docs, text_emb_pairs, embeddings_model) -> FAISS`
- Called when `VECTORSTORE_DIR` does not exist (cold start).
- `embeddings_ndarray = np.array([emb for _, emb in text_emb_pairs], dtype=np.float32)`
- `return create_vectorstore(docs, embeddings_ndarray, embeddings_model)` — from `kitai.index`.

#### `update_vectorstore(text_emb_pairs, metadatas, embeddings_model) -> FAISS`
- Called when `VECTORSTORE_DIR` already exists (incremental update).
- `store = FAISS.load_local(str(VECTORSTORE_DIR), embeddings_model, allow_dangerous_deserialization=True)`
- `store.add_embeddings(text_emb_pairs, metadatas)` — grows index in-place, no rebuild.
- Return `store`.

---

### `main()` — orchestration

```
1.  load_dotenv()
2.  client = OpenAI()
3.  embeddings_model = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS)
     — stored as query encoder; NOT called for document embedding
4.  registry = load_registry()
5.  all_df = load_all_feed_files()           # exits if no feed files
6.  new_df = find_new_articles(all_df, registry)  # exits if nothing new
7.  new_df = assign_ids(new_df, registry)
8.  docs = build_documents(new_df)
9.  pairs = run_embedding_batch(docs, client)  # raises on failure
10. aligned_pairs, aligned_docs = pairs_to_ordered(pairs, docs)
11. if VECTORSTORE_DIR.exists():
        store = update_vectorstore(aligned_pairs, [doc.metadata for doc in aligned_docs], embeddings_model)
    else:
        store = init_vectorstore(aligned_docs, aligned_pairs, embeddings_model)
12. store.save_local(str(VECTORSTORE_DIR))
13. new_rows = DataFrame from aligned_docs metadata
    updated_registry = concat([registry, new_rows])
    save_registry(updated_registry)           # saved AFTER store — failure stays consistent
14. log: "Embedded {len(aligned_docs)} articles. Total in registry: {len(updated_registry)}."
```

**Idempotent:** re-running with no new feeds exits 0. The registry is the single source of truth — FAISS contents ≡ guids in `feeds_registry.tsv`.

---

## Registry TSV Schema

**File:** `data/vectorstore/feeds_registry.tsv`

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Globally unique monotonic integer. Never reused. Starts at 0. |
| `date` | str (YYYY-MM-DD) | Publication date from `pubDate` |
| `title` | str | Article title |
| `link` | str | Article URL |
| `guid` | str | Stable article identifier — the deduplication key |

Append-only in normal operation. Written with `index=False, sep="\t"` for R compatibility.

---

## Vectorstore Directory Layout

```
data/vectorstore/
  feeds_registry.tsv       # committed — source of truth
  feeds/
    index.faiss            # FAISS flat-L2 binary index (grows via add_embeddings)
    index.pkl              # LangChain InMemoryDocstore + index_to_docstore_id map
```

Both `index.faiss` and `index.pkl` are written by `FAISS.save_local()` and loaded by `FAISS.load_local()`.

---

## CI Workflow — `embed_feeds.yml`

**Trigger:** `workflow_run` on `collect-sector-results` completion + `workflow_dispatch`.
**Guard:** skip if upstream failed (same pattern as `collect_sectors.yml`).

```yaml
name: embed-feeds

on:
  workflow_run:
    workflows: ["collect-sector-results"]
    types: [completed]
  workflow_dispatch:

jobs:
  embed:
    runs-on: ubuntu-latest
    if: >
      github.event_name == 'workflow_dispatch' ||
      github.event.workflow_run.conclusion == 'success'

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install openai python-dotenv pandas numpy \
                      langchain-core langchain-community langchain-openai faiss-cpu
          pip install kitai-0.1.0-py3-none-any.whl

      - name: Embed new feed articles
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python embed_feeds.py

      - name: Commit vectorstore and registry
        run: |
          git config --local user.email "actions@github.com"
          git config --local user.name "GitHub Actions"
          git add data/vectorstore/feeds_registry.tsv \
                  data/vectorstore/feeds/index.faiss \
                  data/vectorstore/feeds/index.pkl
          git diff --staged --quiet || git commit -m "embed feeds"
          git push origin || echo "Nothing to push"
```

Key decisions:
- `poll_until_complete` inside `embed_feeds.py` is synchronous and blocks until the batch resolves. For typical feed volumes (50–200 articles/day), batches complete in 1–5 minutes well within the 6-hour CI job limit.
- Only the three specific files under `data/vectorstore/` are staged — no accidental inclusions.
- `git diff --staged --quiet ||` prevents a non-zero exit when nothing changed (same pattern already in `collect_sectors.yml`).

---

## Invariants

1. **Monotonic IDs:** `next_id = max(registry.id) + 1`. IDs are never recycled.
2. **Registry is ground truth:** FAISS docstore contents ≡ guids in registry. They must not diverge.
3. **`guid` is the dedup key:** Same article appearing in multiple feed files is embedded once only.
4. **Registry saved after store:** If FAISS save fails, registry stays at previous state; next run is consistent.
5. **Idempotency:** Running with no new feeds is a clean no-op. Running after a partial failure re-embeds the affected articles safely.

---

## Failure Modes

| Failure | Behaviour | Recovery |
|---|---|---|
| Batch API fails / expires | `RuntimeError` raised; registry not written | Next run re-attempts same articles |
| Partial batch (some items errored) | Dropped from aligned pairs; not added to store or registry | Re-tried on next run |
| FAISS saved but registry write fails | Duplicates on next run (store has articles; registry doesn't) | Mitigate: write registry to `.tmp` then atomic rename |
| Concurrent CI runs | Both read same registry; may write conflicting states | Unlikely — `workflow_run` trigger naturally serialises runs |
| Cold start with >45 000 articles | Exceeds OpenAI Batch API single-batch limit (50 000 requests) | Split into sequential batches of 45 000 before submission |

---

## Gitignore Decision

**Commit the vectorstore.** The project already commits `data/sector_results/`, `data/charts/`, and TSVs. The FAISS files are the same kind of accumulated pipeline output.

At 3072-dim float32, each article costs ~12 KB in the index. At ~100 articles/day × 200 trading days ≈ 20 000 articles ≈ 240 MB theoretical maximum. In practice the rolling window is bounded — old articles do not need to be kept indefinitely. Evaluate **git LFS** for `*.faiss` if the index exceeds 50 MB.

No `.gitignore` changes are needed — `data/` is not currently gitignored.

---

## kitai API Reference (relevant subset)

```python
from kitai.batch import (
    build_embedding_tasks,     # (docs, model, dimensions) → list[dict]
    submit_batch_job,          # (client, tasks, endpoint, metadata) → job_id: str
    poll_until_complete,       # (client, [job_id], poll_interval) → list[str]
    download_batch_results,    # (client, job_id) → list[dict]
    parse_embedding_results,   # (results) → list[tuple[str, list[float]]]
    DEFAULT_EMBEDDING_MODEL,   # "text-embedding-3-large"
    DEFAULT_EMBEDDING_DIMENSIONS,  # 3072
)
from kitai.index import (
    create_vectorstore,        # (docs, embeddings_ndarray, embeddings_model) → FAISS
)
from langchain_community.vectorstores import FAISS
# FAISS.save_local(path: str)
# FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)
# FAISS.add_embeddings(text_embeddings: list[tuple[str, list[float]]], metadatas: list[dict])
```

Document invariant: every `Document` must have `metadata["id"]` (unique int) before being passed to `build_embedding_tasks` or `create_vectorstore`.

---

## Related Files

| File | Relevance |
|---|---|
| `create_batch_files_v2.py` | Pattern for feed loading, content format, batch submission, sentinel design |
| `constants.py` | Add `VECTORSTORE_DIR` and `FEEDS_REGISTRY_FILE` here |
| `.github/workflows/collect_sectors.yml` | Template for `workflow_run` trigger and commit step |
| `notebooks/batch_api_guide.ipynb` | kitai.batch full API walkthrough |
| `notebooks/index_guide.ipynb` | kitai.index `create_vectorstore` and `add_embeddings` walkthrough |
