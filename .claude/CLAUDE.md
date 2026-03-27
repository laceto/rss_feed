# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A hybrid R + Python pipeline for financial news analysis:

1. **R (scraping)**: `download.R` fetches CNBC RSS feeds daily → `output/feeds{date}.txt`
2. **Python (sector analysis)**: LLM-powered batch pipeline extracts sector-level trading signals per day
3. **Python (charting)**: Sentiment trend charts generated from consolidated results
4. **Python (embeddings)**: `embed_feeds.py` embeds all feed articles via OpenAI Batch API → FAISS vectorstore at `data/vectorstore/feeds/`
5. **Python (topic clustering)**: `cluster_topics.py` clusters articles from a rolling 45-day window into emergent narratives and tracks their frequency as a time-series signal
6. **Python (daily briefing)**: `daily_briefing.py` surfaces spiking topics, queries the RAG for narrative summaries, and cross-checks against sector sentiment
7. **Python (briefing batch)**: `create_batch_briefings.py` + `retrieve_batch_briefings.py` generate RAG briefings for all historical dates via OpenAI Batch API (async, 50% cheaper)
8. **Python (backfill + labeling)**: `backfill.py` orchestrates historical cluster + briefing runs; `label_topics.py` labels existing unlabeled topic_ids without re-clustering
9. **Python (HF feeds push)**: `push_new_feeds_to_hf.py` appends daily scraped articles to `lacetohf/feeds` on Hugging Face Datasets; `push_feeds_to_hf.py` is the one-time cold-start
10. **Python (HF analysis push)**: `push_new_analysis_to_hf.py` appends new rows to `lacetohf/sector-analysis`, `lacetohf/topic-trends`, and `lacetohf/entity-sentiment`; `push_analysis_to_hf.py` is the cold-start

## Running the Scripts

### R — RSS Scraper
```bash
Rscript download.R
```
Outputs `output/feeds{YYYY-MM-DD}.txt` (tab-separated). R deps in `DESCRIPTION` (`rvest`, `xml2`, `XML`, `dplyr`, `purrr`).

### Python — Virtual Environment
```bash
venv\Scripts\activate   # Windows

python create_batch_files_v2.py        # submit sector batch to OpenAI Batch API
python retrieve_batch_file_results.py  # collect completed batch results
python read_sector_results.py          # flatten results → data/sector_summary.tsv
python visualize_sentiment.py          # generate charts → data/charts/
python export_time_series.py           # bulk TSV exports → data/sector_sentiment_pivot.tsv + data/entity_sentiment_ts.tsv
python build_sector_db.py              # build SQLite db → data/sector_results.db (full rebuild, atomic)
python cluster_topics.py               # cluster topics for today → data/topic_trends.tsv (append-only)
python cluster_topics.py --date 2026-03-13  # cluster for a specific date
python cluster_topics.py --skip-labeling    # skip LLM labeling (dry run / profiling)
python embed_feeds.py                  # embed feed articles → data/vectorstore/feeds/ (init + incremental)
python query_sector.py                 # not a CLI script — import as a module (see below)

python hybrid_rag.py               # CLI hybrid RAG query (BM25 + semantic + query translation)
streamlit run chatbot_rag.py       # Streamlit RAG chatbot (streaming, sidebar controls)

python daily_briefing.py                    # morning briefing for today (RAG + sector cross-check)
python daily_briefing.py --date 2026-03-21  # briefing for a specific date
python daily_briefing.py --no-rag --save    # fast briefing (no API), save to data/briefings/

python label_topics.py             # label all unlabeled topic_ids via OpenAI Batch API
python label_topics.py --dry-run   # show counts only

python visualize_topics.py              # six charts: heatmap + frequency + timeline + sentiment + delta + scatter (top 15, last 200 days)
python visualize_topics.py --top 20     # top 20 topics
python visualize_topics.py --days 90    # last 90 days only
python visualize_topics.py --animate    # also generate animated GIF (chart G) — slow
python visualize_topics.py --animate --fps 8  # animated GIF at 8 fps

python create_batch_briefings.py             # submit RAG briefing batch to OpenAI Batch API
python create_batch_briefings.py --dry-run   # show counts, no submission
python retrieve_batch_briefings.py           # collect completed batch → data/briefings/{date}.json

python backfill.py                          # cluster + brief Sep 2025 → today
python backfill.py --phase1-only            # cluster only (no API cost)
python backfill.py --phase2-only --no-rag   # briefings without RAG (instant)

python push_feeds_to_hf.py             # cold-start: push all feeds → lacetohf/feeds
python push_feeds_to_hf.py --dry-run   # show row count only
python push_new_feeds_to_hf.py         # daily incremental: append today's new articles

python push_analysis_to_hf.py          # cold-start: push sector/topic/entity → 3 HF datasets
python push_analysis_to_hf.py --dry-run
python push_new_analysis_to_hf.py      # daily incremental: append new analysis rows

python trader_assistant.py         # trading signal extractor (LangChain)
python create_batch_files.py       # MA/RSI batch analysis (tickers)
```

Python deps: `openai pydantic python-dotenv pandas matplotlib seaborn streamlit langchain langchain-community langchain-openai rank-bm25 hdbscan scikit-learn datasets huggingface_hub`. API keys in `.env` (gitignored):
```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
HUGGINGFACE_REPO=your-username/feeds
```

## Architecture

### Sector Analysis Pipeline (primary daily pipeline)
```
GitHub Actions — daily-pipeline (weekdays 19:00 UTC)
  → download.R
  → output/feeds{date}.txt

  → push_new_feeds_to_hf.py
      reads:  output/feeds{date}.txt  (today's file only)
      dedup:  guid — skips articles already in remote dataset
      writes: lacetohf/feeds  (appends new rows; idempotent)

  → create_batch_files_v2.py
      reads:  lacetohf/feeds  (HF Dataset — authoritative source after HF push step)
      skips:  dates with existing data/sector_results/{date}.json
      writes: data/pending_sector_batch.txt  (batch job ID)
      writes: data/batch_tasks_sector.jsonl
      commits all to main

GitHub Actions — collect-sector-results (triggered by daily-pipeline completion)
  → retrieve_batch_file_results.py
      reads:  data/pending_sector_batch.txt
      polls:  OpenAI Batch API (exit 0=done, 1=error, 2=retry)
      writes: data/sector_results/{date}.json  (one per date)
      clears: data/pending_sector_batch.txt on success

  → read_sector_results.py
      reads:  data/sector_results/*.json
      writes: data/sector_summary.tsv  (flat, one row per date × sector)

  → visualize_sentiment.py
      reads:  data/sector_summary.tsv
      writes: data/charts/sentiment_heatmap.png
              data/charts/sentiment_trends.png
              data/charts/sentiment_distribution.png

  → export_time_series.py
      reads:  data/sector_summary.tsv
      writes: data/sector_sentiment_pivot.tsv  (wide: date × 19 sectors, last 90 days)
              data/entity_sentiment_ts.tsv     (long: date × entity × sector, last 90 days)
      window: EXPORT_LOOKBACK_DAYS = 90 (constants.py)

  → build_sector_db.py
      reads:  data/sector_results/*.json  (all dates)
      writes: data/sector_results.db  (SQLite, full rebuild, atomic)
      tables: sector_analyses (one row per date × sector)
              sector_entities (one row per entity mention, FK to sector_analyses)

  → cluster_topics.py
      reads:  data/vectorstore/feeds/  (FAISS, rolling 45-day window)
      reads:  data/topic_centroids.json  (prior run centroids for continuity)
      reads:  data/topic_labels.json     (LLM label cache)
      writes: data/topic_trends.tsv      (append-only: date × topic × count)
              data/topic_centroids.json  (updated centroid map)
              data/topic_labels.json     (updated label cache)
              data/topic_clusters/{date}.json  (article → topic_id mapping)
      labels: new clusters via gpt-4o-mini (cached; 0-3 calls/day in steady state)
      aborts: with exit 2 on degenerate runs (noise_ratio > 0.90 or < 3 clusters)

  → push_new_analysis_to_hf.py
      reads:  data/sector_summary.tsv, data/topic_trends.tsv, data/entity_sentiment_ts.tsv
      dedup:  date+sector / date+topic_id / date+entity+sector (composite keys)
      writes: lacetohf/sector-analysis   (new date×sector rows)
              lacetohf/topic-trends      (new date×topic rows)
              lacetohf/entity-sentiment  (new date×entity×sector rows)
      idempotent: re-running same date is always safe
```

### Daily Briefing Batch Pipeline
```
create_batch_briefings.py
    reads:  data/topic_trends.tsv          (spike detection via get_emerging_topics)
    reads:  data/vectorstore/feeds/        (FAISS + BM25 retrieval, local, no API calls)
    skips:  dates with existing data/briefings/{date}.json
    writes: data/pending_briefings_batch.txt   (batch job ID)
    writes: data/pending_briefings_meta.json   (spike metadata: label, spike_ratio,
                                                article_count, pre-retrieved sources)
    writes: data/batch_tasks_briefings.jsonl   (debug copy of submitted tasks)

retrieve_batch_briefings.py
    reads:  data/pending_briefings_batch.txt
    reads:  data/pending_briefings_meta.json
    polls:  OpenAI Batch API (exit 0=done, 1=error, 2=retry — same pattern as sector batch)
    writes: data/briefings/{date}.json    (one per date; includes rag_answer, rag_sources,
                                           sectors from local _sector_crosscheck)
    clears: pending_briefings_batch.txt + pending_briefings_meta.json on full success

custom_id convention: "briefing-YYYY-MM-DD-{topic_id[:8]}"

Retrieval split:
  create step  — FAISS+BM25 retrieval runs locally (no LLM, no cost)
  batch step   — only the final LLM answer generation is sent to the Batch API
  collect step — sector cross-check (_sector_crosscheck) runs locally at collect time
```

**Briefing output schema** (`data/briefings/{date}.json`):
```json
{
  "date": "YYYY-MM-DD",
  "n_spikes": 3,
  "spikes": [
    {
      "topic_id": "uuid",
      "label": "Fed Rate Decision",
      "spike_ratio": 2.5,
      "article_count": 45,
      "rag_answer": "The Federal Reserve...",
      "rag_sources": [{"title","date","link","snippet","guid"}],
      "sectors": [{"sector","trend_direction","trend_delta","mean_sentiment_score"}]
    }
  ]
}
```
Compatible with `daily_briefing.py`'s `build_briefing()` output — same schema.

### Sector Analysis Schema
`SectorAnalysis` (Pydantic, in `create_batch_files_v2.py`) is the LLM output model. `SectorName` in `constants.py` is the single source of truth for the taxonomy:

| Field | Type |
|---|---|
| `entities` | `list[str]` — named companies/orgs |
| `sector` | `SectorName` — 19-value Literal imported from `constants.py` |
| `sentiment` | `Literal["positive", "neutral", "negative"]` |
| `news_category` | `Literal["earnings","M&A","regulation","macro","appointments","products","markets","other"]` |
| `extraction_status` | `Literal["ok", "partial"]` |

`MultiSectorAnalysis` wraps `list[SectorAnalysis]` (1–8 sectors per day).

### Querying Sector Data from External Flows
`query_sector.py` exposes four functions for downstream consumers (dashboards, agents, other scripts):

```python
from query_sector import get_snapshot, get_time_series, list_sectors
from query_sector import get_all_sectors_pivot, export_sector_pivot

# Current read — single latest entry
snap = get_snapshot("Technology Services")
# → {last_date, latest_sentiment, sentiment_score, entities, news_category, data_age_days}

# Trend over a rolling window
ts = get_time_series("Finance", lookback_days=60, include_articles=True)
# → {mean_sentiment_score, trend_direction, trend_delta,
#    top_entities, category_breakdown, time_series, articles}

# Discover valid names
list_sectors()   # sorted list of 19 valid sector names

# Bulk export — all sectors, last 90 days
pivot = get_all_sectors_pivot(lookback_days=90)  # DataFrame: date × 19 sectors
path  = export_sector_pivot()                    # writes data/sector_sentiment_pivot.tsv
```

- `get_snapshot` / `get_time_series` raise `ValueError` (bad name) or `LookupError` (valid name, no data yet)
- `trend_direction`: `"improving"` / `"deteriorating"` / `"stable"` based on first-half vs second-half mean of the window (threshold ±0.20)
- `include_articles=True` joins back to `output/feeds{date}.txt` for raw article headlines
- `get_all_sectors_pivot` columns are always alphabetically sorted (same order as `list_sectors()`)

### Querying Entity Data from External Flows
`query_entity.py` mirrors the sector API but pivots on entity (company, person, org). Entities are dynamic (LLM-extracted, no fixed taxonomy), so lookup is case-insensitive and unknown names raise `LookupError` with similar-name hints rather than a fixed list.

```python
from query_entity import get_entity_snapshot, get_entity_time_series, list_entities
from query_entity import get_all_entities_ts, export_entity_ts

# Discover known entities
names = list_entities()         # sorted list, returns [] if TSV missing

# Current read — all sectors mentioning the entity on its latest date
snap = get_entity_snapshot("nvidia")   # case-insensitive → resolves to "Nvidia"
# → {entity, last_date, data_age_days, dominant_sentiment,
#    mean_sentiment_score, sectors: [{sector, sentiment, sentiment_score, ...}]}

# Trend over a rolling window
ts = get_entity_time_series("Nvidia", lookback_days=90, include_articles=True)
# → {entity, lookback_days, date_range, n_observations,
#    mean_sentiment_score, trend_direction, trend_delta,
#    dominant_sentiment, sectors_seen, category_breakdown,
#    time_series, articles}

# Bulk export — all entities, last 90 days
ets  = get_all_entities_ts(lookback_days=90)  # long DataFrame: date × entity × sector
path = export_entity_ts()                     # writes data/entity_sentiment_ts.tsv
```

- Per-entity functions raise `LookupError` when entity is not found (includes up to 10 similar-name hints) or has no data in the window
- `trend_direction` threshold ±0.20 — same as `query_sector.py`
- `sectors_seen` is ranked by appearance frequency, not alphabetically
- `include_articles=True` joins back to `output/feeds{date}.txt` for raw article headlines
- `export_entity_ts` writes `date` as plain YYYY-MM-DD (R-friendly, no timezone)

### Bulk Export Files
Pre-computed snapshots auto-regenerated by CI after each collect run:

| File | Format | Description |
|---|---|---|
| `data/sector_sentiment_pivot.tsv` | wide (date × 19 sectors) | mean `sentiment_score` per date/sector; NaN = no data |
| `data/entity_sentiment_ts.tsv` | long (date × entity × sector) | one row per mention; filterable by any column |
| `data/sector_results.db` | SQLite | normalized, lossless — `sector_analyses` + `sector_entities` tables |
| `data/topic_trends.tsv` | append-only TSV | date × topic_id × topic_label × article_count × sentiment_score; one row per cluster per day; sentiment_score = mean sector sentiment for member articles (NaN for pre-backfill rows) |
| `data/topic_centroids.json` | JSON | persistent centroid map: topic_id → {label, centroid, first_seen, last_seen} |
| `data/topic_labels.json` | JSON | persistent LLM label cache: topic_id → label string |
| `data/topic_clusters/{date}.json` | JSON array | per-run article → topic_id mapping for the 45-day window |
| `data/briefings/{date}.json` | JSON | one per date: n_spikes, spikes list with rag_answer + rag_sources + sectors |

Rolling window for sector TSVs is `EXPORT_LOOKBACK_DAYS = 90` in `constants.py` — change once to update all callers.
Topic clustering window is `CLUSTER_WINDOW_DAYS = 45` — separate constant, also in `constants.py`.

R read-back:
```r
pivot <- read.delim("data/sector_sentiment_pivot.tsv", row.names = 1)
ets   <- read.delim("data/entity_sentiment_ts.tsv")
anthropic <- subset(ets, entity == "Anthropic")
```

### Sector SQLite Database (`build_sector_db.py`)
Lossless, normalized mirror of all `data/sector_results/*.json`. Full rebuild on every run (< 1 s for 200+ days), written atomically via `.db.tmp` + `os.replace()`.

**Schema:**
```sql
sector_analyses  (id, date, sector, sentiment, sentiment_score,
                  news_category, extraction_status, batch_id)
sector_entities  (id, analysis_id FK → sector_analyses.id, entity)
```
- `date` is always the **filename stem** (`2026-03-12.json` → `"2026-03-12"`), not the JSON body field.
- `sentiment_score` is denormalized (1/0/−1) from `SENTIMENT_SCORE` in `constants.py`.
- Malformed JSON files are logged to stderr and skipped; build continues.
- Indices on `date`, `sector`, `date+sector`, `lower(entity)`.

**Ad-hoc query example:**
```python
import sqlite3
conn = sqlite3.connect("data/sector_results.db")
conn.execute("""
    SELECT sa.date, sa.sector, se.entity
    FROM sector_analyses sa
    JOIN sector_entities se ON se.analysis_id = sa.id
    WHERE lower(se.entity) = 'nvidia'
    ORDER BY sa.date DESC LIMIT 10
""").fetchall()
```

Path constant: `SECTOR_DB_FILE` in `constants.py`.

### Key Files
| File | Purpose |
|---|---|
| `constants.py` | **Single source of truth**: `SectorName` Literal, `SECTOR_TAXONOMY` list, `SENTIMENT_SCORE`, file paths, `EXPORT_LOOKBACK_DAYS`, `SECTOR_DB_FILE`, `VECTORSTORE_DIR`, `FEEDS_REGISTRY_FILE`, clustering params (`CLUSTER_WINDOW_DAYS=45`, `CLUSTER_MIN_SIZE=10`, `CLUSTER_MIN_SAMPLES=3`, `CLUSTER_MAX_NOISE_RATIO=0.90`, `CLUSTER_MIN_CLUSTERS=3`, `CLUSTER_SELECTION_METHOD="leaf"`), topic file paths (`TOPIC_CENTROIDS_FILE`, `TOPIC_LABELS_FILE`, `TOPIC_TRENDS_FILE`, `TOPIC_CLUSTERS_DIR`), briefing batch paths (`PENDING_BRIEFINGS_BATCH_FILE`, `BRIEFINGS_BATCH_META_FILE`, `BATCH_FILE_BRIEFINGS`, `BRIEFINGS_DIR`) |
| `query_sector.py` | `get_snapshot()` + `get_time_series()` + `get_all_sectors_pivot()` + `export_sector_pivot()` |
| `query_entity.py` | `get_entity_snapshot()` + `get_entity_time_series()` + `get_all_entities_ts()` + `export_entity_ts()` |
| `export_time_series.py` | CLI — calls both export functions; run after `read_sector_results.py` |
| `build_sector_db.py` | CLI — full rebuild of `data/sector_results.db` from all `data/sector_results/*.json`; atomic write; run after `export_time_series.py` |
| `cluster_topics.py` | CLI + importable module — daily topic clustering; `run(date)` is the public entry point; see Topic Clustering section below |
| `tests/test_cluster_topics.py` | 39 unit tests for all `cluster_topics.py` public functions (TDD) |
| `embed_feeds.py` | CLI — cold-start build or incremental update of FAISS vectorstore from feed articles |
| `create_batch_files_v2.py` | Reads raw feeds, submits daily sector batch to OpenAI |
| `retrieve_batch_file_results.py` | Collects completed batch; routes to `data/sector_results/` |
| `read_sector_results.py` | Flattens all JSON results → `data/sector_summary.tsv` |
| `visualize_sentiment.py` | Three sentiment charts from `sector_summary.tsv` |
| `hybrid_rag.py` | CLI hybrid RAG: load FAISS + BM25 + query translation + LLM answer; exposes `ask()` public API for external callers |
| `chatbot_rag.py` | Streamlit chatbot wrapping `hybrid_rag.py`; streaming answers, sidebar controls |
| `pyproject.toml` | Makes the project pip-installable (`pip install -e .`) so external scripts can `from hybrid_rag import ask` |
| `daily_briefing.py` | Morning briefing: checks for pre-computed `data/briefings/{date}.json` first (`print_precomputed(date)`), falls back to live spike detection → RAG summary → sector cross-check; `build_briefing(date, top_n, use_rag)` is the live-compute entry point; `--save` forces a fresh re-run |
| `create_batch_briefings.py` | Runs FAISS+BM25 retrieval locally for each spike, builds OpenAI Batch API JSONL, submits batch, saves ID + spike metadata sidecar |
| `retrieve_batch_briefings.py` | Polls batch (exit 0/1/2), downloads results, assembles briefing JSONs with sector cross-check → `data/briefings/{date}.json` |
| `label_topics.py` | Labels all unlabeled topic_ids via OpenAI Batch API (kitai.batch); submits one job for all unlabeled topics, polls until done, updates topic_labels.json + topic_trends.tsv; use after `backfill.py --phase1-only` |
| `visualize_topics.py` | CLI — six static charts + optional animated GIF from `data/topic_trends.tsv`: `topic_spike_heatmap.png` (LogNorm count heatmap) + `topic_frequency_ts.png` (symlog line chart) + `topic_timeline.png` (Gantt) + `topic_sentiment_heatmap.png` (diverging red–green) + `topic_sentiment_delta.png` (7-day momentum) + `topic_signal_scatter.png` (spike × sentiment, latest date); `--animate` adds `topic_signal_scatter_animated.gif` (one frame per qualifying date, fixed axes/norm); `--fps N` (default 4); `--top N` (default 15), `--days N` (default 200) |
| `backfill.py` | Two-phase historical back-fill: Phase 1 = cluster_topics per date (no API), Phase 2 = daily_briefing per date; idempotent (skips already-done dates) |
| `push_feeds_to_hf.py` | Cold-start: push all `output/feeds*.txt` → `lacetohf/feeds`; run once |
| `push_new_feeds_to_hf.py` | Daily incremental: append today's new articles to `lacetohf/feeds`; dedup on `guid`; called by `daily-pipeline` |
| `push_analysis_to_hf.py` | Cold-start: create + push all three analysis datasets to HF; run once |
| `push_new_analysis_to_hf.py` | Daily incremental: append new rows to `lacetohf/sector-analysis`, `lacetohf/topic-trends`, `lacetohf/entity-sentiment`; composite-key dedup; called by `collect-sector-results` |
| `download.R` | RSS scraper, called by GitHub Actions |
| `old/` | Archived/experimental versions — not used in production (includes `chatbot6.py`, `trader_assistant.py`, `utils.py`, `create_batch_files.py`) |

### Strict Schema Note
`create_batch_files_v2.py` uses `_make_openai_strict()` to convert the Pydantic schema to OpenAI strict JSON schema format (adds `additionalProperties: false` + `required[]` recursively). This replaces the former private SDK call `openai.lib._pydantic.to_strict_json_schema`.

### Incremental Sentinel
All batch scripts use file existence as the processed sentinel:

**Sector batch:**
- `create_batch_files_v2.py` skips dates where `data/sector_results/{date}.json` already exists
- `retrieve_batch_file_results.py` clears `data/pending_sector_batch.txt` only after full success

**Briefing batch:**
- `create_batch_briefings.py` skips dates where `data/briefings/{date}.json` already exists; writes two sentinel files: `data/pending_briefings_batch.txt` (batch ID) and `data/pending_briefings_meta.json` (spike metadata + pre-retrieved sources keyed by custom_id)
- `retrieve_batch_briefings.py` clears both sentinel files only after full success; exit code 2 = batch still in progress (CI-safe retry)

### Feed Vectorstore (`embed_feeds.py`)
FAISS vectorstore of all feed articles, built once and updated daily by CI.

| Path | Description |
|---|---|
| `data/vectorstore/feeds/index.faiss` | FAISS flat-L2 index (1536-dim, `text-embedding-3-small`) |
| `data/vectorstore/feeds/index.pkl` | LangChain InMemoryDocstore + `index_to_docstore_id` map |
| `data/vectorstore/feeds_registry.tsv` | Ground truth: one row per embedded article (`id, date, title, link, guid`) |

- **Cold start**: run `python embed_feeds.py` once to build from all existing feeds (took ~25 min for 7714 articles).
- **Incremental**: CI runs `embed_feeds.py` daily — only new guids (not in registry) are embedded and appended via `FAISS.add_embeddings`. No rebuild.
- **Dedup key**: `guid` (stable RSS identifier). Registry is append-only.
- **ID scheme**: monotonic integers starting at 0, assigned at embed time, never reused.
- **Doc content**: `"{date}: {title}: {description}"` — matches sector analysis format.
- **Load for search**:
  ```python
  from langchain_community.vectorstores import FAISS
  from langchain_openai import OpenAIEmbeddings
  from constants import VECTORSTORE_DIR
  store = FAISS.load_local(str(VECTORSTORE_DIR), OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536), allow_dangerous_deserialization=True)
  results = store.similarity_search("Fed rate decision", k=5)
  ```
  Or use `hybrid_rag.py` / `chatbot_rag.py` which handle this automatically via `_OpenAIEmbeddings`.

### Topic Clustering (`cluster_topics.py`)

Unsupervised narrative discovery layered on top of the existing FAISS vectorstore. Runs daily after `build_sector_db.py` in CI. Does **not** re-embed — all vectors come from the existing store.

**How it works:**
1. Load the 45-day rolling window of article vectors from FAISS
2. PCA(50) for dimensionality reduction (deterministic, `random_state=42`)
3. HDBSCAN (`min_cluster_size=10`, `min_samples=3`) → ~19 clusters, ~60% noise (expected for news)
4. Centroid-cosine matching (threshold 0.85) assigns stable `topic_id` across daily runs
5. New clusters get LLM labels via `gpt-4o-mini`; cached labels reused for matched topics
6. New clusters get LLM labels via `gpt-4o-mini`; cached labels reused for matched topics
6a. `compute_topic_sentiment` joins cluster assignments → `sector_summary.tsv` to compute mean sentiment per topic (pure join, no API)
7. Appended to `data/topic_trends.tsv` (append-only, 5 columns: date × topic_id × topic_label × article_count × sentiment_score); spike ratios computed for signal

**Public API:**

```python
from cluster_topics import (
    extract_window_vectors,    # (date, window_days) -> (np.ndarray, pd.DataFrame)
    reduce_dimensions,         # (vectors, n_components=50) -> np.ndarray
    run_hdbscan,               # (X) -> (labels, noise_ratio) | raises ClusteringAborted
    compute_centroids,         # (vectors, labels) -> dict[int, np.ndarray]
    match_topics,              # (new_centroids, prior_topics, threshold=0.85) -> dict
    get_label,                 # (topic_id, cache, articles, llm_fn=None) -> str
    compute_topic_sentiment,   # (cluster_date, sector_summary_path, topic_clusters_dir) -> dict[str, float]
    append_trends,             # (date, rows, path) -> None | raises DuplicateDateError
    get_emerging_topics,       # (date, trends_df) -> list[dict]
    load_centroids,            # (path) -> dict  — returns {} if absent
    save_centroids,            # (data, path) -> None  — atomic write
    load_label_cache,          # (path) -> dict  — returns {} if absent
    save_label_cache,          # (data, path) -> None  — atomic write
    run,                       # (target_date, ...) -> summary_dict  — full pipeline
    ClusteringAborted,         # exception: degenerate clustering run
    DuplicateDateError,        # exception: append_trends called twice for same date
)
```

**`run()` — full pipeline for one day:**
```python
from cluster_topics import run
from datetime import date

summary = run(target_date=date(2026, 3, 13), skip_labeling=False)
# summary keys: date, window_articles, n_clusters, noise_ratio,
#               new_labels, matched_topics, topics_with_sentiment, cluster_sizes
```

**`get_emerging_topics()` — spike signal:**
```python
import pandas as pd
from cluster_topics import get_emerging_topics
from datetime import date

trends = pd.read_csv("data/topic_trends.tsv", sep="\t")
signals = get_emerging_topics(date.today(), trends)
# signals: list of {topic_id, label, spike_ratio, article_count, sentiment_score}
# sorted by spike_ratio descending; topics with article_count < 5 excluded
# sentiment_score is None when column absent from TSV (backward compatible)
```

**Exceptions:**
- `ClusteringAborted` — raised when `noise_ratio > CLUSTER_MAX_NOISE_RATIO` (0.90) or `n_clusters < CLUSTER_MIN_CLUSTERS` (3). CI treats exit code 2 as a warning, not a failure.
- `DuplicateDateError` — raised by `append_trends` if the target date already exists in the TSV. Delete the rows manually and re-run.

**Topic continuity invariant:** `topic_id` (UUID) is assigned once and never changes for the same narrative across runs. Labels may be updated but IDs are stable. Prior centroids are stored in `data/topic_centroids.json`.

**Empirical parameters (tuned on 7,972-article CNBC corpus):**

| Constant | Value | Rationale |
|---|---|---|
| `CLUSTER_WINDOW_DAYS` | 45 | ~1,700 articles; below 30d HDBSCAN collapses to 2 clusters |
| `CLUSTER_MIN_SIZE` | 10 | ~19 clusters at median size 18; above 20 too few clusters form |
| `CLUSTER_MIN_SAMPLES` | 3 | balanced noise/cluster tradeoff |
| `CLUSTER_MAX_NOISE_RATIO` | 0.90 | 60–70% noise is expected baseline for news data |
| `CLUSTER_MIN_CLUSTERS` | 3 | abort on truly degenerate runs only |
| `CLUSTER_SELECTION_METHOD` | `"leaf"` | HDBSCAN default `"eom"` over-merges to ~3 clusters on this corpus; `"leaf"` recovers ~19 |

### External Caller API (`ask()`)

Other Python projects on the same machine can query the RAG pipeline without Streamlit:

```python
# Option A — editable install (clean, versioned)
# pip install -e /path/to/rss_feed   (once, in the caller's venv)
from hybrid_rag import ask

result = ask("What happened to oil prices after Maduro left?")
# result = {"answer": str, "sources": list[dict], "queries": list[str]}
# sources keys: title, date, link, snippet, guid
# queries[0] is always the original unmodified query

# Option B — sys.path (no install needed)
import sys
sys.path.insert(0, r"C:\Users\l_ace\Desktop\projects\rss_feed")
from hybrid_rag import ask
result = ask("Fed rate decision", strategy="decompose", k_semantic=8)
```

`ask()` parameters: `query`, `strategy` ("expand"/"decompose"/"step_back"/"none"), `k_semantic`, `k_bm25`, `weights_sparse`.
Resources (FAISS, BM25 corpus) are loaded once on first call and cached for the process lifetime.

### Required Directories
- `output/` — daily feed files written by `download.R`
- `data/sector_results/` — created at runtime by collection script
- `data/charts/` — created at runtime by visualization script
- `data/` — must exist for `create_batch_files.py` (holds `batch_tasks_tickers.jsonl`, `book.txt`)
- `data/vectorstore/feeds/` — pre-built FAISS index used by `hybrid_rag.py`, `chatbot_rag.py`, and `cluster_topics.py`; built/updated by `embed_feeds.py`
- `data/topic_clusters/` — created at runtime by `cluster_topics.py`
- `data/briefings/` — created at runtime by `retrieve_batch_briefings.py` or `daily_briefing.py --save`

### CI/CD
| Workflow | Trigger | What it does |
|---|---|---|
| `daily-pipeline` | cron `0 19 * * 1-5` + manual | download.R → **push feeds to HF** → create_batch_files_v2.py → commit |
| `collect-sector-results` | on `daily-pipeline` completion + manual | retrieve → flatten → charts → **export TSVs** → **build SQLite db** → **cluster topics** → **push analysis to HF** → commit |
| `embed-feeds` | on `collect-sector-results` completion + manual | embed new feed articles → update FAISS store + registry → commit |
| `daily-briefing` | on `embed-feeds` completion + cron `0 13 * * 1-5` + manual | daily_briefing.py --save → commit briefing JSON |
