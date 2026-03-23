# Topic Clustering Pipeline for RSS News (Implementation Plan)

## Objective

Build an end-to-end pipeline that transforms raw RSS news into **clustered market narratives (topics)** and **time-series signals** usable for trading insights.

The edge is not in the clusters themselves — it is in:

- **changes in topic frequency** (emerging narratives)
- **novel topics outside the fixed 19-sector taxonomy**
- **divergence between narrative intensity and market pricing**

---

## Relationship to the Existing Pipeline

This is an additive layer on top of the sector analysis pipeline already in production. The table below shows what is already done vs. what is new:

| Step | Status | Notes |
|---|---|---|
| RSS ingestion | **Done** — `download.R` | TSV in `output/feeds{date}.txt` |
| Text preparation | **Done** — feed format includes title + description | `"{date}: {title}: {description}"` is the doc content format |
| Embeddings | **Done** — `embed_feeds.py` | FAISS vectorstore at `data/vectorstore/feeds/`, `text-embedding-3-small`, 1536-dim, 7714+ articles |
| Entity extraction | **Done** — `query_entity.py` | LLM-extracted, case-insensitive lookup, time-series API |
| Sector labeling | **Done** — `query_sector.py` | 19 fixed sectors, `data/sector_results/{date}.json` |
| Dimensionality reduction | **New** | PCA(50) on existing FAISS vectors |
| Clustering | **New** | HDBSCAN on rolling window |
| Topic labeling | **New** | LLM labels with persistent cache |
| Topic time-series | **New** | `data/topic_trends.tsv` |
| Emerging topic signal | **New** | spike = today / rolling_7d_avg |

**Critical: do not re-embed articles.** The vectorstore already exists. All clustering work must load vectors from `data/vectorstore/feeds/` via the FAISS index — creating a second embedding store wastes API cost and creates a data consistency problem.

---

## High-Level Architecture

```text
[Existing]
RSS feeds (TSV) → embed_feeds.py → FAISS vectorstore (1536-dim)
                                  → feeds_registry.tsv (guid, date, title, id)

[New — cluster_topics.py]
FAISS vectors → PCA (50 dims) → HDBSCAN (rolling 45-day window)
             → centroid-cosine topic continuity matching
             → LLM labeling (new clusters only, cached labels reused)
             → data/topic_clusters/{date}.json
             → data/topic_trends.tsv
             → emerging topic signal
```

---

## 1. Data Ingestion & Storage

**Status: already done.** No new work required.

- Feed files: `output/feeds{date}.txt` (TSV: title, description, link, guid, type, id, sponsored, pubDate)
- FAISS registry: `data/vectorstore/feeds_registry.tsv` (id, date, title, link, guid)
- Embeddings: `data/vectorstore/feeds/index.faiss` + `index.pkl`

The registry is the join key between cluster output and article metadata. Do not introduce `rss_raw.parquet` — the existing format is sufficient.

---

## 2. Text Preparation

**Status: already done.** The vectorstore was built from:

```python
doc_content = f"{date}: {title}: {description}"
```

This is the canonical doc content format for this project. No changes needed.

---

## 3. Embedding Generation

**Status: already done.**

- Model: `text-embedding-3-small` (1536-dim) — **not** `text-embedding-3-large`
- Storage: FAISS flat-L2 index at `data/vectorstore/feeds/`
- Incremental updates: `embed_feeds.py` appends only new guids each day

To load vectors for clustering:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from constants import VECTORSTORE_DIR

store = FAISS.load_local(
    str(VECTORSTORE_DIR),
    OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536),
    allow_dangerous_deserialization=True,
)
# Extract all vectors as numpy array
import numpy as np
vectors = np.array([store.index.reconstruct(i) for i in range(store.index.ntotal)])
```

---

## 4. Dimensionality Reduction

**Status: new.**

### Goal

Reduce noise and improve clustering quality. HDBSCAN degrades in high dimensions.

### Approach

```text
Embedding (1536 dims) -> PCA -> 50 dims -> HDBSCAN
```

PCA is fast, deterministic, and stable — correct choice for a daily CI job. UMAP produces tighter clusters but is stochastic and slower; defer to post-MVP.

### Parameters

- `n_components = 50` (MVP)
- Fit PCA on the rolling window corpus, not the full history (prevents old articles influencing current cluster geometry)

### Output

- `X_reduced` — numpy array (n_articles, 50)

---

## 5. Rolling Window & Clustering

**Status: new.**

### Why Rolling Window

Daily article count (~50–80 for CNBC) is too small for HDBSCAN to form stable clusters. A rolling window provides enough mass.

Recommended window: **45 days** (~2,250–3,600 articles). Profile on the actual corpus before locking this in.

Parameters to add to `constants.py`:

```python
CLUSTER_WINDOW_DAYS: int = 45
CLUSTER_MIN_SIZE: int = 20
CLUSTER_MIN_SAMPLES: int = 5
```

### Algorithm: HDBSCAN

```python
import hdbscan
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=CLUSTER_MIN_SIZE,
    min_samples=CLUSTER_MIN_SAMPLES,
    metric="euclidean",
)
labels = clusterer.fit_predict(X_reduced)
```

### Behavior

- Automatically finds number of clusters
- Labels noise as `-1`
- Noise ratio > 0.80 signals a bad run — abort and log to stderr, do not write output

### Output

- `df["cluster"]` — integer cluster ID for the current run

---

## 6. Topic Continuity

**Status: new. Must be implemented before any time-series work.**

### Problem

HDBSCAN assigns arbitrary integer IDs on each run. Cluster 3 today is not the same as cluster 3 tomorrow. Without continuity tracking, the time-series is meaningless.

### Solution: Centroid-Cosine Matching

For each run:
1. Compute each cluster's centroid (mean of member article embeddings in PCA space)
2. Load centroids from the previous run's output
3. Match today's centroids to prior centroids via cosine similarity
4. If `similarity >= 0.85`: assign the same persistent `topic_id`
5. If no match: assign a new `topic_id` (UUID or monotonic int)
6. Save updated centroid map to `data/topic_centroids.json`

```python
{
  "topic_id": "t_042",
  "label": "AI chip supply constraints",
  "centroid": [...],  # list[float], PCA space
  "first_seen": "2026-01-15",
  "last_seen": "2026-03-22"
}
```

### Invariant

`topic_id` is stable across runs for the same narrative. Label changes are allowed (via re-labeling) but `topic_id` never changes once assigned.

---

## 7. Topic Labeling (LLM)

**Status: new.**

### Goal

Convert clusters to human-readable topic labels. Label only new clusters — reuse cached labels for matched persistent topics.

### Steps

For each cluster with no existing label:

1. Sample 10–15 representative articles (highest HDBSCAN membership probability)
2. Extract titles
3. Prompt LLM:

```text
"These are news headlines from the same topic cluster:
{titles}

Give a short topic label of 3–5 words. Be specific (e.g. 'Fed rate pause bets', not 'economic news')."
```

4. Store in persistent cache:

```python
# data/topic_labels.json
{ "t_042": "AI chip supply constraints", ... }
```

### Cost Control

- Only new `topic_id`s (no prior centroid match) trigger an LLM call
- In steady state, most topics persist → 0–3 LLM calls/day
- Cache is append-only; stale topics are never evicted (label history is valuable)

---

## 8. Time-Series Construction

**Status: new.**

### Goal

Track topic frequency over time. Each daily run appends one row per topic.

### Schema

```python
# data/topic_trends.tsv
# date    topic_id    topic_label             article_count
2026-03-20  t_042  AI chip supply constraints  14
2026-03-20  t_018  Fed rate pause bets         9
```

### Invariant

`topic_trends.tsv` is append-only. Each collect run appends today's rows. Never rewrite historical rows.

---

## 9. Signal Generation

**Status: new.**

### A. Emerging Topics

```python
# spike ratio: today vs rolling 7-day average
spike = today_count / rolling_7d_avg
# threshold: spike >= 2.0 → "emerging"
```

Output: dict of `{topic_id: spike_ratio}` sorted descending.

### B. Narrative Momentum

Increasing topic frequency over 7-day window. Computed as linear regression slope on daily counts.

### C. Sector-Topic Co-occurrence

Join cluster membership with `data/sector_results/{date}.json` on article date. For each topic, show the top 2 sectors it co-occurs with. This links emergent narratives to actionable sector exposures.

```python
# Example output
{
  "topic_id": "t_042",
  "label": "AI chip supply constraints",
  "spike_ratio": 3.2,
  "top_sectors": ["Technology Services", "Hardware & Semiconductors"]
}
```

### D. Noise Filtering

- Ignore cluster `-1`
- Ignore topics with `article_count < 5` on the signal day

---

## 10. Validation & Backtesting

### Goal

Verify if topic spikes have predictive power before surfacing them as signals.

### Steps

- Map topics to tickers via entity co-occurrence (entities already extracted in `query_entity.py`)
- Join with price data (yfinance or existing data source)
- Test: topic spike on day T → return on days T+1, T+3, T+5

### Metrics

- Hit rate (% of spikes where return is positive)
- Average return per spike
- Max drawdown from spike-triggered positions

---

## 11. Incremental Updates (Daily Pipeline)

### Steps

1. `embed_feeds.py` runs — new articles appended to FAISS + registry
2. `cluster_topics.py` runs — loads vectors from FAISS for the 45-day rolling window, runs PCA + HDBSCAN, matches centroids, labels new clusters, appends to `topic_trends.tsv`
3. Output committed to main by CI

### Why Rolling Window

- Prevents topic drift (stale narratives age out naturally)
- Keeps cluster geometry relevant to recent market conditions
- Window size controlled by `CLUSTER_WINDOW_DAYS` in `constants.py`

---

## 12. Storage Design

All new files live under `data/` to match existing conventions.

| File | Format | Description |
|---|---|---|
| `data/topic_clusters/{date}.json` | JSON | Article → topic_id mapping for that date's window run |
| `data/topic_centroids.json` | JSON | Persistent centroid map: topic_id → {label, centroid, first_seen, last_seen} |
| `data/topic_labels.json` | JSON | Persistent label cache: topic_id → label string |
| `data/topic_trends.tsv` | TSV (append-only) | date × topic_id × label × article_count |

No Parquet files. The existing TSV + JSON + SQLite pattern is sufficient and consistent with the project.

---

## 13. Monitoring & Quality Checks

### Per-Run Health Checks (abort if failing)

| Check | Abort threshold | Log level |
|---|---|---|
| Noise ratio (`cluster == -1`) | > 0.80 | ERROR — skip writing output |
| Number of clusters | 0 | ERROR — skip writing output |
| Rolling window article count | < 500 | WARNING — proceed but flag in output |

### Quality Metrics (write to stdout for CI logs)

- Total clusters formed
- % noise points
- Cluster size distribution (min, median, max)
- New topics created today (required LLM call)
- Topics matched to prior run
- Topics dropped (no match, below min_size)

---

## 14. New Dependencies

```
hdbscan
scikit-learn   # PCA — likely already installed via langchain
```

Add to `requirements.txt` after verifying against venv.

---

## 15. Future Enhancements

| Enhancement | Value | Prerequisite |
|---|---|---|
| UMAP instead of PCA | Tighter clusters, better separation | Validate stability in CI |
| BERTopic | Replaces steps 4–7, adds `get_topic_over_time()` | Evaluate migration cost vs. benefit |
| Sentiment per topic | Cross-reference with sector sentiment | Topic → sector co-occurrence (step 9C) |
| Topic → ticker mapping | Actionable signal | Entity co-occurrence + `query_entity.py` |
| Cross-asset linkage | Macro narrative detection | Requires multi-source data |

---

## Implementation Breakdown

**Goal:** Build `cluster_topics.py` — a standalone CLI and importable module that produces stable topic time-series and emerging narrative signals from existing FAISS vectors, integrated into daily CI.

Each leaf node is: concrete (specific file/function/test), small (one focused session), and verifiable (clear done condition).

```
A1  Configure parameters via corpus profiling
  A1.1  Load vectorstore, reconstruct all vectors + join registry metadata into a DataFrame
  A1.2  Profile article counts per 30/45/60-day rolling window; print distribution
  A1.3  Run HDBSCAN test passes on each window; record noise_ratio and n_clusters for each
  A1.4  Pick window size (target: noise_ratio < 0.30, n_clusters >= 5);
        add CLUSTER_WINDOW_DAYS / CLUSTER_MIN_SIZE / CLUSTER_MIN_SAMPLES to constants.py

A2  Core clustering module — TDD (Red -> Green -> Refactor)
  A2.1  [Red]    Write test: extract_window_vectors(date, window_days) returns
                 (np.ndarray, pd.DataFrame) with matching row counts
  A2.2  [Green]  Implement extract_window_vectors() — loads FAISS + registry,
                 filters to date window, returns (vectors, metadata)
  A2.3  [Red]    Write test: reduce_dimensions(vectors, n_components=50)
                 returns shape (n, 50); deterministic (fixed random_state)
  A2.4  [Green]  Implement reduce_dimensions() — sklearn PCA, fit on input only (not all-time)
  A2.5  [Red]    Write test: run_hdbscan(X_reduced) returns labels array len == n;
                 noise_ratio computed correctly
  A2.6  [Red]    Write test: run_hdbscan() raises ClusteringAborted
                 when noise_ratio > 0.80 or n_clusters == 0
  A2.7  [Green]  Implement run_hdbscan() + ClusteringAborted exception
  A2.8  [Refactor] Extract into well-named functions; add docstrings with invariants

A3  Topic continuity — TDD
  A3.1  [Red]    Write test: compute_centroids(vectors, labels) returns
                 dict[int, np.ndarray]; excludes label -1; centroid shape == (n_features,)
  A3.2  [Green]  Implement compute_centroids() — np.mean over member rows per cluster
  A3.3  [Red]    Write test: match_topics(new_centroids, prior_topics, threshold=0.85):
                 high-similarity pairs get existing topic_id; unmatched get new UUID;
                 threshold boundary is exact; no double-assignment
  A3.4  [Green]  Implement match_topics() — cosine similarity matrix,
                 greedy assignment (highest-similarity first)
  A3.5  [Red]    Write test: load_centroids() returns {} when file absent;
                 save/load round-trips correctly (centroid as list, not np.ndarray)
  A3.6  [Green]  Implement load_centroids() / save_centroids() for data/topic_centroids.json

A4  LLM labeling with persistent cache — TDD
  A4.1  [Red]    Write test: get_label(topic_id, cache) returns cached string if key exists;
                 does NOT call LLM
  A4.2  [Red]    Write test: get_label(topic_id, cache, articles) calls mock LLM once for
                 new topic_id and stores result in cache
  A4.3  [Green]  Implement get_label() — check cache first; call LLM only on miss; update cache
  A4.4  [Green]  Implement _label_via_llm(articles) — sample top-N by HDBSCAN membership
                 probability; prompt: "3-5 word label for these headlines"
  A4.5  [Red/Green] Write test + implement load_label_cache() / save_label_cache()
                 for data/topic_labels.json
  A4.6  [Refactor] Label generation isolated from cache I/O

A5  Time-series output — TDD
  A5.1  [Red]    Write test: append_trends(date, rows, path) creates file with header
                 if absent; appends without duplicate header on second call
  A5.2  [Red]    Write test: append_trends() with same date called twice
                 raises DuplicateDateError (idempotency guard)
  A5.3  [Green]  Implement append_trends() — TSV columns: date, topic_id, topic_label,
                 article_count
  A5.4  [Refactor] Atomic write via temp file + os.replace (same pattern as build_sector_db.py)

A6  Signal generation — TDD
  A6.1  [Red]    Write test: compute_spike(topic_id, trends_df, date) returns correct ratio;
                 returns None when < 3 days of history
  A6.2  [Green]  Implement compute_spike() — rolling 7-day mean from trends_df
  A6.3  [Red]    Write test: get_sector_cooccurrence(article_guids, date) returns
                 top-2 sectors by count from sector_results JSON
  A6.4  [Green]  Implement get_sector_cooccurrence() — loads data/sector_results/{date}.json;
                 joins on entity overlap (same date approximation)
  A6.5  [Green]  Implement get_emerging_topics(date) — assembles
                 {topic_id, label, spike_ratio, top_sectors};
                 filters noise (article_count < 5); sorts by spike_ratio desc

A7  CLI entry point and CI integration
  A7.1  Implement __main__ block: --date arg (defaults to today);
        print quality metrics to stdout; exit 1 on ClusteringAborted
  A7.2  Add hdbscan to requirements.txt;
        verify scikit-learn already in venv (sklearn.decomposition.PCA)
  A7.3  Add python cluster_topics.py step to
        .github/workflows/collect-sector-results.yml after build_sector_db.py
  A7.4  Add data/topic_trends.tsv, data/topic_centroids.json, data/topic_labels.json
        to the CI git commit step

A8  Backtesting — gate before promoting signal to production
  A8.1  Build topic-to-ticker map: join topic article_ids with query_entity output
        for the same dates
  A8.2  Fetch price data for mapped tickers via yfinance (T, T+1, T+3, T+5)
  A8.3  Compute hit_rate, avg_return at T+1/T+3/T+5, max drawdown per spike event
  A8.4  Document pass threshold: promote signal only if hit_rate > 0.55
        or avg_return_T+3 > 0.5%
```

### Dependency Order

`A1 -> A2 -> A3 -> A4 -> A5 -> A6 -> A7 -> A8`

A1 must complete before A2 (parameters inform HDBSCAN config).
A3 (continuity) must complete before A5 (time-series needs stable `topic_id`).
A4 (labels) can run in parallel with A5 after A3.
A8 (backtesting) is independent of A7 and can begin as soon as A5 produces data.

---

## Final Deliverables

- `cluster_topics.py` — standalone CLI, importable module
- `data/topic_trends.tsv` — append-only time-series of topic frequencies
- `data/topic_centroids.json` — persistent topic identity across runs
- `data/topic_labels.json` — persistent LLM label cache
- Emerging topic signal API (dict of spike ratios, callable from `hybrid_rag.py` context)

---

## Key Open Question

Does centroid-cosine matching at threshold 0.85 produce stable enough topic IDs on the actual CNBC article corpus, or does narrative drift require a lower threshold with manual override labels?

This must be answered empirically on the existing 7,714-article corpus before the time-series step is built.
