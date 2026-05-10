# API Reference

## query_sector.py

```python
from query_sector import get_snapshot, get_time_series, list_sectors
from query_sector import get_all_sectors_pivot, export_sector_pivot

snap = get_snapshot("Technology Services")
# → {last_date, latest_sentiment, sentiment_score, entities, news_category, data_age_days}

ts = get_time_series("Finance", lookback_days=60, include_articles=True)
# → {mean_sentiment_score, trend_direction, trend_delta,
#    top_entities, category_breakdown, time_series, articles}

list_sectors()          # sorted list of 19 valid sector names

pivot = get_all_sectors_pivot(lookback_days=90)  # DataFrame: date × 19 sectors
path  = export_sector_pivot()                    # writes data/sector_sentiment_pivot.tsv
```

- `get_snapshot` / `get_time_series` raise `ValueError` (bad name) or `LookupError` (no data)
- `trend_direction`: `"improving"` / `"deteriorating"` / `"stable"` (threshold ±0.20, first vs second half of window)
- `include_articles=True` joins back to `output/feeds{date}.txt`
- Columns in pivot always alphabetically sorted (same order as `list_sectors()`)
- Not a CLI script — import only

## query_entity.py

```python
from query_entity import get_entity_snapshot, get_entity_time_series, list_entities
from query_entity import get_all_entities_ts, export_entity_ts

names = list_entities()         # sorted list; returns [] if TSV missing

snap = get_entity_snapshot("nvidia")  # case-insensitive → resolves to "Nvidia"
# → {entity, last_date, data_age_days, dominant_sentiment,
#    mean_sentiment_score, sectors: [{sector, sentiment, sentiment_score, ...}]}

ts = get_entity_time_series("Nvidia", lookback_days=90, include_articles=True)
# → {entity, lookback_days, date_range, n_observations,
#    mean_sentiment_score, trend_direction, trend_delta,
#    dominant_sentiment, sectors_seen, category_breakdown,
#    time_series, articles}

ets  = get_all_entities_ts(lookback_days=90)
path = export_entity_ts()   # writes data/entity_sentiment_ts.tsv
```

- Raises `LookupError` with up to 10 similar-name hints when entity not found or no data in window
- `trend_direction` threshold ±0.20 — same as `query_sector.py`
- `sectors_seen` ranked by appearance frequency
- `export_entity_ts` writes `date` as plain YYYY-MM-DD (R-friendly, no timezone)

## hybrid_rag.py — ask()

```python
# Option A: editable install
# pip install -e /path/to/rss_feed
from hybrid_rag import ask

result = ask("What happened to oil prices after Maduro left?")
# result = {"answer": str, "sources": list[dict], "queries": list[str]}
# sources keys: title, date, link, snippet, guid
# queries[0] is always the original unmodified query

# Option B: sys.path
import sys
sys.path.insert(0, r"C:\Users\l_ace\Desktop\projects\rss_feed")
from hybrid_rag import ask
result = ask("Fed rate decision", strategy="decompose", k_semantic=8)
```

Parameters: `query`, `strategy` ("expand"/"decompose"/"step_back"/"none"), `k_semantic`, `k_bm25`, `weights_sparse`.

Resources (FAISS, BM25 corpus) are loaded once on first call and cached for the process lifetime.

## cluster_topics.py — Public API

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
    load_centroids, save_centroids,
    load_label_cache, save_label_cache,
    run,                       # (target_date, ...) -> summary_dict — full pipeline
    ClusteringAborted,
    DuplicateDateError,
)

# Full pipeline for one day:
from cluster_topics import run
from datetime import date
summary = run(target_date=date(2026, 3, 13), skip_labeling=False)
# summary keys: date, window_articles, n_clusters, noise_ratio,
#               new_labels, matched_topics, topics_with_sentiment, cluster_sizes

# Spike signal:
import pandas as pd
from cluster_topics import get_emerging_topics
trends = pd.read_csv("data/topic_trends.tsv", sep="\t")
signals = get_emerging_topics(date.today(), trends)
# signals: list of {topic_id, label, spike_ratio, article_count, sentiment_score}
# sorted by spike_ratio descending; topics with article_count < 5 excluded
```

## SQLite Ad-hoc Queries

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

## R Read-back

```r
pivot <- read.delim("data/sector_sentiment_pivot.tsv", row.names = 1)
ets   <- read.delim("data/entity_sentiment_ts.tsv")
anthropic <- subset(ets, entity == "Anthropic")
```
