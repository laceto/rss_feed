# Architecture

## Sector Analysis Pipeline (primary daily pipeline)

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
      writes: lacetohf/sector-analysis, lacetohf/topic-trends, lacetohf/entity-sentiment
      idempotent: re-running same date is always safe

GitHub Actions — embed-feeds (triggered by collect-sector-results)
  → embed_feeds.py (incremental FAISS update)

GitHub Actions — daily-briefing (triggered by embed-feeds + cron 0 13 * * 1-5)
  → daily_briefing.py --save → data/briefings/{date}.json
```

## Daily Briefing Batch Pipeline

```
create_batch_briefings.py
    reads:  data/topic_trends.tsv          (spike detection via get_emerging_topics)
    reads:  data/vectorstore/feeds/        (FAISS + BM25 retrieval, local, no API calls)
    skips:  dates with existing data/briefings/{date}.json
    writes: data/pending_briefings_batch.txt   (batch job ID)
    writes: data/pending_briefings_meta.json   (spike metadata + pre-retrieved sources)
    writes: data/batch_tasks_briefings.jsonl

retrieve_batch_briefings.py
    reads:  data/pending_briefings_batch.txt
    reads:  data/pending_briefings_meta.json
    polls:  OpenAI Batch API (exit 0=done, 1=error, 2=retry)
    writes: data/briefings/{date}.json
    clears: both sentinel files on full success
```

## Incremental Sentinel Pattern

All batch scripts use **file existence as the processed sentinel**.

**Sector batch:**
- `create_batch_files_v2.py` skips dates where `data/sector_results/{date}.json` already exists
- `retrieve_batch_file_results.py` clears `data/pending_sector_batch.txt` only after full success

**Briefing batch:**
- `create_batch_briefings.py` skips dates where `data/briefings/{date}.json` already exists
- `retrieve_batch_briefings.py` clears both sentinel files only after full success
- Exit code 2 = batch still in progress (CI-safe retry); 0 = done; 1 = error

## Retrieval Split (briefings)

- **create step** — FAISS+BM25 retrieval runs locally (no LLM, no cost)
- **batch step** — only the final LLM answer generation is sent to the Batch API
- **collect step** — sector cross-check runs locally at collect time
