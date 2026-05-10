# Key Files

## pipeline/ — Shared Library

| Module | Purpose |
|---|---|
| `pipeline/constants.py` | **Single source of truth**: `SectorName` Literal, `SECTOR_TAXONOMY`, `SENTIMENT_SCORE`, all file paths, `EXPORT_LOOKBACK_DAYS=90`, `SECTOR_DB_FILE`, `VECTORSTORE_DIR`, `FEEDS_REGISTRY_FILE`, clustering params, topic paths, briefing batch paths |
| `pipeline/query_sector.py` | Module API: `get_snapshot`, `get_time_series`, `get_all_sectors_pivot`, `export_sector_pivot` |
| `pipeline/query_entity.py` | Module API: `get_entity_snapshot`, `get_entity_time_series`, `get_all_entities_ts`, `export_entity_ts` |
| `pipeline/cluster_topics.py` | Daily topic clustering; `run(date)` is the public entry point; exposes `get_emerging_topics`, `ClusteringAborted`, `DuplicateDateError` |
| `pipeline/hybrid_rag.py` | Hybrid RAG (BM25 + FAISS + query translation); exposes `ask()` public API |
| `pipeline/__init__.py` | Package entry point; documents public API surface |

Root shims (`hybrid_rag.py`, `cluster_topics.py`) re-export from `pipeline/` for backward compat.

## batch/ — OpenAI Batch API

| File | Purpose |
|---|---|
| `batch/create_batch_files_v2.py` | Reads raw feeds from HF, submits daily sector batch to OpenAI |
| `batch/retrieve_batch_file_results.py` | Polls and collects sector batch → `data/sector_results/{date}.json` |
| `batch/create_batch_briefings.py` | Local FAISS+BM25 retrieval per spike + OpenAI Batch API submission |
| `batch/retrieve_batch_briefings.py` | Polls batch; assembles briefing JSONs → `data/briefings/{date}.json` |

## results/ — Flatten + Export

| File | Purpose |
|---|---|
| `results/read_sector_results.py` | Flattens JSON → `data/sector_summary.tsv` |
| `results/export_time_series.py` | CLI: writes bulk TSV exports (sector pivot + entity TS) |
| `results/build_sector_db.py` | Full rebuild of `data/sector_results.db` (SQLite); atomic; run after export |

## ingest/ — HuggingFace Push

| File | Purpose |
|---|---|
| `ingest/push_new_feeds_to_hf.py` | Daily incremental: append today's new articles to `lacetohf/feeds`; dedup on `guid` |
| `ingest/push_feeds_to_hf.py` | Cold-start: push all feeds to `lacetohf/feeds` |
| `ingest/push_new_analysis_to_hf.py` | Daily incremental: append new rows to 3 HF datasets; composite-key dedup |
| `ingest/push_analysis_to_hf.py` | Cold-start: create 3 HF repos + push all analysis data |

## enrich/ — Embed, Cluster, Brief

| File | Purpose |
|---|---|
| `enrich/embed_feeds.py` | Cold-start build or incremental update of FAISS vectorstore |
| `enrich/cluster_topics.py` | CLI entry point for daily topic clustering (delegates to `pipeline/cluster_topics.py`) |
| `enrich/label_topics.py` | Labels unlabeled topic_ids via OpenAI Batch API; use after `backfill --phase1-only` |
| `enrich/daily_briefing.py` | Morning briefing: pre-computed JSON → live spike detection → RAG → sector cross-check |
| `enrich/backfill.py` | Two-phase historical backfill (Phase 1: cluster, Phase 2: brief); idempotent |

## output/ — Visualization + RAG Interface

| File | Purpose |
|---|---|
| `output/visualize_sentiment.py` | Three charts from `sector_summary.tsv` → `data/charts/` |
| `output/visualize_topics.py` | Six static charts + optional animated GIF from `topic_trends.tsv` |
| `output/chatbot_rag.py` | Streamlit chatbot wrapping `pipeline.hybrid_rag`; run with `streamlit run output/chatbot_rag.py` |

## scraper/ — R Scraper

| File | Purpose |
|---|---|
| `scraper/download.R` | Fetches CNBC RSS feeds → `output/feeds{date}.txt`; run with `Rscript scraper/download.R` |
| `scraper/DESCRIPTION` | R package deps (`rvest`, `xml2`, `XML`, `dplyr`, `purrr`) |
| `scraper/README.md` | Interface contract: writes `output/feeds{date}.txt` → HF |

## Root Config Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Makes `pipeline/` pip-installable (`pip install -e .`) |
| `Justfile` | Named tasks for every pipeline step; sets `PYTHONPATH=.` |
| `CLAUDE.md` | Agent router — routes to task-specific rule files |
| `hybrid_rag.py` | Backward-compat shim: `from hybrid_rag import ask` still works |
| `cluster_topics.py` | Backward-compat shim: `from cluster_topics import run` still works |

## Tests

| File | Purpose |
|---|---|
| `tests/test_cluster_topics.py` | 61 unit tests for `pipeline/cluster_topics.py` (TDD) |
| `tests/test_build_sector_db.py` | Unit tests for `results/build_sector_db.py` |

## Required Directories

- `output/` — daily feed files written by `scraper/download.R`
- `data/sector_results/` — created at runtime by collect script
- `data/charts/` — created at runtime by visualization script
- `data/vectorstore/feeds/` — pre-built FAISS index; built/updated by `enrich/embed_feeds.py`
- `data/topic_clusters/` — created at runtime by `pipeline/cluster_topics.py`
- `data/briefings/` — created at runtime by `batch/retrieve_batch_briefings.py` or `enrich/daily_briefing.py --save`
