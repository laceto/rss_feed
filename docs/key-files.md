# Key Files

## Source Files

| File | Purpose |
|---|---|
| `constants.py` | **Single source of truth**: `SectorName` Literal, `SECTOR_TAXONOMY`, `SENTIMENT_SCORE`, all file paths, `EXPORT_LOOKBACK_DAYS=90`, `SECTOR_DB_FILE`, `VECTORSTORE_DIR`, `FEEDS_REGISTRY_FILE`, clustering params, topic paths, briefing batch paths |
| `create_batch_files_v2.py` | Reads raw feeds, submits daily sector batch to OpenAI |
| `retrieve_batch_file_results.py` | Collects completed batch → `data/sector_results/{date}.json` |
| `read_sector_results.py` | Flattens JSON → `data/sector_summary.tsv` |
| `visualize_sentiment.py` | Three charts from `sector_summary.tsv` → `data/charts/` |
| `export_time_series.py` | CLI: writes bulk TSV exports |
| `build_sector_db.py` | CLI: full rebuild of `data/sector_results.db`; atomic; run after export_time_series.py |
| `query_sector.py` | Module API: get_snapshot, get_time_series, get_all_sectors_pivot, export_sector_pivot |
| `query_entity.py` | Module API: get_entity_snapshot, get_entity_time_series, get_all_entities_ts, export_entity_ts |
| `embed_feeds.py` | CLI: cold-start build or incremental update of FAISS vectorstore |
| `cluster_topics.py` | CLI + importable: daily topic clustering; `run(date)` is the public entry point |
| `hybrid_rag.py` | CLI hybrid RAG: loads FAISS + BM25, query translation; exposes `ask()` public API |
| `chatbot_rag.py` | Streamlit chatbot wrapping hybrid_rag.py; run with `streamlit run chatbot_rag.py` |
| `daily_briefing.py` | Morning briefing: pre-computed JSON → live spike detection → RAG → sector cross-check |
| `create_batch_briefings.py` | Local FAISS+BM25 retrieval per spike + OpenAI Batch API submission |
| `retrieve_batch_briefings.py` | Polls batch; assembles briefing JSONs; writes `data/briefings/{date}.json` |
| `label_topics.py` | Labels unlabeled topic_ids via OpenAI Batch API; use after `backfill.py --phase1-only` |
| `visualize_topics.py` | Six static charts + optional animated GIF from `topic_trends.tsv` |
| `backfill.py` | Two-phase historical backfill (Phase 1: cluster, Phase 2: brief); idempotent |
| `push_feeds_to_hf.py` | Cold-start: push all feeds to lacetohf/feeds |
| `push_new_feeds_to_hf.py` | Daily incremental: append today's new articles; dedup on guid |
| `push_analysis_to_hf.py` | Cold-start: create 3 HF repos + push all analysis data |
| `push_new_analysis_to_hf.py` | Daily incremental: append new rows to 3 HF datasets; composite-key dedup |
| `download.R` | RSS scraper, called by GitHub Actions |
| `tests/test_cluster_topics.py` | 50 unit tests for all cluster_topics.py public functions |
| `pyproject.toml` | Makes the project pip-installable (`pip install -e .`) |
| `old/` | Archived/experimental versions — NOT used in production |

## Required Directories

- `output/` — daily feed files written by `download.R`
- `data/sector_results/` — created at runtime by collection script
- `data/charts/` — created at runtime by visualization script
- `data/vectorstore/feeds/` — pre-built FAISS index; built/updated by `embed_feeds.py`
- `data/topic_clusters/` — created at runtime by `cluster_topics.py`
- `data/briefings/` — created at runtime by `retrieve_batch_briefings.py` or `daily_briefing.py --save`

## pipeline/ Package

Installable library layer (`pip install -e .`). All submodules use explicit parameters — no module-level side effects on import. `cluster_topics` is NOT in the package; import it from the top-level module.

| Module | Purpose |
|---|---|
| `pipeline/batch_sector.py` | SectorAnalysis/MultiSectorAnalysis Pydantic models, `submit_batch(tasks, client)` |
| `pipeline/batch_collect.py` | `check_batch_status(batch_id, client) -> dict`, `download_results(batch_id, client)` |
| `pipeline/batch_briefings.py` | briefing batch helpers |
| `pipeline/hf_io.py` | HuggingFace I/O helpers |
| `pipeline/query_sector.py` | sector query API + shared helpers |
| `pipeline/query_entity.py` | entity query API |
| `pipeline/sector_io.py` | load_sector_json, insert_sector_date, build_sector_db, etc. |
| `pipeline/sentiment_charts.py` | chart functions |
| `pipeline/topic_charts.py` | all plot_* functions; matplotlib.use("Agg") at module import |
