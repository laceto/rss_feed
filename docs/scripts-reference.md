# Scripts Reference

## Environment

```bash
venv\Scripts\activate   # Windows
export PYTHONPATH=.     # required so pipeline/ imports resolve
```

Secrets in `.env` (gitignored): `OPENAI_API_KEY`, `HF_TOKEN`, `HUGGINGFACE_REPO`.

## Justfile (recommended)

Install [just](https://github.com/casey/just), then use named tasks instead of bare paths.
`just` sets `PYTHONPATH=.` automatically.

```bash
just scrape              # run R scraper
just push-feeds          # push today's feeds to HF
just batch-submit        # submit sector batch
just batch-collect       # collect sector batch results
just flatten             # read_sector_results → sector_summary.tsv
just charts              # visualize_sentiment
just export-ts           # export time-series TSVs
just build-db            # build SQLite db
just cluster             # cluster topics for today
just topic-charts        # visualize_topics
just push-analysis       # push analysis to HF
just embed               # embed new feed articles
just briefing --save     # generate daily briefing
just collect             # run full collect sequence locally
```

## R — RSS Scraper

```bash
Rscript scraper/download.R   # run from repo root
```
Outputs `output/feeds{YYYY-MM-DD}.txt`. R deps in `scraper/DESCRIPTION`.

## Sector Analysis Pipeline

```bash
python batch/create_batch_files_v2.py        # submit sector batch to OpenAI Batch API
python batch/retrieve_batch_file_results.py  # collect completed batch results
python results/read_sector_results.py        # flatten results → data/sector_summary.tsv
python output/visualize_sentiment.py         # generate charts → data/charts/
python results/export_time_series.py         # bulk TSV exports
python results/build_sector_db.py            # build SQLite db → data/sector_results.db
```

## Topic Clustering

```bash
python enrich/cluster_topics.py                    # cluster for today
python enrich/cluster_topics.py --date 2026-03-13  # cluster for a specific date
python enrich/cluster_topics.py --skip-labeling    # skip LLM labeling (dry run)

python enrich/label_topics.py             # label all unlabeled topic_ids via OpenAI Batch API
python enrich/label_topics.py --dry-run   # show counts only
```

## Embeddings

```bash
python enrich/embed_feeds.py   # cold-start build or incremental update of FAISS vectorstore
```

## RAG

```bash
python -m pipeline.hybrid_rag           # CLI hybrid RAG query (module entrypoint)
streamlit run output/chatbot_rag.py     # Streamlit chatbot
```

## Daily Briefing

```bash
python enrich/daily_briefing.py                    # morning briefing for today
python enrich/daily_briefing.py --date 2026-03-21  # briefing for a specific date
python enrich/daily_briefing.py --no-rag --save    # fast briefing, save to data/briefings/
```

## Briefing Batch

```bash
python batch/create_batch_briefings.py             # submit RAG briefing batch
python batch/create_batch_briefings.py --dry-run   # show counts, no submission
python batch/retrieve_batch_briefings.py           # collect completed batch
```

## Visualization

```bash
python output/visualize_topics.py              # six charts (top 15, last 200 days)
python output/visualize_topics.py --top 20     # top 20 topics
python output/visualize_topics.py --days 90    # last 90 days only
python output/visualize_topics.py --animate    # also generate animated GIF — slow
python output/visualize_topics.py --animate --fps 8
```

## Backfill

```bash
python enrich/backfill.py                          # cluster + brief Sep 2025 → today
python enrich/backfill.py --phase1-only            # cluster only (no API cost)
python enrich/backfill.py --phase2-only --no-rag   # briefings without RAG (instant)
```

## HuggingFace Push

```bash
python ingest/push_feeds_to_hf.py             # cold-start: push all feeds → lacetohf/feeds
python ingest/push_feeds_to_hf.py --dry-run
python ingest/push_new_feeds_to_hf.py         # daily incremental: append today's new articles

python ingest/push_analysis_to_hf.py          # cold-start: push sector/topic/entity datasets
python ingest/push_analysis_to_hf.py --dry-run
python ingest/push_new_analysis_to_hf.py      # daily incremental: append new analysis rows
```

## HF Datasets (live)

- `lacetohf/feeds` — raw articles; dedup key: `guid`
- `lacetohf/sector-analysis` — dedup key: `date+sector`
- `lacetohf/topic-trends` — dedup key: `date+topic_id`
- `lacetohf/entity-sentiment` — dedup key: `date+entity+sector`
