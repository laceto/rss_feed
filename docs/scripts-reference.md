# Scripts Reference

## Environment

```bash
venv\Scripts\activate   # Windows
```

Secrets in `.env` (gitignored): `OPENAI_API_KEY`, `HF_TOKEN`, `HUGGINGFACE_REPO`.

## R — RSS Scraper

```bash
Rscript download.R
```
Outputs `output/feeds{YYYY-MM-DD}.txt` (tab-separated). R deps in `DESCRIPTION`.

## Sector Analysis Pipeline

```bash
python create_batch_files_v2.py        # submit sector batch to OpenAI Batch API
python retrieve_batch_file_results.py  # collect completed batch results
python read_sector_results.py          # flatten results → data/sector_summary.tsv
python visualize_sentiment.py          # generate charts → data/charts/
python export_time_series.py           # bulk TSV exports
python build_sector_db.py              # build SQLite db → data/sector_results.db
```

## Topic Clustering

```bash
python cluster_topics.py                    # cluster for today
python cluster_topics.py --date 2026-03-13  # cluster for a specific date
python cluster_topics.py --skip-labeling    # skip LLM labeling (dry run / profiling)

python label_topics.py             # label all unlabeled topic_ids via OpenAI Batch API
python label_topics.py --dry-run   # show counts only
```

## Embeddings

```bash
python embed_feeds.py   # cold-start build or incremental update of FAISS vectorstore
```

## RAG

```bash
python hybrid_rag.py               # CLI hybrid RAG query
streamlit run chatbot_rag.py       # Streamlit chatbot
```

## Daily Briefing

```bash
python daily_briefing.py                    # morning briefing for today
python daily_briefing.py --date 2026-03-21  # briefing for a specific date
python daily_briefing.py --no-rag --save    # fast briefing, save to data/briefings/
```

## Briefing Batch

```bash
python create_batch_briefings.py             # submit RAG briefing batch
python create_batch_briefings.py --dry-run   # show counts, no submission
python retrieve_batch_briefings.py           # collect completed batch
```

## Visualization

```bash
python visualize_topics.py              # six charts (top 15, last 200 days)
python visualize_topics.py --top 20     # top 20 topics
python visualize_topics.py --days 90    # last 90 days only
python visualize_topics.py --animate    # also generate animated GIF — slow
python visualize_topics.py --animate --fps 8
```

## Backfill

```bash
python backfill.py                          # cluster + brief Sep 2025 → today
python backfill.py --phase1-only            # cluster only (no API cost)
python backfill.py --phase2-only --no-rag   # briefings without RAG (instant)
```

## HuggingFace Push

```bash
python push_feeds_to_hf.py             # cold-start: push all feeds → lacetohf/feeds
python push_feeds_to_hf.py --dry-run
python push_new_feeds_to_hf.py         # daily incremental: append today's new articles

python push_analysis_to_hf.py          # cold-start: push sector/topic/entity datasets
python push_analysis_to_hf.py --dry-run
python push_new_analysis_to_hf.py      # daily incremental: append new analysis rows
```

## HF Datasets (live)

- `lacetohf/feeds` — raw articles; dedup key: `guid`
- `lacetohf/sector-analysis` — dedup key: `date+sector`
- `lacetohf/topic-trends` — dedup key: `date+topic_id`
- `lacetohf/entity-sentiment` — dedup key: `date+entity+sector`
