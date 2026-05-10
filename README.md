# RSS Feed — Financial News Analysis Pipeline

A hybrid R + Python pipeline that scrapes CNBC RSS feeds daily, extracts sector-level trading signals via LLM, clusters emerging narratives, and generates a morning briefing — all automated through GitHub Actions.

## What It Does

| Step | Tool | Output |
|---|---|---|
| Scrape feeds | R (`scraper/download.R`) | `output/feeds{date}.txt` |
| Push to Hugging Face | Python | `lacetohf/feeds` (public dataset) |
| Sector sentiment extraction | OpenAI Batch API | `data/sector_results/{date}.json` |
| Embed articles | OpenAI + FAISS | `data/vectorstore/feeds/` |
| Topic clustering | HDBSCAN | `data/topic_trends.tsv` |
| Daily briefing | RAG + sector cross-check | `data/briefings/{date}.json` |
| Push analysis to HF | Python | 3 public HF datasets |

## Hugging Face Datasets (public)

| Dataset | Description |
|---|---|
| [`lacetohf/feeds`](https://huggingface.co/datasets/lacetohf/feeds) | Raw CNBC feed articles (~150/day) |
| [`lacetohf/sector-analysis`](https://huggingface.co/datasets/lacetohf/sector-analysis) | Sector sentiment per date |
| [`lacetohf/topic-trends`](https://huggingface.co/datasets/lacetohf/topic-trends) | Topic cluster counts per date |
| [`lacetohf/entity-sentiment`](https://huggingface.co/datasets/lacetohf/entity-sentiment) | Entity mentions per date × sector |

## Quick Start

```bash
git clone https://github.com/laceto/rss_feed
cd rss_feed
python -m venv venv
venv\Scripts\activate        # Windows

pip install openai pydantic python-dotenv pandas matplotlib seaborn \
            streamlit langchain langchain-community langchain-openai \
            rank-bm25 hdbscan scikit-learn datasets huggingface_hub
pip install git+https://github.com/laceto/kitai.git
pip install -e .   # installs pipeline/ package so imports resolve
```

Create `.env`:
```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
HUGGINGFACE_REPO=lacetohf/feeds
```

## Repo Layout

```
pipeline/     shared library (constants, cluster_topics, hybrid_rag, query_sector, query_entity)
scraper/      R scraper (download.R, DESCRIPTION)
batch/        OpenAI Batch API scripts (submit + retrieve)
results/      flatten + export (read_sector_results, export_time_series, build_sector_db)
ingest/       HuggingFace push scripts
enrich/       embed, cluster, label, briefing, backfill
output/       visualizations, chatbot, hybrid_rag CLI
tests/        pytest suite
Justfile      named tasks — use `just <task>` instead of raw python paths
```

## Key Scripts

### Daily pipeline (automated by CI, or run with `just`)

```bash
just scrape                            # Rscript scraper/download.R
just push-feeds                        # push to HF
just batch-submit                      # submit sector batch
just batch-collect                     # collect results
just flatten && just export-ts         # sector_summary.tsv + TSV exports
just build-db                          # build SQLite db
just cluster                           # cluster today's topics
just push-analysis                     # push analysis to HF
just embed                             # embed new articles
just briefing --save                   # generate briefing
```

Or run directly (set `PYTHONPATH=.` first):

```bash
Rscript scraper/download.R
python ingest/push_new_feeds_to_hf.py
python batch/create_batch_files_v2.py
python batch/retrieve_batch_file_results.py
python results/read_sector_results.py
python results/export_time_series.py
python results/build_sector_db.py
python enrich/cluster_topics.py
python ingest/push_new_analysis_to_hf.py
python enrich/embed_feeds.py
python enrich/daily_briefing.py --save
```

### Querying data

```python
from pipeline.query_sector import get_snapshot, get_time_series
from pipeline.query_entity import get_entity_snapshot, get_entity_time_series

# Sector snapshot
snap = get_snapshot("Technology Services")

# Entity trend
ts = get_entity_time_series("Nvidia", lookback_days=60)
```

### RAG chatbot

```bash
streamlit run output/chatbot_rag.py
```

### Morning briefing

```bash
python enrich/daily_briefing.py                    # today
python enrich/daily_briefing.py --date 2026-03-25  # specific date
```

### Visualizations

```bash
python output/visualize_sentiment.py
python output/visualize_topics.py              # six charts
python output/visualize_topics.py --animate    # + animated GIF
```

## Data Files

| File | Description |
|---|---|
| `data/sector_summary.tsv` | Flat sector sentiment — one row per date × sector |
| `data/sector_sentiment_pivot.tsv` | Wide format: date × 19 sectors (last 90 days) |
| `data/entity_sentiment_ts.tsv` | Entity mentions: date × entity × sector (last 90 days) |
| `data/sector_results.db` | SQLite: lossless normalized mirror of all JSON results |
| `data/topic_trends.tsv` | Topic cluster counts — append-only, one row per topic per day |
| `data/briefings/{date}.json` | Daily briefing: spiking topics + RAG narrative + sector signal |
| `data/vectorstore/feeds/` | FAISS index of all embedded feed articles |

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for pipeline diagrams and data flow.
See [`docs/api-reference.md`](docs/api-reference.md) for the full Python API.
See [`docs/workflows.md`](docs/workflows.md) for GitHub Actions CI/CD details.

## Dependencies

**R** — deps in `scraper/DESCRIPTION` (`rvest`, `xml2`, `XML`, `dplyr`, `purrr`).

**Python**:
```
openai pydantic python-dotenv pandas matplotlib seaborn streamlit
langchain langchain-community langchain-openai rank-bm25
hdbscan scikit-learn datasets huggingface_hub
kitai (private: github.com/laceto/kitai)
```
