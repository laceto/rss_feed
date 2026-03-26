# RSS Feed — Financial News Analysis Pipeline

A hybrid R + Python pipeline that scrapes CNBC RSS feeds daily, extracts sector-level trading signals via LLM, clusters emerging narratives, and generates a morning briefing — all automated through GitHub Actions.

## What It Does

| Step | Tool | Output |
|---|---|---|
| Scrape feeds | R (`download.R`) | `output/feeds{date}.txt` |
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
# Clone and set up Python environment
git clone https://github.com/laceto/rss_feed
cd rss_feed
python -m venv venv
venv\Scripts\activate        # Windows

pip install openai pydantic python-dotenv pandas matplotlib seaborn \
            streamlit langchain langchain-community langchain-openai \
            rank-bm25 hdbscan scikit-learn datasets huggingface_hub
pip install git+https://github.com/laceto/kitai.git
```

Create `.env`:
```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
HUGGINGFACE_REPO=lacetohf/feeds
```

## Key Scripts

### Daily pipeline (automated by CI)
```bash
Rscript download.R                     # scrape feeds
python push_new_feeds_to_hf.py         # push to HF
python create_batch_files_v2.py        # submit sector batch
python retrieve_batch_file_results.py  # collect results
python read_sector_results.py          # flatten → sector_summary.tsv
python export_time_series.py           # export TSVs
python build_sector_db.py              # build SQLite db
python cluster_topics.py               # cluster today's topics
python push_new_analysis_to_hf.py      # push analysis to HF
python embed_feeds.py                  # embed new articles
python daily_briefing.py --save        # generate briefing
```

### Querying data
```python
from query_sector import get_snapshot, get_time_series
from query_entity import get_entity_snapshot, get_entity_time_series

# Sector snapshot
snap = get_snapshot("Technology Services")

# Entity trend
ts = get_entity_time_series("Nvidia", lookback_days=60)
```

### RAG chatbot
```bash
streamlit run chatbot_rag.py
```

### Morning briefing
```bash
python daily_briefing.py                    # today
python daily_briefing.py --date 2026-03-25  # specific date
```

### Visualizations
```bash
python visualize_sentiment.py   # sector sentiment charts
python visualize_topics.py      # topic heatmap + frequency chart
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

See [`CLAUDE.md`](CLAUDE.md) for the full architecture, pipeline diagrams, and API reference.
See [`docs/workflows.md`](docs/workflows.md) for the GitHub Actions CI/CD details.

## R Dependencies
Listed in `DESCRIPTION` (`rvest`, `xml2`, `XML`, `dplyr`, `purrr`).

## Python Dependencies
```
openai pydantic python-dotenv pandas matplotlib seaborn streamlit
langchain langchain-community langchain-openai rank-bm25
hdbscan scikit-learn datasets huggingface_hub
kitai (private: github.com/laceto/kitai)
```
