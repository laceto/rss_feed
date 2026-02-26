# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A hybrid R + Python pipeline for financial news analysis:

1. **R (scraping)**: `download.R` fetches CNBC RSS feeds daily → `output/feeds{date}.txt`
2. **Python (sector analysis)**: LLM-powered batch pipeline extracts sector-level trading signals per day
3. **Python (charting)**: Sentiment trend charts generated from consolidated results

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
python query_sector.py                 # not a CLI script — import as a module (see below)

python trader_assistant.py         # trading signal extractor (LangChain)
python create_batch_files.py       # MA/RSI batch analysis (tickers)
streamlit run chatbot6.py          # RAG chatbot (requires pre-built vectorstore)
```

Python deps: `openai pydantic python-dotenv pandas matplotlib seaborn`. API keys in `.env` (gitignored):
```
OPENAI_API_KEY=sk-...
```

## Architecture

### Sector Analysis Pipeline (primary daily pipeline)
```
GitHub Actions — daily-pipeline (weekdays 19:00 UTC)
  → download.R
  → output/feeds{date}.txt

  → create_batch_files_v2.py
      reads:  output/feeds*.txt  (top-level only — never output/enriched/)
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
```

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
`query_sector.py` exposes two functions for downstream consumers (dashboards, agents, other scripts):

```python
from query_sector import get_snapshot, get_time_series, list_sectors

# Current read — single latest entry
snap = get_snapshot("Technology Services")
# → {last_date, latest_sentiment, sentiment_score, entities, news_category, data_age_days}

# Trend over a rolling window
ts = get_time_series("Finance", lookback_days=60, include_articles=True)
# → {mean_sentiment_score, trend_direction, trend_delta,
#    top_entities, category_breakdown, time_series, articles}

# Discover valid names
list_sectors()   # sorted list of 19 valid sector names
```

- Both functions raise `ValueError` (bad name) or `LookupError` (valid name, no data yet)
- `trend_direction`: `"improving"` / `"deteriorating"` / `"stable"` based on first-half vs second-half mean of the window (threshold ±0.20)
- `include_articles=True` joins back to `output/feeds{date}.txt` for raw article headlines

### Key Files
| File | Purpose |
|---|---|
| `constants.py` | **Single source of truth**: `SectorName` Literal, `SECTOR_TAXONOMY` list, `SENTIMENT_SCORE`, file paths |
| `query_sector.py` | `get_snapshot()` + `get_time_series()` — sector lookup for external flows |
| `create_batch_files_v2.py` | Reads raw feeds, submits daily sector batch to OpenAI |
| `retrieve_batch_file_results.py` | Collects completed batch; routes to `data/sector_results/` |
| `read_sector_results.py` | Flattens all JSON results → `data/sector_summary.tsv` |
| `visualize_sentiment.py` | Three sentiment charts from `sector_summary.tsv` |
| `utils.py` | Shared helpers: `get_file_paths`, `df_to_docs`, `excel_to_docs` |
| `trader_assistant.py` | News → trading signal LangChain chain |
| `create_batch_files.py` | MA/RSI ticker analysis via OpenAI Batch API |
| `chatbot6.py` | Streamlit RAG chatbot with LangGraph + hybrid FAISS/BM25 retrieval |
| `download.R` | RSS scraper, called by GitHub Actions |
| `old/` | Archived/experimental versions — not used in production |

### Strict Schema Note
`create_batch_files_v2.py` uses `_make_openai_strict()` to convert the Pydantic schema to OpenAI strict JSON schema format (adds `additionalProperties: false` + `required[]` recursively). This replaces the former private SDK call `openai.lib._pydantic.to_strict_json_schema`.

### Incremental Sentinel
Both batch scripts use file existence as the processed sentinel:
- `create_batch_files_v2.py` skips dates where `data/sector_results/{date}.json` already exists
- `retrieve_batch_file_results.py` clears `data/pending_sector_batch.txt` only after full success

### Required Directories
- `output/` — daily feed files written by `download.R`
- `data/sector_results/` — created at runtime by collection script
- `data/charts/` — created at runtime by visualization script
- `data/` — must exist for `create_batch_files.py` (holds `batch_tasks_tickers.jsonl`, `book.txt`)
- `vectorstore/book/` — pre-built FAISS index for `chatbot6.py` (not regenerated at runtime)

### CI/CD
| Workflow | Trigger | What it does |
|---|---|---|
| `daily-pipeline` | cron `0 19 * * 1-5` + manual | download.R → create_batch_files_v2.py → commit |
| `collect-sector-results` | on `daily-pipeline` completion + manual | retrieve → flatten → charts → commit |
