# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A hybrid R + Python pipeline for financial news analysis:

1. **R (scraping)**: `download.R` fetches CNBC RSS feeds daily → `output/feeds{date}.txt` (tab-separated)
2. **Python (analysis)**: LLM-powered scripts consume those feed files to extract trading signals and structured financial analysis

## Running the Scripts

### R — RSS Scraper
```bash
Rscript download.R
```
Outputs `output/feeds{YYYY-MM-DD}.txt`. R package deps are declared in `DESCRIPTION` (`rvest`, `xml2`, `XML`, `dplyr`, `purrr`).

### Python — Virtual Environment
```bash
# Activate venv (Windows)
venv\Scripts\activate

# Run trading signal extractor (reads all output/*.txt feed files)
python trader_assistant.py

# Run batch job creator (reads output_dict_list.txt of ticker indicator dicts)
python create_batch_files.py

# Run RAG chatbot (requires pre-built vectorstore at ./vectorstore/book)
streamlit run chatbot6.py
```

Python deps are in `requirements.txt`. API keys go in `.env` (already gitignored):
```
OPENAI_API_KEY=sk-...
```

## Architecture

### Data Flow
```
GitHub Actions (weekdays 19:00 UTC)
  → download.R
  → output/feeds{date}.txt   (tab-separated: title, description, link, guid, type, id, sponsored, pubDate)

trader_assistant.py
  → reads all output/*.txt via utils.get_file_paths()
  → concatenates into pandas DataFrame
  → converts rows to LangChain Documents via utils.df_to_docs()
  → LangChain chain: ChatPromptTemplate | ChatOpenAI | OutputFixingParser
  → outputs structured JSON: {company_or_sector, trading_decision, signal, motivation, news_topic}

create_batch_files.py
  → reads output_dict_list.txt (list of ticker indicator dicts: ema_st/mt/lt, rsi, spreads)
  → builds JSONL batch file → data/batch_tasks_tickers.jsonl
  → uploads to OpenAI Files API, creates batch job

old/retrieve_batch_results.py
  → reads data/batch_job_results_tickers.jsonl
  → parses OpenAI batch results

chatbot6.py (Streamlit RAG)
  → loads FAISS vectorstore from ./vectorstore/book (pre-built, FakeEmbeddings for loading)
  → loads ./data/book.txt and chunks it for BM25
  → hybrid retriever: EnsembleRetriever(BM25 + semantic FAISS, weights 0.5/0.5)
  → LangGraph: StateGraph with retrieve → generate nodes + MemorySaver checkpointer
  → streams responses via ChatOpenAI(model="gpt-4", streaming=True)
```

### Key Files
| File | Purpose |
|---|---|
| `utils.py` | Shared helpers: `get_file_paths`, `df_to_docs`, `excel_to_docs`, `add_metadata_to_docs` |
| `enrich_feeds.py` | Batch-enrich feed files → `output/enriched/feeds{date}.txt` (see below) |
| `trader_assistant.py` | News → trading signal LangChain chain |
| `create_batch_files.py` | MA/RSI analysis via OpenAI Batch API |
| `chatbot6.py` | Streamlit RAG chatbot with LangGraph + hybrid retrieval |
| `download.R` | RSS scraper, called by GitHub Actions |
| `old/` | Archived/experimental versions — not used in production |

### Pydantic Models in `create_batch_files.py`
`TraderAnalysis` is a deeply nested Pydantic model representing the full structured MA/RSI report: `Overview`, `MovingAverageStructureAnalysis`, `RSIAndMomentumAssessment`, `TrendStrengthRSIMatrix`, `OverallMarketContext`, `RiskOpportunityAssessment`, `TacticalTradeConsiderations`, `SummarySignal`.

### Required Directories and Artifacts
- `output/` — created by `download.R` if absent; holds daily feed files
- `data/` — must exist before running `create_batch_files.py` (holds JSONL and `book.txt`)
- `vectorstore/book/` — pre-built FAISS index required by `chatbot6.py` (not regenerated at runtime)
- `output_dict_list.txt` — JSON list of ticker indicator dicts, input to `create_batch_files.py`

### Feed Enrichment Pipeline (`enrich_feeds.py`)

Uses direct async LangChain calls (`asyncio.gather` with `MAX_CONCURRENCY=20`) to enrich feed rows in one blocking pass.

```
output/feeds{date}.txt      (raw, immutable)
        ↓  enrich_feeds.py
output/enriched/feeds{date}.txt  (guid + entities|sector|sentiment|news_category|extraction_status)
```

**Join raw + enriched:** `raw_df.merge(enriched_df, on="guid", how="left")`

**Deduplication:** rows with duplicate normalized descriptions (across all new files in a single run) are dropped before calling the LLM. Deduped rows appear as NaN in the merged dataframe.

**Incremental sentinel:** enriched file presence = processed. All-failed dates are NOT written and are retried on the next run.

**`FeedEnrichment` Pydantic model** in `enrich_feeds.py` is the single source of truth for the enriched schema. Add/remove fields there first; writing and joining code follows from it.

```bash
python enrich_feeds.py   # process all unprocessed dates
```

### CI/CD
`.github/workflows/main.yml` runs `download.R` on weekdays at 19:00 UTC, commits the resulting feed file back to `main`. The second workflow (`call_openai.yaml`) is manual-dispatch only.
