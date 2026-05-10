# Justfile — stable named tasks for the rss_feed pipeline
# Install: https://github.com/casey/just
# Usage: just <task>
# All tasks run from the repo root with PYTHONPATH=. so pipeline/ imports resolve.

export PYTHONPATH := "."

# --- daily-pipeline ---

scrape:
    Rscript scraper/download.R

push-feeds:
    python ingest/push_new_feeds_to_hf.py

push-feeds-cold:
    python ingest/push_feeds_to_hf.py

batch-submit:
    python batch/create_batch_files_v2.py

# --- collect-sector-results ---

batch-collect:
    python batch/retrieve_batch_file_results.py

flatten:
    python results/read_sector_results.py

charts:
    python output/visualize_sentiment.py

export-ts:
    python results/export_time_series.py

build-db:
    python results/build_sector_db.py

cluster:
    python enrich/cluster_topics.py

topic-charts:
    python output/visualize_topics.py

push-analysis:
    python ingest/push_new_analysis_to_hf.py

push-analysis-cold:
    python ingest/push_analysis_to_hf.py

# --- embed-feeds ---

embed:
    python enrich/embed_feeds.py

# --- daily-briefing ---

briefing *args:
    python enrich/daily_briefing.py {{args}}

# --- batch briefings ---

briefing-submit:
    python batch/create_batch_briefings.py

briefing-collect:
    python batch/retrieve_batch_briefings.py

# --- maintenance ---

label-topics *args:
    python enrich/label_topics.py {{args}}

backfill *args:
    python enrich/backfill.py {{args}}

# --- query / explore ---

chatbot:
    streamlit run output/chatbot_rag.py

# --- full collect sequence (local) ---

collect: batch-collect flatten charts export-ts build-db cluster topic-charts push-analysis
