# GitHub Actions Workflows

Four workflows form a chained daily pipeline. Each triggers the next on success.

## Pipeline Overview

```
19:00 UTC (weekdays)
    │
    ▼
[daily-pipeline]           scrape + push feeds + submit sector batch
    │ on: completed (success)
    ▼
[collect-sector-results]   poll batch + flatten + charts + cluster + push HF
    │ on: completed (success)
    ▼
[embed-feeds]              embed new articles → FAISS vectorstore
    │ on: completed (success)
    ▼
[daily-briefing]           generate morning briefing JSON
    │ also: cron 13:00 UTC weekdays
```

All four workflows also support `workflow_dispatch` for manual runs.

---

## 1. `daily-pipeline`

**File:** `.github/workflows/main.yml`
**Trigger:** `cron 0 19 * * 1-5` (weekdays 19:00 UTC) + manual

### Steps

| Step | Script | Output |
|---|---|---|
| Download feeds | `Rscript download.R` | `output/feeds{date}.txt` |
| Push feeds to HF | `push_new_feeds_to_hf.py` | `lacetohf/feeds` (appends new rows) |
| Verify HF updated | inline Python | prints row count + latest date |
| Submit sector batch | `create_batch_files_v2.py` | `data/pending_sector_batch.txt`, `data/batch_tasks_sector.jsonl` |
| Commit | git | commits all outputs to `main` |

### Secrets required
- `OPENAI_API_KEY`
- `HF_TOKEN`
- `HUGGINGFACE_REPO`

### Notes
- `create_batch_files_v2.py` reads feeds from `lacetohf/feeds` (not local files)
- Skips dates where `data/sector_results/{date}.json` already exists (idempotent)
- The HF verify step prints row count + latest pubDate for observability

---

## 2. `collect-sector-results`

**File:** `.github/workflows/collect_sectors.yml`
**Trigger:** on `daily-pipeline` completion (success only) + manual

### Steps

| Step | Script | Output |
|---|---|---|
| Check pending | checks `data/pending_sector_batch.txt` | sets `pending` output |
| Collect batch (retry loop) | `retrieve_batch_file_results.py` | `data/sector_results/{date}.json` |
| Build sector summary | `read_sector_results.py` | `data/sector_summary.tsv` |
| Generate charts | `visualize_sentiment.py` | `data/charts/*.png` |
| Export time series | `export_time_series.py` | `data/sector_sentiment_pivot.tsv`, `data/entity_sentiment_ts.tsv` |
| Build SQLite db | `build_sector_db.py` | `data/sector_results.db` |
| Cluster topics | `cluster_topics.py` | `data/topic_trends.tsv`, `data/topic_clusters/{date}.json` |
| Push analysis to HF | `push_new_analysis_to_hf.py` | `lacetohf/sector-analysis`, `lacetohf/topic-trends`, `lacetohf/entity-sentiment` |
| Commit | git | commits all outputs to `main` |

### Secrets required
- `OPENAI_API_KEY`
- `HF_TOKEN`
- `HUGGINGFACE_REPO`

### Retry logic
`retrieve_batch_file_results.py` uses exit codes:
- `0` = batch complete → proceed
- `1` = fatal error → abort workflow
- `2` = batch still in progress → sleep 5 min, retry (up to 12 attempts = 1 hour)

### Notes
- All steps after "Check pending" are gated on `steps.check.outputs.pending == 'true'`
- `cluster_topics.py` exit code 2 = `ClusteringAborted` — treated as warning, not failure
- `push_new_analysis_to_hf.py` uses composite-key dedup — re-running same date is safe

---

## 3. `embed-feeds`

**File:** `.github/workflows/embed_feeds.yml`
**Trigger:** on `collect-sector-results` completion (success only) + manual

### Steps

| Step | Script | Output |
|---|---|---|
| Embed new articles | `embed_feeds.py` | `data/vectorstore/feeds/index.faiss`, `data/vectorstore/feeds/index.pkl`, `data/vectorstore/feeds_registry.tsv` |
| Commit | git | commits vectorstore + registry to `main` |

### Secrets required
- `OPENAI_API_KEY`

### Notes
- `embed_feeds.py` is incremental — only articles not in `feeds_registry.tsv` (by `guid`) are embedded
- Uses OpenAI Batch API internally via `kitai.batch` (`poll_until_complete` blocks until done)
- Typical volume: 50–200 articles/day → 1–5 minutes
- Commit uses `git pull --rebase` before push to handle concurrent writes from other workflows

---

## 4. `daily-briefing`

**File:** `.github/workflows/daily_briefing.yml`
**Trigger:** on `embed-feeds` completion (success only) + `cron 0 13 * * 1-5` (13:00 UTC weekdays) + manual

### Manual inputs
| Input | Description | Default |
|---|---|---|
| `date` | Target date `YYYY-MM-DD` | today |
| `top` | Number of top spikes | `5` |

### Steps

| Step | Script | Output |
|---|---|---|
| Check topic trends | checks `data/topic_trends.tsv` exists | sets `exists` output |
| Build daily briefing | `daily_briefing.py --save` | `data/briefings/{date}.json` |
| Commit briefing | git | commits briefing JSON to `main` |

### Secrets required
- `OPENAI_API_KEY`

### Notes
- `daily_briefing.py` first checks for a pre-computed `data/briefings/{date}.json` — if found, prints it without any API calls
- `--save` forces a fresh live run (spike detection → RAG → sector cross-check)
- Commit uses `git pull --rebase` before push
- The 13:00 UTC schedule ensures the briefing is available by 9 AM ET even if the overnight pipeline ran late

---

## Secrets Reference

| Secret | Used by | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | all workflows | sector batch, embedding, RAG, clustering labels |
| `HF_TOKEN` | `daily-pipeline`, `collect-sector-results` | push to Hugging Face Datasets |
| `HUGGINGFACE_REPO` | `daily-pipeline`, `collect-sector-results` | target HF repo (`lacetohf/feeds`) |
| `GITHUB_TOKEN` | all workflows (auto) | git push to `main` |

---

## Manual Trigger Guide

```bash
# Trigger any workflow manually via GitHub CLI
gh workflow run main.yml                          # daily pipeline
gh workflow run collect_sectors.yml               # collect + analyze
gh workflow run embed_feeds.yml                   # embed articles
gh workflow run daily_briefing.yml                # today's briefing
gh workflow run daily_briefing.yml -f date=2026-03-25 -f top=5  # specific date

# Watch a run live
gh run list --workflow=main.yml --limit=3
gh run watch <run-id>
```

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `collect-sector-results` exits early with "No pending batch" | `daily-pipeline` had nothing new to process | Normal — no action needed |
| `collect-sector-results` retries 12 times then stops | OpenAI batch took > 1 hour | Re-run manually once batch completes |
| `cluster_topics.py` exits 2 | Degenerate clustering (too few articles or too much noise) | Check feed volume; re-run next day |
| `embed-feeds` push fails | Race condition with another workflow commit | `git pull --rebase` handles this automatically |
| HF push fails with auth error | `HF_TOKEN` expired or wrong | Rotate token in GitHub secrets |
