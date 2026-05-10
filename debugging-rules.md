# Debugging Rules

## Pipeline Failures (CI)

### First: Check Exit Codes

Scripts use structured exit codes — do not treat all non-zero exits as errors:
- Exit 0 = success
- Exit 1 = error (something is actually broken)
- Exit 2 = in progress / degenerate run (CI retries; NOT a failure)

### Second: Check Sentinel Files

Before investigating code, check which sentinel files exist:
- `data/pending_sector_batch.txt` — sector batch in flight; may still be processing
- `data/pending_briefings_batch.txt` — briefing batch in flight
- `data/pending_briefings_meta.json` — briefing metadata sidecar

If a sentinel file exists but the retrieve script exited 1, the batch errored on OpenAI's side. Delete the sentinel to reset and re-submit.

### Third: Check Data Files

- `data/sector_results/{date}.json` — missing means that date was never collected
- `data/sector_summary.tsv` — missing rows = json file missing or malformed
- `data/topic_trends.tsv` — `DuplicateDateError` means the script ran twice for the same date; delete the duplicate rows and re-run
- `data/briefings/{date}.json` — missing means briefing batch not yet collected or spiked topics < threshold

## Test Failures

1. Read the failing test name — it describes the contract
2. Do NOT change the test
3. Trace through source code to find where the contract is violated
4. Fix the minimum code to satisfy the test
5. Run `pytest tests/ -v` again
6. If all pass → STOP (do not refactor)

## Vectorstore Issues

If `embed_feeds.py` or `hybrid_rag.py` errors on FAISS load:
- Check `data/vectorstore/feeds/index.faiss` and `index.pkl` both exist
- Check `data/vectorstore/feeds_registry.tsv` exists and has header `id,date,title,link,guid`
- Cold-start rebuild: `python embed_feeds.py` (takes ~25 min for 7000+ articles)

## Clustering Degenerate Runs (exit 2)

`ClusteringAborted` fires when:
- `noise_ratio > 0.90` — too few articles in the rolling window (likely a holiday gap)
- `n_clusters < 3` — HDBSCAN collapsed; window may be too small

Fix: do not change clustering parameters. Check whether the rolling window has enough articles (need ~1700; below ~30 days HDBSCAN collapses).

## When Done

- Document the root cause in the commit message
- Do not "fix" by deleting files unless the file is genuinely corrupt
