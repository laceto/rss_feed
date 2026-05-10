# scraper/

R-based CNBC RSS scraper. Runs daily via GitHub Actions.

## What it does

1. Fetches CNBC RSS feeds (5 endpoints)
2. Writes `output/feeds{YYYY-MM-DD}.txt` (tab-separated, repo root)
3. `push_new_feeds_to_hf.py` (repo root) then pushes new articles to HuggingFace

## Interface contract

**Output**: `output/feeds{date}.txt` — tab-separated, columns: `title, description, link, guid, type, id, sponsored, pubDate`

**Handoff**: `lacetohf/feeds` (HuggingFace Dataset) is the authoritative source for downstream analysis. The analysis pipeline reads from HF, not from local feed files.

## Dependencies

R package deps declared in `scraper/DESCRIPTION`. Installed in CI via:
```yaml
- uses: r-lib/actions/setup-r-dependencies@v2
  with:
    working-directory: scraper
```

## Running locally

```bash
Rscript scraper/download.R   # from repo root
```
