# CI/CD Rules

## Workflow Chain

Workflows fire in order. Do not break the trigger chain:

```
daily-pipeline  →  collect-sector-results  →  embed-feeds  →  daily-briefing
```

- `daily-pipeline`: cron weekdays 19:00 UTC — download.R → push feeds to HF → sector batch submit → commit
- `collect-sector-results`: triggered by daily-pipeline — retrieve → flatten → charts → export TSVs → build SQLite → cluster topics → push analysis to HF → commit
- `embed-feeds`: triggered by collect-sector-results — embed new articles → update FAISS + registry → commit
- `daily-briefing`: triggered by embed-feeds + cron 0 13 * * 1-5 — daily_briefing.py --save → commit

## Required Settings (all workflows)

```yaml
env:
  FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true
```

This must be present on every workflow. Do not remove it.

## Exit Code Contract

Retrieve scripts must exit with:
- `0` = done, downstream steps can proceed
- `1` = error, fail the job
- `2` = still in progress, CI should retry (do not fail the job)

Do not change this contract. Downstream workflow steps depend on it.

## Commit Rules in CI

- Commits made by CI actions use `git commit -m "..."` directly
- Never add `Co-Authored-By:` lines to commit messages — not in CI, not locally

## What NOT to Do

- Do not add `--no-verify` to git commands in workflows
- Do not merge `daily-pipeline` and `collect-sector-results` into one workflow — the batch processing requires async separation
- Do not change retry logic without updating exit codes in the corresponding retrieve script
- Do not add secrets inline — use GitHub Actions secrets (`${{ secrets.OPENAI_API_KEY }}`)
