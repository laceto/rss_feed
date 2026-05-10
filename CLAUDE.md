# RSS Feed Pipeline — Agent Router

Hybrid R + Python financial news analysis pipeline. This file routes you to the right context.

## Identify Your Task

**CODING (Python)** — Adding a new script, extending the analysis pipeline, fixing a Python bug
→ READ: `coding-rules.md`
→ ALSO READ: `docs/architecture.md` for data flow context

**CODING (R / scraper)** — Modifying `scraper/download.R` or the feed ingestion logic
→ READ: `scraper/README.md` for the interface contract
→ READ: `coding-rules.md` scraper section

**TESTING** — Writing new tests or expanding test coverage
→ READ: `testing-rules.md`

**DEBUGGING** — A CI workflow failed, a script errored, or output looks wrong
→ READ: `debugging-rules.md`
→ ALSO READ: `docs/architecture.md` for pipeline flow

**CI/CD** — Modifying GitHub Actions workflows
→ READ: `ci-rules.md`

**DATA / QUERY** — Exploring pipeline outputs, running queries, calling query APIs
→ READ: `docs/api-reference.md`
→ READ: `docs/scripts-reference.md` for CLI commands

## Repo Layout

```
pipeline/     shared library (constants, cluster_topics, hybrid_rag, query_sector, query_entity)
scraper/      R scraper (download.R, DESCRIPTION)
batch/        OpenAI Batch API scripts (submit + retrieve)
results/      flatten + export (read_sector_results, export_time_series, build_sector_db)
ingest/       HuggingFace push scripts
enrich/       embed, cluster, label, briefing, backfill
output/       visualizations, chatbot, hybrid_rag CLI
notebooks/    user guide notebooks — start with START_HERE.ipynb
tests/        pytest suite
docs/         reference documentation
Justfile      stable named tasks — use `just <task>` instead of bare python paths
```

Root holds only: `pyproject.toml`, `CLAUDE.md`, `*.md` rule files, backward-compat shims (`hybrid_rag.py`, `cluster_topics.py`).

## Reference Docs (load only what your task requires)

| File | Contents |
|---|---|
| `docs/architecture.md` | Pipeline diagrams, data flow, incremental sentinel pattern |
| `docs/schemas.md` | SectorAnalysis, briefing JSON, SQLite schema, bulk export files |
| `docs/api-reference.md` | query_sector, query_entity, ask(), cluster_topics public API |
| `docs/scripts-reference.md` | All CLI commands with flags |
| `docs/key-files.md` | Key files table, required directories, pipeline/ package |
| `docs/vectorstore.md` | FAISS vectorstore details |
| `docs/topic-clustering.md` | cluster_topics.py parameters, exceptions, continuity invariant |

## Instructions

1. Identify your task above
2. Load the rule file for that task
3. Load only the reference docs the task actually requires
4. Do not load all docs — load what you need
5. If unsure which category fits, ask — do not guess
