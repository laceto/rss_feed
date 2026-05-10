# Coding Rules

## Root-Only Rule

Root (`/`) holds only config + package files. Do NOT add new `.py` scripts here.
- **Config**: `pyproject.toml`, `CLAUDE.md`, `*.md` rule files, `.env`
- **Package**: `pipeline/` (shared library), `scraper/` (R), backward-compat shims (`hybrid_rag.py`, `cluster_topics.py`)
- **New CLI scripts** → pick the right stage dir: `batch/`, `results/`, `ingest/`, `enrich/`, or `output/`
- **New shared modules** (imported by multiple scripts) → `pipeline/`; use relative imports inside (`from .constants import X`)
- Run scripts via `just <task>` or `PYTHONPATH=. python <stage>/script.py` — never bare `python script.py` from root

## Single Source of Truth

- All paths, taxonomy values, and tunable constants live in `constants.py`. Never hardcode them.
- `SectorName` Literal is the only valid taxonomy — 19 values, import from `constants.py`.
- Sector/entity trend direction threshold ±0.20 — defined once; used in both `query_sector.py` and `query_entity.py`.

## File I/O Invariants

- **Atomic writes**: use `path.with_suffix('.tmp')` + `os.replace()` for any file that might be read by CI mid-write.
- **Append-only files** (e.g., `topic_trends.tsv`, `feeds_registry.tsv`): never overwrite; detect and reject duplicate rows.
- **File existence sentinel**: batch scripts skip already-processed dates by checking if the output file exists. Do not add other sentinel mechanisms.

## Batch Script Pattern

All batch scripts follow the submit/retrieve split:
1. **Submit script**: builds JSONL tasks, submits to OpenAI Batch API, writes batch ID to `data/pending_*.txt`, writes metadata sidecar. Skips dates with existing output files.
2. **Retrieve script**: reads pending ID, polls status, exits with 0 (done), 1 (error), or 2 (in progress). On success: writes output files, clears pending sentinels.

Never merge submit + retrieve into one script. CI re-runs the retrieve script until exit 0.

## New Scripts

- Must be idempotent: re-running the same date is always safe.
- Print strings must use ASCII arrows (`->`) not Unicode (`→`) — Windows cp1252 console.
- Use `argparse` for CLI flags; follow the `--date`, `--dry-run`, `--save` patterns already established.
- Never add module-level side effects; all side effects belong inside `if __name__ == "__main__"` or explicit function calls.

## Module Boundaries

- `constants.py` is a leaf import — no imports from pipeline scripts.
- `query_sector.py` and `query_entity.py` are pure read-only modules; they must not write files.
- The `pipeline/` package uses explicit parameters — no module-level resource loading.
- `cluster_topics` is NOT in the `pipeline/` package; import it from the top-level module.

## R Scraper (`scraper/`)

- R files live in `scraper/`: `download.R` and `DESCRIPTION`
- Run from repo root: `Rscript scraper/download.R` — CWD-relative paths in the script (`output/`) resolve correctly
- R package deps are declared in `scraper/DESCRIPTION`; CI installs them with `working-directory: scraper`
- The handoff to Python is via `lacetohf/feeds` (HuggingFace Dataset) — `push_new_feeds_to_hf.py` at repo root handles the push
- Do not add Python imports to R scripts; do not add R calls to Python scripts

## TDD Requirement

Write the failing test first. Do not write implementation code without a failing test that demands it.
- Tests live in `tests/`
- Run with `pytest` from project root
- See `testing-rules.md` for the full TDD cycle

## When Done

Stop. Do not write tests here — go to `testing-rules.md` for that.
