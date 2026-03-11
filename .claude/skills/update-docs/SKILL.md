---
name: update-docs
description: 'Updates all project documentation to reflect the current state of the codebase. Use when asked to "update docs", "update documentation", "sync docs", "refresh the README", "update the user guide", or "update CLAUDE.md". Reads every source file public API and docstring, diffs against existing docs, and rewrites README.md, CLAUDE.md, user-guide notebooks (.ipynb), and any other doc files — preserving each file'\''s existing format and style.'
---

# Update Docs

Synchronises all project documentation with the current codebase.  Run this
skill whenever the source code has changed and the docs may be stale.

## When to Use This Skill

- User says "update docs", "update documentation", "sync the docs"
- User says "update README", "update CLAUDE.md", "update the user guide"
- After a significant feature addition or refactor
- Before a release or code review

## What This Skill Updates

| File / pattern | Action |
|---|---|
| `README.md` | Rewrite to reflect current public API, module layout, usage examples |
| `CLAUDE.md` | Update architecture notes, import paths, invariants, debugging guidance |
| `*_guide.ipynb` / `*guide*.ipynb` | Update module-map cell, API examples, and any stale function references |
| Other `*.md` docs found in the repo root or `docs/` | Update to match current behaviour |

## Step-by-Step Workflow

### 1. Discover all documentation targets

Search the repo for files to update:

- `README.md` (repo root)
- `CLAUDE.md` (repo root or `.claude/`)
- `**/*guide*.ipynb` — user-guide notebooks
- `**/*findings*.md` — code-review finding plans (update only if stale)
- Any other `*.md` outside `venv/` and `.claude/skills/`

Read every target file in full before making any changes.

### 2. Read the current source code

For each module in the package (e.g. `kitai/`):

- Read the file completely.
- Extract the **public API**: every non-underscore function/class with its
  signature, docstring, args, return type, and raised exceptions.
- Note any module-level constants or invariants documented in the module
  docstring.

Do **not** guess — only document what is actually in the source.

### 3. Identify what has changed

Compare the extracted API against what each doc file currently describes:

- New functions / classes not yet documented → add them.
- Removed or renamed symbols still referenced → remove / update them.
- Changed signatures, return types, or raised exceptions → update them.
- Changed module-level invariants or import paths → propagate to all docs.

### 4. Update each documentation file

Apply the minimum diff needed to make each file accurate.  Preserve:

- The file's existing **section structure and headings**.
- The file's **tone and style** (terse vs. verbose, prose vs. table).
- Code block language tags and indentation conventions.
- Notebook cell types (markdown vs. code) and cell order.

Rules:

- Do **not** delete sections that are still valid.
- Do **not** reformat the whole file — surgical edits only.
- Do **not** add new sections that duplicate existing ones.
- For notebooks: update code cells that call renamed/changed functions;
  update markdown cells that describe the old API; keep all other cells intact.

### 5. Verify consistency across files

After all edits:

- Cross-check that the same function is described identically in README,
  CLAUDE.md, and any notebook that covers it.
- Ensure import paths in code examples match the actual package structure
  (e.g. `from kitai.batch import submit_batch_job`, not a bare import).
- Confirm every raised exception listed in a docstring is also noted in the
  relevant doc section.

### 6. Report what changed

Summarise the updates in a short bullet list:

```
Updated:
  - README.md  — added kitai.batch section; updated kitai.index invariants
  - CLAUDE.md  — added batch.py to package layout; updated import paths
  - retriever_guide.ipynb — fixed create_hybrid_retriever call signature
```

## Invariants to Preserve Across All Docs

- Import paths must use the installed package name, not bare module names.
- Every public function description must include: purpose, args, return type,
  raised exceptions.
- Logging guidance: all modules use `logging.getLogger(__name__)`; no module
  configures handlers — callers do.

## Common Mistakes to Avoid

| Mistake | Correct approach |
|---|---|
| Describing a function from memory | Always read the source file first |
| Full file rewrite when only one section changed | Surgical edits with Edit tool |
| Updating a notebook's output cells | Leave output cells as-is |
| Using bare imports in examples (`from retriever import …`) | Use package-qualified imports (`from kitai.retriever import …`) |
| Documenting private helpers (`_build_query_chain`) | Only document public symbols |
