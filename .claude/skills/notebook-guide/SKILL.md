---
name: notebook-guide
description: 'Generates a user guide Jupyter notebook (.ipynb) for a Python module or file. Use when asked to "create a user guide notebook", "make a guide notebook", "document this module as a notebook", "write a notebook guide for @file", or "create a notebook for [module]". Analyzes public API, docstrings, and existing guide notebooks in the repo, then produces a well-structured .ipynb with a module map, runnable examples, API-key-gated cells, shared sample data, error handling demos, and an end-to-end pattern section.'
---

# Notebook Guide Skill

Produces a comprehensive, runnable Jupyter notebook that serves as a user guide for a Python module. Follows the conventions of existing guide notebooks in the repo and builds the notebook cell-by-cell using `NotebookEdit` to avoid JSON encoding issues.

## When to Use This Skill

- User says "create a user guide notebook for @file" or "guide notebook for [module]"
- User says "make a notebook guide", "document this module as a notebook"
- User asks to "write a notebook for [kitai.retriever / kitai.index / any Python file]"
- User wants runnable examples and documentation in .ipynb format

## Prerequisites

- Target Python file exists and is readable
- Project contains at least one existing `*_guide.ipynb` for style reference

---

## Step-by-Step Workflow

### Step 1 — Read the target file and existing guides

1. Read the target Python file in full.
2. Glob for `*_guide.ipynb` files and read **one** existing guide to capture style conventions (cell layout, section naming, guard patterns, sample data shape).
3. Check `MEMORY.md` for any project-specific import paths or known constraints (e.g. `langchain_classic` vs `langchain`).

### Step 2 — Plan the notebook structure

Before writing any cells, identify:

| Item | Source |
|---|---|
| Public API | All `def` names not prefixed with `_` |
| Return types | Function signatures and docstrings |
| Raises | Docstring `Raises:` sections — these become the error reference section |
| API-key dependency | Any function that calls an LLM or external service |
| Natural groupings | Functions that share data flow (e.g. build → search → reorder) |

Determine the section list. Standard sections:

1. Setup
2. Sample data / corpus (shared across all sections)
3. One section per public function or logical group
4. End-to-end pattern (combines 2+ functions in a realistic pipeline)
5. Error handling reference

### Step 3 — Scaffold the notebook file

**Always** create the initial file with a Python script (not the `Write` tool directly) to guarantee valid JSON:

```python
import json
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": [{
        "cell_type": "markdown", "id": "md-placeholder",
        "metadata": {}, "source": ["placeholder"]
    }]
}
with open("MODULE_guide.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

Then populate every cell using `NotebookEdit` — **never** embed multi-line Python code inside a `Write` JSON string, as unescaped quotes and backslashes will corrupt the file.

### Step 4 — Write the title cell (first NotebookEdit)

Replace the placeholder cell. Use this template:

````markdown
# `kitai.MODULE` — User Guide

One-sentence description of what this module does.

```
kitai/MODULE.py
│
├── Group A
│   ├── function_a()     signature summary
│   └── function_b()     signature summary
│
└── Group B
    └── function_c()     signature summary
```

| Function | Needs X | Needs LLM | Best for |
|---|---|---|---|
| `function_a` | yes | no | ... |

**Prerequisites:**
- Sections N–M run with **no API key**.
- Section X requires `OPENAI_API_KEY`.

## Sections
1. [Setup](#1-setup)
2. [Sample data](#2-sample-data)
...
````

### Step 5 — Write remaining cells

Add cells in this order using `NotebookEdit` with `edit_mode="insert"` and `cell_id` set to the last inserted cell:

#### Setup cell (code)
```python
import logging
import os
# stdlib imports first, then third-party, then kitai imports
from kitai.MODULE import fn_a, fn_b, fn_c

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    force=True,
)
```

#### Sample data cell (code)
- Define a small `DOCS` / `TEXTS` / `SAMPLES` list used throughout every section.
- For LangChain modules: use `Document` objects with structured `metadata` fields that showcase the module's filtering/retrieval features.
- For vector stores: generate synthetic embeddings with `np.random.default_rng(seed=0)` and use `FakeEmbeddings` so no API key is needed.

#### Per-function sections
Each function gets:
1. A **markdown cell** explaining:
   - What the function does
   - When to choose it over alternatives
   - Key parameters (table if >3 params)
   - Any invariants or gotchas
2. A **code cell** showing the happy path
3. (Optional) extra markdown + code cells for variants, edge cases, or comparisons

#### API-key-gated cells
For any function requiring an LLM or external service, wrap the entire cell body:

```python
if not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY not set — skipping.")
else:
    # ... demo code ...
```

If a second prerequisite exists (e.g. `chromadb`), use a combined flag:

```python
_has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
_has_chroma  = importlib.util.find_spec("chromadb") is not None

if not (_has_api_key and _has_chroma):
    print("Skipping — requires OPENAI_API_KEY and chromadb.")
else:
    ...
```

Document the limitation in the preceding markdown cell and suggest the install command.

#### End-to-end pattern section
Define a small helper function that chains 2+ public functions together. Run it against the sample data. Show how to feed the output to the next step. Keep it self-contained — no hidden state from earlier cells.

#### Error handling reference section
Always the last section. Contains:
1. A markdown cell with a table mapping each function → guard condition → error type (sourced from docstring `Raises:` sections).
2. One or more code cells that call each guard with bad input inside `try/except` and `print` the error, confirming raise-on-bad-input behaviour.

### Step 6 — Validate

After building all cells, run a smoke-test to confirm the notebook is valid JSON and all cells execute without error:

```python
import json, sys

# 1. Parse notebook
with open("MODULE_guide.ipynb", encoding="utf-8") as f:
    nb = json.load(f)
print(f"Cells: {len(nb['cells'])}")

# 2. Unique IDs
ids = [c['id'] for c in nb['cells']]
assert len(ids) == len(set(ids)), "Duplicate cell IDs"

# 3. Execute all cells in one namespace
ns = {}
for i, cell in enumerate([c for c in nb['cells'] if c['cell_type'] == 'code']):
    src = ''.join(cell['source'])
    exec(compile(src, f'cell_{i}', 'exec'), ns)

print("All cells OK")
```

If any cell fails, fix it with `NotebookEdit` (replace mode on the failing cell) and re-run.

---

## Conventions for This Project

These are confirmed patterns from existing guide notebooks (`index_guide.ipynb`, `query_translation_guide.ipynb`, `retriever_guide.ipynb`):

| Convention | Rule |
|---|---|
| File name | `{module_name}_guide.ipynb` in the project root |
| Cell ratio | Aim for ~1:1 markdown to code cells |
| Section headers | `## N — function_name` with anchor comment |
| Sample data | Defined once in section 2, reused in all later sections |
| Synthetic embeddings | `np.random.default_rng(seed=0)` + `FakeEmbeddings(size=DIM)` |
| LangChain imports | Use `langchain_classic.*` NOT `langchain.*` for retrievers/chains/schema |
| Print format | `f"Label : {value}"` — colon-aligned labels for readability |
| No print() in source | Demos use `print(f"...")` in cells; the module itself uses `logger` |
| API key guard | `if not os.environ.get("OPENAI_API_KEY"):` before any LLM call |
| Threshold retriever note | Always note that `FakeEmbeddings` produces random scores; threshold may return empty |

---

## Quality Checklist

Before declaring done:

- [ ] Notebook JSON is valid (`json.load` succeeds)
- [ ] All cell IDs are unique
- [ ] All no-API-key cells execute without error
- [ ] Every public function has at least one code example
- [ ] API-key-gated cells print a clear skip message when key is absent
- [ ] Error handling section covers every `Raises:` entry in the docstrings
- [ ] End-to-end section is self-contained
- [ ] Module docstring / public API matches what was documented
- [ ] File is saved as `{module_name}_guide.ipynb` in project root

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `JSONDecodeError` on load | Double-quotes in code cells written via `Write` | Rebuild affected cells with `NotebookEdit` |
| `ModuleNotFoundError: langchain.retrievers` | Wrong package (Anthropic SDK shadows classic LangChain) | Use `langchain_classic.*` instead |
| `ValueError: Self query retriever ... not supported` | FAISS has no filter translator in `langchain_classic` | Switch demo to Chroma; document limitation in markdown |
| Cell executes but produces no output | Missing `print()` call | Add explicit print statements — notebooks don't auto-display mid-cell values |
| Smoke test fails on API-gated cell | Cell runs even without key | Ensure guard is `if not os.environ.get(...)` at the very top of the cell body |
| Duplicate cell IDs | `NotebookEdit` auto-generates IDs | IDs are auto-assigned; duplicates only occur if manually set — verify with `json.load` |
