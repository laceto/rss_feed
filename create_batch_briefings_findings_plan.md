# Code Review: create_batch_briefings.py

**Review Date:** 2026-03-27
**Reviewer:** Claude Code
**File:** `create_batch_briefings.py`
**Scope:** Migration opportunities relative to `pipeline/` package; private API coupling

---

## Executive Summary

`create_batch_briefings.py` is well-structured and already uses `kitai.batch` for
submission. The architecture split (local retrieval + deferred LLM batch) is sound
and correctly documented. There are two actionable findings: one coupling violation
against a private API (High) and one dead `sys.path.insert` left over from before
the project became pip-installable (Medium). Everything else — atomic-write-free
sentinel pattern, retrieval isolation, dry-run gate, skip logic — is correct and
should be preserved.

---

## Findings

### 🟠 High Priority Issues (Count: 1)

#### Issue 1: `from hybrid_rag import _get_resources` imports a private function across a module boundary
**Severity:** High
**Category:** Maintainability / API coupling
**Lines:** 71–72

**Description:**
`create_batch_briefings.py` imports `_get_resources` — a private function
(leading underscore) — directly from `hybrid_rag`:

```python
from hybrid_rag import (
    K_BM25,
    K_SEMANTIC,
    WEIGHTS_SPARSE,
    CHAT_MODEL,
    _get_resources,    # ← private
)
```

`_get_resources()` is the lazy-init cache for FAISS vectorstore, BM25 corpus,
and OpenAI clients. The public `ask()` API calls it internally but does not
expose the resources dict to callers — so `create_batch_briefings.py` cannot
use `ask()` for retrieval-only work; it legitimately needs the resources dict.
That said, importing a private symbol from another module creates invisible
coupling: any rename, refactor, or removal of `_get_resources` inside
`hybrid_rag.py` silently breaks `create_batch_briefings.py` with an
`ImportError` at startup.

**Impact:**
- Refactoring `hybrid_rag.py` internals can silently break briefing submission
  without any static type error.
- The coupling is invisible — `_get_resources` appears nowhere in `hybrid_rag`'s
  documented public API (CLAUDE.md, README, module docstring).
- Any caller that needs retrieval-only access must discover and import a private
  symbol, which is a non-obvious pattern to follow.

**Recommendation:**
Expose a public `get_resources() -> dict` function in `hybrid_rag.py` that simply
delegates to `_get_resources()`. This is a one-line change in `hybrid_rag.py` and
a one-word rename in `create_batch_briefings.py`.

**Proposed Solution:**

`hybrid_rag.py` — add after `_get_resources`:
```python
def get_resources() -> dict:
    """Return the cached RAG resource dict (FAISS store, BM25 corpus, clients).

    Callers that need retrieval without LLM generation (e.g. batch submission
    scripts) use this to load FAISS + BM25 without triggering a full ask() call.

    Returns:
        dict with keys: openai_client, chat_model, vs, corpus
    """
    return _get_resources()
```

`create_batch_briefings.py` — rename the import:
```python
from hybrid_rag import (
    K_BM25,
    K_SEMANTIC,
    WEIGHTS_SPARSE,
    CHAT_MODEL,
    get_resources,     # ← now public
)
```

And at line 286:
```python
res = get_resources()
```

Net change: +6 lines in `hybrid_rag.py`, 1 rename in `create_batch_briefings.py`.
The module docstring for `create_batch_briefings.py` already references
`_get_resources()` at line 36 — update that comment too.

---

### 🟡 Medium Priority Issues (Count: 1)

#### Issue 2: `sys.path.insert(0, PROJECT_ROOT)` is dead code — project is pip-installable
**Severity:** Medium
**Category:** Maintainability / Dead code
**Lines:** 55–56

**Description:**
```python
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
```

This was needed before `pyproject.toml` made the project pip-installable
(`pip install -e .`). Now that `pyproject.toml` exists and the project is
installed in editable mode, `sys.path.insert` is redundant — Python already
finds all root-level modules via the editable install's `.pth` file.

Keeping it has two small costs:
1. `PROJECT_ROOT` is a module-level name that pollutes the namespace and
   appears to be configuration when it is actually dead.
2. The `sys.path.insert` silently takes precedence over any installed version
   of the project if someone accidentally installs a different copy — a latent
   confusion source.

**Impact:**
- Minor: no runtime error, but dead code creates a false impression that the
  path manipulation is necessary.

**Recommendation:**
Delete both lines. If `PROJECT_ROOT` is needed elsewhere (it is not, in this
file), derive it locally at that point of use.

**Proposed Solution:**
```python
# Delete lines 55-56:
# PROJECT_ROOT = Path(__file__).parent   ← remove
# sys.path.insert(0, str(PROJECT_ROOT))  ← remove
```

Also remove `Path` from the import if it becomes unused after this deletion
(check: `Path` is still used at line 252 for `trends_path = Path(TOPIC_TRENDS_FILE)`,
so the `from pathlib import Path` import stays).

---

## Positive Observations

- Already uses `kitai.batch.submit_batch_job` — no raw OpenAI batch SDK calls.
  This is consistent with the rest of the pipeline post-refactor.
- Retrieval/LLM split is clean: FAISS+BM25 runs locally in this script;
  only the LLM generation task is sent to the Batch API. Cost and latency
  are correctly partitioned.
- `_retrieve_docs_for_label` catches all retrieval exceptions and returns `[]`
  — the batch job can still be submitted for other spikes even if one label's
  retrieval fails.
- `build_tasks` skips unlabeled topics with a clear message pointing to
  `label_topics.py` — correct fault isolation.
- Sentinel files (`pending_briefings_batch.txt` + `pending_briefings_meta.json`)
  are written only after a successful `submit_batch_job()` call — no partial
  state is persisted on failure.
- `--dry-run` exits before loading FAISS, which is expensive (several seconds).
  This is the correct gate position.
- `openai_client` is reused from the resources dict for `submit_batch_job()`
  rather than creating a second client — one connection.
- `MAX_CONTEXT_CHARS` safety cap prevents oversized batch tasks.

---

## Action Plan

### Phase 1: High Priority (immediate — 10 minutes)
- [ ] Add `get_resources()` public wrapper to `hybrid_rag.py` (Issue 1)
- [ ] Replace `_get_resources` with `get_resources` in the import block
      and at line 286 (Issue 1)
- [ ] Update the module docstring line 36: `_get_resources()` → `get_resources()` (Issue 1)

### Phase 2: Medium Priority (next PR)
- [ ] Delete `PROJECT_ROOT = Path(__file__).parent` and
      `sys.path.insert(0, str(PROJECT_ROOT))` (lines 55–56) (Issue 2)

---

## Technical Debt Estimate

- **Total Issues:** 2 (0 critical, 1 high, 1 medium, 0 low)
- **Estimated Fix Time:** 10–15 minutes
- **Risk Level:** Low — logic is unchanged; only API surface changes
- **Recommended Refactor:** Yes — trivially safe
