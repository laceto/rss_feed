# Code Review: daily_briefing.py

**Review Date:** 2026-03-27
**Reviewer:** Claude Code
**File:** `daily_briefing.py`
**Scope:** Pipeline package migration; single-source-of-truth violations

---

## Executive Summary

`daily_briefing.py` is well-structured and correctly implements the pre-computed
fast-path / live-compute fallback pattern. The sector cross-check, RAG isolation,
and fault-tolerant formatting are all sound. There are two actionable findings:
`BRIEFINGS_DIR` is redefined locally despite being a named constant in
`constants.py` (High), and `sys.path.insert` is dead code left over from before
the project was pip-installable (Medium). Both fixes are mechanical with zero
logic change.

---

## Findings

### 🟠 High Priority Issues (Count: 1)

#### Issue 1: `BRIEFINGS_DIR` is redefined locally — `constants.BRIEFINGS_DIR` already exists
**Severity:** High
**Category:** Maintainability / Single source of truth
**Lines:** 44–45, 51

**Description:**
`daily_briefing.py` defines its own path at module level:

```python
# daily_briefing.py line 44-45
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ...imports...

BRIEFINGS_DIR = PROJECT_ROOT / "data" / "briefings"    # line 51
```

`constants.py` already defines the authoritative version at line 124:

```python
# constants.py
BRIEFINGS_DIR = Path("data") / "briefings"
```

`BRIEFINGS_DIR` is already imported from `constants` in every other script that
uses it (`create_batch_briefings.py`, `retrieve_batch_briefings.py`,
`backfill.py`). `daily_briefing.py` is the only holdout with a local copy.

The two paths are equivalent at runtime (both resolve relative to the working
directory), but the local redefinition means that if `constants.BRIEFINGS_DIR`
ever changes (e.g. a data directory reorganisation), `daily_briefing.py` will
silently use a stale location — the same failure mode as the `_SENTIMENT_SCORE`
duplication in `cluster_topics.py`.

**Impact:**
- `daily_briefing.py` becomes out-of-sync with the rest of the pipeline if
  `BRIEFINGS_DIR` changes in `constants.py`.
- The `PROJECT_ROOT` variable exists solely to compute `BRIEFINGS_DIR` — once
  `BRIEFINGS_DIR` is imported, `PROJECT_ROOT` becomes dead code.

**Recommendation:**
Add `BRIEFINGS_DIR` to the existing `constants` import and delete line 51.

**Proposed Solution:**
```python
# constants import block — add BRIEFINGS_DIR:
from constants import BRIEFINGS_DIR, TOPIC_TRENDS_FILE

# Delete line 51:
# BRIEFINGS_DIR = PROJECT_ROOT / "data" / "briefings"   ← remove
```

Net change: +1 import name, -1 line. Zero logic change.

---

### 🟡 Medium Priority Issues (Count: 1)

#### Issue 2: `sys.path.insert(0, PROJECT_ROOT)` is dead code — project is pip-installable
**Severity:** Medium
**Category:** Maintainability / Dead code
**Lines:** 44–45

**Description:**
```python
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
```

The same pattern found in `create_batch_briefings.py` (already fixed). The
project is pip-installable via `pyproject.toml` (`pip install -e .`), so
Python already finds all root-level modules through the editable install's
`.pth` file. The `sys.path.insert` call is redundant and adds a confusing
implicit dependency.

After Issue 1 is fixed, `PROJECT_ROOT` has no remaining usages and the entire
two-line block becomes dead. `Path` remains used at line 243
(`trends_path = Path(TOPIC_TRENDS_FILE)`) so the `from pathlib import Path`
import stays — but note that `TOPIC_TRENDS_FILE` is already a `Path` object
from `constants.py`, making the `Path(...)` wrapper at line 243 also redundant
(though harmless).

**Recommendation:**
Delete lines 44–45 after applying Issue 1's fix. The `Path(TOPIC_TRENDS_FILE)`
at line 243 can optionally be simplified to just `TOPIC_TRENDS_FILE`.

**Proposed Solution:**
```python
# Delete lines 44-45:
# PROJECT_ROOT = Path(__file__).parent    ← remove
# sys.path.insert(0, str(PROJECT_ROOT))   ← remove

# Line 243 — optional simplification:
# Before:
trends_path = Path(TOPIC_TRENDS_FILE)
# After (TOPIC_TRENDS_FILE is already a Path):
trends_path = TOPIC_TRENDS_FILE
```

If `trends_path` is simplified, `Path` is no longer used anywhere in the file
and `from pathlib import Path` can also be removed from the imports.

---

## Positive Observations

- Pre-computed fast-path (`print_precomputed`) correctly checks for
  `data/briefings/{date}.json` before triggering any LLM calls — the same
  skip-sentinel pattern used throughout the pipeline.
- `_rag_summary` wraps `from hybrid_rag import ask` in a `try/except ImportError`
  — the briefing still runs without RAG if `hybrid_rag` is unavailable.
- `--no-rag` exits before loading FAISS (lazy import inside `_rag_summary`) —
  no expensive resource initialisation for fast/offline runs.
- `_sector_crosscheck` catches `LookupError` per sector and continues — missing
  sector data never blocks the briefing.
- `build_briefing` is a proper pure function (returns a dict, no I/O) —
  `_save` is separated, making the function easily testable.
- `_print_briefing` handles the zero-spikes case cleanly with a helpful message.
- `--save` only writes the file when `n_spikes > 0` — no empty briefing files.

---

## Action Plan

### Phase 1: High Priority (immediate — 5 minutes)
- [ ] Add `BRIEFINGS_DIR` to the constants import (Issue 1)
- [ ] Delete `BRIEFINGS_DIR = PROJECT_ROOT / "data" / "briefings"` line 51 (Issue 1)

### Phase 2: Medium Priority (same PR)
- [ ] Delete `PROJECT_ROOT = Path(__file__).parent` and
      `sys.path.insert(0, str(PROJECT_ROOT))` (lines 44–45) (Issue 2)
- [ ] Simplify `Path(TOPIC_TRENDS_FILE)` → `TOPIC_TRENDS_FILE` at line 243 (Issue 2)
- [ ] Remove `from pathlib import Path` if no longer used (Issue 2)

---

## Technical Debt Estimate

- **Total Issues:** 2 (0 critical, 1 high, 1 medium)
- **Estimated Fix Time:** 5–10 minutes
- **Risk Level:** Low — logic is unchanged; only import consolidation
- **Recommended Refactor:** Yes — trivially safe
