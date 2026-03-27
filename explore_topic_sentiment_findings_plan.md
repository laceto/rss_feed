# Code Review: explore_topic_sentiment.py

**Review Date:** 2026-03-27
**Reviewer:** Claude Code
**File:** `explore_topic_sentiment.py`
**Scope:** Constants deduplication; dead code

---

## Executive Summary

`explore_topic_sentiment.py` is a standalone manual validation script — it is
not part of CI and has hardcoded `TARGET_DATES`. The join logic and visualisation
are clear and well-commented. There are two maintainability findings: four
module-level constants are redefined locally despite having canonical definitions
in `constants.py` (High), and one dead dict (`SCORE_LABEL`) is never referenced
anywhere (Low). Both fixes are mechanical.

---

## Findings

### 🟠 High Priority Issues (Count: 1)

#### Issue 1: Four constants redefined locally — all exist in `constants.py`
**Severity:** High
**Category:** Maintainability / Single source of truth
**Lines:** 32–37

**Description:**
The config block at lines 32–37 defines four names that already exist in
`constants.py`:

```python
# explore_topic_sentiment.py lines 32-37
PROJECT_ROOT       = Path(__file__).parent
TOPIC_CLUSTERS_DIR = PROJECT_ROOT / "data" / "topic_clusters"
SECTOR_SUMMARY     = PROJECT_ROOT / "data" / "sector_summary.tsv"
TOPIC_TRENDS       = PROJECT_ROOT / "data" / "topic_trends.tsv"

SENTIMENT_SCORE: dict[str, float] = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
```

`constants.py` already defines all four:

```python
# constants.py
TOPIC_CLUSTERS_DIR  = Path("data") / "topic_clusters"   # line 114
SECTOR_SUMMARY_FILE = Path("data") / "sector_summary.tsv"  # line 67
TOPIC_TRENDS_FILE   = Path("data") / "topic_trends.tsv"    # line 113
SENTIMENT_SCORE     = {"positive": 1, "neutral": 0, "negative": -1}  # int-typed
```

The `SENTIMENT_SCORE` duplication is the same pattern fixed in `cluster_topics.py`
earlier in this session. The `float` vs `int` type difference is functionally
irrelevant — `pandas.Series.map()` promotes int to float on use.

Note the local names differ from the canonical names:
- `SECTOR_SUMMARY` → `SECTOR_SUMMARY_FILE`
- `TOPIC_TRENDS` → `TOPIC_TRENDS_FILE`

All usages in this file must be renamed when migrating.

**Impact:**
- If any of these paths change in `constants.py`, `explore_topic_sentiment.py`
  silently uses stale locations.
- `PROJECT_ROOT` becomes dead code once all four local constants are replaced by
  imports — it exists solely to construct the path constants.

**Recommendation:**
Replace the config block with a constants import. Rename `SECTOR_SUMMARY` →
`SECTOR_SUMMARY_FILE` and `TOPIC_TRENDS` → `TOPIC_TRENDS_FILE` at all usage sites.

**Proposed Solution:**
```python
# Replace lines 32-37 with:
from constants import (
    SECTOR_SUMMARY_FILE,
    SENTIMENT_SCORE,
    TOPIC_CLUSTERS_DIR,
    TOPIC_TRENDS_FILE,
)
```

Usage sites to rename:
| Old name | New name | Lines |
|---|---|---|
| `SECTOR_SUMMARY` | `SECTOR_SUMMARY_FILE` | 53 |
| `TOPIC_TRENDS` | `TOPIC_TRENDS_FILE` | 235 |
| `TOPIC_CLUSTERS_DIR` | `TOPIC_CLUSTERS_DIR` | 73, 210 (no change — same name) |
| `SENTIMENT_SCORE` | `SENTIMENT_SCORE` | 54 (no change — same name) |

Remove `from pathlib import Path` if `Path` is no longer used after deleting
`PROJECT_ROOT`. Check: `Path` is not used anywhere else in this file, so the
import can be removed.

Net change: -6 lines (config block + `from pathlib import Path`), +4 import
names, 2 renames. Zero logic change.

---

### 🟢 Low Priority Issues (Count: 1)

#### Issue 2: `SCORE_LABEL` dict is dead code — never referenced
**Severity:** Low
**Category:** Maintainability / Dead code
**Lines:** 118–121

**Description:**
```python
SCORE_LABEL = {
    lambda s: s > 0.20:  "positive",
    lambda s: s < -0.20: "negative",
}
```

This dict is defined but never used anywhere in the file. The `score_to_label()`
function immediately below it (lines 123–127) implements the same logic as
standalone conditional branches and is the only thing actually called. The
`SCORE_LABEL` dict also has a subtle bug: using lambdas as dict keys works in
Python (lambdas are hashable), but the dict cannot be iterated to dispatch on
values because dict keys are compared by identity, not by calling. This pattern
looks like an abandoned attempt at a dispatch table.

**Recommendation:**
Delete lines 118–121.

---

## Positive Observations

- The join is correctly designed: articles are joined to day-level sentiment
  (average across all sectors per date), not directly to per-article sentiment
  which doesn't exist. The comment at lines 88–90 explains the indirect join
  precisely.
- `compute_topic_sentiment` returns an empty DataFrame (not an exception) when
  the cluster file is absent — correct for a validation loop that should
  continue to the next date.
- `main()` runs `compute_topic_sentiment` twice per date (once for per-date
  output, once for the cross-date summary) — a minor redundancy but acceptable
  for a validation script where clarity trumps efficiency.
- `spot_check_top_spike` picks the topic with the highest *absolute* score
  (not just the highest positive score), which is the correct choice for
  surfacing strong signals in either direction.
- `bar()` ASCII visualisation handles `NaN` gracefully and produces a
  zero-centred display — appropriate for a ±1 scale.

---

## Action Plan

### Phase 1: High Priority (immediate — 5 minutes)
- [ ] Add `from constants import SECTOR_SUMMARY_FILE, SENTIMENT_SCORE, TOPIC_CLUSTERS_DIR, TOPIC_TRENDS_FILE` (Issue 1)
- [ ] Delete the config block lines 32–37 (Issue 1)
- [ ] Delete `from pathlib import Path` (Issue 1)
- [ ] Rename `SECTOR_SUMMARY` → `SECTOR_SUMMARY_FILE` at line 53 (Issue 1)
- [ ] Rename `TOPIC_TRENDS` → `TOPIC_TRENDS_FILE` at line 235 (Issue 1)

### Phase 2: Low Priority (same PR)
- [ ] Delete `SCORE_LABEL` dict lines 118–121 (Issue 2)

---

## Technical Debt Estimate

- **Total Issues:** 2 (0 critical, 1 high, 0 medium, 1 low)
- **Estimated Fix Time:** 5–10 minutes
- **Risk Level:** Low — this is a standalone validation script; no CI dependency
- **Recommended Refactor:** Yes — trivially safe
