# Code Review: cluster_topics.py

**Review Date:** 2026-03-27
**Reviewer:** Claude Code
**File:** `cluster_topics.py`
**Scope:** Migration opportunities relative to `pipeline/` package and `constants.py`

---

## Executive Summary

`cluster_topics.py` is a well-designed, self-contained module. It was intentionally
excluded from the `pipeline/` package (it is registered as a top-level module in
`pyproject.toml` and imported directly by other scripts). There is no duplication
with any pipeline submodule — the pipeline package does not cover FAISS loading,
HDBSCAN clustering, centroid matching, or label caching.

There is one concrete constants-duplication finding: `_SENTIMENT_SCORE` at line 469
is a private float-typed copy of `SENTIMENT_SCORE` already defined in `constants.py`.
`constants.SENTIMENT_SCORE` is imported by 12 other modules in this project and is
the single source of truth for the sentiment map. `cluster_topics.py` is the only
file that defines its own copy instead of importing it.

---

## Findings

### 🟠 High Priority Issues (Count: 1)

#### Issue 1: `_SENTIMENT_SCORE` duplicates `constants.SENTIMENT_SCORE`
**Severity:** High
**Category:** Maintainability / Single source of truth
**Lines:** 469–470

**Description:**
`cluster_topics.py` defines its own private sentiment map at module level:

```python
# cluster_topics.py line 468-469
# Sentiment numeric map (mirrors SENTIMENT_SCORE in constants.py)
_SENTIMENT_SCORE: dict[str, float] = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
```

`constants.py` already defines the authoritative version:

```python
# constants.py line 48
SENTIMENT_SCORE: dict[str, int] = {"positive": 1, "neutral": 0, "negative": -1}
```

The comment even acknowledges this: *"mirrors SENTIMENT_SCORE in constants.py"* —
a comment that is itself a code smell (mirroring means two truths, not one).

The type difference (`float` vs `int`) is functionally irrelevant: `pandas.Series.mean()`
promotes integers to float regardless of the input dtype, so `_SENTIMENT_SCORE` provides
no actual benefit over the `int`-valued constant.

`SENTIMENT_SCORE` is NOT currently in the constants import block of `cluster_topics.py`
(lines 42–56) — adding it is a one-line change.

**Impact:**
- If the sentiment mapping ever changes in `constants.py`, `compute_topic_sentiment`
  will silently use stale values.
- The explanatory comment is load-bearing (without it, readers wonder why there
  are two maps); removing the duplicate removes the need for the comment entirely.

**Recommendation:**
Add `SENTIMENT_SCORE` to the existing constants import and delete `_SENTIMENT_SCORE`.
The single usage site in `compute_topic_sentiment` (line 517) requires one rename.

**Proposed Solution:**
```python
# constants import block (lines 42-56) — add SENTIMENT_SCORE:
from constants import (
    CLUSTER_MAX_NOISE_RATIO,
    CLUSTER_MIN_CLUSTERS,
    CLUSTER_MIN_SAMPLES,
    CLUSTER_MIN_SIZE,
    CLUSTER_SELECTION_METHOD,
    CLUSTER_WINDOW_DAYS,
    FEEDS_REGISTRY_FILE,
    SECTOR_SUMMARY_FILE,
    SENTIMENT_SCORE,          # ← add
    TOPIC_CENTROIDS_FILE,
    TOPIC_CLUSTERS_DIR,
    TOPIC_LABELS_FILE,
    TOPIC_TRENDS_FILE,
    VECTORSTORE_DIR,
)

# Delete lines 467-469 (_SENTIMENT_SCORE definition and its comment).

# compute_topic_sentiment line 517 — rename:
sector_df["sentiment_score"] = sector_df["sentiment"].map(SENTIMENT_SCORE)
```

Net change: +1 import name, -3 lines, 1 rename. Logic is identical.

---

### 🟢 Low Priority Issues (Count: 1)

#### Issue 2: `compute_topic_sentiment` loads `sector_summary.tsv` directly — no reuse of `pipeline.query_sector`
**Severity:** Low
**Category:** Code organisation / Documentation
**Lines:** 516–523

**Description:**
`compute_topic_sentiment` does a raw `pd.read_csv` + `.map(SENTIMENT_SCORE)` on
`sector_summary.tsv`. `pipeline.query_sector._load_summary()` performs an
equivalent load. The private function cannot be used here for two valid reasons:

1. `_load_summary()` is private (leading `_`); calling it from outside `pipeline/`
   would couple `cluster_topics` to a private implementation detail.
2. `compute_topic_sentiment` accepts `sector_summary_path` as an explicit parameter
   (injectable for testing); `_load_summary()` uses the constant from `constants.py`
   internally (not injectable).

**No code change is needed.** This is a documentation gap: add a one-line comment
explaining why the direct read is intentional.

**Proposed Solution:**
```python
# Direct read rather than pipeline.query_sector._load_summary() because:
#   (a) _load_summary is private, and (b) this function needs an injectable path.
sector_df = pd.read_csv(Path(sector_summary_path), sep="\t")
```

---

## Positive Observations

- `cluster_topics.py` is intentionally kept as a top-level module and explicitly
  excluded from the `pipeline/` package — this boundary is correctly maintained.
- All file writes are atomic (`*.tmp` → `os.replace()`): `save_centroids`,
  `save_label_cache`, `append_trends`.
- `DuplicateDateError` and `ClusteringAborted` are properly typed exception classes
  that callers (now including `backfill.py`) can catch precisely.
- `_SENTIMENT_SCORE`'s comment ("mirrors SENTIMENT_SCORE in constants.py") at least
  makes the duplication visible — it guided this review to the correct finding.
- All public functions take explicit path parameters; no hidden global-path coupling.
- `_label_via_llm` returns `"Unknown topic"` on API failure rather than crashing —
  correct fault isolation for a labeling step that should never block the pipeline.

---

## Action Plan

### Phase 1: High Priority (immediate — 5 minutes)
- [ ] Add `SENTIMENT_SCORE` to the constants import block (Issue 1)
- [ ] Delete `_SENTIMENT_SCORE` definition and its comment (lines 467–469) (Issue 1)
- [ ] Rename `_SENTIMENT_SCORE` → `SENTIMENT_SCORE` at line 517 (Issue 1)

### Phase 2: Low Priority (backlog)
- [ ] Add a one-line comment to `compute_topic_sentiment` explaining the
      intentional direct read (Issue 2)

---

## Technical Debt Estimate

- **Total Issues:** 2 (0 critical, 1 high, 0 medium, 1 low)
- **Estimated Fix Time:** 5–10 minutes
- **Risk Level:** Low — rename of a module-private constant; logic is unchanged
- **Recommended Refactor:** Yes — trivially safe
