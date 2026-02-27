# Code Review: query_sector.py + query_entity.py

**Review Date:** 2026-02-27
**Reviewer:** Claude Code
**Files:** `query_sector.py`, `query_entity.py`, `visualize_sentiment.py`
**Focus question:** How to create a time series of sentiment values per sector,
exportable as a file and queryable via an API.

---

## Executive Summary

Both query modules are well-structured, clearly documented, and handle error
cases correctly. The main quality risk is **cross-module duplication**: four
helpers are copy-pasted verbatim between `query_sector.py` and `query_entity.py`,
and `SENTIMENT_SCORE` is independently redeclared in `visualize_sentiment.py`
despite existing in `constants.py`. These are the only structural concerns.

The focus question — exporting an all-sector pivot time series — is answered in
the Action Plan below. The infrastructure is already in place: `visualize_sentiment.py`
builds exactly this pivot internally (line 131–135) but discards it. The work
is exposing and persisting it.

---

## Findings

### 🟠 High Priority Issues (Count: 2)

#### Issue 1: `_trend_direction` and `_load_articles_for_dates` duplicated verbatim

**Severity:** High
**Category:** Maintainability / DRY violation
**Lines:** `query_sector.py:130–172`, `query_entity.py:177–225`

**Description:**
Both functions are copy-pasted with only minor comment changes. A bug fix or
behaviour change (e.g. the `±0.20` threshold) must be applied twice, with no
compiler or linter enforcement that they stay in sync. The `query_entity.py`
copies already include a comment acknowledging this ("duplicated to keep this
module self-contained"), which is a correct trade-off *if the modules are
deployed independently* — but in this repo they are always co-deployed.

**Impact:**
- Threshold drift: changing `_TREND_THRESHOLD` in one file silently leaves the
  other unchanged.
- Bug fixes must be applied twice.
- New developers will not know which copy to trust.

**Recommendation:**
Extract shared helpers to `constants.py` or a new `_query_utils.py`. Keep both
query modules self-contained for import, but import from the shared module.

**Proposed solution:**
```python
# constants.py — add at the bottom:
_TREND_THRESHOLD = 0.20

def trend_direction(scores: "pd.Series") -> "tuple[str, float]":
    """Shared implementation — imported by query_sector and query_entity."""
    import pandas as pd
    if len(scores) < 2:
        return "stable", 0.0
    mid = len(scores) // 2
    delta = round(float(scores.iloc[mid:].mean() - scores.iloc[:mid].mean()), 4)
    if delta > _TREND_THRESHOLD:
        return "improving", delta
    if delta < -_TREND_THRESHOLD:
        return "deteriorating", delta
    return "stable", delta
```
Then in each query module:
```python
from constants import trend_direction as _trend_direction
```

---

#### Issue 2: `SENTIMENT_SCORE` redeclared in `visualize_sentiment.py`

**Severity:** High
**Category:** Single source of truth violation
**Lines:** `visualize_sentiment.py:32`

**Description:**
`constants.py` is explicitly documented as the single source of truth for
`SENTIMENT_SCORE`. `visualize_sentiment.py` ignores this and declares its own
copy. If a new sentiment label is added (e.g. "mixed"), `constants.py` would
be updated but the chart script would silently produce `NaN` sentiment scores.

**Current code:**
```python
# visualize_sentiment.py line 32
SENTIMENT_SCORE: dict[str, int] = {"positive": 1, "neutral": 0, "negative": -1}
```

**Proposed solution:**
```python
from constants import SENTIMENT_SCORE, SENTIMENT_COLORS
# Remove the local declarations entirely
```

---

### 🟡 Medium Priority Issues (Count: 2)

#### Issue 3: `_load_entity_df()` called once per public function call — no caching

**Severity:** Medium
**Category:** Performance
**Lines:** `query_entity.py:260`, `query_entity.py:323`

**Description:**
Every call to `get_entity_snapshot()` or `get_entity_time_series()` reads and
explodes `sector_summary.tsv` from disk from scratch. For the current file size
this is fast, but in a Streamlit app or agent loop that calls both functions in
the same request, the file is parsed twice and the explode operation runs twice.

**Recommendation:**
Use `functools.lru_cache` (with `maxsize=1`) on a function that returns the
tuple `(mtime, df)` — cache invalidates automatically when the file changes.

**Proposed solution:**
```python
import functools, os

@functools.lru_cache(maxsize=1)
def _cached_entity_df(mtime: float) -> pd.DataFrame:
    """Inner loader — keyed by file mtime so cache auto-invalidates on update."""
    ...  # actual load + explode logic here

def _load_entity_df() -> pd.DataFrame:
    mtime = SECTOR_SUMMARY_FILE.stat().st_mtime if SECTOR_SUMMARY_FILE.exists() else 0.0
    return _cached_entity_df(mtime)
```

---

#### Issue 4: `_resolve_entity` hint-building is O(n × m)

**Severity:** Medium
**Category:** Performance
**Lines:** `query_entity.py:164–168`

**Description:**
The set comprehension calls `df.loc[df["entity_lower"] == el, "entity"].iloc[0]`
for every distinct lowercase entity. For 1099 entities this is ~1099 linear
scans of the DataFrame. For the current dataset this is negligible, but it
degrades as the entity set grows.

**Proposed solution:**
Pre-build a `{lowercase → canonical}` dict once:
```python
def _resolve_entity(name: str, df: pd.DataFrame) -> str:
    canon_map = (
        df.sort_values("date")
          .drop_duplicates(subset="entity_lower", keep="first")
          .set_index("entity_lower")["entity"]
          .to_dict()
    )
    name_lower = name.lower()
    if name_lower in canon_map:
        return canon_map[name_lower]
    similar = sorted(
        v for k, v in canon_map.items()
        if name_lower in k or k.startswith(name_lower)
    )[:10]
    hint = f"  Similar entities: {similar}" if similar else ""
    raise LookupError(
        f"Entity '{name}' not found in sector data.{hint}\n"
        "Call list_entities() to browse all known entities."
    )
```

---

### 🟢 Low Priority Issues (Count: 2)

#### Issue 5: `iterrows()` used for time_series list construction

**Severity:** Low
**Category:** Performance / Pythonic idiom
**Lines:** `query_sector.py:262–271`, `query_entity.py:358–367`

**Description:**
`iterrows()` is the slowest way to iterate a DataFrame. For small results
(< 1000 rows) it doesn't matter, but it signals unfamiliarity with the
vectorised API.

**Proposed solution:**
```python
time_series = rows[["date", "sentiment", "sentiment_score", "news_category"]].assign(
    date=rows["date"].dt.date.astype(str),
    sentiment_score=rows["sentiment_score"].astype(int),
).to_dict("records")
```

---

#### Issue 6: Inconsistent `stacklevel` in `warnings.warn` calls

**Severity:** Low
**Category:** Observability
**Lines:** `query_sector.py:104` (stacklevel=3), `query_sector.py:159` (stacklevel=4)

**Description:**
`stacklevel` controls which call frame is reported in the warning. The values
differ between `_load_summary()` and `_load_articles_for_dates()` without
explanation. Incorrect `stacklevel` causes the warning to point to the wrong
line in user code, making debugging harder.

**Recommendation:**
Document the chosen stacklevel with a comment, or standardise on `stacklevel=2`
(points at the public API caller) across all internal helpers.

---

## Positive Observations

- Module docstrings are exemplary — they include usage examples, full schema,
  and a debugging section. Rare and valuable.
- `constants.py` as a single source of truth for the taxonomy is the correct
  architectural decision; `SECTOR_TAXONOMY` derived from `SectorName.__args__`
  eliminates drift between Pydantic validation and runtime validation.
- `LookupError` vs `ValueError` distinction in `query_sector.py` (wrong name
  vs valid name / no data) is semantically precise and makes error handling in
  callers straightforward.
- `_load_summary()` returning an empty DataFrame (with correct columns) instead
  of raising on missing file is the right defensive pattern.
- `query_entity._resolve_entity` returning canonical casing from the *oldest*
  date is a well-reasoned, deterministic choice.

---

## Answering the Focus Question: Exporting a Per-Sector Sentiment Time Series

### What you want

A **wide-format pivot** — `date × sector` — where each cell is the mean
`sentiment_score` for that date and sector:

```
date        | Commercial Services | Communications | Electronic Technology | ...
2026-01-28  | 0.0                 | 1.0            | 1.0                   | ...
2026-01-29  | NaN                 | 0.0            | 1.0                   | ...
```

- `NaN` = the LLM found no news for that sector on that date.
- The pivot can be read directly into R (`read_tsv`), Excel, or
  `pd.read_csv(sep="\t", index_col=0, parse_dates=True)`.

### What already exists

`visualize_sentiment.py` builds this exact pivot at line 131–135 (`chart_trends`),
but only as a local variable. The pivot is discarded after the chart renders.

### What to add to `query_sector.py`

Two new public functions (add after `get_time_series`):

```python
def get_all_sectors_pivot(
    lookback_days: int | None = None,
    freq: str = "D",
) -> pd.DataFrame:
    """Return a date × sector pivot of daily mean sentiment scores.

    Each cell is the mean sentiment_score (-1, 0, or 1) for that sector
    on that date. NaN means no data for that sector on that date.

    Args:
        lookback_days: Calendar days to look back from today.
                       None = all available history.
        freq:          Pandas offset alias for the output date frequency.
                       "D" = daily (default), "W" = weekly, "M" = monthly.

    Returns:
        DataFrame with DatetimeIndex and one column per sector.
        Columns are sorted alphabetically.
        Empty DataFrame if the TSV is missing.
    """
    df = _load_summary()
    if df.empty:
        return pd.DataFrame()

    if lookback_days is not None:
        cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))
        df = df[df["date"] >= cutoff]

    if df.empty:
        return pd.DataFrame()

    pivot = (
        df.groupby(["date", "sector"])["sentiment_score"]
        .mean()
        .unstack("sector")
        .sort_index()
    )
    # Reindex to a regular calendar grid so gaps are explicit NaN, not missing
    if freq != "D" or not pivot.empty:
        pivot = pivot.asfreq(freq)

    return pivot[sorted(pivot.columns)]


def export_sector_pivot(
    output_path: "Path | str" = Path("data") / "sector_sentiment_pivot.tsv",
    lookback_days: int | None = None,
    freq: str = "D",
) -> Path:
    """Write the date × sector sentiment pivot to a TSV file.

    Suitable for downstream consumption by R, Excel, or any charting tool.
    Dates with no data for a sector are written as empty cells (NaN).

    Args:
        output_path:  Destination path. Default: data/sector_sentiment_pivot.tsv
        lookback_days: None = full history.
        freq:          "D" | "W" | "M" — output date granularity.

    Returns:
        The resolved output path (for chaining / logging).

    Raises:
        RuntimeError: If the pivot is empty (no data / missing TSV).
    """
    pivot = get_all_sectors_pivot(lookback_days=lookback_days, freq=freq)
    if pivot.empty:
        raise RuntimeError(
            "Pivot is empty — sector_summary.tsv may be missing or the "
            "lookback window contains no data."
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out, sep="\t")
    return out
```

### How to query the pivot

```python
from query_sector import get_all_sectors_pivot, export_sector_pivot

# In-memory — for dashboards or agents
pivot = get_all_sectors_pivot(lookback_days=60)
print(pivot["Electronic Technology"].dropna())   # single sector
print(pivot.mean())                              # mean score per sector over window

# Export to file — for R, Excel, downstream scripts
path = export_sector_pivot(lookback_days=90, freq="W")   # weekly granularity
print(f"Written to {path}")
```

### Reading the export in R

```r
library(readr)
library(dplyr)

pivot <- read_tsv("data/sector_sentiment_pivot.tsv") |>
  mutate(date = as.Date(date))
```

### Register the new file in `constants.py`

```python
# constants.py — add to File paths section:
SECTOR_PIVOT_FILE = Path("data") / "sector_sentiment_pivot.tsv"
```

And update `CLAUDE.md` `Key Files` table when these are added.

---

## Action Plan

### Phase 1: Answer the focus question (immediate)
- [ ] Add `get_all_sectors_pivot()` to `query_sector.py`
- [ ] Add `export_sector_pivot()` to `query_sector.py`
- [ ] Add `SECTOR_PIVOT_FILE` path to `constants.py`
- [ ] Add usage examples to `CLAUDE.md`

### Phase 2: DRY / single source of truth (this sprint)
- [ ] Extract `_trend_direction` + `_load_articles_for_dates` to `constants.py`
  or a new `_query_utils.py`; update both query modules to import
- [ ] Remove `SENTIMENT_SCORE` redeclaration from `visualize_sentiment.py`;
  import from `constants.py` instead

### Phase 3: Performance (next sprint, only if load becomes an issue)
- [ ] Add `mtime`-keyed `lru_cache` to `_load_entity_df()` / `_load_summary()`
- [ ] Replace `_resolve_entity` O(n²) hint scan with pre-built `{lower → canonical}` dict

### Phase 4: Polish (backlog)
- [ ] Replace `iterrows()` with vectorised `.to_dict("records")` in both modules
- [ ] Standardise `warnings.warn(stacklevel=...)` with explanatory comments

---

## Technical Debt Estimate

| Metric | Value |
|---|---|
| Total issues | 6 (0 critical, 2 high, 2 medium, 2 low) |
| Estimated fix time — Phase 1 (pivot export) | 30 min |
| Estimated fix time — Phase 2 (DRY) | 1 h |
| Estimated fix time — Phase 3+4 | 1–2 h |
| Risk level | Low |
| Recommended full refactor | No — incremental fixes sufficient |
