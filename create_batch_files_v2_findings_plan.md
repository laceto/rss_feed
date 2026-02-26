# Code Review: create_batch_files_v2.py

**Review Date:** 2026-02-26
**Reviewer:** Claude Code
**File:** `create_batch_files_v2.py`

## Executive Summary

`create_batch_files_v2.py` submits a daily sector-analysis batch job to the OpenAI Batch API. It aggregates all news descriptions per date into a single prompt and extracts a `MultiSectorAnalysis` structured response. The core analytical idea is sound, but three bugs combine to directly cause the rate-limit failures: the data loader silently reads enriched files (wrong schema), there is no incremental state tracking so every run re-submits all historical dates, and private OpenAI SDK internals are used for schema serialization. Beyond the bugs, the script has high cognitive debt from five reassignments of `dfs`, no results-retrieval companion, and no way to limit or monitor the size of the prompts sent.

---

## Findings

### 🔴 Critical Issues (Count: 1)

#### Issue 1: `get_file_paths('output', ...)` now recurses into `output/enriched/`
**Severity:** Critical
**Category:** Correctness / Data Integrity
**Lines:** 13–14

**Description:**
`utils.get_file_paths` uses `os.walk`, which descends into every subdirectory. Since `output/enriched/` was created by the new pipeline and contains 136 `.txt` files with a completely different schema (`guid`, `entities`, `sector`, …), those files are now silently included in the `file_list`. When `pd.read_csv` reads them, the `description`, `title`, and `pubDate` columns are absent, causing either a `KeyError` crash or — if the column names happen to survive — polluting `joined_contents` with garbage rows that inflate token counts and corrupt the sector analysis.

**Current Code:**
```python
file_list = get_file_paths('output', file_pattern='.txt')
dfs = [pd.read_csv(file, sep='\t') for file in file_list]
```

**Impact:**
- `KeyError: 'description'` crash on line 15 or silent data corruption.
- If not crashing: enriched-file rows are concatenated into `joined_contents`, massively inflating token counts and triggering the enqueued-token rate limit.
- Every subsequent run is affected as long as `output/enriched/` is non-empty.

**Recommendation:**
Read only the top-level `output/` directory (not subdirectories) by globbing directly, matching the same pattern used in `enrich_feeds.py`.

**Proposed Fix:**
```python
from pathlib import Path

RAW_DIR = Path("output")
file_list = sorted(RAW_DIR.glob("feeds*.txt"))   # top-level only; no recursion
dfs = [pd.read_csv(f, sep="\t") for f in file_list]
```

---

### 🟠 High Priority Issues (Count: 3)

#### Issue 2: No incremental state tracking — all dates re-submitted every run
**Severity:** High
**Category:** Performance / Cost / Rate Limits
**Lines:** 82–115

**Description:**
The script collects every date present in `output/`, builds a task for each, and submits them all as one batch job on every execution. With 136 dates already present, each run enqueues the full history. This is the primary mechanism behind hitting the 2 M enqueued-token limit: previous batch jobs may still be in-flight when the next run fires.

**Impact:**
- Hitting `Enqueued token limit reached` on the second and subsequent runs.
- Paying to re-analyse dates whose results already exist.
- No mechanism to correlate which batch job covered which dates, so results retrieval is ambiguous.

**Recommendation:**
Introduce a results directory (`data/sector_results/`) and use the same sentinel pattern as `enrich_feeds.py`: if `data/sector_results/{date}.json` exists, skip that date.

**Proposed Fix (sketch):**
```python
RESULTS_DIR = Path("data/sector_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Filter to only unprocessed dates
unprocessed = [(date, content) for date, content in zip(dates, joined_contents)
               if not (RESULTS_DIR / f"{date}.json").exists()]
```

---

#### Issue 3: Private OpenAI SDK internals used for schema serialization
**Severity:** High
**Category:** Maintainability / Fragility
**Lines:** 10–11, 66

**Description:**
`from openai.lib._pydantic import to_strict_json_schema` and `from openai.lib._parsing._completions import type_to_response_format_param` import from underscore-prefixed (private) modules. These are not part of the public API and are not covered by semantic versioning guarantees — they can disappear or change signature in any patch release. The OpenAI SDK 2.x already changed internal structure once.

**Current Code:**
```python
from openai.lib._parsing._completions import type_to_response_format_param
from openai.lib._pydantic import to_strict_json_schema
...
Structured_Response = to_strict_json_schema(MultiSectorAnalysis)
```

**Impact:**
- Silent breakage on any `openai` package update.
- `ImportError` already observed in some SDK versions.

**Recommendation:**
Build the JSON schema from the Pydantic model using the public `model_json_schema()` method, and construct the `response_format` dict manually — this is stable across SDK versions.

**Proposed Fix:**
```python
# No private imports needed
schema = MultiSectorAnalysis.model_json_schema()

# In the task body:
"response_format": {
    "type": "json_schema",
    "json_schema": {
        "name": "multi_sector_analysis",
        "schema": schema,
        "strict": True,
    },
},
```

---

#### Issue 4: `dfs` variable reassigned five times with different types and semantics
**Severity:** High
**Category:** Cognitive Debt / Maintainability
**Lines:** 14–23

**Description:**
The name `dfs` is reused for five conceptually distinct objects across seven lines: a list of raw DataFrames → deduped DataFrames → a single concatenated DataFrame → grouped DataFrames → description-prepended DataFrames. This makes it impossible to follow the data transformation pipeline or add a step without reading all five assignments.

**Current Code:**
```python
dfs = [pd.read_csv(file, sep='\t') for file in file_list]    # List[DataFrame]
dfs = [df.drop_duplicates(subset=["description"]) for df in dfs]  # List[DataFrame]
df = pd.concat(dfs, ignore_index=True)                         # DataFrame
...
dfs = [g.copy() for _, g in df.groupby('pubDate')]             # List[DataFrame]
dfs = [df.assign(...) for df in dfs]                           # List[DataFrame]
dfs = [df.assign(...) for df in dfs]                           # List[DataFrame]
```

**Impact:**
- High cognitive load to trace what `dfs` means at any given line.
- Refactoring any single step risks breaking the chain silently.
- `df` and `dfs` coexist, increasing confusion.

**Recommendation:**
Use descriptive names for each stage of the pipeline.

**Proposed Fix:**
```python
raw_dfs = [pd.read_csv(f, sep="\t") for f in file_list]
combined = pd.concat(raw_dfs, ignore_index=True)
combined = combined.drop_duplicates(subset=["description"])
combined["pubDate"] = pd.to_datetime(combined["pubDate"]).dt.strftime("%Y-%m-%d")
combined["description"] = combined["pubDate"] + ": " + combined["title"] + ": " + combined["description"]

daily_dfs = {date: group for date, group in combined.groupby("pubDate")}
```

---

### 🟡 Medium Priority Issues (Count: 3)

#### Issue 5: `joined_contents` has no token budget — long days can exceed context window
**Severity:** Medium
**Category:** Correctness / Robustness
**Lines:** 24

**Description:**
A day's news items are concatenated with `"."` as separator. There is no guard on total length. A high-news day (e.g. the `feeds2025-11-21.txt` file which has 112 rows) produces a user message of ~15,000 tokens — close to `gpt-4.1-nano`'s 16K context limit. If the system prompt is also long, the request will fail.

**Recommendation:**
Truncate `joined_contents` per date to a safe token budget before building tasks (e.g. 10,000 chars ≈ 2,500 tokens leaves room for the schema and system prompt).

```python
MAX_CHARS = 10_000
joined_contents = [
    content[:MAX_CHARS] for content in
    (".\n".join(doc.page_content for doc in sublist) for sublist in docs)
]
```

---

#### Issue 6: No results-retrieval counterpart in this file
**Severity:** Medium
**Category:** Completeness / Observability
**Lines:** entire file

**Description:**
The script creates and uploads a batch job but has no way to collect results. The retrieval logic lives in `old/retrieve_batch_results.py` which is generic and unmaintained. The batch job ID printed to stdout is ephemeral — if the terminal is closed the ID is lost. Without a stored ID, the results are unrecoverable without listing all batches manually.

**Recommendation:**
Write the batch job ID to a file (`data/pending_sector_batch.txt`) immediately after creation, so a companion `collect_sector_results.py` script can read it and poll for completion.

```python
Path("data/pending_sector_batch.txt").write_text(batch_job.id)
log.info("Batch job ID saved → data/pending_sector_batch.txt")
```

---

#### Issue 7: List comprehension used purely for side effects (printing)
**Severity:** Medium
**Category:** Cognitive Debt / Style
**Line:** 27

**Description:**
`[print(info) for info in joined_contents]` constructs and discards a list of `None` values just to invoke `print`. This pattern is an anti-pattern in Python — list comprehensions are for building lists, not for side effects.

**Current Code:**
```python
[print(info) for info in joined_contents]
```

**Proposed Fix:**
```python
for content in joined_contents:
    log.debug("Sample content (first 200 chars): %s", content[:200])
```

---

### 🟢 Low Priority Issues (Count: 2)

#### Issue 8: Duplicate import of `BaseModel, Field`
**Severity:** Low
**Category:** Style
**Lines:** 7, 9

`from pydantic import BaseModel, Field` appears twice. Remove the duplicate.

---

#### Issue 9: Commented-out debug code left in production path
**Severity:** Low
**Category:** Maintainability
**Lines:** 25, 117

`# joined_contents = joined_contents[:5]` and `# print(tasks[:3])` are debug guards left commented in. They should either be removed or replaced with a `--dry-run` CLI flag.

---

## Positive Observations

- The `MultiSectorAnalysis` / `Sector` Pydantic schema is well-designed: all fields have clear descriptions, `Literal` types enforce valid enum values, and the model correctly limits to 8 sectors to avoid hallucination sprawl.
- Deduplication by `description` at both the per-file and global level (lines 15, 17) is correct — the same story appears across multiple days' feed files.
- Using the OpenAI Batch API (not synchronous calls) is the right choice for this volume: ~one request per trading day is well within limits when run incrementally.

---

## Action Plan

### Phase 1: Critical Fixes — Fix Before Next Run

- [ ] **Replace `get_file_paths` with direct glob on top-level `output/`** (Issue 1)
      Prevents reading enriched files; eliminates the crash / data corruption / token inflation.

### Phase 2: High Priority — Fix This Week

- [ ] **Add incremental sentinel (`data/sector_results/{date}.json`)** (Issue 2)
      Prevents re-submitting historical dates; directly solves the rate-limit problem.
- [ ] **Replace private SDK imports with `model_json_schema()`** (Issue 3)
      Eliminates fragile private-API dependency.
- [ ] **Rename pipeline variables (`raw_dfs`, `combined`, `daily_dfs`)** (Issue 4)
      Reduces cognitive load for all future edits.

### Phase 3: Medium Priority — Next Session

- [ ] **Add `MAX_CHARS` guard on `joined_contents`** (Issue 5)
- [ ] **Persist batch job ID to disk; write companion collect script** (Issue 6)
- [ ] **Replace list-comprehension print with `log.debug`** (Issue 7)

### Phase 4: Low Priority — Backlog

- [ ] Remove duplicate `pydantic` import (Issue 8)
- [ ] Remove commented-out debug lines or replace with `--dry-run` flag (Issue 9)

---

## Technical Debt Estimate

- **Total Issues:** 9 (1 critical, 3 high, 3 medium, 2 low)
- **Estimated Fix Time:** 3–4 hours
- **Risk Level:** High (the critical issue causes active failures on every run)
- **Recommended Refactor:** Yes — Phase 1 + 2 items together constitute a near-complete rewrite of the data loading and submission loop; doing them incrementally is messier than a clean pass.
