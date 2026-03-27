# Code Review: backfill.py

**Review Date:** 2026-03-27
**Reviewer:** Claude Code
**File:** `backfill.py`
**Scope:** Migration opportunities to use `pipeline/` package and `constants.py`

---

## Executive Summary

`backfill.py` is a clean, focused orchestration script. Its subprocess-based design
gives free process isolation — one failing date does not abort the loop — but it also
means the script depends on other scripts' file names and misses the structured
exception types (`ClusteringAborted`, `DuplicateDateError`) that `cluster_topics.py`
already exposes as a module API. Two hardcoded paths bypass `constants.py`, creating
a second source of truth for file locations. Fixing these two issues is low-risk and
eliminates the most significant maintenance liability.

---

## Findings

### 🟠 High Priority Issues (Count: 2)

#### Issue 1: Hardcoded file paths duplicate `constants.py`
**Severity:** High
**Category:** Maintainability / Single source of truth
**Lines:** 32, 58–59

**Description:**
`TRENDS_FILE` and `briefings_dir` are derived from `PROJECT_ROOT` rather than
imported from `constants.py`, which already defines `TOPIC_TRENDS_FILE` and
`BRIEFINGS_DIR` as the authoritative locations. If the paths change in
`constants.py`, `backfill.py` silently uses the old paths.

**Current Code:**
```python
PROJECT_ROOT = Path(__file__).parent
TRENDS_FILE  = PROJECT_ROOT / "data" / "topic_trends.tsv"   # line 32

def _briefing_dates() -> set[str]:
    briefings_dir = PROJECT_ROOT / "data" / "briefings"     # line 58–59
```

**Impact:**
- Two sources of truth for `data/topic_trends.tsv` and `data/briefings/`
- Silent divergence if paths are updated in `constants.py`

**Recommendation:**
Import both constants and drop the local derivations.

**Proposed Solution:**
```python
from constants import BRIEFINGS_DIR, TOPIC_TRENDS_FILE

def _clustered_dates() -> set[str]:
    if not TOPIC_TRENDS_FILE.exists():
        return set()
    df = pd.read_csv(TOPIC_TRENDS_FILE, sep="\t")
    return set(df["date"].astype(str).unique())

def _briefing_dates() -> set[str]:
    if not BRIEFINGS_DIR.exists():
        return set()
    return {p.stem for p in BRIEFINGS_DIR.glob("*.json")}
```

Remove `PROJECT_ROOT` and `TRENDS_FILE` module-level variables — they are no
longer needed once `BRIEFINGS_DIR` is imported.

---

#### Issue 2: Subprocess call to `cluster_topics.py` bypasses the module API
**Severity:** High
**Category:** Cognitive debt / Coupling
**Lines:** 89

**Description:**
`cluster_topics.py` is registered in `pyproject.toml` as an importable module and
exposes a clean public API (`run()`, `ClusteringAborted`). The subprocess call
couples `backfill.py` to a file name and to the CLI argument interface. The exit
code 2 / `ClusteringAborted` mapping already exists in the module — there is no
reason to go through the shell.

Direct import also enables catching `DuplicateDateError` separately (currently
treated as a generic error), and eliminates the `sys.executable` path injection.

**Current Code:**
```python
rc = _run([sys.executable, "cluster_topics.py", "--date", ds, "--skip-labeling"])

if rc == 0:
    stats["clustered"] += 1
elif rc == 2:
    stats["aborted"] += 1
else:
    stats["errors"] += 1
```

**Impact:**
- If `cluster_topics.py` is renamed, `backfill.py` breaks silently at runtime
- Exit codes are an implicit protocol — any unrelated failure returns `rc != 0`
- `sys.executable` and `cwd=PROJECT_ROOT` are required to locate the script

**Recommendation:**
Replace with a direct `try/except` block using the module's exception types.

**Proposed Solution:**
```python
from cluster_topics import run as cluster_run, ClusteringAborted, DuplicateDateError

def phase1_cluster(dates: list[date], sleep_s: float) -> dict:
    done  = _clustered_dates()
    stats = {"skipped": 0, "clustered": 0, "aborted": 0, "errors": 0}
    total = len(dates)

    for i, d in enumerate(dates, 1):
        ds = str(d)
        if ds in done:
            print(f"[{i:>3}/{total}] {ds}  SKIP")
            stats["skipped"] += 1
            continue

        print(f"[{i:>3}/{total}] {ds}  clustering ...", flush=True)
        try:
            cluster_run(target_date=d, skip_labeling=True)
            stats["clustered"] += 1
            print(f"[{i:>3}/{total}] {ds}  OK")
        except ClusteringAborted:
            stats["aborted"] += 1
            print(f"[{i:>3}/{total}] {ds}  ABORTED (degenerate / no articles)")
        except DuplicateDateError:
            stats["skipped"] += 1  # already done, just not in TRENDS_FILE query
            print(f"[{i:>3}/{total}] {ds}  SKIP (duplicate date in trends)")
        except Exception as exc:
            stats["errors"] += 1
            print(f"[{i:>3}/{total}] {ds}  ERROR {exc}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    return stats
```

After this change `sys`, `subprocess`, and `_run()` can all be removed if
`phase2_briefing` is also updated (see Medium issue below).

---

### 🟡 Medium Priority Issues (Count: 1)

#### Issue 3: Subprocess call to `daily_briefing.py`
**Severity:** Medium
**Category:** Coupling
**Lines:** 123–125

**Description:**
`daily_briefing.py` exposes `build_briefing(date, top_n, use_rag)` as a public
API. A direct call is possible, but `--save` file-writing is currently handled by
the CLI layer, so a direct import requires the caller to write
`data/briefings/{date}.json` explicitly.

This is a valid trade-off: subprocess is simpler here because the file-writing
logic stays in one place. The main risk is the same as Issue 2 — file-name
coupling and opaque exit codes.

**Current Code:**
```python
cmd = [sys.executable, "daily_briefing.py", "--date", ds, "--save"]
if not use_rag:
    cmd.append("--no-rag")
rc = _run(cmd)
```

**Recommendation (option A — direct import):**
```python
import json
from daily_briefing import build_briefing
from constants import BRIEFINGS_DIR

briefing = build_briefing(d, top_n=5, use_rag=use_rag)
if briefing and briefing.get("spikes"):
    out = BRIEFINGS_DIR / f"{ds}.json"
    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(briefing, indent=2, default=str), encoding="utf-8")
    stats["generated"] += 1
else:
    stats["no_spikes"] += 1
```

**Recommendation (option B — keep subprocess, fix coupling):**
If the subprocess approach is intentionally kept for `daily_briefing` (e.g. to
avoid loading the RAG model into the backfill process), reference the script via
an importable constant rather than a hardcoded string:

```python
# At module level — easier to find and update
_DAILY_BRIEFING_SCRIPT = Path(__file__).parent / "daily_briefing.py"

cmd = [sys.executable, str(_DAILY_BRIEFING_SCRIPT), "--date", ds, "--save"]
```

Either option reduces implicit coupling. Option A is preferred when the backfill
machine has FAISS loaded; option B when memory isolation matters.

---

### 🟢 Low Priority Issues (Count: 1)

#### Issue 4: `DEFAULT_START` hardcoded to a project-specific date
**Severity:** Low
**Category:** Documentation
**Lines:** 34

**Description:**
`DEFAULT_START = date(2025, 9, 1)` is a project-specific constant that will
become stale as the project ages. It is already overridable via `--start`, but
the magic date is unexplained.

**Proposed Solution:**
Add a comment explaining why Sep 2025:

```python
# Sep 2025 = first date with sector results in this deployment
DEFAULT_START = date(2025, 9, 1)
```

---

## Positive Observations

- Idempotent by design — both `_clustered_dates()` and `_briefing_dates()` check
  existing output before running; safe to re-run after partial failure.
- Exit code semantics for `ClusteringAborted` (rc==2 = non-fatal, counted as
  `aborted`) are correctly modelled even via subprocess.
- `--sleep` parameter and the `time.sleep(sleep_s)` guard prevent runaway API
  calls during large back-fills.
- Clean `argparse` setup with sensible defaults.

---

## Action Plan

### Phase 1: High Priority (immediate)
- [ ] Replace `TRENDS_FILE` and `briefings_dir` with `constants.TOPIC_TRENDS_FILE`
      and `constants.BRIEFINGS_DIR`
- [ ] Replace subprocess call to `cluster_topics.py` with direct
      `from cluster_topics import run, ClusteringAborted, DuplicateDateError`
      and `try/except` block

### Phase 2: Medium Priority (next pass)
- [ ] Evaluate subprocess vs direct import for `daily_briefing.py`; if keeping
      subprocess, pin the script path to `Path(__file__).parent / "daily_briefing.py"`

### Phase 3: Low Priority (backlog)
- [ ] Add inline comment explaining `DEFAULT_START = date(2025, 9, 1)`

---

## Technical Debt Estimate

- **Total Issues:** 4 (0 critical, 2 high, 1 medium, 1 low)
- **Estimated Fix Time:** 1–2 hours
- **Risk Level:** Low — all changes are mechanical; tests for `cluster_topics`
  cover the exception types
- **Recommended Refactor:** Incremental; Phases 1+2 together in one commit
