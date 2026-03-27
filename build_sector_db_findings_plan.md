# Code Review: build_sector_db.py

**Review Date:** 2026-03-27
**Reviewer:** Claude Code
**File:** `build_sector_db.py`
**Scope:** Migration to `pipeline.sector_io`

---

## Executive Summary

`build_sector_db.py` is well-written and correct. Its atomic write pattern, error
handling, and date-from-filename invariant are all sound. However, every line of
implementation logic (`_DDL`, `_load_json`, `_insert_date`, `build`) has been
faithfully extracted into `pipeline/sector_io.py` during the recent refactor — making
this file a full duplicate of the package module. The only unique element is the
`main()` CLI guard.

The fix is mechanical: strip all implementation code and reduce `build_sector_db.py`
to a thin CLI wrapper (~25 lines) that delegates to `pipeline.sector_io.build_sector_db`.

---

## Findings

### 🟠 High Priority Issues (Count: 1)

#### Issue 1: Complete logic duplication with `pipeline.sector_io`
**Severity:** High
**Category:** Maintainability / DRY
**Lines:** 36–169

**Description:**
Four implementation units in `build_sector_db.py` are byte-for-byte duplicates of
their counterparts in `pipeline/sector_io.py`:

| `build_sector_db.py` | `pipeline/sector_io.py` |
|---|---|
| `_DDL` (lines 36–59) | `_SECTOR_DB_DDL` (lines 28–51) |
| `_load_json()` (lines 64–82) | `load_sector_json()` (lines 57–75) |
| `_insert_date()` (lines 85–129) | `insert_sector_date()` (lines 78–127) |
| `build()` (lines 135–169) | `build_sector_db()` (lines 130–167) |

Any bug fix or schema change must now be applied in both places or the
two implementations will silently diverge.

**Current Code (representative excerpt — `build()` vs `build_sector_db()`):**
```python
# build_sector_db.py — build()
def build(db_path: Path = SECTOR_DB_FILE, results_dir: Path = SECTOR_RESULTS_DIR) -> int:
    json_files = sorted(results_dir.glob("*.json"))
    tmp_path = db_path.with_suffix(".db.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    conn = sqlite3.connect(tmp_path)
    conn.executescript(_DDL)
    total = 0
    for path in json_files:
        data = _load_json(path)
        if data is None:
            continue
        date = path.stem
        total += _insert_date(conn, date, data)
    conn.commit()
    conn.close()
    os.replace(tmp_path, db_path)
    return total

# pipeline/sector_io.py — build_sector_db()  (identical logic, different names)
def build_sector_db(db_path: Path, results_dir: Path) -> int:
    ...  # same body, using load_sector_json / insert_sector_date / _SECTOR_DB_DDL
```

**Impact:**
- Any schema change to the DDL must be applied in both files.
- Bug fixes to `_load_json` or `_insert_date` must be applied in both files.
- Two sources of truth for the canonical DB build logic.

**Recommendation:**
Strip all implementation code. Reduce `build_sector_db.py` to a CLI wrapper that
imports from `pipeline.sector_io`. The `main()` logic is unique and stays.

**Proposed Solution:**
```python
"""
build_sector_db.py

CLI wrapper for pipeline.sector_io.build_sector_db.

Reads all data/sector_results/{date}.json files and builds a SQLite
database at data/sector_results.db.

Usage (CLI):
    python build_sector_db.py

To use programmatically, import from the package:
    from pipeline.sector_io import build_sector_db
"""

import sys

from constants import SECTOR_DB_FILE, SECTOR_RESULTS_DIR
from pipeline.sector_io import build_sector_db


def main() -> None:
    json_count = sum(1 for _ in SECTOR_RESULTS_DIR.glob("*.json"))
    if json_count == 0:
        print(
            f"[build_sector_db] No JSON files found in {SECTOR_RESULTS_DIR}. "
            "Nothing to do.",
            file=sys.stderr,
        )
        return

    total = build_sector_db(SECTOR_DB_FILE, SECTOR_RESULTS_DIR)
    print(f"Built {SECTOR_DB_FILE} — {total} rows across {json_count} dates.")


if __name__ == "__main__":
    main()
```

Result: 194 lines → ~30 lines; zero logic change; one source of truth.

---

### 🟡 Medium Priority Issues (Count: 1)

#### Issue 2: Documented public API `from build_sector_db import build` is superseded
**Severity:** Medium
**Category:** Documentation / API discoverability
**Lines:** 10

**Description:**
The module docstring advertises:
```python
# Can also be imported:
from build_sector_db import build
```
After the refactor this API still works, but callers should prefer
`from pipeline.sector_io import build_sector_db` — the canonical location.
Without a forwarding note, new code will continue importing from the
script rather than the package.

**Recommendation:**
Update the import example in the docstring after applying the fix from Issue 1.
The proposed solution above already includes the corrected docstring.

---

### 🟢 Low Priority Issues (Count: 1)

#### Issue 3: `build()` has default arguments coupling it to module-level constants
**Severity:** Low
**Category:** API design
**Lines:** 135–138

**Description:**
```python
def build(
    db_path: Path = SECTOR_DB_FILE,
    results_dir: Path = SECTOR_RESULTS_DIR,
) -> int:
```
`pipeline.sector_io.build_sector_db` correctly requires explicit arguments
(no defaults), forcing callers to be explicit about paths. The default
arguments in the root-level `build()` are convenient for the CLI but hide
the path dependency from library callers.

This is a non-issue after Issue 1 is fixed (the defaults move to `main()` where
they belong), but worth noting as a design anti-pattern to avoid in new functions.

---

## Positive Observations

- Atomic write pattern (`db.tmp` → `os.replace()`) is correctly implemented and
  was faithfully preserved in `pipeline/sector_io.py`.
- `date` taken from filename stem (not JSON body) — correct invariant, documented
  in both places.
- Error isolation in `_load_json` (log to stderr, return `None`, continue) is
  the right approach for a batch rebuild that must not abort on one bad file.
- `main()` pre-flight check (count JSON files before doing any work) prevents a
  silent no-op that produces an empty database.

---

## Action Plan

### Phase 1: High Priority (immediate — 15 minutes)
- [ ] Replace all implementation code in `build_sector_db.py` with the
      proposed 30-line CLI wrapper (Issue 1)
- [ ] Update the module docstring import example to point to `pipeline.sector_io`
      (Issue 2 — resolved by same edit)

### Phase 2: Low Priority (backlog)
- [ ] Note the explicit-args pattern in a comment when writing new pipeline
      functions — no code change needed (Issue 3)

---

## Technical Debt Estimate

- **Total Issues:** 3 (0 critical, 1 high, 1 medium, 1 low)
- **Estimated Fix Time:** 15–20 minutes
- **Risk Level:** Low — logic is unchanged; only the import chain changes
- **Recommended Refactor:** Yes — trivially safe, high leverage
