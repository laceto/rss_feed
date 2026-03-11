# Code Review: chatbot_rag.py

**Review Date:** 2026-03-11
**Reviewer:** Claude Code
**File:** `chatbot_rag.py`

## Executive Summary

The chatbot is well-structured and correctly delegates all RAG business logic to `hybrid_rag.py`. Separation of concerns is good: vectorstore loading is cached, the hybrid retriever honours live sidebar values, and session state is clearly documented. Two issues stand out: the spinner wraps the entire retrieve+generate flow, so the user sees no progress until the full answer is ready (addressed by streaming); and source-document rendering is duplicated verbatim between the history loop and the new-message handler, which will cause drift. Several minor issues exist around imports, closures, and documentation.

## Findings

### 🟠 High Priority Issues (Count: 2)

#### Issue 1: No streaming — user waits silently for the full LLM response
**Severity:** High
**Category:** Performance / UX
**Lines:** 287–301

**Description:**
`answer_query()` blocks until the entire LLM response is complete. Combined with
`st.spinner`, the user sees a frozen spinner for the entire generation time with no
indication of progress.

**Current Code:**
```python
with st.spinner("Retrieving and generating answer…"):
    ...
    answer = answer_query(prompt, ordered, openai_client)

st.write(answer)
```

**Impact:**
- Poor perceived responsiveness — modern chat UX expectation is token-by-token output.
- The spinner hides all progress for retrieval AND generation together.

**Recommendation:**
Add a `_stream_answer()` generator in `chatbot_rag.py` (streaming is a UI concern, not
a `hybrid_rag.py` concern). Use `st.write_stream()` which returns the full concatenated
string for session state persistence. Confine the spinner to steps 1–3 (translation +
retrieval + reorder) only; streaming replaces the spinner for step 4.

---

#### Issue 2: `_api_key` captured by closure — invisible cache dependency
**Severity:** High
**Category:** Correctness / Cognitive Debt
**Lines:** 138–153

**Description:**
`_load_base_resources()` reads `_api_key` from the module scope via closure rather
than accepting it as a parameter. `@st.cache_resource` keys on function *arguments*
only, so the API key is not part of the cache key. If the key changes between runs
(e.g. dotenv reload), the cached clients still hold the old key silently.

**Current Code:**
```python
@st.cache_resource(show_spinner="Loading vectorstore …")
def _load_base_resources(folder, index_name, embedding_model, embed_dimensions):
    openai_client = _OpenAIClient(api_key=_api_key)   # _api_key from closure
    ...
```

**Impact:**
- Stale API key in cached client if key rotates without server restart.
- Hidden assumption: only discoverable by reading the closure carefully.

**Recommendation:**
Pass `api_key` as an explicit parameter so it is part of the cache key.

**Proposed Solution:**
```python
@st.cache_resource(show_spinner="Loading vectorstore …")
def _load_base_resources(folder, index_name, embedding_model, embed_dimensions, api_key):
    openai_client = _OpenAIClient(api_key=api_key)
    ...

vs, corpus, openai_client, chat_model = _load_base_resources(
    VECTORSTORE_DIR, FAISS_INDEX_NAME, EMBEDDING_MODEL, EMBED_DIMENSIONS, _api_key
)
```

---

### 🟡 Medium Priority Issues (Count: 2)

#### Issue 3: Source-document rendering duplicated in two places
**Severity:** Medium
**Category:** Maintainability / DRY
**Lines:** 261–273 (history loop) and 321–331 (new message handler)

**Description:**
The expander that renders source documents is written twice — once when replaying
history and once when handling a new message. Any change to the display format
(e.g. adding a field) must be applied in both places.

**Recommendation:**
Extract to `_render_sources(sources: list[dict]) -> None` and call from both sites.

**Proposed Solution:**
```python
def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Source documents ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            st.markdown(
                f"**[{i}]** {src['title']}  \n*{src['date']}*"
                + (f" · [{src['link']}]({src['link']})" if src["link"] else "")
            )
            st.caption(src["snippet"])
            if i < len(sources):
                st.divider()
```

---

#### Issue 4: Augmented-query rendering duplicated in two places
**Severity:** Medium
**Category:** Maintainability / DRY
**Lines:** 254–259 (history loop) and 303–308 (new message handler)

**Description:**
Same pattern as Issue 3 — the augmented-query expander is duplicated.

**Recommendation:**
Extract to `_render_queries(queries: list[str]) -> None`.

---

### 🟢 Low Priority Issues (Count: 3)

#### Issue 5: `answer_query` imported but unused after streaming
**Severity:** Low
**Category:** Cleanliness
**Lines:** 86

**Description:**
`answer_query` is imported from `hybrid_rag` but will be unused once streaming is
added. Unused imports add noise to the dependency surface.

**Recommendation:**
Remove the import after introducing `_stream_answer()`.

---

#### Issue 6: `Path` imported but never used directly
**Severity:** Low
**Category:** Cleanliness
**Lines:** 69

**Description:**
`from pathlib import Path` is present but `Path` is only used in the type hint of
`_load_base_resources`. At runtime, `VECTORSTORE_DIR` (already a `Path`) is passed
in — no `Path()` calls are made in this file.

**Recommendation:**
Remove if the type hint is dropped; keep if the hint is kept (it is useful
documentation for the cache function signature).

---

#### Issue 7: Module docstring architecture diagram still shows blocking `answer_query`
**Severity:** Low
**Category:** Documentation
**Lines:** 34

**Description:**
After adding streaming, the diagram line `answer_query(openai_client) → str` will be
stale. The docstring is a key orientation tool and should stay current.

**Recommendation:**
Update to `_stream_answer(openai_client) → Generator[str]` after the change.

---

## Positive Observations

- All RAG business logic lives in `hybrid_rag.py` — no duplication of retrieval code.
- `@st.cache_resource` is used correctly for the expensive vectorstore load.
- Session state shape is fully documented in the module docstring — easy to reason about.
- Early `st.stop()` on missing API key prevents confusing downstream errors.
- Sidebar defaults are driven by `hybrid_rag` constants — single source of truth maintained.
- `# -*- coding: utf-8 -*-` declaration prevents cp1252 decode errors on Windows.

## Action Plan

### Phase 1: High Priority (Immediate)
- [x] Add `_stream_answer()` generator; replace `answer_query` + `st.write` with `st.write_stream` (Issue 1)
- [x] Pass `api_key` as explicit param to `_load_base_resources` (Issue 2)

### Phase 2: Medium Priority (Next session)
- [ ] Extract `_render_sources()` helper (Issue 3)
- [ ] Extract `_render_queries()` helper (Issue 4)

### Phase 3: Low Priority (Backlog)
- [ ] Remove unused `answer_query` import after streaming lands (Issue 5)
- [ ] Decide whether to keep `Path` import with type hint (Issue 6)
- [ ] Update module docstring architecture diagram (Issue 7)

## Technical Debt Estimate

- **Total Issues:** 7 (0 critical, 2 high, 2 medium, 3 low)
- **Estimated Fix Time:** 2–3 hours
- **Risk Level:** Low
- **Recommended Refactor:** No — incremental improvements sufficient
