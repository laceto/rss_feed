# Testing Rules

## TDD Cycle (mandatory)

1. **Red**: write the test first — it must fail before any implementation exists
2. **Green**: write the minimum code to make it pass — no more
3. **Refactor**: clean up with the safety net in place

Never write implementation code without a failing test that demands it.

## Test Structure

```
tests/
  test_cluster_topics.py   # 50 unit tests — reference for style
  test_*.py                # one file per module
```

- One `class Test<Function>` per public function
- Test names: `test_<behavior>_<given_context>`
- Arrange-Act-Assert (AAA) pattern

## Coverage Requirements

Every public function needs:
- Happy path
- Edge cases (empty input, boundary values)
- Failure modes (expected exceptions like `ClusteringAborted`, `DuplicateDateError`, `ValueError`, `LookupError`)

## Mocking Rules

- Mock `OpenAI` client calls — tests must not make real API calls
- Mock FAISS `load_local` — tests must not require the vectorstore on disk
- Never mock the file system when a `tmp_path` fixture suffices
- Never mock the thing you are testing

## Running Tests

```bash
cd C:\Users\l_ace\Desktop\projects\rss_feed
venv\Scripts\activate
pytest tests/ -v
```

## When Done

- All tests pass → STOP, do not refactor
- Any tests fail → LOAD: `debugging-rules.md`, section "Test Failures"
