"""
tests/test_track_metadata.py

TDD tests for enrich/track_metadata.py.
Run with: pytest tests/test_track_metadata.py -v

The webai TrackResearcher is replaced by a fake — no API calls, no webai install needed.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "enrich"))
import track_metadata as tm


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _write_tracks(path: Path, rows: list[tuple[str, str, int]]) -> Path:
    pd.DataFrame(rows, columns=["artist", "title", "count"]).to_csv(path, index=False)
    return path


class _FakeInfo(SimpleNamespace):
    def model_dump(self) -> dict:
        return dict(self.__dict__)


class _FakeResearcher:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def research_track(self, artist, title):
        self.calls.append((artist, title))
        return _FakeInfo(
            query=title, artist="A", title="T", styles=["Minimal", "Tech House"],
            sources=["https://x", "https://y"], status="ok", error="", confidence=0.9,
        )


ROWS = [
    ("(unknown)", "(unknown)", 29),
    ("(unknown)", "Jeff Samuel - Lost", 1),
    ("(unknown)", "John Gaiser - Half Life", 1),
    ("(unknown)", "John Gaiser - Seepage", 1),
]


# ── load_tracks ──────────────────────────────────────────────────────────────

class TestLoadTracks:
    def test_returns_first_n_rows_given_limit(self, tmp_path):
        path = _write_tracks(tmp_path / "t.csv", ROWS)
        df = tm.load_tracks(path, limit=2)
        assert list(df["title"]) == ["(unknown)", "Jeff Samuel - Lost"]

    def test_offset_skips_rows(self, tmp_path):
        path = _write_tracks(tmp_path / "t.csv", ROWS)
        df = tm.load_tracks(path, limit=2, offset=2)
        assert list(df["title"]) == ["John Gaiser - Half Life", "John Gaiser - Seepage"]

    def test_missing_columns_raise_value_error(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1]}).to_csv(path, index=False)
        with pytest.raises(ValueError):
            tm.load_tracks(path, limit=10)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            tm.load_tracks(tmp_path / "nope.csv", limit=10)


# ── enrich_tracks ────────────────────────────────────────────────────────────

class TestEnrichTracks:
    def test_rows_are_flattened_with_input_columns(self, tmp_path):
        df = tm.load_tracks(_write_tracks(tmp_path / "t.csv", ROWS[1:2]), limit=10)
        out = tm.enrich_tracks(df, _FakeResearcher(), existing=None)

        assert len(out) == 1
        row = out.iloc[0]
        assert row["input_artist"] == "(unknown)"
        assert row["input_title"] == "Jeff Samuel - Lost"
        # list fields are joined for CSV output
        assert row["styles"] == "Minimal; Tech House"
        assert row["sources"] == "https://x; https://y"

    def test_already_enriched_rows_are_not_researched_again(self, tmp_path):
        df = tm.load_tracks(_write_tracks(tmp_path / "t.csv", ROWS[1:3]), limit=10)
        existing = pd.DataFrame(
            [{"input_artist": "(unknown)", "input_title": "Jeff Samuel - Lost", "status": "ok"}]
        )
        fake = _FakeResearcher()
        out = tm.enrich_tracks(df, fake, existing=existing)

        assert fake.calls == [("(unknown)", "John Gaiser - Half Life")]
        assert len(out) == 2  # existing row kept + new row appended

    def test_errored_rows_are_retried(self, tmp_path):
        df = tm.load_tracks(_write_tracks(tmp_path / "t.csv", ROWS[1:2]), limit=10)
        existing = pd.DataFrame(
            [{"input_artist": "(unknown)", "input_title": "Jeff Samuel - Lost", "status": "error"}]
        )
        fake = _FakeResearcher()
        out = tm.enrich_tracks(df, fake, existing=existing)

        assert len(fake.calls) == 1
        assert len(out) == 1
        assert out.iloc[0]["status"] == "ok"


# ── save_atomic ──────────────────────────────────────────────────────────────

class TestSaveAtomic:
    def test_writes_csv_and_leaves_no_tmp_file(self, tmp_path):
        target = tmp_path / "out.csv"
        tm.save_atomic(pd.DataFrame({"a": [1]}), target)
        assert pd.read_csv(target)["a"].tolist() == [1]
        assert not target.with_suffix(".tmp").exists()


class TestEnrichTracksReleaseYear:
    def test_release_year_is_integer_given_missing_values(self, tmp_path):
        df = tm.load_tracks(_write_tracks(tmp_path / "t.csv", ROWS[1:3]), limit=10)
        existing = pd.DataFrame(
            [{"input_artist": "(unknown)", "input_title": "Jeff Samuel - Lost",
              "status": "ok", "release_year": "2005"}]
        )
        out = tm.enrich_tracks(df, _FakeResearcher(), existing=existing)
        assert str(out["release_year"].dtype) == "Int64"
        assert out.iloc[0]["release_year"] == 2005
        assert pd.isna(out.iloc[1]["release_year"])
