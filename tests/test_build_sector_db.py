"""
tests/test_build_sector_db.py

TDD tests for build_sector_db.py.
Run with: pytest tests/test_build_sector_db.py -v

All tests use tmp_path fixtures (pytest) so they never touch the real DB.
"""

import json
import sqlite3
from pathlib import Path

import pytest

import build_sector_db as bsd


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _write_json(directory: Path, filename: str, payload: dict) -> Path:
    """Write a JSON fixture file and return its path."""
    path = directory / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


SAMPLE_DATE_1 = {
    "date": "2026-01-01",
    "batch_id": "batch_abc123",
    "sectors": [
        {
            "entities": ["Apple", "Microsoft"],
            "sector": "Technology Services",
            "sentiment": "positive",
            "news_category": "earnings",
            "extraction_status": "ok",
        },
        {
            "entities": ["ExxonMobil"],
            "sector": "Energy Minerals",
            "sentiment": "negative",
            "news_category": "macro",
            "extraction_status": "ok",
        },
    ],
}

SAMPLE_DATE_2 = {
    "date": "2026-01-02",
    "batch_id": "batch_def456",
    "sectors": [
        {
            "entities": ["Google"],
            "sector": "Electronic Technology",
            "sentiment": "neutral",
            "news_category": "products",
            "extraction_status": "partial",
        },
    ],
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildEmpty:
    def test_build_empty_dir_returns_zero(self, tmp_path):
        """Empty results dir → build returns 0, no db written."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        db_path = tmp_path / "test.db"

        count = bsd.build(db_path, results_dir)

        assert count == 0


class TestSchema:
    def test_tables_exist(self, tmp_path):
        """DB must have sector_analyses and sector_entities tables."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        _write_json(results_dir, "2026-01-01.json", SAMPLE_DATE_1)
        db_path = tmp_path / "test.db"

        bsd.build(db_path, results_dir)

        conn = sqlite3.connect(db_path)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert "sector_analyses" in tables
        assert "sector_entities" in tables

    def test_sector_analyses_columns(self, tmp_path):
        """sector_analyses must have all required columns."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        _write_json(results_dir, "2026-01-01.json", SAMPLE_DATE_1)
        db_path = tmp_path / "test.db"

        bsd.build(db_path, results_dir)

        conn = sqlite3.connect(db_path)
        col_names = {row[1] for row in conn.execute(
            "PRAGMA table_info(sector_analyses)"
        ).fetchall()}
        conn.close()

        expected = {"id", "date", "sector", "sentiment", "sentiment_score",
                    "news_category", "extraction_status", "batch_id"}
        assert expected.issubset(col_names)


class TestRowCounts:
    def test_single_date_row_count(self, tmp_path):
        """One JSON with 2 sectors → 2 rows in sector_analyses."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        _write_json(results_dir, "2026-01-01.json", SAMPLE_DATE_1)
        db_path = tmp_path / "test.db"

        count = bsd.build(db_path, results_dir)

        assert count == 2
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT COUNT(*) FROM sector_analyses").fetchone()[0] == 2
        conn.close()

    def test_two_dates_row_count(self, tmp_path):
        """Two JSON files → total rows across both."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        _write_json(results_dir, "2026-01-01.json", SAMPLE_DATE_1)
        _write_json(results_dir, "2026-01-02.json", SAMPLE_DATE_2)
        db_path = tmp_path / "test.db"

        count = bsd.build(db_path, results_dir)

        assert count == 3  # 2 + 1


class TestEntitiesNormalized:
    def test_entities_split_into_rows(self, tmp_path):
        """Entities list → one row per entity in sector_entities."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        _write_json(results_dir, "2026-01-01.json", SAMPLE_DATE_1)
        db_path = tmp_path / "test.db"

        bsd.build(db_path, results_dir)

        conn = sqlite3.connect(db_path)
        # SAMPLE_DATE_1 has 2 + 1 = 3 entities total
        entity_count = conn.execute("SELECT COUNT(*) FROM sector_entities").fetchone()[0]
        entities = {row[0] for row in conn.execute("SELECT entity FROM sector_entities").fetchall()}
        conn.close()

        assert entity_count == 3
        assert entities == {"Apple", "Microsoft", "ExxonMobil"}

    def test_entities_linked_to_analysis(self, tmp_path):
        """sector_entities.analysis_id must reference a real sector_analyses row."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        _write_json(results_dir, "2026-01-01.json", SAMPLE_DATE_1)
        db_path = tmp_path / "test.db"

        bsd.build(db_path, results_dir)

        conn = sqlite3.connect(db_path)
        orphans = conn.execute("""
            SELECT COUNT(*) FROM sector_entities se
            LEFT JOIN sector_analyses sa ON se.analysis_id = sa.id
            WHERE sa.id IS NULL
        """).fetchone()[0]
        conn.close()

        assert orphans == 0


class TestSentimentScore:
    def test_scores_mapped_correctly(self, tmp_path):
        """positive→1, neutral→0, negative→-1 in sentiment_score column."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        _write_json(results_dir, "2026-01-01.json", SAMPLE_DATE_1)
        _write_json(results_dir, "2026-01-02.json", SAMPLE_DATE_2)
        db_path = tmp_path / "test.db"

        bsd.build(db_path, results_dir)

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT sentiment, sentiment_score FROM sector_analyses ORDER BY sentiment"
        ).fetchall()
        conn.close()

        score_map = {sentiment: score for sentiment, score in rows}
        assert score_map["positive"] == 1
        assert score_map["negative"] == -1
        assert score_map["neutral"] == 0


class TestDateFromFilename:
    def test_date_comes_from_filename_not_json_body(self, tmp_path):
        """date column must equal the filename stem, even if JSON body has a different date."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        # JSON body says 2026-01-01 but filename says 2099-12-31
        payload = dict(SAMPLE_DATE_1)
        payload["date"] = "2026-01-01"  # intentionally differs from filename
        _write_json(results_dir, "2099-12-31.json", payload)
        db_path = tmp_path / "test.db"

        bsd.build(db_path, results_dir)

        conn = sqlite3.connect(db_path)
        dates = {row[0] for row in conn.execute("SELECT DISTINCT date FROM sector_analyses").fetchall()}
        conn.close()

        assert "2099-12-31" in dates
        assert "2026-01-01" not in dates


class TestMalformedJsonSkipped:
    def test_bad_file_does_not_abort_build(self, tmp_path, capsys):
        """A malformed JSON file is skipped; valid files are still processed."""
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        (results_dir / "2026-01-00.json").write_text("NOT VALID JSON {{{", encoding="utf-8")
        _write_json(results_dir, "2026-01-01.json", SAMPLE_DATE_1)
        db_path = tmp_path / "test.db"

        count = bsd.build(db_path, results_dir)

        # Valid file still processed
        assert count == 2
        # Error logged to stderr
        captured = capsys.readouterr()
        assert "2026-01-00.json" in captured.err


class TestAtomicWrite:
    def test_original_db_unchanged_on_empty_rebuild(self, tmp_path):
        """
        If a rebuild produces 0 rows (empty dir), the pre-existing DB is
        replaced with the new (empty) one — the atomic write still completes.
        Invariant: build always replaces, never leaves a partial file.
        """
        results_dir = tmp_path / "sector_results"
        results_dir.mkdir()
        db_path = tmp_path / "test.db"

        # First build with data
        _write_json(results_dir, "2026-01-01.json", SAMPLE_DATE_1)
        bsd.build(db_path, results_dir)

        # Remove files, rebuild — should produce a valid (empty) db, not corrupt
        (results_dir / "2026-01-01.json").unlink()
        bsd.build(db_path, results_dir)

        # DB file must still be a valid SQLite file
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM sector_analyses").fetchone()[0]
        conn.close()
        assert count == 0
