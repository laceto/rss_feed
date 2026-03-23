"""
tests/test_cluster_topics.py

Unit tests for cluster_topics.py.
TDD order: Red (this file) → Green (implement) → Refactor.

All tests use synthetic data — no FAISS, no OpenAI, no filesystem I/O
unless explicitly testing I/O functions.

Coverage:
  A2 — extract_window_vectors, reduce_dimensions, run_hdbscan, ClusteringAborted
  A3 — compute_centroids, match_topics, load/save centroids
  A4 — get_label, label cache I/O
  A5 — append_trends, DuplicateDateError
  A6 — compute_spike, get_emerging_topics
"""

import json
import os
import tempfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vectors(n: int, dim: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _make_registry(n: int, reference_date: date | None = None) -> pd.DataFrame:
    ref = reference_date or date(2026, 3, 1)
    dates = [ref - timedelta(days=i % 30) for i in range(n)]
    return pd.DataFrame({
        "id": list(range(n)),
        "date": pd.to_datetime(dates),
        "title": [f"Article {i}" for i in range(n)],
        "link": [f"http://example.com/{i}" for i in range(n)],
        "guid": [f"guid-{i}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# A2.1 / A2.2 — extract_window_vectors
# ---------------------------------------------------------------------------

class TestExtractWindowVectors:
    def test_returns_tuple_of_ndarray_and_dataframe(self, tmp_path, monkeypatch):
        """extract_window_vectors returns (np.ndarray, pd.DataFrame) for valid window."""
        from cluster_topics import extract_window_vectors

        n = 100
        registry = _make_registry(n, reference_date=date(2026, 3, 1))
        vectors_all = _make_vectors(n, dim=1536)

        # Patch the FAISS store and registry loading
        mock_store = MagicMock()
        mock_store.index.ntotal = n
        mock_store.index.reconstruct.side_effect = lambda i: vectors_all[i]

        monkeypatch.setattr("cluster_topics._load_store", lambda: mock_store)
        monkeypatch.setattr("cluster_topics._load_registry", lambda: registry)

        vectors, meta = extract_window_vectors(date(2026, 3, 1), window_days=30)

        assert isinstance(vectors, np.ndarray)
        assert isinstance(meta, pd.DataFrame)
        assert vectors.shape[0] == meta.shape[0]
        assert vectors.shape[1] == 1536

    def test_filters_to_window(self, tmp_path, monkeypatch):
        """Only articles within window_days of target_date are returned."""
        from cluster_topics import extract_window_vectors

        # 60 articles: 30 inside window, 30 outside
        ref = date(2026, 3, 1)
        dates_in  = [ref - timedelta(days=i) for i in range(30)]   # last 30 days
        dates_out = [ref - timedelta(days=60 + i) for i in range(30)]  # >30 days ago
        all_dates = dates_in + dates_out

        registry = pd.DataFrame({
            "id": list(range(60)),
            "date": pd.to_datetime(all_dates),
            "title": [f"Art {i}" for i in range(60)],
            "link": [""] * 60,
            "guid": [f"g{i}" for i in range(60)],
        })
        vectors_all = _make_vectors(60, dim=1536)

        mock_store = MagicMock()
        mock_store.index.ntotal = 60
        mock_store.index.reconstruct.side_effect = lambda i: vectors_all[i]

        monkeypatch.setattr("cluster_topics._load_store", lambda: mock_store)
        monkeypatch.setattr("cluster_topics._load_registry", lambda: registry)

        vectors, meta = extract_window_vectors(ref, window_days=30)

        assert len(meta) == 30
        assert all(pd.to_datetime(meta["date"]).dt.date >= ref - timedelta(days=30))

    def test_row_counts_match(self, monkeypatch):
        """vectors.shape[0] == len(meta) — always."""
        from cluster_topics import extract_window_vectors

        n = 50
        registry = _make_registry(n, reference_date=date(2026, 3, 1))
        vectors_all = _make_vectors(n, dim=1536)

        mock_store = MagicMock()
        mock_store.index.ntotal = n
        mock_store.index.reconstruct.side_effect = lambda i: vectors_all[i]

        monkeypatch.setattr("cluster_topics._load_store", lambda: mock_store)
        monkeypatch.setattr("cluster_topics._load_registry", lambda: registry)

        vectors, meta = extract_window_vectors(date(2026, 3, 1), window_days=45)
        assert vectors.shape[0] == len(meta)


# ---------------------------------------------------------------------------
# A2.3 / A2.4 — reduce_dimensions
# ---------------------------------------------------------------------------

class TestReduceDimensions:
    def test_output_shape(self):
        """reduce_dimensions returns (n, n_components) array."""
        from cluster_topics import reduce_dimensions

        X = _make_vectors(200, dim=1536)
        result = reduce_dimensions(X, n_components=50)
        assert result.shape == (200, 50)

    def test_deterministic(self):
        """Same input → same output (fixed random_state)."""
        from cluster_topics import reduce_dimensions

        X = _make_vectors(100, dim=1536)
        r1 = reduce_dimensions(X, n_components=50)
        r2 = reduce_dimensions(X, n_components=50)
        np.testing.assert_array_equal(r1, r2)

    def test_default_n_components(self):
        """Default n_components=50."""
        from cluster_topics import reduce_dimensions

        X = _make_vectors(100, dim=1536)
        result = reduce_dimensions(X)
        assert result.shape[1] == 50


# ---------------------------------------------------------------------------
# A2.5 / A2.6 / A2.7 — run_hdbscan + ClusteringAborted
# ---------------------------------------------------------------------------

class TestRunHdbscan:
    def test_returns_labels_correct_length(self):
        """run_hdbscan returns integer labels array of length n."""
        from cluster_topics import run_hdbscan

        # Build tight clusters so HDBSCAN finds something
        rng = np.random.default_rng(42)
        centers = rng.standard_normal((5, 50))
        X = np.vstack([
            centers[i] + rng.standard_normal((40, 50)) * 0.1
            for i in range(5)
        ]).astype(np.float32)

        labels, noise_ratio = run_hdbscan(X, min_cluster_size=10, min_samples=3)
        assert len(labels) == len(X)
        assert labels.dtype in (np.int32, np.int64, int, np.intp)

    def test_noise_ratio_in_range(self):
        """noise_ratio is between 0.0 and 1.0 for clusterable data."""
        from cluster_topics import run_hdbscan

        # Build tight clusters — HDBSCAN should find them without aborting
        rng = np.random.default_rng(42)
        centers = rng.standard_normal((5, 50))
        X = np.vstack([
            centers[i] + rng.standard_normal((40, 50)) * 0.05
            for i in range(5)
        ]).astype(np.float32)

        _, noise_ratio = run_hdbscan(X, min_cluster_size=10, min_samples=3)
        assert 0.0 <= noise_ratio <= 1.0

    def test_raises_clustering_aborted_on_all_noise(self):
        """ClusteringAborted raised when noise_ratio > max_noise_ratio."""
        from cluster_topics import run_hdbscan, ClusteringAborted

        # Uniform random data → HDBSCAN produces mostly noise
        rng = np.random.default_rng(99)
        X = rng.uniform(-100, 100, size=(300, 50)).astype(np.float32)

        # Force abort by setting very low threshold
        with pytest.raises(ClusteringAborted, match="noise"):
            run_hdbscan(X, min_cluster_size=10, min_samples=3, max_noise_ratio=0.01)

    def test_raises_clustering_aborted_on_zero_clusters(self):
        """ClusteringAborted raised when n_clusters < min_clusters threshold.

        Note: zero clusters implies 100% noise, so either the noise check or
        the cluster check will fire. We test that ClusteringAborted is raised;
        the specific message is not asserted since ordering is an implementation detail.
        """
        from cluster_topics import run_hdbscan, ClusteringAborted

        # 5 articles, min_cluster_size=100 → HDBSCAN labels all as noise
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, 50)).astype(np.float32)

        with pytest.raises(ClusteringAborted):
            run_hdbscan(X, min_cluster_size=100, min_samples=50)


# ---------------------------------------------------------------------------
# A3.1 / A3.2 — compute_centroids
# ---------------------------------------------------------------------------

class TestComputeCentroids:
    def test_returns_dict_of_arrays(self):
        """compute_centroids returns dict[int, np.ndarray]."""
        from cluster_topics import compute_centroids

        vectors = _make_vectors(60, dim=50)
        labels = np.array([0] * 20 + [1] * 20 + [-1] * 20)
        result = compute_centroids(vectors, labels)

        assert isinstance(result, dict)
        assert 0 in result and 1 in result
        assert result[0].shape == (50,)

    def test_excludes_noise(self):
        """Noise label -1 is never a key in the result."""
        from cluster_topics import compute_centroids

        vectors = _make_vectors(30, dim=50)
        labels = np.array([0] * 10 + [-1] * 20)
        result = compute_centroids(vectors, labels)

        assert -1 not in result
        assert 0 in result

    def test_centroid_is_mean(self):
        """Centroid equals mean of member vectors."""
        from cluster_topics import compute_centroids

        vectors = np.array([
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],  # label -1
        ], dtype=np.float32)
        labels = np.array([0, 0, -1])
        result = compute_centroids(vectors, labels)

        expected = np.array([2.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(result[0], expected)

    def test_empty_labels(self):
        """All-noise input returns empty dict."""
        from cluster_topics import compute_centroids

        vectors = _make_vectors(10, dim=50)
        labels = np.full(10, -1)
        result = compute_centroids(vectors, labels)
        assert result == {}


# ---------------------------------------------------------------------------
# A3.3 / A3.4 — match_topics
# ---------------------------------------------------------------------------

class TestMatchTopics:
    def _make_centroid_entry(self, vec: np.ndarray, topic_id: str,
                              label: str = "test", first_seen: str = "2026-01-01") -> dict:
        return {
            "topic_id": topic_id,
            "label": label,
            "centroid": vec.tolist(),
            "first_seen": first_seen,
            "last_seen": first_seen,
        }

    def test_high_similarity_reuses_topic_id(self):
        """Near-identical centroids should be matched → same topic_id."""
        from cluster_topics import match_topics

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        prior = {"t001": self._make_centroid_entry(vec, "t001", first_seen="2026-01-01")}

        # Slightly perturbed — still above 0.85 cosine similarity
        new_vec = vec + np.array([0.01, 0.0, 0.0], dtype=np.float32)
        new_centroids = {0: new_vec}

        result = match_topics(new_centroids, prior, threshold=0.85,
                               run_date=date(2026, 3, 1))
        assert result[0]["topic_id"] == "t001"

    def test_orthogonal_centroid_gets_new_topic_id(self):
        """Orthogonal centroid (similarity=0) must get a new UUID."""
        from cluster_topics import match_topics

        vec_prior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_new   = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        prior = {"t001": self._make_centroid_entry(vec_prior, "t001")}

        result = match_topics({0: vec_new}, prior, threshold=0.85,
                               run_date=date(2026, 3, 1))
        assert result[0]["topic_id"] != "t001"
        assert len(result[0]["topic_id"]) > 0

    def test_no_double_assignment(self):
        """One prior topic cannot match two new clusters."""
        from cluster_topics import match_topics

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        prior = {"t001": self._make_centroid_entry(vec, "t001")}

        # Two nearly-identical new clusters competing for same prior
        c0 = vec + np.array([0.01, 0.0, 0.0], dtype=np.float32)
        c1 = vec + np.array([0.02, 0.0, 0.0], dtype=np.float32)

        result = match_topics({0: c0, 1: c1}, prior, threshold=0.85,
                               run_date=date(2026, 3, 1))
        ids = [result[k]["topic_id"] for k in result]
        assert len(set(ids)) == 2  # both get distinct IDs; one inherits, one is new

    def test_empty_prior_all_new(self):
        """With no prior topics, all new clusters get fresh UUIDs."""
        from cluster_topics import match_topics

        new_centroids = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 1.0], dtype=np.float32),
        }
        result = match_topics(new_centroids, {}, threshold=0.85,
                               run_date=date(2026, 3, 1))
        assert len(result) == 2
        ids = [result[k]["topic_id"] for k in result]
        assert len(set(ids)) == 2


# ---------------------------------------------------------------------------
# A3.5 / A3.6 — load/save centroids
# ---------------------------------------------------------------------------

class TestCentroidsIO:
    def test_load_returns_empty_dict_when_absent(self, tmp_path):
        """load_centroids() returns {} when file does not exist."""
        from cluster_topics import load_centroids

        result = load_centroids(tmp_path / "nonexistent.json")
        assert result == {}

    def test_roundtrip(self, tmp_path):
        """save_centroids + load_centroids preserves all fields."""
        from cluster_topics import save_centroids, load_centroids

        data = {
            "t001": {
                "topic_id": "t001",
                "label": "Iran oil geopolitics",
                "centroid": [0.1, 0.2, 0.3],
                "first_seen": "2026-01-15",
                "last_seen": "2026-03-01",
            }
        }
        path = tmp_path / "centroids.json"
        save_centroids(data, path)
        loaded = load_centroids(path)

        assert loaded == data


# ---------------------------------------------------------------------------
# A4.1 / A4.2 / A4.3 — get_label (cache-first, LLM on miss)
# ---------------------------------------------------------------------------

class TestGetLabel:
    def test_returns_cached_label_without_llm(self):
        """get_label returns cached value and does not call LLM."""
        from cluster_topics import get_label

        cache = {"t001": "Iran oil geopolitics"}
        mock_llm = MagicMock()

        result = get_label("t001", cache, articles=[], llm_fn=mock_llm)

        assert result == "Iran oil geopolitics"
        mock_llm.assert_not_called()

    def test_calls_llm_on_cache_miss_and_stores(self):
        """get_label calls llm_fn for unknown topic and stores result in cache."""
        from cluster_topics import get_label

        cache = {}
        mock_llm = MagicMock(return_value="Fed rate pause bets")
        articles = ["Fed holds rates steady", "Powell signals no hike"]

        result = get_label("t999", cache, articles=articles, llm_fn=mock_llm)

        assert result == "Fed rate pause bets"
        assert cache["t999"] == "Fed rate pause bets"
        mock_llm.assert_called_once()

    def test_llm_receives_article_titles(self):
        """LLM function receives the article list."""
        from cluster_topics import get_label

        cache = {}
        received = []

        def capture_llm(articles):
            received.extend(articles)
            return "test label"

        articles = ["Title A", "Title B"]
        get_label("t000", cache, articles=articles, llm_fn=capture_llm)

        assert received == articles


# ---------------------------------------------------------------------------
# A4.5 — label cache I/O
# ---------------------------------------------------------------------------

class TestLabelCacheIO:
    def test_load_returns_empty_dict_when_absent(self, tmp_path):
        from cluster_topics import load_label_cache

        result = load_label_cache(tmp_path / "labels.json")
        assert result == {}

    def test_roundtrip(self, tmp_path):
        from cluster_topics import save_label_cache, load_label_cache

        data = {"t001": "Iran oil geopolitics", "t002": "Fed rate pause bets"}
        path = tmp_path / "labels.json"
        save_label_cache(data, path)
        assert load_label_cache(path) == data


# ---------------------------------------------------------------------------
# A5.1 / A5.2 / A5.3 — append_trends + DuplicateDateError
# ---------------------------------------------------------------------------

class TestAppendTrends:
    def _make_rows(self, dt: date) -> list[dict]:
        return [
            {"date": str(dt), "topic_id": "t001", "topic_label": "Iran oil", "article_count": 12},
            {"date": str(dt), "topic_id": "t002", "topic_label": "Fed pause", "article_count": 8},
        ]

    def test_creates_file_with_header_if_absent(self, tmp_path):
        """append_trends creates TSV with header when file doesn't exist."""
        from cluster_topics import append_trends

        path = tmp_path / "trends.tsv"
        append_trends(date(2026, 3, 1), self._make_rows(date(2026, 3, 1)), path)

        df = pd.read_csv(path, sep="\t")
        assert list(df.columns) == ["date", "topic_id", "topic_label", "article_count"]
        assert len(df) == 2

    def test_appends_without_duplicate_header(self, tmp_path):
        """Second call appends rows; no extra header row inserted."""
        from cluster_topics import append_trends

        path = tmp_path / "trends.tsv"
        append_trends(date(2026, 3, 1), self._make_rows(date(2026, 3, 1)), path)
        append_trends(date(2026, 3, 2), self._make_rows(date(2026, 3, 2)), path)

        df = pd.read_csv(path, sep="\t")
        assert len(df) == 4
        # Ensure no row has "date" as value (which would be a stray header)
        assert not (df["date"] == "date").any()

    def test_raises_duplicate_date_error(self, tmp_path):
        """Calling append_trends twice with same date raises DuplicateDateError."""
        from cluster_topics import append_trends, DuplicateDateError

        path = tmp_path / "trends.tsv"
        append_trends(date(2026, 3, 1), self._make_rows(date(2026, 3, 1)), path)

        with pytest.raises(DuplicateDateError):
            append_trends(date(2026, 3, 1), self._make_rows(date(2026, 3, 1)), path)

    def test_atomic_write(self, tmp_path):
        """Verify that the file is written atomically (no temp file left behind)."""
        from cluster_topics import append_trends

        path = tmp_path / "trends.tsv"
        append_trends(date(2026, 3, 1), self._make_rows(date(2026, 3, 1)), path)

        tmp_file = path.with_suffix(".tsv.tmp")
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# A6.1 / A6.2 — compute_spike
# ---------------------------------------------------------------------------

class TestComputeSpike:
    def _make_trends(self, topic_id: str, counts: dict[date, int]) -> pd.DataFrame:
        """Build a minimal trends DataFrame for a single topic."""
        rows = [
            {"date": str(d), "topic_id": topic_id, "topic_label": "test", "article_count": c}
            for d, c in counts.items()
        ]
        return pd.DataFrame(rows)

    def test_correct_spike_ratio(self):
        """compute_spike returns today / mean(last 7 days excluding today)."""
        from cluster_topics import compute_spike

        ref = date(2026, 3, 10)
        counts = {ref - timedelta(days=i): 10 for i in range(1, 8)}  # avg=10
        counts[ref] = 40  # today: spike of 4x

        trends = self._make_trends("t001", counts)
        ratio = compute_spike("t001", trends, ref)

        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_returns_none_with_insufficient_history(self):
        """Returns None when fewer than 3 days of history exist."""
        from cluster_topics import compute_spike

        ref = date(2026, 3, 10)
        counts = {ref - timedelta(days=1): 5, ref: 20}  # only 1 prior day
        trends = self._make_trends("t001", counts)

        assert compute_spike("t001", trends, ref) is None

    def test_returns_none_for_unknown_topic(self):
        """Returns None for a topic_id not in trends."""
        from cluster_topics import compute_spike

        trends = pd.DataFrame(columns=["date", "topic_id", "topic_label", "article_count"])
        assert compute_spike("t_nonexistent", trends, date(2026, 3, 1)) is None

    def test_zero_average_returns_none(self):
        """Returns None when rolling average is 0 (avoids divide-by-zero)."""
        from cluster_topics import compute_spike

        ref = date(2026, 3, 10)
        counts = {ref - timedelta(days=i): 0 for i in range(1, 8)}
        counts[ref] = 5
        trends = self._make_trends("t001", counts)

        assert compute_spike("t001", trends, ref) is None


# ---------------------------------------------------------------------------
# A6.5 — get_emerging_topics
# ---------------------------------------------------------------------------

class TestGetEmergingTopics:
    def _make_trends_df(self) -> pd.DataFrame:
        ref = date(2026, 3, 10)
        rows = []
        # t001: spike (40 today vs avg 10) → ratio 4.0
        for i in range(1, 8):
            rows.append({"date": str(ref - timedelta(days=i)), "topic_id": "t001",
                          "topic_label": "Iran oil", "article_count": 10})
        rows.append({"date": str(ref), "topic_id": "t001",
                     "topic_label": "Iran oil", "article_count": 40})
        # t002: stable (10 today vs avg 10) → ratio 1.0
        for i in range(1, 8):
            rows.append({"date": str(ref - timedelta(days=i)), "topic_id": "t002",
                          "topic_label": "Fed pause", "article_count": 10})
        rows.append({"date": str(ref), "topic_id": "t002",
                     "topic_label": "Fed pause", "article_count": 10})
        # t003: today count < 5 → filtered
        rows.append({"date": str(ref), "topic_id": "t003",
                     "topic_label": "Tiny cluster", "article_count": 3})
        return pd.DataFrame(rows)

    def test_sorted_by_spike_ratio_descending(self):
        """Results sorted highest spike_ratio first."""
        from cluster_topics import get_emerging_topics

        trends = self._make_trends_df()
        results = get_emerging_topics(date(2026, 3, 10), trends)

        ratios = [r["spike_ratio"] for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_filters_tiny_clusters(self):
        """Topics with article_count < 5 on target date are excluded."""
        from cluster_topics import get_emerging_topics

        trends = self._make_trends_df()
        results = get_emerging_topics(date(2026, 3, 10), trends)

        ids = [r["topic_id"] for r in results]
        assert "t003" not in ids

    def test_result_keys(self):
        """Each result dict has required keys."""
        from cluster_topics import get_emerging_topics

        trends = self._make_trends_df()
        results = get_emerging_topics(date(2026, 3, 10), trends)

        for r in results:
            assert "topic_id" in r
            assert "label" in r
            assert "spike_ratio" in r
