"""
cluster_topics.py

Daily topic clustering pipeline over the existing FAISS feed vectorstore.

High-level flow (called via __main__ or imported):
  1. extract_window_vectors   — load articles from rolling window
  2. reduce_dimensions        — PCA(50) for noise reduction
  3. run_hdbscan              — cluster; abort on degenerate output
  4. compute_centroids        — mean embedding per cluster
  5. match_topics             — cosine-similarity continuity across runs
  6. get_label                — LLM label (cache-first, LLM only for new topics)
  7a. compute_topic_sentiment — mean sector sentiment per topic (pure join, no API)
  7b. append_trends           — append date × topic × count × sentiment to topic_trends.tsv
  8. get_emerging_topics      — spike ratio + sentiment signal for downstream consumers

Invariants:
  - topic_id is stable once assigned; never changes for the same narrative
  - topic_trends.tsv is append-only; DuplicateDateError on same-date re-run
  - ClusteringAborted is raised (not swallowed) on degenerate clustering;
    callers (CI) should treat this as a non-fatal warning, not an error
  - No article is ever re-embedded; all vectors come from the existing FAISS store

Dependencies: hdbscan, scikit-learn, langchain-community, langchain-openai,
              openai, python-dotenv, pandas, numpy (all in requirements.txt)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import hdbscan
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from constants import (
    CLUSTER_MAX_NOISE_RATIO,
    CLUSTER_MIN_CLUSTERS,
    CLUSTER_MIN_SAMPLES,
    CLUSTER_MIN_SIZE,
    CLUSTER_SELECTION_METHOD,
    CLUSTER_WINDOW_DAYS,
    FEEDS_REGISTRY_FILE,
    SECTOR_SUMMARY_FILE,
    SENTIMENT_SCORE,
    TOPIC_CENTROIDS_FILE,
    TOPIC_CLUSTERS_DIR,
    TOPIC_LABELS_FILE,
    TOPIC_TRENDS_FILE,
    VECTORSTORE_DIR,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ClusteringAborted(RuntimeError):
    """Raised when a clustering run produces degenerate output.

    Callers should log this as a warning and skip writing output for the day,
    rather than crashing the pipeline.
    """


class DuplicateDateError(ValueError):
    """Raised by append_trends when the target date already exists in the TSV."""


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded, cached for process lifetime)
# ---------------------------------------------------------------------------

_store = None
_registry: pd.DataFrame | None = None


def _load_store():
    """Load FAISS vectorstore (singleton — loaded once per process)."""
    global _store
    if _store is None:
        from dotenv import load_dotenv

        load_dotenv()
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings

        _store = FAISS.load_local(
            str(VECTORSTORE_DIR),
            OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536),
            allow_dangerous_deserialization=True,
        )
    return _store


def _load_registry() -> pd.DataFrame:
    """Load feeds_registry.tsv (singleton — loaded once per process)."""
    global _registry
    if _registry is None:
        reg = pd.read_csv(FEEDS_REGISTRY_FILE, sep="\t")
        reg["date"] = pd.to_datetime(reg["date"], errors="coerce")
        _registry = reg.dropna(subset=["date"]).reset_index(drop=True)
    return _registry


# ---------------------------------------------------------------------------
# A2 — Core clustering
# ---------------------------------------------------------------------------


def extract_window_vectors(
    target_date: date,
    window_days: int = CLUSTER_WINDOW_DAYS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load article vectors for the rolling window ending on target_date.

    Args:
        target_date:  The run date (inclusive upper bound).
        window_days:  How many calendar days to look back (default: CLUSTER_WINDOW_DAYS).

    Returns:
        (vectors, metadata) where:
          vectors  — np.ndarray shape (n, 1536), float32
          metadata — pd.DataFrame with columns: id, date, title, link, guid
                     Row i of vectors corresponds to row i of metadata.

    Invariant: vectors.shape[0] == len(metadata)
    """
    store = _load_store()
    registry = _load_registry()

    cutoff = target_date - timedelta(days=window_days)
    mask = (registry["date"].dt.date >= cutoff) & (registry["date"].dt.date <= target_date)
    sub = registry[mask].reset_index(drop=True)

    ids = sub["id"].astype(int).tolist()
    vectors = np.array(
        [store.index.reconstruct(i) for i in ids],
        dtype=np.float32,
    )

    assert vectors.shape[0] == len(sub), "Row count mismatch between vectors and metadata"
    return vectors, sub


def reduce_dimensions(
    vectors: np.ndarray,
    n_components: int = 50,
) -> np.ndarray:
    """Reduce embedding dimensionality via PCA.

    Args:
        vectors:      Input array shape (n, d).
        n_components: Target dimensionality (default 50).

    Returns:
        Array shape (n, n_components). Deterministic (random_state=42).
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(vectors)


def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = CLUSTER_MIN_SIZE,
    min_samples: int = CLUSTER_MIN_SAMPLES,
    max_noise_ratio: float = CLUSTER_MAX_NOISE_RATIO,
    min_clusters: int = CLUSTER_MIN_CLUSTERS,
    cluster_selection_method: str = CLUSTER_SELECTION_METHOD,
) -> tuple[np.ndarray, float]:
    """Cluster reduced vectors with HDBSCAN.

    Args:
        X:                       Reduced vectors, shape (n, d).
        min_cluster_size:        HDBSCAN min_cluster_size.
        min_samples:             HDBSCAN min_samples.
        max_noise_ratio:         Abort if (n_noise / n) > this value.
        min_clusters:            Abort if n_clusters < this value.
        cluster_selection_method: HDBSCAN cluster_selection_method.
                                  'leaf' finds finer-grained clusters (~19 on
                                  this corpus); 'eom' (HDBSCAN default) tends
                                  to over-merge to ~3.

    Returns:
        (labels, noise_ratio) where labels[i] is the integer cluster ID
        for article i (-1 = noise), and noise_ratio = n_noise / n.

    Raises:
        ClusteringAborted: If noise_ratio > max_noise_ratio or
                           n_clusters < min_clusters.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(X)

    n = len(labels)
    noise_count = int((labels == -1).sum())
    noise_ratio = noise_count / n if n > 0 else 1.0
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if noise_ratio > max_noise_ratio:
        raise ClusteringAborted(
            f"noise ratio {noise_ratio:.1%} exceeds abort threshold {max_noise_ratio:.1%}"
        )
    if n_clusters < min_clusters:
        raise ClusteringAborted(
            f"only {n_clusters} cluster(s) formed (minimum: {min_clusters})"
        )

    return labels, noise_ratio


# ---------------------------------------------------------------------------
# A3 — Topic continuity
# ---------------------------------------------------------------------------


def compute_centroids(
    vectors: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute the mean embedding vector for each cluster.

    Args:
        vectors: Array shape (n, d).
        labels:  Integer cluster labels, length n. -1 = noise (excluded).

    Returns:
        dict mapping cluster_id → centroid ndarray of shape (d,).
        Noise label -1 is never a key.
    """
    centroids: dict[int, np.ndarray] = {}
    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        centroids[int(cid)] = vectors[mask].mean(axis=0)
    return centroids


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_topics(
    new_centroids: dict[int, np.ndarray],
    prior_topics: dict[str, dict],
    threshold: float = 0.85,
    run_date: date | None = None,
) -> dict[int, dict]:
    """Assign stable topic_ids to today's clusters by matching to prior centroids.

    Algorithm:
      - Build cosine similarity matrix (new clusters × prior topics).
      - Greedy assignment: highest similarity first; each prior topic can match
        at most one new cluster (no double-assignment).
      - Matched pairs above threshold inherit the prior topic_id.
      - Unmatched new clusters receive a fresh UUID.

    Args:
        new_centroids:  {cluster_int_id: centroid_ndarray}
        prior_topics:   {topic_id_str: centroid_entry_dict}
                        centroid_entry_dict must have "centroid" (list[float])
        threshold:      Minimum cosine similarity to declare a match.
        run_date:       Date of this run (used to set last_seen / first_seen).

    Returns:
        dict mapping cluster_int_id → centroid_entry_dict with fields:
          topic_id, label, centroid (list[float]), first_seen, last_seen
    """
    today_str = str(run_date or date.today())
    new_ids = list(new_centroids.keys())
    prior_ids = list(prior_topics.keys())

    result: dict[int, dict] = {}

    if not prior_ids:
        for cid, vec in new_centroids.items():
            result[cid] = _new_topic_entry(vec, today_str)
        return result

    # Build similarity matrix: rows = new clusters, cols = prior topics
    sim_matrix = np.zeros((len(new_ids), len(prior_ids)), dtype=np.float64)
    prior_vecs = [np.array(prior_topics[pid]["centroid"], dtype=np.float32)
                  for pid in prior_ids]

    for r, cid in enumerate(new_ids):
        for c, pvec in enumerate(prior_vecs):
            sim_matrix[r, c] = _cosine_similarity(new_centroids[cid], pvec)

    # Greedy assignment
    used_prior: set[int] = set()
    used_new:   set[int] = set()
    assignments: dict[int, str] = {}  # new_cluster_idx → prior_topic_id

    # Sort all (similarity, row, col) descending
    candidates = sorted(
        ((sim_matrix[r, c], r, c) for r in range(len(new_ids)) for c in range(len(prior_ids))),
        reverse=True,
    )
    for sim, r, c in candidates:
        if sim < threshold:
            break
        if r in used_new or c in used_prior:
            continue
        assignments[r] = prior_ids[c]
        used_new.add(r)
        used_prior.add(c)

    # Build result
    for r, cid in enumerate(new_ids):
        vec = new_centroids[cid]
        if r in assignments:
            prior_id = assignments[r]
            prior_entry = prior_topics[prior_id]
            result[cid] = {
                "topic_id": prior_id,
                "label": prior_entry.get("label", ""),
                "centroid": vec.tolist(),
                "first_seen": prior_entry["first_seen"],
                "last_seen": today_str,
            }
        else:
            result[cid] = _new_topic_entry(vec, today_str)

    return result


def _new_topic_entry(vec: np.ndarray, today_str: str) -> dict:
    return {
        "topic_id": str(uuid.uuid4()),
        "label": "",
        "centroid": vec.tolist(),
        "first_seen": today_str,
        "last_seen": today_str,
    }


def load_centroids(path: Path | str = TOPIC_CENTROIDS_FILE) -> dict[str, dict]:
    """Load topic centroid map from JSON.

    Returns {} when the file does not exist (first run).
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_centroids(data: dict[str, dict], path: Path | str = TOPIC_CENTROIDS_FILE) -> None:
    """Atomically write topic centroid map to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# A4 — LLM labeling with persistent cache
# ---------------------------------------------------------------------------


def get_label(
    topic_id: str,
    cache: dict[str, str],
    articles: list[str],
    llm_fn: Callable[[list[str]], str] | None = None,
) -> str:
    """Return a human-readable label for topic_id.

    Cache-first: if topic_id is already in cache, return immediately without
    calling the LLM. On a cache miss, call llm_fn(articles) and store the
    result before returning.

    Args:
        topic_id: Stable identifier for the topic.
        cache:    Mutable dict — updated in-place on LLM call.
        articles: List of article title strings (used only on cache miss).
        llm_fn:   Callable(articles) → label_str. Defaults to _label_via_llm.

    Returns:
        Label string (≤ 5 words, plain text).
    """
    if topic_id in cache:
        return cache[topic_id]

    fn = llm_fn or _label_via_llm
    label = fn(articles)
    cache[topic_id] = label
    return label


def _label_via_llm(articles: list[str]) -> str:
    """Call OpenAI to generate a 3–5 word topic label from article titles.

    Samples up to 15 titles. Returns "Unknown topic" on API failure rather
    than crashing — a missing label does not block the pipeline.
    """
    from dotenv import load_dotenv

    load_dotenv()
    from openai import OpenAI

    client = OpenAI()
    sample = articles[:15]
    headlines = "\n".join(f"- {t}" for t in sample)
    prompt = (
        "These are news headlines from the same topic cluster:\n"
        f"{headlines}\n\n"
        "Give a short topic label of 3–5 words. "
        "Be specific (e.g. 'Fed rate pause bets', not 'economic news'). "
        "Return only the label, no explanation."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[cluster_topics] LLM labeling failed: {exc}", flush=True)
        return "Unknown topic"


def load_label_cache(path: Path | str = TOPIC_LABELS_FILE) -> dict[str, str]:
    """Load label cache from JSON. Returns {} when file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_label_cache(cache: dict[str, str], path: Path | str = TOPIC_LABELS_FILE) -> None:
    """Atomically write label cache to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# A5 — Time-series output
# ---------------------------------------------------------------------------

_TRENDS_COLUMNS = ["date", "topic_id", "topic_label", "article_count", "sentiment_score"]


def compute_topic_sentiment(
    cluster_date: str,
    sector_summary_path: Path | str = SECTOR_SUMMARY_FILE,
    topic_clusters_dir: Path | str = TOPIC_CLUSTERS_DIR,
) -> dict[str, float]:
    """Compute the mean sentiment score for each topic cluster on cluster_date.

    Join path:
      topic_clusters/{cluster_date}.json  — article guid → topic_id + article date
      sector_summary.tsv                  — date × sector → sentiment
      _SENTIMENT_SCORE                    — "positive"→+1, "neutral"→0, "negative"→-1

    For each topic: collect all member articles, look up the mean sector sentiment
    on each article's date, then average across all articles in the cluster. Articles
    from dates with no sector data are excluded from the mean (not zeroed).

    Args:
        cluster_date:       Date string YYYY-MM-DD (matches cluster JSON filename).
        sector_summary_path: Path to sector_summary.tsv.
        topic_clusters_dir: Directory containing per-date cluster JSONs.

    Returns:
        {topic_id: mean_sentiment_score} for all non-noise topics in the cluster.
        Returns {} when the cluster file does not exist.

    Invariant:
        - Noise articles (topic_id=None) are excluded.
        - Only articles with matching sector data contribute to the mean.
        - Score range is [-1.0, +1.0].
    """
    cluster_file = Path(topic_clusters_dir) / f"{cluster_date}.json"
    if not cluster_file.exists():
        return {}

    articles = pd.DataFrame(json.loads(cluster_file.read_text(encoding="utf-8")))
    # Drop noise articles
    articles = articles[articles["topic_id"].notna()].copy()
    if articles.empty:
        return {}

    # Normalise article date to YYYY-MM-DD
    articles["article_date"] = articles["date"].astype(str).str[:10]

    # Build day-level mean sentiment score from sector_summary.
    # Direct read rather than pipeline.query_sector._load_summary() because
    # _load_summary is private and uses a hardcoded path (not injectable).
    sector_df = pd.read_csv(Path(sector_summary_path), sep="\t")
    sector_df["sentiment_score"] = sector_df["sentiment"].map(SENTIMENT_SCORE)
    day_scores = (
        sector_df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"date": "article_date", "sentiment_score": "day_score"})
    )

    # Join articles → day scores; articles with no match get NaN
    merged = articles.merge(day_scores, on="article_date", how="left")

    # Aggregate per topic (NaN rows excluded from mean via skipna=True default)
    result: dict[str, float] = {}
    for topic_id, grp in merged.groupby("topic_id"):
        mean_score = grp["day_score"].mean()  # NaN if all articles unmatched
        if not pd.isna(mean_score):
            result[str(topic_id)] = float(mean_score)

    return result


def compute_topic_sentiment_detail(
    cluster_date: str,
    sector_df: pd.DataFrame,
    topic_clusters_dir: Path | str = TOPIC_CLUSTERS_DIR,
) -> pd.DataFrame:
    """Diagnostic variant of compute_topic_sentiment — returns a full DataFrame.

    Same join as compute_topic_sentiment but exposes per-topic match statistics
    for validation and exploration use cases.

    Args:
        cluster_date:       Date string YYYY-MM-DD (matches cluster JSON filename).
        sector_df:          Pre-loaded sector DataFrame with a numeric
                            sentiment_score column (e.g. from pipeline.query_sector.load_summary()).
        topic_clusters_dir: Directory containing per-date cluster JSONs.

    Returns:
        DataFrame with columns: topic_id, n_articles, n_matched, mean_score,
        coverage_pct — sorted by mean_score ascending.
        Returns an empty DataFrame when the cluster file does not exist.
    """
    cluster_file = Path(topic_clusters_dir) / f"{cluster_date}.json"
    if not cluster_file.exists():
        return pd.DataFrame()

    articles = pd.DataFrame(json.loads(cluster_file.read_text(encoding="utf-8")))
    articles = articles[articles["topic_id"].notna()].copy()
    if articles.empty:
        return pd.DataFrame()

    articles["article_date"] = articles["date"].astype(str).str[:10]

    # Day-level mean sentiment: average across all sectors per date.
    # sector_df is injected by the caller so path resolution stays outside this function.
    day_scores = (
        sector_df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"date": "article_date", "sentiment_score": "day_score"})
    )

    merged = articles.merge(day_scores, on="article_date", how="left")

    rows = []
    for topic_id, grp in merged.groupby("topic_id"):
        matched = grp["day_score"].notna().sum()
        rows.append({
            "topic_id":     topic_id,
            "n_articles":   len(grp),
            "n_matched":    int(matched),
            "mean_score":   grp["day_score"].mean(),   # NaN-aware
            "coverage_pct": round(100 * matched / len(grp), 1),
        })

    return pd.DataFrame(rows).sort_values("mean_score")


def append_trends(
    run_date: date,
    rows: list[dict],
    path: Path | str = TOPIC_TRENDS_FILE,
) -> None:
    """Append today's topic counts and sentiment scores to the trends TSV.

    Args:
        run_date: The date these counts belong to.
        rows:     List of dicts with keys: date, topic_id, topic_label,
                  article_count, sentiment_score.
                  sentiment_score may be omitted (written as NaN).
        path:     TSV path (default: TOPIC_TRENDS_FILE).

    Raises:
        DuplicateDateError: If run_date already has rows in the file.

    Invariant:
        - File is created with header if absent.
        - Rows are appended; existing rows are never modified.
        - Write is atomic via temp file + os.replace.
        - Existing rows missing sentiment_score get NaN (backward compatible).
    """
    path = Path(path)
    date_str = str(run_date)

    # Check for duplicate date
    if path.exists():
        existing = pd.read_csv(path, sep="\t", dtype=str)
        if date_str in existing["date"].values:
            raise DuplicateDateError(
                f"topic_trends already contains rows for {date_str}. "
                "Delete them manually before re-running."
            )

    # Build new rows DataFrame — fill missing sentiment_score with NaN
    new_df = pd.DataFrame(rows)
    for col in _TRENDS_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = float("nan")
    new_df = new_df[_TRENDS_COLUMNS]

    if path.exists():
        existing = pd.read_csv(path, sep="\t", dtype={"article_count": int})
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df

    # Atomic write
    tmp = path.with_suffix(".tsv.tmp")
    combined.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# A6 — Signal generation
# ---------------------------------------------------------------------------


def compute_spike(
    topic_id: str,
    trends: pd.DataFrame,
    target_date: date,
    lookback_days: int = 7,
    min_history: int = 3,
) -> float | None:
    """Compute the spike ratio for a topic on target_date.

    spike = today_count / mean(prior lookback_days counts)

    Args:
        topic_id:     Topic identifier.
        trends:       DataFrame with columns: date, topic_id, article_count.
        target_date:  The date to compute the spike for.
        lookback_days: Number of prior days used for the rolling mean.
        min_history:  Minimum prior days required; returns None if below this.

    Returns:
        spike_ratio (float) or None if:
          - topic_id not found in trends
          - fewer than min_history prior observations exist
          - rolling average is 0 (avoids divide-by-zero)
    """
    topic_rows = trends[trends["topic_id"] == topic_id].copy()
    if topic_rows.empty:
        return None

    topic_rows["date"] = pd.to_datetime(topic_rows["date"]).dt.date

    today_rows = topic_rows[topic_rows["date"] == target_date]
    if today_rows.empty:
        return None
    today_count = int(today_rows["article_count"].iloc[0])

    cutoff = target_date - timedelta(days=lookback_days)
    prior = topic_rows[
        (topic_rows["date"] >= cutoff) & (topic_rows["date"] < target_date)
    ]
    if len(prior) < min_history:
        return None

    avg = prior["article_count"].mean()
    if avg == 0:
        return None

    return today_count / avg


def get_emerging_topics(
    target_date: date,
    trends: pd.DataFrame,
    min_article_count: int = 5,
    spike_lookback: int = 7,
) -> list[dict]:
    """Return topics with the highest spike ratios on target_date.

    Filters out:
      - Topics with article_count < min_article_count on target_date.
      - Topics with insufficient history (compute_spike returns None).

    Returns:
        List of dicts sorted by spike_ratio descending, each with:
          topic_id, label, spike_ratio, article_count, sentiment_score
        sentiment_score is None when the column is absent from trends
        (backward compatible with pre-sentiment TSV files).
    """
    date_str = str(target_date)
    today_rows = trends[trends["date"] == date_str]
    has_sentiment = "sentiment_score" in trends.columns

    results = []
    for _, row in today_rows.iterrows():
        if int(row["article_count"]) < min_article_count:
            continue
        ratio = compute_spike(
            row["topic_id"], trends, target_date, lookback_days=spike_lookback
        )
        if ratio is None:
            continue

        score = None
        if has_sentiment:
            raw = row["sentiment_score"]
            score = None if pd.isna(raw) else float(raw)

        results.append({
            "topic_id": row["topic_id"],
            "label": row["topic_label"],
            "spike_ratio": round(ratio, 3),
            "article_count": int(row["article_count"]),
            "sentiment_score": score,
        })

    return sorted(results, key=lambda r: r["spike_ratio"], reverse=True)


# ---------------------------------------------------------------------------
# A7 — Full pipeline run (called by __main__ and CI)
# ---------------------------------------------------------------------------


def run(
    target_date: date | None = None,
    window_days: int = CLUSTER_WINDOW_DAYS,
    centroids_path: Path = TOPIC_CENTROIDS_FILE,
    labels_path: Path = TOPIC_LABELS_FILE,
    trends_path: Path = TOPIC_TRENDS_FILE,
    clusters_dir: Path = TOPIC_CLUSTERS_DIR,
    sector_summary_path: Path = SECTOR_SUMMARY_FILE,
    skip_labeling: bool = False,
) -> dict[str, Any]:
    """Execute the full clustering pipeline for one day.

    Returns a summary dict with quality metrics (written to stdout by __main__).
    Raises ClusteringAborted on degenerate runs — callers decide whether to
    treat this as a warning or error.
    """
    if target_date is None:
        target_date = date.today()

    # Step 1-2: vectors + dim reduction
    vectors, meta = extract_window_vectors(target_date, window_days=window_days)
    X = reduce_dimensions(vectors)

    # Step 3: cluster
    labels, noise_ratio = run_hdbscan(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Step 4: centroids
    centroids = compute_centroids(vectors, labels)

    # Step 5: continuity matching
    prior_topics = load_centroids(centroids_path)
    topic_map = match_topics(centroids, prior_topics, run_date=target_date)
    # topic_map: {cluster_int_id → centroid_entry}

    # Step 6: labeling
    label_cache = load_label_cache(labels_path)
    new_labels_count = 0
    for cid, entry in topic_map.items():
        tid = entry["topic_id"]
        if tid not in label_cache and not skip_labeling:
            # Sample top articles (by HDBSCAN membership probability or just index)
            member_mask = labels == cid
            member_titles = meta[member_mask]["title"].head(15).tolist()
            get_label(tid, label_cache, articles=member_titles)
            new_labels_count += 1
        elif tid in label_cache:
            entry["label"] = label_cache[tid]

    # Update centroids with labels
    updated_centroids: dict[str, dict] = {}
    for cid, entry in topic_map.items():
        tid = entry["topic_id"]
        entry["label"] = label_cache.get(tid, entry.get("label", ""))
        updated_centroids[tid] = entry

    # Write persistent state
    save_centroids(updated_centroids, centroids_path)
    save_label_cache(label_cache, labels_path)

    # Write per-date cluster assignment
    clusters_dir = Path(clusters_dir)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    cluster_output = meta[["id", "guid", "date", "title"]].copy()
    label_series = pd.Series(labels, index=meta.index, name="cluster_id")
    cluster_output = cluster_output.join(label_series)
    # Map cluster_id → topic_id
    cid_to_tid = {cid: entry["topic_id"] for cid, entry in topic_map.items()}
    cluster_output["topic_id"] = cluster_output["cluster_id"].map(cid_to_tid)
    cluster_output.to_json(
        clusters_dir / f"{target_date}.json", orient="records", date_format="iso"
    )

    # Step 7a: topic sentiment (pure join — no API calls; runs after cluster JSON is written)
    topic_sentiment = compute_topic_sentiment(
        str(target_date),
        sector_summary_path=sector_summary_path,
        topic_clusters_dir=clusters_dir,
    )

    # Step 7b: trends
    trend_rows = []
    for cid, entry in topic_map.items():
        member_count = int((labels == cid).sum())
        tid = entry["topic_id"]
        trend_rows.append({
            "date": str(target_date),
            "topic_id": tid,
            "topic_label": label_cache.get(tid, entry.get("label", "")),
            "article_count": member_count,
            "sentiment_score": topic_sentiment.get(tid),  # None → NaN in TSV
        })

    try:
        append_trends(target_date, trend_rows, trends_path)
    except DuplicateDateError as exc:
        print(f"[cluster_topics] WARNING: {exc}", flush=True)

    n_with_sentiment = sum(1 for v in topic_sentiment.values() if v is not None)
    summary = {
        "date": str(target_date),
        "window_articles": len(meta),
        "n_clusters": n_clusters,
        "noise_ratio": round(noise_ratio, 3),
        "new_labels": new_labels_count,
        "matched_topics": sum(1 for e in topic_map.values()
                              if e["topic_id"] in prior_topics),
        "topics_with_sentiment": n_with_sentiment,
        "cluster_sizes": sorted(
            [int((labels == cid).sum()) for cid in topic_map], reverse=True
        ),
    }
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run topic clustering for one day.")
    parser.add_argument(
        "--date",
        default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        help="Skip LLM labeling (useful for profiling / CI dry runs)",
    )
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    try:
        summary = run(target_date=target, skip_labeling=args.skip_labeling)
    except ClusteringAborted as exc:
        print(f"[cluster_topics] ABORTED: {exc}", flush=True)
        sys.exit(2)

    print("[cluster_topics] Run complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
