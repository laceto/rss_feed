"""
constants.py
Single source of truth for shared taxonomy, scores, and file paths.

Import pattern:
    from constants import SectorName, SECTOR_TAXONOMY, SENTIMENT_SCORE

SectorName  — Pydantic/typing Literal, used in model field annotations
SECTOR_TAXONOMY — plain list[str] derived from SectorName, used for validation
                  and iteration (e.g. query_sector.py, visualize_sentiment.py)

Invariant: SECTOR_TAXONOMY is always derived from SectorName.__args__ —
never edit one without the other, and prefer editing only SectorName.
"""

from pathlib import Path
from typing import Literal

# ── Sector taxonomy ─────────────────────────────────────────────────────────────

SectorName = Literal[
    "Commercial Services",
    "Communications",
    "Consumer Durables",
    "Consumer Non-Durables",
    "Consumer Services",
    "Distribution Services",
    "Electronic Technology",
    "Energy Minerals",
    "Finance",
    "Health Services",
    "Health Technology",
    "Industrial Services",
    "Non-Energy Minerals",
    "Process Industries",
    "Producer Manufacturing",
    "Retail Trade",
    "Technology Services",
    "Transportation",
    "Utilities",
]

# Derived from the Literal — do not maintain separately
SECTOR_TAXONOMY: list[str] = list(SectorName.__args__)

# ── Sentiment ───────────────────────────────────────────────────────────────────

SENTIMENT_SCORE: dict[str, int] = {"positive": 1, "neutral": 0, "negative": -1}

SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#27ae60",
    "neutral":  "#95a5a6",
    "negative": "#c0392b",
}

# ── News categories ─────────────────────────────────────────────────────────────

NEWS_CATEGORIES: list[str] = [
    "earnings", "M&A", "regulation", "macro",
    "appointments", "products", "markets", "other",
]

# ── File paths ──────────────────────────────────────────────────────────────────

RAW_FEED_DIR       = Path("output")
SECTOR_RESULTS_DIR = Path("data") / "sector_results"
SECTOR_SUMMARY_FILE = Path("data") / "sector_summary.tsv"
PENDING_BATCH_FILE  = Path("data") / "pending_sector_batch.txt"
BATCH_FILE          = Path("data") / "batch_tasks_sector.jsonl"
CHARTS_DIR          = Path("data") / "charts"

# ── Bulk export settings ─────────────────────────────────────────────────────
# Rolling window used by export_time_series.py and the export_* functions.
# Change here to update all callers at once.

EXPORT_LOOKBACK_DAYS: int = 90
SECTOR_PIVOT_FILE       = Path("data") / "sector_sentiment_pivot.tsv"
ENTITY_TS_FILE          = Path("data") / "entity_sentiment_ts.tsv"
SECTOR_DB_FILE          = Path("data") / "sector_results.db"

# ── Feed vectorstore ─────────────────────────────────────────────────────────
# FAISS index and registry produced by embed_feeds.py.
# Change VECTORSTORE_DIR here to move the store; embed_feeds.py picks it up.

VECTORSTORE_DIR     = Path("data") / "vectorstore" / "feeds"
FEEDS_REGISTRY_FILE = Path("data") / "vectorstore" / "feeds_registry.tsv"

# ── Topic clustering ──────────────────────────────────────────────────────────
# Parameters for cluster_topics.py. Change here to update all callers.
#
# CLUSTER_WINDOW_DAYS    — rolling window of articles fed to each clustering run.
#                          Empirically tuned: 45d yields ~1,700 articles and 19
#                          coherent clusters with ~60% noise (expected for news).
# CLUSTER_MIN_SIZE       — HDBSCAN min_cluster_size. Below 10, cluster count
#                          explodes (50+); above 20, too few clusters form.
# CLUSTER_MIN_SAMPLES    — HDBSCAN min_samples. Controls cluster conservatism.
# CLUSTER_MAX_NOISE_RATIO — abort threshold. 60-70% noise is normal for news
#                           data; abort only on truly degenerate runs (>90%).
# CLUSTER_MIN_CLUSTERS   — abort threshold for too-few-clusters runs.

CLUSTER_WINDOW_DAYS:     int   = 45
CLUSTER_MIN_SIZE:        int   = 10
CLUSTER_MIN_SAMPLES:     int   = 3
CLUSTER_MAX_NOISE_RATIO: float = 0.90
CLUSTER_MIN_CLUSTERS:    int   = 3

TOPIC_CENTROIDS_FILE = Path("data") / "topic_centroids.json"
TOPIC_LABELS_FILE    = Path("data") / "topic_labels.json"
TOPIC_TRENDS_FILE    = Path("data") / "topic_trends.tsv"
TOPIC_CLUSTERS_DIR   = Path("data") / "topic_clusters"
