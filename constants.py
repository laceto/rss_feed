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
