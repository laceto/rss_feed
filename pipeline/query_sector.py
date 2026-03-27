"""Sector query API — snapshot, time-series, and bulk pivot utilities.

Extracted from query_sector.py.

The shared helpers _trend_direction and _load_articles_for_dates live here
and are imported by pipeline.query_entity to avoid duplication.

Two public query functions:

    get_snapshot(sector)
        Latest available signal for a sector — single day, fast.

    get_time_series(sector, lookback_days, include_articles)
        Trend over a rolling window — entity analysis, category breakdown,
        optional raw article headlines.

Both raise ValueError (bad sector name) or LookupError (no data found).
"""

from __future__ import annotations

import warnings
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from constants import (
    EXPORT_LOOKBACK_DAYS,
    RAW_FEED_DIR,
    SECTOR_PIVOT_FILE,
    SECTOR_SUMMARY_FILE,
    SECTOR_TAXONOMY,
    SENTIMENT_SCORE,
)

# ── Trend threshold ─────────────────────────────────────────────────────────────

_TREND_THRESHOLD = 0.20  # minimum delta between halves to call a trend


# ── Private helpers — also used by pipeline.query_entity ──────────────────────


def _trend_direction(scores: pd.Series) -> tuple[str, float]:
    """Return (direction, delta) based on first-half vs second-half mean.

    direction: "improving" | "deteriorating" | "stable"
    delta: second_half_mean - first_half_mean  (positive = improving)

    Shared by pipeline.query_entity to ensure identical trend logic.
    """
    if len(scores) < 2:
        return "stable", 0.0
    mid = len(scores) // 2
    first_half = scores.iloc[:mid].mean()
    second_half = scores.iloc[mid:].mean()
    delta = round(float(second_half - first_half), 4)
    if delta > _TREND_THRESHOLD:
        return "improving", delta
    if delta < -_TREND_THRESHOLD:
        return "deteriorating", delta
    return "stable", delta


def _load_articles_for_dates(dates: list[str]) -> list[dict]:
    """Load raw feed articles for the given dates from output/feeds{date}.txt.

    Returns a flat list of article dicts. Dates with missing files are skipped
    with a warning — missing files don't invalidate the sector signal.

    Shared by pipeline.query_entity to ensure identical article-loading logic.
    """
    articles = []
    for d in sorted(dates):
        feed_file = RAW_FEED_DIR / f"feeds{d}.txt"
        if not feed_file.exists():
            warnings.warn(f"Raw feed not found for {d}: {feed_file}", stacklevel=4)
            continue
        try:
            df = pd.read_csv(feed_file, sep="\t")
            for _, row in df.iterrows():
                articles.append({
                    "date":        d,
                    "title":       row.get("title", ""),
                    "description": row.get("description", ""),
                    "link":        row.get("link", ""),
                })
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not read {feed_file}: {exc}", stacklevel=4)
    return articles


def _validate_sector(name: str) -> None:
    """Raise ValueError with valid options if name is not in the taxonomy."""
    if name not in SECTOR_TAXONOMY:
        valid = ", ".join(sorted(SECTOR_TAXONOMY))
        raise ValueError(
            f"Unknown sector '{name}'.\nValid values: {valid}"
        )


def _load_summary() -> pd.DataFrame:
    """Load sector_summary.tsv and prepare numeric sentiment_score column.

    Returns an empty DataFrame (with correct columns) if the file is missing
    rather than raising — callers handle the empty case explicitly.
    """
    if not SECTOR_SUMMARY_FILE.exists():
        warnings.warn(
            f"{SECTOR_SUMMARY_FILE} not found. "
            "Run read_sector_results.py to generate it.",
            stacklevel=3,
        )
        return pd.DataFrame(
            columns=["date", "sector", "entities", "sentiment",
                     "news_category", "extraction_status", "sentiment_score"]
        )

    df = pd.read_csv(SECTOR_SUMMARY_FILE, sep="\t", parse_dates=["date"])
    df["sentiment_score"] = df["sentiment"].map(SENTIMENT_SCORE)
    return df


def load_summary() -> pd.DataFrame:
    """Load sector_summary.tsv with a numeric sentiment_score column.

    Public wrapper around _load_summary() for callers that need the raw
    sector DataFrame without the full query API (e.g. validation scripts,
    batch submission steps that only need the data, not the query layer).

    Returns:
        DataFrame with columns: date, sector, entities, sentiment,
        news_category, extraction_status, sentiment_score.
        Returns an empty DataFrame (with correct columns) if the file is missing.
    """
    return _load_summary()


def _split_entities(pipe_str: str) -> list[str]:
    """Split a pipe-separated entity string into a clean list."""
    if not isinstance(pipe_str, str) or not pipe_str.strip():
        return []
    return [e.strip() for e in pipe_str.split("|") if e.strip()]


def _top_entities(series: pd.Series, n: int = 10) -> list[str]:
    """Return the top-n entities by mention frequency across a Series of pipe-strings."""
    counter: Counter = Counter()
    for val in series.dropna():
        counter.update(_split_entities(val))
    return [entity for entity, _ in counter.most_common(n)]


# ── Public API ────────────────────────────────────────────────────────────────


def list_sectors() -> list[str]:
    """Return the full list of valid sector names (sorted alphabetically)."""
    return sorted(SECTOR_TAXONOMY)


def get_snapshot(sector: str) -> dict:
    """Return the latest available signal for a sector.

    Reads the most recent row in sector_summary.tsv for the given sector.
    Suitable for "current read" queries — fast single-row lookup.

    Args:
        sector: Must exactly match a value in constants.SECTOR_TAXONOMY.

    Returns:
        dict with keys: sector, last_date, latest_sentiment, sentiment_score,
        entities, news_category, extraction_status, data_age_days.

    Raises:
        ValueError: If sector name is not in the taxonomy.
        LookupError: If sector exists in the taxonomy but has no data yet.
    """
    _validate_sector(sector)
    df = _load_summary()

    rows = df[df["sector"] == sector].sort_values("date")
    if rows.empty:
        raise LookupError(
            f"No data found for sector '{sector}'. "
            "The pipeline may not have seen this sector in the feed yet."
        )

    latest = rows.iloc[-1]
    last_date_val: date = (
        latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]
    )
    age_days = (date.today() - last_date_val).days

    return {
        "sector":            sector,
        "last_date":         str(last_date_val),
        "latest_sentiment":  latest["sentiment"],
        "sentiment_score":   int(latest["sentiment_score"]),
        "entities":          _split_entities(latest.get("entities", "")),
        "news_category":     latest.get("news_category", ""),
        "extraction_status": latest.get("extraction_status", ""),
        "data_age_days":     age_days,
    }


def get_time_series(
    sector: str,
    lookback_days: int = 30,
    include_articles: bool = False,
) -> dict:
    """Return sentiment trend and entity analysis for a sector over a rolling window.

    Args:
        sector:           Must exactly match a value in constants.SECTOR_TAXONOMY.
        lookback_days:    Number of calendar days to look back from today.
        include_articles: If True, also fetch raw article headlines/descriptions
                          from output/feeds{date}.txt for each date in the window.

    Returns:
        dict with keys: sector, lookback_days, date_range, n_observations,
        mean_sentiment_score, trend_direction, trend_delta, dominant_sentiment,
        top_entities, category_breakdown, time_series, articles.

    Raises:
        ValueError: If sector name is not in the taxonomy.
        LookupError: If sector has no data within the lookback window.
    """
    _validate_sector(sector)
    df = _load_summary()

    cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))
    rows = (
        df[(df["sector"] == sector) & (df["date"] >= cutoff)]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if rows.empty:
        raise LookupError(
            f"No data for sector '{sector}' in the last {lookback_days} days. "
            "Try a longer lookback or check that the pipeline has run recently."
        )

    scores = rows["sentiment_score"].dropna()
    direction, delta = _trend_direction(scores)
    dominant = rows["sentiment"].value_counts().idxmax()
    entities_top = _top_entities(rows["entities"])
    category_counts = rows["news_category"].value_counts().to_dict()

    time_series = [
        {
            "date":            str(row["date"].date()),
            "sentiment":       row["sentiment"],
            "sentiment_score": int(row["sentiment_score"]),
            "entities":        _split_entities(row.get("entities", "")),
            "news_category":   row.get("news_category", ""),
        }
        for _, row in rows.iterrows()
    ]

    result = {
        "sector":               sector,
        "lookback_days":        lookback_days,
        "date_range": {
            "from": str(rows["date"].min().date()),
            "to":   str(rows["date"].max().date()),
        },
        "n_observations":       len(rows),
        "mean_sentiment_score": round(float(scores.mean()), 4),
        "trend_direction":      direction,
        "trend_delta":          delta,
        "dominant_sentiment":   dominant,
        "top_entities":         entities_top,
        "category_breakdown":   {k: int(v) for k, v in category_counts.items()},
        "time_series":          time_series,
        "articles":             None,
    }

    if include_articles:
        dates = [entry["date"] for entry in time_series]
        result["articles"] = _load_articles_for_dates(dates)

    return result


# ── Bulk export API ────────────────────────────────────────────────────────────


def get_all_sectors_pivot(
    lookback_days: int | None = None,
    freq: str = "D",
) -> pd.DataFrame:
    """Wide date x sector pivot of mean daily sentiment scores.

    Columns are all 19 sectors (alphabetically sorted). NaN = no data for that
    sector on that date. Use freq="W" or freq="M" to aggregate to weekly/monthly.

    Args:
        lookback_days: Calendar days to look back from today.
                       None -> full history (all dates in TSV).
        freq:          Resampling frequency: "D" daily (default), "W" weekly,
                       "M" monthly.

    Returns:
        pd.DataFrame indexed by date with one column per sector.
        Empty DataFrame if sector_summary.tsv is missing.

    Invariant: columns are always alphabetically sorted — same order as list_sectors().
    """
    df = _load_summary()
    if df.empty:
        return pd.DataFrame()

    if lookback_days is not None:
        cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))
        df = df[df["date"] >= cutoff]

    if df.empty:
        return pd.DataFrame()

    pivot = (
        df.groupby(["date", "sector"])["sentiment_score"]
        .mean()
        .unstack("sector")
        .sort_index()
        .asfreq(freq)
    )
    return pivot[sorted(pivot.columns)]


def export_sector_pivot(
    output_path: "Path | str | None" = None,
    lookback_days: int | None = EXPORT_LOOKBACK_DAYS,
    freq: str = "D",
) -> Path:
    """Write the date x sector pivot to a TSV file and return its path.

    Defaults to SECTOR_PIVOT_FILE (data/sector_sentiment_pivot.tsv) and a
    rolling 90-day window (EXPORT_LOOKBACK_DAYS). Override either via arguments.

    Args:
        output_path:   Destination path. Defaults to constants.SECTOR_PIVOT_FILE.
        lookback_days: Window in calendar days. None -> full history.
        freq:          Resampling frequency (see get_all_sectors_pivot).

    Returns:
        Path to the written file.

    Raises:
        RuntimeError: If the pivot is empty (TSV missing or no data in window).
    """
    out = Path(output_path) if output_path is not None else SECTOR_PIVOT_FILE
    pivot = get_all_sectors_pivot(lookback_days=lookback_days, freq=freq)
    if pivot.empty:
        raise RuntimeError(
            "Sector pivot is empty — sector_summary.tsv is missing or "
            f"contains no data in the last {lookback_days} days."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out, sep="\t")
    return out
