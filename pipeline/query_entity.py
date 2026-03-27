"""Entity query API — snapshot, time-series, and bulk export utilities.

Extracted from query_entity.py.

Entities are dynamic (LLM-extracted, no fixed taxonomy), so lookup is
case-insensitive and unknown names raise LookupError with similar-name
hints rather than a fixed list of valid values.

Deduplication:
  _trend_direction and _load_articles_for_dates are imported from
  pipeline.query_sector — they are identical to the originals; importing
  them here keeps the logic in one place.

Two public query functions:

    get_entity_snapshot(entity)
        All sectors mentioning the entity on its most recent date — fast.

    get_entity_time_series(entity, lookback_days, include_articles)
        Trend over a rolling window: sentiment direction, category breakdown,
        optional raw article headlines.

Both raise LookupError if entity is not found (includes similar-name hints).
"""

from __future__ import annotations

import warnings
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from constants import (
    ENTITY_TS_FILE,
    EXPORT_LOOKBACK_DAYS,
    SECTOR_SUMMARY_FILE,
    SENTIMENT_SCORE,
)
from .query_sector import _load_articles_for_dates, _trend_direction


# ── Private helpers ─────────────────────────────────────────────────────────────


def _load_entity_df() -> pd.DataFrame:
    """Load sector_summary.tsv and explode the pipe-separated entities column.

    Returns a DataFrame with one row per (entity x date x sector).
    Adds a lowercase 'entity_lower' column for case-insensitive matching.
    Whitespace is stripped from entity names; original casing is preserved.

    Returns an empty DataFrame if the TSV is missing (warns instead of raising
    so that list_entities() can return [] gracefully).
    """
    if not SECTOR_SUMMARY_FILE.exists():
        warnings.warn(
            f"{SECTOR_SUMMARY_FILE} not found. "
            "Run read_sector_results.py to generate it.",
            stacklevel=3,
        )
        return pd.DataFrame(
            columns=[
                "date", "sector", "entities", "entity", "entity_lower",
                "sentiment", "news_category", "extraction_status",
                "sentiment_score",
            ]
        )

    df = pd.read_csv(SECTOR_SUMMARY_FILE, sep="\t", parse_dates=["date"])
    df["sentiment_score"] = df["sentiment"].map(SENTIMENT_SCORE)

    # Explode pipe-separated entities into individual rows.
    # Rows with empty/null entities get an empty list and are dropped.
    df["entity"] = df["entities"].apply(
        lambda v: [e.strip() for e in v.split("|") if e.strip()]
        if isinstance(v, str) and v.strip()
        else []
    )
    df = df.explode("entity").dropna(subset=["entity"])
    df = df[df["entity"] != ""]

    df["entity_lower"] = df["entity"].str.lower()
    return df.reset_index(drop=True)


def _resolve_entity(name: str, df: pd.DataFrame) -> str:
    """Return the canonical casing of an entity via case-insensitive lookup.

    'Canonical' is the first spelling seen in the DataFrame after sorting by
    date (oldest first), which gives stable, reproducible output.

    Args:
        name: Entity name supplied by the caller (any casing).
        df:   Exploded entity DataFrame from _load_entity_df().

    Returns:
        Canonical string as stored in the TSV.

    Raises:
        LookupError: If no exact (case-insensitive) match exists.
            The error message includes up to 10 similar entity names
            (case-insensitive prefix or substring matches).
    """
    name_lower = name.lower()

    # Exact case-insensitive match
    matches = df[df["entity_lower"] == name_lower]
    if not matches.empty:
        oldest_first = matches.sort_values("date")
        return str(oldest_first.iloc[0]["entity"])

    # Build a hint list from prefix + substring matches
    all_lower = df["entity_lower"].unique()
    similar = sorted({
        df.loc[df["entity_lower"] == el, "entity"].iloc[0]
        for el in all_lower
        if name_lower in el or el.startswith(name_lower)
    })[:10]

    hint = f"  Similar entities: {similar}" if similar else ""
    raise LookupError(
        f"Entity '{name}' not found in sector data.{hint}\n"
        "Call list_entities() to browse all known entities."
    )


# ── Public API ────────────────────────────────────────────────────────────────


def list_entities() -> list[str]:
    """Return all unique entity names seen in sector_summary.tsv (sorted).

    Returns [] gracefully if the TSV is missing rather than raising, so that
    callers can safely use this as a discovery mechanism.
    """
    df = _load_entity_df()
    if df.empty:
        return []
    return sorted(df["entity"].unique().tolist())


def get_entity_snapshot(entity: str) -> dict:
    """Return all sector entries that mention the entity on its most recent date.

    Performs a case-insensitive lookup; the returned dict uses canonical casing.
    'Most recent date' is the latest calendar date the entity appears in any
    sector entry.

    Args:
        entity: Entity name (case-insensitive). Use list_entities() to discover
                valid names.

    Returns:
        dict with keys: entity, last_date, data_age_days, dominant_sentiment,
        mean_sentiment_score, sectors (list).

    Raises:
        LookupError: If the entity is not found in the data (includes hints).
    """
    df = _load_entity_df()

    if df.empty:
        raise LookupError(
            f"Entity '{entity}' not found — sector_summary.tsv is missing or empty."
        )

    canonical = _resolve_entity(entity, df)
    entity_rows = df[df["entity_lower"] == canonical.lower()].sort_values("date")

    last_ts = entity_rows["date"].max()
    last_date_val: date = last_ts.date() if hasattr(last_ts, "date") else last_ts
    age_days = (date.today() - last_date_val).days

    day_rows = entity_rows[entity_rows["date"] == last_ts]

    dominant = day_rows["sentiment"].value_counts().idxmax()
    mean_score = round(float(day_rows["sentiment_score"].mean()), 4)

    sectors = [
        {
            "sector":            str(row["sector"]),
            "sentiment":         str(row["sentiment"]),
            "sentiment_score":   int(row["sentiment_score"]),
            "news_category":     str(row.get("news_category", "")),
            "extraction_status": str(row.get("extraction_status", "")),
        }
        for _, row in day_rows.iterrows()
    ]

    return {
        "entity":               canonical,
        "last_date":            str(last_date_val),
        "data_age_days":        age_days,
        "dominant_sentiment":   dominant,
        "mean_sentiment_score": mean_score,
        "sectors":              sectors,
    }


def get_entity_time_series(
    entity: str,
    lookback_days: int = 30,
    include_articles: bool = False,
) -> dict:
    """Return sentiment trend for an entity over a rolling window.

    One row per (date x sector) in which the entity appears within the window.

    Args:
        entity:           Entity name (case-insensitive).
        lookback_days:    Number of calendar days to look back from today.
        include_articles: If True, also fetch raw article headlines from
                          output/feeds{date}.txt for every date in the window.

    Returns:
        dict with keys: entity, lookback_days, date_range, n_observations,
        mean_sentiment_score, trend_direction, trend_delta, dominant_sentiment,
        sectors_seen, category_breakdown, time_series, articles.

    Raises:
        LookupError: If entity is not found, or has no data within the window.
    """
    df = _load_entity_df()

    if df.empty:
        raise LookupError(
            f"Entity '{entity}' not found — sector_summary.tsv is missing or empty."
        )

    canonical = _resolve_entity(entity, df)
    cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))

    rows = (
        df[
            (df["entity_lower"] == canonical.lower()) &
            (df["date"] >= cutoff)
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if rows.empty:
        raise LookupError(
            f"No data for entity '{canonical}' in the last {lookback_days} days. "
            "Try a longer lookback or check that the pipeline has run recently."
        )

    scores = rows["sentiment_score"].dropna()
    direction, delta = _trend_direction(scores)
    dominant = rows["sentiment"].value_counts().idxmax()

    # Sectors ranked by appearance frequency
    sector_counts: Counter = Counter(rows["sector"].tolist())
    sectors_seen = [s for s, _ in sector_counts.most_common()]

    category_counts = rows["news_category"].value_counts().to_dict()

    time_series = [
        {
            "date":            str(row["date"].date()),
            "sector":          str(row["sector"]),
            "sentiment":       str(row["sentiment"]),
            "sentiment_score": int(row["sentiment_score"]),
            "news_category":   str(row.get("news_category", "")),
        }
        for _, row in rows.iterrows()
    ]

    result = {
        "entity":               canonical,
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
        "sectors_seen":         sectors_seen,
        "category_breakdown":   {k: int(v) for k, v in category_counts.items()},
        "time_series":          time_series,
        "articles":             None,
    }

    if include_articles:
        dates = sorted({entry["date"] for entry in time_series})
        result["articles"] = _load_articles_for_dates(dates)

    return result


# ── Bulk export API ─────────────────────────────────────────────────────────────


def get_all_entities_ts(
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Long-format time series for every entity seen in sector_summary.tsv.

    One row per (date x entity x sector) mention within the window.
    Columns: date, entity, sector, sentiment, sentiment_score, news_category.

    Args:
        lookback_days: Calendar days to look back from today.
                       None -> full history (all dates in TSV).

    Returns:
        pd.DataFrame sorted by (date, entity).
        Empty DataFrame if sector_summary.tsv is missing.
    """
    df = _load_entity_df()
    if df.empty:
        return pd.DataFrame()

    if lookback_days is not None:
        cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))
        df = df[df["date"] >= cutoff]

    if df.empty:
        return pd.DataFrame()

    cols = ["date", "entity", "sector", "sentiment", "sentiment_score", "news_category"]
    return (
        df[[c for c in cols if c in df.columns]]
        .sort_values(["date", "entity"])
        .reset_index(drop=True)
    )


def export_entity_ts(
    output_path: "Path | str | None" = None,
    lookback_days: int | None = EXPORT_LOOKBACK_DAYS,
) -> Path:
    """Write the long-format entity time series to a TSV file and return its path.

    Defaults to ENTITY_TS_FILE (data/entity_sentiment_ts.tsv) and a rolling
    90-day window (EXPORT_LOOKBACK_DAYS). Override either via arguments.

    The date column is written as a plain YYYY-MM-DD string (not a timestamp)
    so the file is directly importable in R without parse errors.

    Args:
        output_path:   Destination path. Defaults to constants.ENTITY_TS_FILE.
        lookback_days: Window in calendar days. None -> full history.

    Returns:
        Path to the written file.

    Raises:
        RuntimeError: If the result is empty (TSV missing or no data in window).
    """
    out = Path(output_path) if output_path is not None else ENTITY_TS_FILE
    ts = get_all_entities_ts(lookback_days=lookback_days)
    if ts.empty:
        raise RuntimeError(
            "Entity time series is empty — sector_summary.tsv is missing or "
            f"contains no data in the last {lookback_days} days."
        )
    # Write date as plain string (R-friendly), without the DataFrame index
    ts["date"] = ts["date"].dt.strftime("%Y-%m-%d")
    out.parent.mkdir(parents=True, exist_ok=True)
    ts.to_csv(out, sep="\t", index=False)
    return out
