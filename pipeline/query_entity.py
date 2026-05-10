"""
query_entity.py
Query accumulated sector data by entity name. Entry point for external flows.

Mirrors query_sector.py but pivots on entity (company, person, org) rather
than sector. Entities are dynamic — extracted by the LLM, not from a fixed
taxonomy — so lookup is case-insensitive and unknown names raise LookupError
with a list of similar known entities rather than a ValueError with valid values.

Two public functions:

    get_entity_snapshot(entity)
        All sectors that mention the entity on its most recent date — fast.
        Use when you need: "what is the current read on Nvidia?"

    get_entity_time_series(entity, lookback_days=30, include_articles=False)
        Trend over a rolling window — sentiment direction and category breakdown.
        Use when you need: "how has sentiment around the Fed changed over time?"

Both raise LookupError if the entity is not found (includes similar-name hints).

Usage:
    from query_entity import get_entity_snapshot, get_entity_time_series, list_entities

    names = list_entities()
    snap  = get_entity_snapshot("nvidia")           # case-insensitive
    ts    = get_entity_time_series("Nvidia", lookback_days=90, include_articles=True)

Schema — get_entity_snapshot():
    {
        entity              : str   # canonical casing as found in TSV
        last_date           : str   # "YYYY-MM-DD" of most recent entry
        data_age_days       : int   # days since last_date vs today
        dominant_sentiment  : str   # most frequent across all sectors that day
        mean_sentiment_score: float
        sectors             : list[{
            sector           : str
            sentiment        : str
            sentiment_score  : int
            news_category    : str
            extraction_status: str
        }]
    }

Schema — get_entity_time_series():
    {
        entity              : str
        lookback_days       : int
        date_range          : {from: str, to: str}
        n_observations      : int   # total (date × sector) rows in window
        mean_sentiment_score: float
        trend_direction     : str   # "improving"|"deteriorating"|"stable"
        trend_delta         : float # second_half_mean − first_half_mean
        dominant_sentiment  : str
        sectors_seen        : list[str]  # unique sectors, ranked by frequency
        category_breakdown  : dict[str, int]
        time_series         : list[{
            date            : str
            sector          : str
            sentiment       : str
            sentiment_score : int
            news_category   : str
        }]
        articles            : list[{date, title, description, link}] | None
    }

Debugging:
    - "entity not found" → the entity name may differ in casing or spelling;
      call list_entities() to browse known names.
    - "No data within window" → try a longer lookback_days or check that the
      pipeline has run recently (run read_sector_results.py).
    - articles=None when include_articles=True → the raw feed file for that
      date is missing from output/; the entity signal itself is still valid.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import warnings

import pandas as pd

from constants import (
    SENTIMENT_SCORE, SECTOR_SUMMARY_FILE, RAW_FEED_DIR,
    EXPORT_LOOKBACK_DAYS, ENTITY_TS_FILE,
)

# ── Trend threshold (must match query_sector._TREND_THRESHOLD) ────────────────

_TREND_THRESHOLD = 0.20  # minimum delta between halves to call a trend


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_entity_df() -> pd.DataFrame:
    """Load sector_summary.tsv and explode the pipe-separated entities column.

    Returns a DataFrame with one row per (entity × date × sector).
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
        # Canonical casing = value from the oldest date available
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


def _trend_direction(scores: pd.Series) -> tuple[str, float]:
    """Return (direction, delta) based on first-half vs second-half mean.

    direction: "improving" | "deteriorating" | "stable"
    delta: second_half_mean - first_half_mean  (positive = improving)

    Identical to query_sector._trend_direction; duplicated to keep this module
    self-contained and independently testable.
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
    with a warning — missing files don't invalidate the entity signal.

    Identical to query_sector._load_articles_for_dates; duplicated to keep
    this module self-contained.
    """
    articles = []
    for d in sorted(dates):
        feed_file = RAW_FEED_DIR / f"feeds{d}.txt"
        if not feed_file.exists():
            warnings.warn(f"Raw feed not found for {d}: {feed_file}", stacklevel=4)
            continue
        try:
            feed_df = pd.read_csv(feed_file, sep="\t")
            for _, row in feed_df.iterrows():
                articles.append({
                    "date":        d,
                    "title":       row.get("title", ""),
                    "description": row.get("description", ""),
                    "link":        row.get("link", ""),
                })
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not read {feed_file}: {exc}", stacklevel=4)
    return articles


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

    # Latest date this entity appears
    last_ts = entity_rows["date"].max()
    last_date: date = last_ts.date() if hasattr(last_ts, "date") else last_ts
    age_days = (date.today() - last_date).days

    # All sectors on that exact date
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
        "last_date":            str(last_date),
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

    One row per (date × sector) in which the entity appears within the window.

    Args:
        entity:           Entity name (case-insensitive).
        lookback_days:    Number of calendar days to look back from today.
        include_articles: If True, also fetch raw article headlines from
                          output/feeds{date}.txt for every date in the window.

    Returns:
        dict — see module docstring for full schema.

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


# ── Bulk export API ────────────────────────────────────────────────────────────

def get_all_entities_ts(
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Long-format time series for every entity seen in sector_summary.tsv.

    One row per (date × entity × sector) mention within the window.
    Columns: date, entity, sector, sentiment, sentiment_score, news_category.

    The result is queryable by filtering on any column, e.g.:
        df[df["entity"] == "Anthropic"]
        df[df["sector"] == "Technology Services"]

    Args:
        lookback_days: Calendar days to look back from today.
                       None → full history (all dates in TSV).

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
        lookback_days: Window in calendar days. None → full history.

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
