"""
explore_topic_sentiment.py

Manual validation of the indirect topic-sentiment join.

Join path:
  topic_clusters/{date}.json   — article guid → topic_id, article date
  sector_summary.tsv           — date × sector → sentiment
  SENTIMENT_SCORE              — "positive"→+1, "neutral"→0, "negative"→-1

For each topic on a given date:
  1. Collect all articles assigned to that topic (from the 45-day cluster window)
  2. For each article, look up sector sentiment for the article's date
  3. Average all sentiment scores → topic_sentiment_score

Output tables:
  A. Per-topic sentiment for the target date (sorted by spike_ratio)
  B. Cross-date comparison: do different topics get meaningfully different scores on
     the same date? (Answers the key open question from the brainstorm)
  C. Spot-check: for the top spike, show which articles drove the sentiment
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
TOPIC_CLUSTERS_DIR = PROJECT_ROOT / "data" / "topic_clusters"
SECTOR_SUMMARY     = PROJECT_ROOT / "data" / "sector_summary.tsv"
TOPIC_TRENDS       = PROJECT_ROOT / "data" / "topic_trends.tsv"

SENTIMENT_SCORE: dict[str, float] = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# Dates to inspect — pick a few recent ones with known spikes
TARGET_DATES = [
    "2026-03-05",
    "2026-02-19",
    "2026-01-15",
    "2025-12-12",
    "2025-11-06",
]


# ── Core join ──────────────────────────────────────────────────────────────────

def load_sector_scores() -> pd.DataFrame:
    """Return sector_summary with a numeric sentiment_score column."""
    df = pd.read_csv(SECTOR_SUMMARY, sep="\t")
    df["sentiment_score"] = df["sentiment"].map(SENTIMENT_SCORE)
    # date-level mean (one score per date, across all sectors)
    day_mean = (
        df.groupby("date")["sentiment_score"]
        .mean()
        .rename("day_mean_score")
    )
    return df, day_mean


def compute_topic_sentiment(
    cluster_date: str,
    sector_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute mean sentiment score for every topic in cluster_date's JSON.

    Returns DataFrame: topic_id, n_articles, n_matched, mean_score, coverage_pct
    """
    cluster_file = TOPIC_CLUSTERS_DIR / f"{cluster_date}.json"
    if not cluster_file.exists():
        print(f"[warn] No cluster file for {cluster_date}")
        return pd.DataFrame()

    articles = pd.DataFrame(json.loads(cluster_file.read_text(encoding="utf-8")))

    # Keep only clustered articles (noise has topic_id = null)
    articles = articles[articles["topic_id"].notna()].copy()

    # Normalise article date to YYYY-MM-DD (strip time component if present)
    articles["article_date"] = (
        articles["date"].astype(str).str[:10]
    )

    # Day-level mean sentiment: average across all sectors for each date
    # This is the "indirect join" — we don't have per-article sentiment,
    # only per-date-per-sector sentiment.
    day_scores = (
        sector_df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"date": "article_date", "sentiment_score": "day_score"})
    )

    # Join articles to day-level scores
    merged = articles.merge(day_scores, on="article_date", how="left")

    # Aggregate per topic
    rows = []
    for topic_id, grp in merged.groupby("topic_id"):
        matched = grp["day_score"].notna().sum()
        rows.append({
            "topic_id":     topic_id,
            "n_articles":   len(grp),
            "n_matched":    matched,
            "mean_score":   grp["day_score"].mean(),          # NaN-aware
            "coverage_pct": round(100 * matched / len(grp), 1),
        })

    return pd.DataFrame(rows).sort_values("mean_score")


# ── Pretty output ──────────────────────────────────────────────────────────────

SCORE_LABEL = {
    lambda s: s > 0.20:  "positive",
    lambda s: s < -0.20: "negative",
}

def score_to_label(s: float) -> str:
    if pd.isna(s):      return "n/a"
    if s >  0.20:       return "positive"
    if s < -0.20:       return "negative"
    return "neutral/mixed"


def bar(score: float, width: int = 20) -> str:
    """ASCII sentiment bar centred on 0."""
    if pd.isna(score):
        return " " * width
    half = width // 2
    pos = int(round(score * half))
    if pos > 0:
        return " " * half + "+" * pos + " " * (half - pos)
    elif pos < 0:
        return " " * (half + pos) + "-" * (-pos) + " " * half
    else:
        return " " * half + "|" + " " * (half - 1)


def print_topic_table(
    cluster_date: str,
    topic_df: pd.DataFrame,
    trends: pd.DataFrame,
    day_mean: pd.Series,
) -> None:
    """Print per-topic sentiment for one date."""
    # Enrich with labels from topic_trends
    day_trends = trends[trends["date"] == cluster_date].set_index("topic_id")
    topic_df = topic_df.copy()
    topic_df["label"] = topic_df["topic_id"].map(
        day_trends.get("topic_label", pd.Series(dtype=str))
    ).fillna("(unlabeled)")
    topic_df["spike_ratio"] = topic_df["topic_id"].map(
        day_trends.get("spike_ratio", pd.Series(dtype=float))
    )
    topic_df = topic_df.sort_values("mean_score")

    dm = day_mean.get(cluster_date, float("nan"))

    print(f"\n{'='*80}")
    print(f"  {cluster_date}   |   market day score: {dm:+.3f} ({score_to_label(dm)})")
    print(f"{'='*80}")
    print(f"  {'label':<38} {'score':>6}  {'label':>14}  cov%  {'bar'}")
    print(f"  {'-'*38} {'-'*6}  {'-'*14}  {'-'*4}  {'-'*20}")

    for _, row in topic_df.iterrows():
        label     = str(row["label"])[:38]
        score     = row["mean_score"]
        slabel    = score_to_label(score)
        cov       = row["coverage_pct"]
        diff      = (score - dm) if not pd.isna(score) and not pd.isna(dm) else float("nan")
        diff_str  = f"({diff:+.2f} vs mkt)" if not pd.isna(diff) else ""
        print(
            f"  {label:<38} {score:>+6.3f}  {slabel:>14}  {cov:>4.0f}%  "
            f"{bar(score)}  {diff_str}"
        )

    # Key question: how much spread is there?
    scores = topic_df["mean_score"].dropna()
    if len(scores) >= 2:
        print(f"\n  Spread: min={scores.min():+.3f}  max={scores.max():+.3f}  "
              f"range={scores.max()-scores.min():.3f}  std={scores.std():.3f}")
        verdict = "DISTINCT signals (range > 0.30)" if scores.max()-scores.min() > 0.30 else "FLAT -- topics share day-level sentiment (range <= 0.30)"
        print(f"  -> {verdict}")


def spot_check_top_spike(
    cluster_date: str,
    topic_df: pd.DataFrame,
    trends: pd.DataFrame,
    sector_df: pd.DataFrame,
) -> None:
    """For the highest-scored topic on the date, show which article dates drove it."""
    day_trends = trends[trends["date"] == cluster_date].set_index("topic_id")
    if day_trends.empty:
        return

    # Pick the topic with highest absolute score
    top = topic_df.reindex(
        topic_df["mean_score"].abs().sort_values(ascending=False).index
    ).iloc[0]
    label = day_trends["topic_label"].get(top["topic_id"], "(unlabeled)")

    print(f"\n  Spot-check: '{label}'  (score={top['mean_score']:+.3f})")

    cluster_file = TOPIC_CLUSTERS_DIR / f"{cluster_date}.json"
    articles = pd.DataFrame(json.loads(cluster_file.read_text(encoding="utf-8")))
    arts = articles[articles["topic_id"] == top["topic_id"]].copy()
    arts["article_date"] = arts["date"].astype(str).str[:10]

    day_scores = (
        sector_df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"date": "article_date", "sentiment_score": "day_score"})
    )
    arts = arts.merge(day_scores, on="article_date", how="left")

    print(f"  {'article_date':<12} {'day_score':>10}  {'title'}")
    print(f"  {'-'*12} {'-'*10}  {'-'*50}")
    for _, r in arts.sort_values("article_date").iterrows():
        score_str = f"{r['day_score']:+.3f}" if not pd.isna(r["day_score"]) else "  n/a"
        print(f"  {r['article_date']:<12} {score_str:>10}  {str(r['title'])[:60]}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading sector scores...")
    sector_df, day_mean = load_sector_scores()
    trends = pd.read_csv(TOPIC_TRENDS, sep="\t")

    for target_date in TARGET_DATES:
        topic_df = compute_topic_sentiment(target_date, sector_df)
        if topic_df.empty:
            print(f"\n[skip] {target_date} — no cluster file")
            continue

        print_topic_table(target_date, topic_df, trends, day_mean)
        spot_check_top_spike(target_date, topic_df, trends, sector_df)

    # ── Cross-date summary: answer the key open question ──────────────────────
    print(f"\n\n{'='*80}")
    print("  CROSS-DATE SUMMARY: Is topic sentiment distinct within a day?")
    print(f"{'='*80}")
    print(f"  {'date':<12} {'n_topics':>8} {'range':>8} {'std':>8}  verdict")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")

    for target_date in TARGET_DATES:
        topic_df = compute_topic_sentiment(target_date, sector_df)
        if topic_df.empty:
            continue
        scores = topic_df["mean_score"].dropna()
        if len(scores) < 2:
            continue
        rng = scores.max() - scores.min()
        std = scores.std()
        verdict = "DISTINCT" if rng > 0.30 else "FLAT"
        print(f"  {target_date:<12} {len(scores):>8} {rng:>8.3f} {std:>8.3f}  {verdict}")

    print()


if __name__ == "__main__":
    main()
