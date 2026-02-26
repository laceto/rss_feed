"""
visualize_sentiment.py
Generates sentiment trend charts by sector from the consolidated sector summary.

Runs after read_sector_results.py, which writes data/sector_summary.tsv.

Input:  data/sector_summary.tsv
Output: data/charts/
    sentiment_heatmap.png       — sector × week, colored by mean sentiment score
    sentiment_trends.png        — rolling sentiment score per sector (small multiples)
    sentiment_distribution.png  — % positive / neutral / negative breakdown per sector

Sentiment encoding: positive=+1, neutral=0, negative=-1

Debugging:
- "No data to plot" warnings mean a sector had < MIN_DATA_POINTS entries — expected for
  infrequent sectors; check data/sector_summary.tsv for the affected sector.
- All three charts are independent — a failure in one does not abort the others.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

INPUT_FILE = Path("data") / "sector_summary.tsv"
CHARTS_DIR = Path("data") / "charts"

SENTIMENT_SCORE: dict[str, int] = {"positive": 1, "neutral": 0, "negative": -1}
SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#27ae60",
    "neutral": "#95a5a6",
    "negative": "#c0392b",
}
ROLLING_WINDOW = 5      # trading days for rolling average in trend chart
MIN_DATA_POINTS = 3     # sectors with fewer points are excluded from trend chart


# ── Data loading ────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load sector_summary.tsv and add a numeric sentiment_score column.

    Exits with a clear message if the input file is missing — run
    read_sector_results.py first to generate it.
    """
    if not INPUT_FILE.exists():
        print(f"[error] Input file not found: {INPUT_FILE}")
        print("Run read_sector_results.py first to generate it.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, sep="\t", parse_dates=["date"])
    df["sentiment_score"] = df["sentiment"].map(SENTIMENT_SCORE)
    df = df.dropna(subset=["sentiment_score", "date", "sector"])
    df = df.sort_values("date").reset_index(drop=True)

    print(
        f"Loaded {len(df)} rows | "
        f"{df['date'].nunique()} date(s) | "
        f"{df['sector'].nunique()} sector(s)"
    )
    return df


# ── Chart 1: Heatmap ────────────────────────────────────────────────────────────

def chart_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of mean sentiment score by sector (Y) and week (X).

    Weekly aggregation smooths daily noise and keeps the chart readable
    for long date ranges.
    Color scale: red (−1 negative) → white (0 neutral) → green (+1 positive).
    """
    df_w = df.copy()
    df_w["week"] = df["date"].dt.to_period("W").dt.start_time

    pivot = (
        df_w.groupby(["week", "sector"])["sentiment_score"]
        .mean()
        .unstack("sector")
    )

    if pivot.empty:
        print("[warn] heatmap: no data to plot.")
        return

    fig_w = max(12, len(pivot) * 0.45)
    fig_h = max(5, len(pivot.columns) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot.T,
        ax=ax,
        cmap="RdYlGn",
        vmin=-1, vmax=1, center=0,
        linewidths=0.4,
        linecolor="#e0e0e0",
        cbar_kws={"label": "Sentiment score  (−1 neg → 0 neutral → +1 pos)", "shrink": 0.6},
    )

    # Format x-axis tick labels as "Mon 'YY"
    week_labels = [w.strftime("%b '%y") for w in pivot.index]
    ax.set_xticklabels(week_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title("Sector Sentiment — Weekly Heatmap", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    out = CHARTS_DIR / "sentiment_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Chart 2: Trend lines ────────────────────────────────────────────────────────

def chart_trends(df: pd.DataFrame) -> None:
    """Small-multiples line chart: rolling sentiment score over time per sector.

    Each subplot shows one sector. The shaded area above zero is green (positive
    periods) and below zero is red (negative periods), making regime shifts
    immediately visible.

    Sectors with fewer than MIN_DATA_POINTS data points are excluded.
    """
    daily = (
        df.groupby(["date", "sector"])["sentiment_score"]
        .mean()
        .unstack("sector")
        .sort_index()
    )

    valid = [c for c in daily.columns if daily[c].notna().sum() >= MIN_DATA_POINTS]
    if not valid:
        print("[warn] trends: no sectors with enough data points.")
        return

    daily = daily[valid]
    smoothed = daily.rolling(window=ROLLING_WINDOW, min_periods=1, center=True).mean()

    ncols = 3
    nrows = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(18, nrows * 3.2),
        sharex=True,
        squeeze=False,
    )

    for i, sector in enumerate(sorted(valid)):
        ax = axes[i // ncols][i % ncols]
        series = smoothed[sector].dropna()

        ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
        ax.fill_between(
            series.index, series, 0,
            where=(series >= 0), color=SENTIMENT_COLORS["positive"], alpha=0.3,
        )
        ax.fill_between(
            series.index, series, 0,
            where=(series < 0), color=SENTIMENT_COLORS["negative"], alpha=0.3,
        )
        ax.plot(series.index, series, color="#2c3e50", linewidth=1.2, zorder=2)

        ax.set_title(sector, fontsize=9, fontweight="bold")
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["neg", "0", "pos"], fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    # Hide unused subplot cells
    for j in range(len(valid), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"Sector Sentiment Trends  ({ROLLING_WINDOW}-day rolling average)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = CHARTS_DIR / "sentiment_trends.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Chart 3: Distribution ───────────────────────────────────────────────────────

def chart_distribution(df: pd.DataFrame) -> None:
    """Stacked horizontal bar: % positive / neutral / negative per sector.

    Normalizing to 100% makes sectors with different news volumes comparable.
    Sectors are sorted by positive share (highest at top) to create a natural
    ranking of sector sentiment.
    """
    counts = (
        df.groupby(["sector", "sentiment"])
        .size()
        .unstack("sentiment")
        .fillna(0)
    )

    # Ensure all three columns exist even if a value never appeared
    for col in ["positive", "neutral", "negative"]:
        if col not in counts.columns:
            counts[col] = 0.0

    proportions = counts.div(counts.sum(axis=1), axis=0) * 100
    proportions = (
        proportions[["positive", "neutral", "negative"]]
        .sort_values("positive", ascending=True)  # best sector at top
    )

    if proportions.empty:
        print("[warn] distribution: no data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(proportions) * 0.55)))

    left = pd.Series(0.0, index=proportions.index)
    for col, color in SENTIMENT_COLORS.items():
        ax.barh(
            proportions.index,
            proportions[col],
            left=left,
            color=color,
            label=col.capitalize(),
            height=0.65,
        )
        left = left + proportions[col]

    ax.set_xlim(0, 100)
    ax.axvline(50, color="#888888", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("Share of news items (%)", fontsize=10)
    ax.set_title(
        "Sentiment Distribution by Sector", fontsize=14, fontweight="bold", pad=12
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    out = CHARTS_DIR / "sentiment_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Entry point ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Sentiment Trend Visualizations ===")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    chart_heatmap(df)
    chart_trends(df)
    chart_distribution(df)

    print(f"\nAll charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
