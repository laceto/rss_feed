"""
visualize_topics.py

Generates two topic-trend charts from data/topic_trends.tsv:

  A) topic_spike_heatmap.png  — dates × top-N topics, cell = article_count
     Shows which narratives dominated which days at a glance.

  B) topic_frequency_ts.png   — line chart, article_count per top-N topic over time
     Shows the rise and fall of each narrative as a story arc.

Usage:
    python visualize_topics.py              # top 15 topics, last 90 days
    python visualize_topics.py --top 20     # top 20 topics
    python visualize_topics.py --days 60    # last 60 days only

Outputs: data/charts/topic_spike_heatmap.png
         data/charts/topic_frequency_ts.png

Invariants:
    - Reads only data/topic_trends.tsv — no API calls.
    - Topics with empty labels fall back to first 8 chars of topic_id.
    - Missing date/topic combinations are shown as 0 (no data), not NaN.
    - Charts are written atomically via a temp file to avoid partial writes.

Failure modes:
    - topic_trends.tsv absent or empty: exits with message.
    - Fewer unique topics than --top: uses all available.
    - data/charts/ created automatically if absent.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for CI / headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

TOPIC_TRENDS_FILE = Path("data") / "topic_trends.tsv"
CHARTS_DIR        = Path("data") / "charts"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_trends(lookback_days: int) -> pd.DataFrame:
    """Load and validate topic_trends.tsv; apply date window and label fallback."""
    if not TOPIC_TRENDS_FILE.exists():
        raise FileNotFoundError(
            f"{TOPIC_TRENDS_FILE} not found. "
            "Run backfill.py --phase1-only (or cluster_topics.py) first."
        )

    df = pd.read_csv(TOPIC_TRENDS_FILE, sep="\t", parse_dates=["date"])
    if df.empty:
        raise ValueError("topic_trends.tsv is empty — no data to visualize.")

    # Apply lookback window
    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff].copy()

    # Label fallback: use first 8 chars of topic_id for unlabeled topics
    df["label"] = df["topic_label"].fillna("").str.strip()
    df["label"] = df.apply(
        lambda r: r["label"] if r["label"] else r["topic_id"][:8], axis=1
    )

    print(f"Loaded {len(df):,} rows | "
          f"{df['date'].nunique()} dates | "
          f"{df['topic_id'].nunique()} unique topics")
    return df


def pick_top_topics(df: pd.DataFrame, top_n: int) -> list[str]:
    """Select top-N topics by total article count across the window.

    Returns topic_id list ordered by descending total article count.
    """
    totals = df.groupby("topic_id")["article_count"].sum().nlargest(top_n)
    return totals.index.tolist()


# ── Chart A: spike heatmap ────────────────────────────────────────────────────

def plot_spike_heatmap(df: pd.DataFrame, top_ids: list[str], out_path: Path) -> None:
    """Heatmap: dates (rows) × topic labels (cols), colour = article_count.

    Missing combinations are shown as 0 (topic not active that day).
    """
    subset = df[df["topic_id"].isin(top_ids)].copy()

    # Build a stable label map (topic_id -> label, pick most recent non-empty)
    label_map = (
        subset.sort_values("date")
              .groupby("topic_id")["label"]
              .last()
    )

    # Pivot: index=date, columns=label, values=article_count
    pivot = (
        subset.pivot_table(index="date", columns="topic_id",
                           values="article_count", aggfunc="max")
              .fillna(0)
              .rename(columns=label_map)
    )
    pivot.index = pivot.index.strftime("%Y-%m-%d")

    # Truncate long labels for readability
    pivot.columns = [c[:35] for c in pivot.columns]

    n_dates, n_topics = pivot.shape
    fig_h = max(6, n_dates * 0.22)
    fig_w = max(10, n_topics * 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="#e0e0e0",
        cbar_kws={"label": "article count", "shrink": 0.6},
        vmin=0,
        annot=False,
    )
    ax.set_title("Topic Activity Heatmap (article count per day)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=40, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _save_figure(fig, out_path)
    print(f"  Saved: {out_path}")


# ── Chart B: frequency time series ───────────────────────────────────────────

def plot_frequency_ts(df: pd.DataFrame, top_ids: list[str], out_path: Path) -> None:
    """Line chart: article_count per top topic over time.

    Each topic gets its own line. Topics with no label use their topic_id prefix.
    """
    subset = df[df["topic_id"].isin(top_ids)].copy()

    label_map = (
        subset.sort_values("date")
              .groupby("topic_id")["label"]
              .last()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = sns.color_palette("tab20", n_colors=len(top_ids))

    for tid, color in zip(top_ids, palette):
        topic_df = subset[subset["topic_id"] == tid].sort_values("date")
        label = label_map.get(tid, tid[:8])
        ax.plot(
            topic_df["date"],
            topic_df["article_count"],
            label=label[:40],
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=3,
            alpha=0.85,
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=40, ha="right", fontsize=8)
    ax.set_title("Topic Article Frequency Over Time", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Article count")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=7, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path)
    print(f"  Saved: {out_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Write figure atomically: render to temp file, then os.replace()."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    os.replace(tmp, out_path)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Visualize topic trends")
    p.add_argument("--top",  type=int, default=15, help="Top N topics (default: 15)")
    p.add_argument("--days", type=int, default=90, help="Lookback window in days (default: 90)")
    args = p.parse_args()

    print(f"=== Topic Visualizations (top={args.top}, days={args.days}) ===")

    df = load_trends(args.days)
    top_ids = pick_top_topics(df, args.top)
    actual_n = len(top_ids)
    print(f"Selected {actual_n} topic(s) for charts.")

    print("\nGenerating spike heatmap...")
    plot_spike_heatmap(df, top_ids, CHARTS_DIR / "topic_spike_heatmap.png")

    print("Generating frequency time series...")
    plot_frequency_ts(df, top_ids, CHARTS_DIR / "topic_frequency_ts.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
