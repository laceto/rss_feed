"""
visualize_topics.py

Generates two topic-trend charts from data/topic_trends.tsv:

  A) topic_spike_heatmap.png  — dates × top-N topics, cell = article_count
     Shows which narratives dominated which days at a glance.

  B) topic_frequency_ts.png   — line chart, article_count per top-N topic over time
     Shows the rise and fall of each narrative as a story arc.

  C) topic_timeline.png       — Gantt chart, one bar per topic: first → last appearance
     Shows topic lifespans; bar darkness encodes total article volume.

Usage:
    python visualize_topics.py              # top 15 topics, last 200 days
    python visualize_topics.py --top 20     # top 20 topics
    python visualize_topics.py --days 90    # last 90 days only

Outputs: data/charts/topic_spike_heatmap.png
         data/charts/topic_frequency_ts.png
         data/charts/topic_timeline.png

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
import matplotlib.ticker
from matplotlib.colors import LogNorm
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
    # Replace 0/missing with NaN so inactive cells render blank under LogNorm.
    pivot = (
        subset.pivot_table(index="date", columns="topic_id",
                           values="article_count", aggfunc="max")
              .fillna(0)
              .replace(0, float("nan"))
              .rename(columns=label_map)
    )
    pivot.index = pivot.index.strftime("%Y-%m-%d")

    # Truncate long labels for readability
    pivot.columns = [c[:35] for c in pivot.columns]

    n_dates, n_topics = pivot.shape
    # Scale row height down for large windows so the image stays manageable.
    # ≤90 rows → 0.22 in/row; 150 rows → 0.16 in/row; linear interpolation.
    row_h = max(0.14, 0.22 - (n_dates - 90) * 0.001) if n_dates > 90 else 0.22
    fig_h = max(8, n_dates * row_h)
    fig_w = max(10, n_topics * 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        norm=LogNorm(vmin=1),   # log scale — prevents outliers from washing out the palette
        linewidths=0.3,
        linecolor="#e0e0e0",
        cbar_kws={"label": "article count (log scale)", "shrink": 0.6},
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

    Each topic gets its own line with a direct end-of-line label so readers
    don't have to cross-reference a legend box.  Y-axis uses a symlog scale so
    a single outlier spike cannot compress all other lines to the baseline.
    """
    subset = df[df["topic_id"].isin(top_ids)].copy()

    label_map = (
        subset.sort_values("date")
              .groupby("topic_id")["label"]
              .last()
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    palette = sns.color_palette("tab20", n_colors=len(top_ids))

    # Track last (date, value) per topic for end-of-line annotations.
    endpoints: list[tuple] = []

    for tid, color in zip(top_ids, palette):
        topic_df = subset[subset["topic_id"] == tid].sort_values("date")
        label = label_map.get(tid, tid[:8])
        ax.plot(
            topic_df["date"],
            topic_df["article_count"],
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=3,
            alpha=0.85,
        )
        if not topic_df.empty:
            last = topic_df.iloc[-1]
            endpoints.append((last["date"], last["article_count"], label[:40], color))

    # Direct end-of-line labels.
    # Sort by last value so the label order matches the visual order of lines.
    # Spread label y-positions evenly across the data range to avoid overlap.
    endpoints.sort(key=lambda t: t[1])
    y_vals  = [e[1] for e in endpoints]
    y_lo    = max(1, min(y_vals) * 0.5)
    y_hi    = max(y_vals) * 1.1
    n       = len(endpoints)
    import numpy as np
    y_slots = list(np.linspace(y_lo, y_hi, n)) if n > 1 else [y_lo]
    x_max   = df["date"].max()

    for (x, y, lbl, color), y_text in zip(endpoints, y_slots):
        ax.annotate(
            lbl,
            xy=(x, y),
            xytext=(x_max, y_text),
            fontsize=7,
            color=color,
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.4),
        )

    # symlog scale: linear near 0, log elsewhere — outlier spikes don't crush
    # low-activity topics. linthresh = smallest meaningful value (1 article).
    ax.set_yscale("symlog", linthresh=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=40, ha="right", fontsize=8)
    ax.set_title("Topic Article Frequency Over Time", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Article count (symlog)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path)
    print(f"  Saved: {out_path}")


# ── Chart C: topic lifespan timeline (Gantt) ─────────────────────────────────

def plot_topic_timeline(df: pd.DataFrame, top_ids: list[str], out_path: Path) -> None:
    """Gantt chart: one bar per topic showing first → last appearance date.

    Bars are sorted top-to-bottom by first_seen (oldest narrative at the top).
    Bar width encodes total active days; bar colour encodes total article count
    (darker = more coverage overall).
    """
    subset = df[df["topic_id"].isin(top_ids)].copy()

    label_map = (
        subset.sort_values("date")
              .groupby("topic_id")["label"]
              .last()
    )

    # Compute per-topic span and total coverage.
    agg = (
        subset.groupby("topic_id")
              .agg(first_seen=("date", "min"),
                   last_seen=("date", "max"),
                   total_articles=("article_count", "sum"))
              .reset_index()
    )
    agg["label"] = agg["topic_id"].map(label_map)
    agg = agg.sort_values("first_seen")   # oldest narrative at the top

    n = len(agg)
    fig_h = max(5, n * 0.45)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    # Normalise total_articles → [0.3, 1.0] for alpha (more coverage = darker).
    t_min, t_max = agg["total_articles"].min(), agg["total_articles"].max()
    norm_alpha = lambda v: 0.35 + 0.65 * (v - t_min) / max(t_max - t_min, 1)

    palette = sns.color_palette("tab20", n_colors=n)

    for i, (_, row) in enumerate(agg.iterrows()):
        start    = row["first_seen"]
        end      = row["last_seen"]
        duration = (end - start).days + 1
        color    = palette[i]
        alpha    = norm_alpha(row["total_articles"])

        ax.barh(
            y=i,
            width=duration,
            left=start,
            height=0.6,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.4,
        )
        # Label inside bar if wide enough, otherwise to the right.
        label_x = start + pd.Timedelta(days=duration / 2)
        if duration >= 14:
            ax.text(label_x, i, row["label"][:35],
                    ha="center", va="center", fontsize=7,
                    color="black", fontweight="bold")
        else:
            ax.text(end + pd.Timedelta(days=2), i, row["label"][:35],
                    ha="left", va="center", fontsize=7, color=color)

    ax.set_yticks(range(n))
    ax.set_yticklabels([""] * n)   # labels already drawn inline
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.set_title("Topic Lifespans (first → last appearance)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()   # oldest topic at the top

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
    p.add_argument("--days", type=int, default=200, help="Lookback window in days (default: 200)")
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

    print("Generating topic timeline...")
    plot_topic_timeline(df, top_ids, CHARTS_DIR / "topic_timeline.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
