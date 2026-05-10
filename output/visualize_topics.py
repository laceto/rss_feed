"""
visualize_topics.py

Generates six topic-trend charts from data/topic_trends.tsv:

  A) topic_spike_heatmap.png     — dates × top-N topics, cell = article_count
     Shows which narratives dominated which days at a glance.

  B) topic_frequency_ts.png      — line chart, article_count per top-N topic over time
     Shows the rise and fall of each narrative as a story arc.

  C) topic_timeline.png          — Gantt chart, one bar per topic: first → last appearance
     Shows topic lifespans; bar darkness encodes total article volume.

  D) topic_sentiment_heatmap.png — dates × top-N topics, cell = sentiment_score
     Diverging red–green: shows which topics are bearish vs. bullish each day.
     Grey = no sentiment data (pre-backfill rows).

  E) topic_sentiment_delta.png   — dates × top-N topics, cell = 7-day rolling delta
     Shows narrative momentum: is sentiment improving or deteriorating?
     Only topics with ≥ 3 scored observations are shown.

  F) topic_signal_scatter.png    — spike_ratio × sentiment_score on the most recent date
     All topics with spike_ratio ≥ 1.0. Four quadrants expose the signal:
       top-right  = high spike + positive  → momentum / bullish
       top-left   = low  spike + positive  → quiet positive
       bottom-right = high spike + negative → crisis / risk-off
       bottom-left  = low  spike + negative → slow bleed

  G) topic_signal_scatter_animated.gif  — animated version of chart F, one frame per date
     Axes, colour norm, and quadrant split line are fixed globally so bubble movement
     is directly comparable across frames.  Requires --animate flag to generate
     (slow — renders one frame per qualifying date).

Usage:
    python visualize_topics.py              # top 15 topics, last 200 days
    python visualize_topics.py --top 20     # top 20 topics
    python visualize_topics.py --days 90    # last 90 days only
    python visualize_topics.py --animate    # also generate animated GIF (chart G)

Outputs: data/charts/topic_spike_heatmap.png
         data/charts/topic_frequency_ts.png
         data/charts/topic_timeline.png
         data/charts/topic_sentiment_heatmap.png
         data/charts/topic_sentiment_delta.png
         data/charts/topic_signal_scatter.png
         data/charts/topic_signal_scatter_animated.gif  (only with --animate)

Invariants:
    - Reads only data/topic_trends.tsv — no API calls.
    - Topics with empty labels fall back to first 8 chars of topic_id.
    - Missing date/topic combinations shown as 0 for count charts; NaN for sentiment.
    - Sentiment charts skipped gracefully when sentiment_score column is absent.
    - Charts written atomically via temp file + os.replace.

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

import copy

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for CI / headless
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.ticker
from matplotlib.colors import LogNorm, TwoSlopeNorm
import matplotlib.lines as mlines
import numpy as np
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


# ── Sentiment helpers ─────────────────────────────────────────────────────────

def _has_sentiment(df: pd.DataFrame) -> bool:
    """Return True when the DataFrame has usable sentiment_score values."""
    return "sentiment_score" in df.columns and df["sentiment_score"].notna().any()


def _build_sentiment_pivot(df: pd.DataFrame, top_ids: list[str]) -> pd.DataFrame:
    """Pivot sentiment_score: index=date (str), columns=label, values=float.

    NaN = topic not active or sentiment not yet computed for that date.
    """
    subset = df[df["topic_id"].isin(top_ids)].copy()
    label_map = (
        subset.sort_values("date")
              .groupby("topic_id")["label"]
              .last()
    )
    pivot = (
        subset.pivot_table(index="date", columns="topic_id",
                           values="sentiment_score", aggfunc="mean")
              .rename(columns=label_map)
    )
    pivot.index = pivot.index.strftime("%Y-%m-%d")
    pivot.columns = [c[:35] for c in pivot.columns]
    return pivot


def _compute_delta_pivot(sentiment_pivot: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """7-day rolling momentum: score_today − mean(prior window observations).

    min_periods=3 suppresses delta for bursty topics with sparse history.
    Uses observation-count rolling (not calendar-day) so gaps don't inflate windows.
    """
    rolling_mean = (
        sentiment_pivot
        .shift(1)                              # exclude today from its own baseline
        .rolling(window=window, min_periods=3)
        .mean()
    )
    return sentiment_pivot - rolling_mean


def _diverging_norm(pivot: pd.DataFrame) -> TwoSlopeNorm:
    """Build TwoSlopeNorm(vcenter=0) with sensible vmin/vmax.

    Ensures both sides of zero are always represented so the norm never errors.
    """
    finite = pivot.stack().dropna()
    vmin = float(finite.min()) if not finite.empty else -0.1
    vmax = float(finite.max()) if not finite.empty else  0.1
    vmin = min(vmin, -0.01)   # guarantee at least a sliver below zero
    vmax = max(vmax,  0.01)   # guarantee at least a sliver above zero
    return TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)


def _spike_ratios_on_date(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    lookback: int = 7,
    min_history: int = 3,
) -> pd.DataFrame:
    """Return a DataFrame of (topic_id, label, spike_ratio, article_count,
    sentiment_score, sentiment_delta) for all topics active on target_date
    that have a computable spike_ratio.

    spike_ratio = today_count / mean(prior lookback days).
    sentiment_delta = sentiment_score_today - mean(prior lookback sentiment).
    """
    cutoff = target_date - pd.Timedelta(days=lookback)
    rows = []

    # Pre-build label map (most recent label per topic)
    label_map = (
        df.sort_values("date")
          .groupby("topic_id")["label"]
          .last()
          .to_dict()
    )

    # Pre-build sentiment delta map
    has_sent = _has_sentiment(df)

    for tid, grp in df.groupby("topic_id"):
        grp = grp.sort_values("date")
        today = grp[grp["date"] == target_date]
        if today.empty:
            continue
        today_count = int(today["article_count"].iloc[0])
        prior = grp[(grp["date"] >= cutoff) & (grp["date"] < target_date)]
        if len(prior) < min_history or prior["article_count"].mean() == 0:
            continue

        spike = today_count / prior["article_count"].mean()

        score = delta = None
        if has_sent:
            score_val = today["sentiment_score"].iloc[0]
            score = float(score_val) if not pd.isna(score_val) else None
            prior_scores = prior["sentiment_score"].dropna()
            if score is not None and len(prior_scores) >= min_history:
                delta = score - float(prior_scores.mean())

        rows.append({
            "topic_id":        tid,
            "label":           label_map.get(tid, tid[:8]),
            "spike_ratio":     round(spike, 3),
            "article_count":   today_count,
            "sentiment_score": score,
            "sentiment_delta": delta,
        })

    return pd.DataFrame(rows)


# ── Chart D: sentiment heatmap ────────────────────────────────────────────────

def plot_sentiment_heatmap(df: pd.DataFrame, top_ids: list[str], out_path: Path) -> None:
    """Diverging heatmap: dates × top-N topics, cell = sentiment_score.

    Red = bearish (negative), green = bullish (positive), grey = no data.
    Uses TwoSlopeNorm so the neutral midpoint is always white/yellow, not
    distorted by an asymmetric data range.
    """
    if not _has_sentiment(df):
        print("  Skipping sentiment heatmap — no sentiment_score data.")
        return

    pivot = _build_sentiment_pivot(df, top_ids)
    if pivot.empty:
        print("  Skipping sentiment heatmap — empty pivot.")
        return

    norm = _diverging_norm(pivot)
    cmap = copy.copy(plt.get_cmap("RdYlGn"))
    cmap.set_bad(color="#d0d0d0")   # grey for NaN (no data), distinct from neutral white

    n_dates, n_topics = pivot.shape
    row_h = max(0.14, 0.22 - (n_dates - 90) * 0.001) if n_dates > 90 else 0.22
    fig_h = max(8, n_dates * row_h)
    fig_w = max(10, n_topics * 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        norm=norm,
        linewidths=0.3,
        linecolor="#f0f0f0",
        cbar_kws={
            "label": "sentiment score  (-1 = bearish  |  0 = neutral  |  +1 = bullish)",
            "shrink": 0.6,
        },
        annot=False,
    )
    ax.set_title(
        "Topic Sentiment Heatmap  (red = bearish · green = bullish · grey = no data)\n"
        "Note: score = mean sector sentiment on each article's date — not per-article",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=40, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _save_figure(fig, out_path)
    print(f"  Saved: {out_path}")


# ── Chart E: sentiment delta (momentum) heatmap ───────────────────────────────

def plot_sentiment_delta(df: pd.DataFrame, top_ids: list[str], out_path: Path) -> None:
    """Diverging heatmap of 7-day rolling sentiment momentum.

    cell = sentiment_score_today − mean(prior 7 observations).
    Red = deteriorating, green = improving, grey = insufficient history.
    Topics with fewer than 3 scored observations are suppressed.
    """
    if not _has_sentiment(df):
        print("  Skipping sentiment delta — no sentiment_score data.")
        return

    sentiment_pivot = _build_sentiment_pivot(df, top_ids)
    delta_pivot = _compute_delta_pivot(sentiment_pivot)

    # Drop columns that are all-NaN (topics with no delta signal in the window)
    delta_pivot = delta_pivot.dropna(axis=1, how="all")
    if delta_pivot.empty:
        print("  Skipping sentiment delta — insufficient history for any topic.")
        return

    norm = _diverging_norm(delta_pivot)
    cmap = copy.copy(plt.get_cmap("RdYlGn"))
    cmap.set_bad(color="#d0d0d0")

    n_dates, n_topics = delta_pivot.shape
    row_h = max(0.14, 0.22 - (n_dates - 90) * 0.001) if n_dates > 90 else 0.22
    fig_h = max(8, n_dates * row_h)
    fig_w = max(10, n_topics * 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        delta_pivot,
        ax=ax,
        cmap=cmap,
        norm=norm,
        linewidths=0.3,
        linecolor="#f0f0f0",
        cbar_kws={
            "label": "sentiment delta  (score − 7-day rolling mean)  red=worse · green=better",
            "shrink": 0.6,
        },
        annot=False,
    )
    ax.set_title(
        "Topic Sentiment Momentum  (7-day rolling delta)\n"
        "Green = improving narrative · Red = deteriorating · Grey = < 3 observations",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=40, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _save_figure(fig, out_path)
    print(f"  Saved: {out_path}")


# ── Chart F: spike × sentiment signal scatter ─────────────────────────────────

_SCATTER_SPIKE_THRESHOLD = 1.0   # include topics with spike_ratio >= this value
_SCATTER_MAX_TOPICS      = 40    # cap to prevent overplotting


def plot_signal_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter: spike_ratio (X, log) × sentiment_score (Y) on the most recent date.

    All topics with spike_ratio >= 1.0 are plotted (threshold-based, not top-N).
    Dot size = article_count.  Dot colour = sentiment_delta (improving = green).
    Quadrant lines divide the chart into four trading signal zones.
    Top-8 dots by |spike_ratio × sentiment_score| are labelled.
    """
    if not _has_sentiment(df):
        print("  Skipping signal scatter — no sentiment_score data.")
        return

    target_date = df["date"].max()
    scatter_df = _spike_ratios_on_date(df, target_date)

    # Filter: threshold + minimum article count
    scatter_df = scatter_df[
        (scatter_df["spike_ratio"] >= _SCATTER_SPIKE_THRESHOLD) &
        (scatter_df["article_count"] >= 5) &
        scatter_df["sentiment_score"].notna()
    ].copy()

    if scatter_df.empty:
        print("  Skipping signal scatter — no qualifying topics on most recent date.")
        return

    # Cap to prevent overplotting; keep the most extreme spike_ratios
    if len(scatter_df) > _SCATTER_MAX_TOPICS:
        scatter_df = scatter_df.nlargest(_SCATTER_MAX_TOPICS, "spike_ratio")

    # Sentiment delta colour: green=improving, red=deteriorating, grey=unknown
    delta_vals = scatter_df["sentiment_delta"].fillna(0.0)
    delta_norm = _diverging_norm(pd.DataFrame(delta_vals))
    cmap_delta = copy.copy(plt.get_cmap("RdYlGn"))

    # Dot size: article_count scaled to [80, 600]
    cnt = scatter_df["article_count"].astype(float)
    size_min, size_max = 80.0, 600.0
    c_min, c_max = cnt.min(), cnt.max()
    if c_max > c_min:
        sizes = size_min + (cnt - c_min) / (c_max - c_min) * (size_max - size_min)
    else:
        sizes = pd.Series([200.0] * len(cnt), index=cnt.index)

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)

    sc = ax.scatter(
        scatter_df["spike_ratio"],
        scatter_df["sentiment_score"],
        s=sizes,
        c=delta_vals,
        cmap=cmap_delta,
        norm=delta_norm,
        alpha=0.80,
        edgecolors="#444444",
        linewidths=0.5,
        zorder=3,
    )

    # Quadrant guide lines
    ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=2)
    # Vertical spike threshold: median of visible data (adaptive to data range)
    x_split = float(np.median(scatter_df["spike_ratio"]))
    ax.axvline(x_split, color="#888888", linewidth=1.0, linestyle="--", zorder=2)

    # Quadrant labels — placed in axes-fraction coordinates so they stay on-screen
    # regardless of the actual data range on the log-scale X axis.
    _ql_kw = dict(
        fontsize=8, color="#aaaaaa", ha="center", va="center",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.7),
    )
    ax.text(0.20, 0.82, "quiet\npositive",     **_ql_kw)
    ax.text(0.78, 0.82, "momentum /\nbullish", **_ql_kw)
    ax.text(0.20, 0.18, "slow bleed",          **_ql_kw)
    ax.text(0.78, 0.18, "crisis /\nrisk-off",  **_ql_kw)

    # Label top-8 most extreme dots (highest |spike × sentiment|)
    scatter_df = scatter_df.copy()
    scatter_df["_signal"] = (
        scatter_df["spike_ratio"] * scatter_df["sentiment_score"].abs()
    )
    top_label = scatter_df.nlargest(8, "_signal")

    for _, row in top_label.iterrows():
        ax.annotate(
            row["label"][:32],
            xy=(row["spike_ratio"], row["sentiment_score"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=7,
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
        )

    # Colour bar for delta
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("sentiment delta  (green=improving · red=deteriorating · grey=unknown)",
                   fontsize=8)

    # Legend for dot size — use Line2D proxy handles to avoid PathCollection
    # crash on log-scale axes when scatter([], []) produces an empty path.
    legend_handles = []
    for cnt_val, lbl in [(10, "10 articles"), (30, "30 articles"), (60, "60 articles")]:
        if cnt_val <= c_max:
            s = size_min + (cnt_val - c_min) / max(c_max - c_min, 1) * (size_max - size_min)
            # scatter 's' is area in pt²; markersize is diameter in pt → 2*sqrt(s/π)
            ms = 2.0 * np.sqrt(s / np.pi)
            legend_handles.append(mlines.Line2D(
                [], [], linewidth=0, marker="o",
                color="#888888", alpha=0.6,
                markeredgecolor="#444444", markeredgewidth=0.5,
                markersize=ms, label=lbl,
            ))
    if legend_handles:
        ax.legend(handles=legend_handles, title="article count",
                  loc="upper left", fontsize=7, title_fontsize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Spike ratio  (log scale — today vs. 7-day baseline)", fontsize=10)
    ax.set_ylabel("Sentiment score  (-1 = bearish  ·  0 = neutral  ·  +1 = bullish)", fontsize=10)
    ax.set_title(
        f"Topic Signal Landscape  —  {target_date.strftime('%Y-%m-%d')}\n"
        f"All topics with spike ratio >= {_SCATTER_SPIKE_THRESHOLD}  "
        f"(dot size = article count, colour = 7-day sentiment change)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(alpha=0.25)

    _save_figure(fig, out_path)
    print(f"  Saved: {out_path}")


# ── Chart G: animated signal scatter ─────────────────────────────────────────

def plot_signal_scatter_animation(
    df: pd.DataFrame,
    out_path: Path,
    fps: int = 4,
) -> None:
    """Animated GIF of chart F: one frame per date, first → latest.

    Axes (log-scale X, fixed Y [-1.15, 1.15]), colour norm (delta range), and
    the quadrant split line are all computed globally so bubble movement is
    directly comparable across frames — a bubble jumping quadrants means a real
    shift in narrative signal, not a rescaling artefact.

    Args:
        df:       Full trends DataFrame (from load_trends).
        out_path: Destination path for the .gif file.
        fps:      Frames per second (default 4 → ~250 ms/frame).

    Invariants:
        - Skipped gracefully when sentiment_score column is absent.
        - Frames with fewer than 2 qualifying topics are skipped silently.
        - Global axis bounds and colour norm are fixed across all frames.
        - Quadrant split = global median spike_ratio (stable reference line).
        - Dot size scale is fixed globally so size encodes absolute article volume.
        - Saved via PillowWriter (Pillow dep) — no ffmpeg required.
    """
    if not _has_sentiment(df):
        print("  Skipping scatter animation — no sentiment_score data.")
        return

    all_dates = sorted(df["date"].unique())

    # ── Pre-compute per-date scatter DataFrames ──────────────────────────────
    frames_data: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for d in all_dates:
        sdf = _spike_ratios_on_date(df, d)
        if sdf.empty or "spike_ratio" not in sdf.columns:
            continue
        sdf = sdf[
            (sdf["spike_ratio"] >= _SCATTER_SPIKE_THRESHOLD) &
            (sdf["article_count"] >= 5) &
            sdf["sentiment_score"].notna()
        ].copy()
        if len(sdf) >= 2:
            frames_data.append((d, sdf))

    if len(frames_data) < 2:
        print("  Skipping scatter animation — fewer than 2 qualifying frames.")
        return

    print(f"  Building animation: {len(frames_data)} frames @ {fps} fps…")

    # ── Global bounds (fixed axes = comparable across frames) ─────────────────
    all_ratios = pd.concat([f[1]["spike_ratio"]              for f in frames_data])
    all_deltas = pd.concat([f[1]["sentiment_delta"].fillna(0.0) for f in frames_data])
    all_counts = pd.concat([f[1]["article_count"].astype(float) for f in frames_data])

    xmin = all_ratios.min() * 0.8
    xmax = all_ratios.max() * 1.3
    x_split = float(all_ratios.median())   # fixed quadrant line

    delta_norm = _diverging_norm(pd.DataFrame(all_deltas))
    cmap_delta = copy.copy(plt.get_cmap("RdYlGn"))

    c_min, c_max = float(all_counts.min()), float(all_counts.max())
    size_min, size_max = 80.0, 600.0

    # ── Figure setup (persists across frames) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)

    sm = plt.cm.ScalarMappable(cmap=cmap_delta, norm=delta_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(
        "sentiment delta  (green = improving · red = deteriorating · grey = unknown)",
        fontsize=8,
    )

    # ── Per-frame draw function ───────────────────────────────────────────────
    def _draw_frame(i: int) -> None:
        ax.cla()
        target_date, sdf = frames_data[i]

        if len(sdf) > _SCATTER_MAX_TOPICS:
            sdf = sdf.nlargest(_SCATTER_MAX_TOPICS, "spike_ratio")

        delta_vals = sdf["sentiment_delta"].fillna(0.0)
        cnt        = sdf["article_count"].astype(float)
        if c_max > c_min:
            sizes = size_min + (cnt - c_min) / (c_max - c_min) * (size_max - size_min)
        else:
            sizes = pd.Series([200.0] * len(cnt), index=cnt.index)

        ax.scatter(
            sdf["spike_ratio"],
            sdf["sentiment_score"],
            s=sizes,
            c=delta_vals,
            cmap=cmap_delta,
            norm=delta_norm,
            alpha=0.80,
            edgecolors="#444444",
            linewidths=0.5,
            zorder=3,
        )

        # Quadrant guide lines (fixed reference)
        ax.axhline(0,       color="#888888", linewidth=1.0, linestyle="--", zorder=2)
        ax.axvline(x_split, color="#888888", linewidth=1.0, linestyle="--", zorder=2)

        _ql_kw = dict(
            fontsize=8, color="#aaaaaa", ha="center", va="center",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.7),
        )
        ax.text(0.20, 0.82, "quiet\npositive",     **_ql_kw)
        ax.text(0.78, 0.82, "momentum /\nbullish", **_ql_kw)
        ax.text(0.20, 0.18, "slow bleed",          **_ql_kw)
        ax.text(0.78, 0.18, "crisis /\nrisk-off",  **_ql_kw)

        # Label top-5 most extreme dots (|spike × sentiment|)
        sdf = sdf.copy()
        sdf["_signal"] = sdf["spike_ratio"] * sdf["sentiment_score"].abs()
        for _, row in sdf.nlargest(5, "_signal").iterrows():
            ax.annotate(
                row["label"][:32],
                xy=(row["spike_ratio"], row["sentiment_score"]),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=7,
                color="#222222",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
            )

        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel("Spike ratio  (log scale — today vs. 7-day baseline)", fontsize=10)
        ax.set_ylabel(
            "Sentiment score  (-1 = bearish  ·  0 = neutral  ·  +1 = bullish)",
            fontsize=10,
        )
        ax.set_title(
            f"Topic Signal Landscape  —  {target_date.strftime('%Y-%m-%d')}  "
            f"[{i + 1} / {len(frames_data)}]\n"
            f"All topics with spike ratio ≥ {_SCATTER_SPIKE_THRESHOLD}  "
            "(dot size = article count, colour = 7-day sentiment change)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.grid(alpha=0.25)

        if (i + 1) % 10 == 0 or i == len(frames_data) - 1:
            print(f"  Frame {i + 1}/{len(frames_data)}", flush=True)

    anim = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=len(frames_data),
        interval=1000 // fps,
        repeat=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out_path}  ({len(frames_data)} frames, {fps} fps)")


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
    p.add_argument("--top",     type=int,  default=15,  help="Top N topics (default: 15)")
    p.add_argument("--days",    type=int,  default=200, help="Lookback window in days (default: 200)")
    p.add_argument("--animate", action="store_true",
                   help="Also generate animated GIF (chart G) — slow, one frame per qualifying date")
    p.add_argument("--fps",     type=int,  default=4,
                   help="Frames per second for the animated GIF (default: 4)")
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

    print("Generating sentiment heatmap...")
    plot_sentiment_heatmap(df, top_ids, CHARTS_DIR / "topic_sentiment_heatmap.png")

    print("Generating sentiment delta (momentum) heatmap...")
    plot_sentiment_delta(df, top_ids, CHARTS_DIR / "topic_sentiment_delta.png")

    print("Generating signal scatter (spike x sentiment)...")
    plot_signal_scatter(df, CHARTS_DIR / "topic_signal_scatter.png")

    if args.animate:
        # Load full history (no day cap) so the animation spans all available dates.
        df_full = load_trends(lookback_days=10_000)
        print(f"\nGenerating animated signal scatter ({args.fps} fps)...")
        plot_signal_scatter_animation(
            df_full,
            CHARTS_DIR / "topic_signal_scatter_animated.gif",
            fps=args.fps,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
