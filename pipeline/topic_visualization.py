"""High-level topic-chart orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .topic_charts import (
    load_trends,
    pick_top_topics,
    plot_frequency_ts,
    plot_sentiment_delta,
    plot_sentiment_heatmap,
    plot_signal_scatter,
    plot_signal_scatter_animation,
    plot_spike_heatmap,
    plot_topic_timeline,
)

TOPIC_TRENDS_FILE = Path("data") / "topic_trends.tsv"
CHARTS_DIR = Path("data") / "charts"
DEFAULT_TOP_N = 15
DEFAULT_LOOKBACK_DAYS = 200
DEFAULT_ANIMATION_FPS = 4


def parse_topic_visualization_args() -> argparse.Namespace:
    """Parse CLI args for visualize_topics.py."""
    parser = argparse.ArgumentParser(description="Visualize topic trends")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="Top N topics (default: 15)")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Lookback window in days (default: 200)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Also generate animated GIF (chart G) — slow, one frame per qualifying date",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_ANIMATION_FPS,
        help="Frames per second for the animated GIF (default: 4)",
    )
    return parser.parse_args()


def run_topic_visualizations(
    top_n: int = DEFAULT_TOP_N,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    animate: bool = False,
    fps: int = DEFAULT_ANIMATION_FPS,
    trends_file: Path = TOPIC_TRENDS_FILE,
    charts_dir: Path = CHARTS_DIR,
) -> dict[str, Any]:
    """Generate the topic visualization set and return a small execution summary."""
    charts_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Topic Visualizations (top={top_n}, days={lookback_days}) ===")

    df = load_trends(lookback_days, trends_file=trends_file)
    top_ids = pick_top_topics(df, top_n)
    actual_n = len(top_ids)
    print(f"Selected {actual_n} topic(s) for charts.")

    outputs = {
        "topic_spike_heatmap": charts_dir / "topic_spike_heatmap.png",
        "topic_frequency_ts": charts_dir / "topic_frequency_ts.png",
        "topic_timeline": charts_dir / "topic_timeline.png",
        "topic_sentiment_heatmap": charts_dir / "topic_sentiment_heatmap.png",
        "topic_sentiment_delta": charts_dir / "topic_sentiment_delta.png",
        "topic_signal_scatter": charts_dir / "topic_signal_scatter.png",
    }

    print("\nGenerating spike heatmap...")
    plot_spike_heatmap(df, top_ids, outputs["topic_spike_heatmap"])

    print("Generating frequency time series...")
    plot_frequency_ts(df, top_ids, outputs["topic_frequency_ts"])

    print("Generating topic timeline...")
    plot_topic_timeline(df, top_ids, outputs["topic_timeline"])

    print("Generating sentiment heatmap...")
    plot_sentiment_heatmap(df, top_ids, outputs["topic_sentiment_heatmap"])

    print("Generating sentiment delta (momentum) heatmap...")
    plot_sentiment_delta(df, top_ids, outputs["topic_sentiment_delta"])

    print("Generating signal scatter (spike x sentiment)...")
    plot_signal_scatter(df, outputs["topic_signal_scatter"])

    if animate:
        df_full = load_trends(lookback_days=10_000, trends_file=trends_file)
        outputs["topic_signal_scatter_animated"] = (
            charts_dir / "topic_signal_scatter_animated.gif"
        )
        print(f"\nGenerating animated signal scatter ({fps} fps)...")
        plot_signal_scatter_animation(
            df_full,
            outputs["topic_signal_scatter_animated"],
            fps=fps,
        )

    print("\nDone.")
    return {
        "top_n_requested": top_n,
        "top_n_selected": actual_n,
        "lookback_days": lookback_days,
        "animate": animate,
        "fps": fps,
        "outputs": outputs,
    }
