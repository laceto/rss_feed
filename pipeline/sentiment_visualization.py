"""High-level sentiment-chart orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sentiment_charts import (
    chart_distribution,
    chart_heatmap,
    chart_trends,
    load_sentiment_data,
)

INPUT_FILE = Path("data") / "sector_summary.tsv"
CHARTS_DIR = Path("data") / "charts"


def run_sentiment_visualizations(
    input_file: Path = INPUT_FILE,
    charts_dir: Path = CHARTS_DIR,
) -> dict[str, Any]:
    """Generate all sentiment charts and return a small execution summary."""
    print("=== Sentiment Trend Visualizations ===")
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = load_sentiment_data(input_file)

    chart_heatmap(df, charts_dir)
    chart_trends(df, charts_dir)
    chart_distribution(df, charts_dir)

    print(f"\nAll charts saved to {charts_dir}/")
    return {
        "row_count": len(df),
        "date_count": int(df["date"].nunique()),
        "sector_count": int(df["sector"].nunique()),
        "charts_dir": charts_dir,
    }
