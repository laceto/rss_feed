"""Topic chart CLI wrapper."""

from __future__ import annotations

from pipeline.topic_visualization import (
    parse_topic_visualization_args,
    run_topic_visualizations,
)


def main() -> None:
    args = parse_topic_visualization_args()
    run_topic_visualizations(
        top_n=args.top,
        lookback_days=args.days,
        animate=args.animate,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
