"""Sentiment chart CLI wrapper."""

from pipeline.sentiment_visualization import run_sentiment_visualizations


def main() -> None:
    run_sentiment_visualizations()


if __name__ == "__main__":
    main()
