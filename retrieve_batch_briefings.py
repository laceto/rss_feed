"""Briefing batch collection CLI wrapper."""

from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.briefing_batch_collect import run_briefing_batch_collection

load_dotenv()


def main() -> None:
    run_briefing_batch_collection(OpenAI())


if __name__ == "__main__":
    main()
