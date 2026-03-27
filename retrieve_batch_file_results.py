"""Sector batch collection CLI wrapper."""

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.sector_batch_collect import run_sector_batch_collection

load_dotenv()


def main() -> None:
    run_sector_batch_collection(OpenAI())


if __name__ == "__main__":
    main()
