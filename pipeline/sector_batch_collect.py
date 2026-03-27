"""High-level sector batch collection orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI

from .batch_collect import (
    check_batch_status,
    download_results,
    save_batch_results,
)

PENDING_BATCH_FILE = Path("data") / "pending_sector_batch.txt"
RAW_OUTPUT_FILE = Path("data") / "batch_output_sector.jsonl"
RESULTS_DIR = Path("data") / "sector_results"


def _read_pending_sector_batch_id(pending_batch_file: Path) -> str:
    """Read the active sector batch ID with a sector-specific error message."""
    if not pending_batch_file.exists():
        print(
            f"[error] No pending batch found at {pending_batch_file}.\n"
            "Run the sector batch submit step first to create a batch job."
        )
        raise SystemExit(1)
    return pending_batch_file.read_text(encoding="utf-8").strip()


def run_sector_batch_collection(
    client: OpenAI,
    pending_batch_file: Path = PENDING_BATCH_FILE,
    raw_output_file: Path = RAW_OUTPUT_FILE,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Collect a completed sector batch and write per-date JSON result files."""
    print("=== Sector Batch Collection ===")

    batch_id = _read_pending_sector_batch_id(pending_batch_file)
    print(f"Found pending batch: {batch_id}")

    check_batch_status(batch_id, client)
    items = download_results(batch_id, client, output_file=raw_output_file)
    ok_count, failed_count = save_batch_results(items, batch_id, results_dir)

    print(f"\nCollection complete: {ok_count} ok, {failed_count} failed.")

    if failed_count == 0:
        pending_batch_file.unlink()
        print(f"Cleared {pending_batch_file}.")
    else:
        print(
            f"[warn] {failed_count} item(s) failed — keeping {pending_batch_file} for inspection.\n"
            "Manually delete it after investigating the failures above."
        )

    return {
        "batch_id": batch_id,
        "item_count": len(items),
        "ok_count": ok_count,
        "failed_count": failed_count,
        "results_dir": results_dir,
    }
