"""High-level briefing batch collection orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI

from constants import BRIEFINGS_BATCH_META_FILE, BRIEFINGS_DIR, PENDING_BRIEFINGS_BATCH_FILE

from .batch_briefings import (
    check_briefings_batch_status,
    collect_briefing_results,
    read_spike_metadata,
    save_briefings,
)


def _read_pending_briefings_batch_id(pending_batch_file: Path) -> str:
    """Read the active briefing batch ID with a briefing-specific error message."""
    if not pending_batch_file.exists():
        print(
            f"[error] No pending batch found at {pending_batch_file}.\n"
            "Run create_batch_briefings.py first to submit a batch job."
        )
        raise SystemExit(1)
    return pending_batch_file.read_text(encoding="utf-8").strip()


def run_briefing_batch_collection(
    client: OpenAI,
    pending_batch_file: Path = PENDING_BRIEFINGS_BATCH_FILE,
    metadata_file: Path = BRIEFINGS_BATCH_META_FILE,
    briefings_dir: Path = BRIEFINGS_DIR,
) -> dict[str, Any]:
    """Collect a completed briefing batch and write briefing JSON files."""
    print("=== Briefing Batch Collection ===")

    batch_id = _read_pending_briefings_batch_id(pending_batch_file)
    metadata = read_spike_metadata(metadata_file)
    print(f"Pending batch : {batch_id}")
    print(f"Spike metadata: {len(metadata)} tasks loaded from {metadata_file}")

    check_briefings_batch_status(client, batch_id)

    from kitai.batch import download_batch_results

    print("Downloading batch results...")
    items = download_batch_results(client, batch_id)
    print(f"Downloaded {len(items)} item(s).")

    spikes_by_date, ok_count, failed_count = collect_briefing_results(items, metadata)

    print("\nWriting briefing files...")
    written = save_briefings(spikes_by_date, briefings_dir)

    print(
        f"\nCollection complete: {ok_count} ok, {failed_count} failed, "
        f"{written} briefing file(s) written."
    )

    if failed_count == 0:
        pending_batch_file.unlink()
        metadata_file.unlink()
        print("Cleared sentinel files.")
    else:
        print(
            f"[warn] {failed_count} item(s) failed — keeping sentinel files for inspection.\n"
            "Manually delete them after investigating the failures above."
        )

    return {
        "batch_id": batch_id,
        "metadata_count": len(metadata),
        "item_count": len(items),
        "ok_count": ok_count,
        "failed_count": failed_count,
        "written": written,
    }
