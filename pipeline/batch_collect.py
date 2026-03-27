"""OpenAI Batch API result collection — polling, download, and routing.

Extracted from retrieve_batch_file_results.py.

Module-level path constants and the OpenAI client have been converted to
explicit parameters so callers can inject their own paths and credentials.

Polling and download use kitai.batch — check_batch_job returns a structured
status dict; download_batch_results returns a parsed list[dict] directly.

Exit-code contract (preserved from source script):
    sys.exit(0) — sentinel file absent, nothing to do (safe no-op in CI)
    sys.exit(1) — hard error (bad sentinel, failed/expired batch)
    sys.exit(2) — batch still in flight; caller should retry later
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kitai.batch import check_batch_job, download_batch_results
from openai import OpenAI

# ── Constants ──────────────────────────────────────────────────────────────────

# Batch terminal states — anything else means still in flight.
# Kept as a reference constant; internal logic uses kitai's is_terminal / is_complete.
TERMINAL_STATES: frozenset[str] = frozenset({"completed", "failed", "expired", "cancelled"})


# ── Batch status ────────────────────────────────────────────────────────────────


def read_pending_batch_id(pending_file: Path) -> str:
    """Read the active batch job ID from the sentinel file.

    Args:
        pending_file: Path to the sentinel file (e.g. data/pending_sector_batch.txt).

    Returns:
        The batch ID string.

    NOTE: calls sys.exit(1) when the sentinel file is absent.
    """
    if not pending_file.exists():
        print(  # NOTE: uses print()
            f"[error] No pending batch found at {pending_file}.\n"
            "Run create_batch_files_v2.py first to submit a batch job."
        )
        sys.exit(1)  # NOTE: uses sys.exit(1) — sentinel missing
    return pending_file.read_text().strip()


def check_batch_status(batch_id: str, client: OpenAI) -> dict:
    """Check batch progress via kitai.batch.check_batch_job and report status.

    Args:
        batch_id: The batch job ID from read_pending_batch_id().
        client:   Authenticated OpenAI client instance.

    Returns:
        The status dict from kitai.batch.check_batch_job when the batch has
        completed successfully.
        Keys: batch_id, status, is_terminal, is_complete,
              counts (total/completed/failed), output_file_id, error_file_id.

    NOTE: calls sys.exit(2) if the batch is still in flight (retry is safe).
    NOTE: calls sys.exit(1) if the batch failed, expired, or was cancelled.
    """
    status = check_batch_job(client, batch_id)
    counts = status.get("counts", {})

    print(  # NOTE: uses print()
        f"Batch {batch_id} -> status: {status['status']} | "
        f"total: {counts.get('total', '?')} | "
        f"completed: {counts.get('completed', '?')} | "
        f"failed: {counts.get('failed', '?')}"
    )

    if not status.get("is_terminal"):
        print("Batch is still in progress. Re-run this script once it completes.")  # NOTE: uses print()
        sys.exit(2)  # NOTE: uses sys.exit(2) = not ready yet; retry is safe

    if not status.get("is_complete"):
        print(  # NOTE: uses print()
            f"[error] Batch ended with status '{status['status']}'. "
            "No results to collect. Check the OpenAI dashboard for details."
        )
        sys.exit(1)  # NOTE: uses sys.exit(1) — batch failed/expired/cancelled

    return status


# ── Download ────────────────────────────────────────────────────────────────────


def download_results(
    batch_id: str,
    client: OpenAI,
    output_file: Path | None = None,
) -> list[dict]:
    """Download batch output via kitai.batch.download_batch_results.

    Each element is one batch item: {custom_id, response, error}.
    Optionally saves a local JSONL copy to output_file for debugging.

    Args:
        batch_id:    The batch job ID (pass the return value of check_batch_status()).
        client:      Authenticated OpenAI client instance.
        output_file: Optional path for a local JSONL debug copy
                     (e.g. data/batch_output_sector.jsonl). No file written when None.

    Returns:
        List of parsed batch item dicts.

    NOTE: signature changed from download_results(batch, output_file, client) —
          now takes batch_id (str) instead of the batch object, and output_file
          is optional and follows client in the argument list.
    """
    print(f"Downloading batch results for {batch_id}...")  # NOTE: uses print()
    items = download_batch_results(client, batch_id)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"Raw output saved -> {output_file}")  # NOTE: uses print()

    print(f"Parsed {len(items)} item(s) from batch output.")  # NOTE: uses print()
    return items


# ── Parsing and routing ──────────────────────────────────────────────────────────


def date_from_custom_id(custom_id: str) -> str | None:
    """Extract date string from 'sector-YYYY-MM-DD' custom_id format.

    Returns:
        The YYYY-MM-DD date string, or None if the format does not match.
        Returning None is defensive — should not happen with tasks generated
        by build_batch_tasks().
    """
    prefix = "sector-"
    if custom_id.startswith(prefix):
        return custom_id[len(prefix):]
    return None


def parse_sectors(content_str: str) -> dict | None:
    """Parse the LLM JSON response string into a dict.

    Returns:
        Parsed dict on success, or None if JSON parsing fails.
        Since batch tasks use strict JSON schema mode, malformed JSON
        here indicates an API-level issue.
    """
    try:
        parsed = json.loads(content_str)
        if "sectors" not in parsed:
            print("[warn] Response JSON has no 'sectors' key — storing as-is.")  # NOTE: uses print()
        return parsed
    except json.JSONDecodeError as exc:
        print(f"[error] Failed to parse response JSON: {exc}")  # NOTE: uses print()
        return None


def save_batch_results(
    items: list[dict],
    batch_id: str,
    results_dir: Path,
) -> tuple[int, int]:
    """Parse each batch item and write to results_dir/{date}.json.

    Args:
        items:       Parsed batch items from download_results().
        batch_id:    The batch job ID (written into each result file).
        results_dir: Directory where per-date result files are written.

    Returns:
        (ok_count, failed_count)

    Each output file format:
        {
            "date": "YYYY-MM-DD",
            "batch_id": "batch_...",
            "sectors": [{ entities, sector, sentiment, news_category, extraction_status }, ...]
        }
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    failed_count = 0

    for item in items:
        custom_id = item.get("custom_id", "unknown")
        date = date_from_custom_id(custom_id)

        if date is None:
            print(f"[warn] Unrecognised custom_id '{custom_id}' — skipping.")  # NOTE: uses print()
            failed_count += 1
            continue

        # ── Check for item-level error ──────────────────────────────────────
        if item.get("error"):
            print(f"[fail] {date}: batch item error -> {item['error']}")  # NOTE: uses print()
            failed_count += 1
            continue

        response = item.get("response")
        if response is None:
            print(f"[fail] {date}: response is null (no error field either).")  # NOTE: uses print()
            failed_count += 1
            continue

        status_code = response.get("status_code")
        if status_code != 200:
            print(f"[fail] {date}: HTTP {status_code} from API.")  # NOTE: uses print()
            failed_count += 1
            continue

        # ── Parse content ───────────────────────────────────────────────────
        try:
            content_str = response["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            print(f"[fail] {date}: unexpected response structure -> {exc}")  # NOTE: uses print()
            failed_count += 1
            continue

        sectors_data = parse_sectors(content_str)
        if sectors_data is None:
            failed_count += 1
            continue

        # ── Write result file ────────────────────────────────────────────────
        result = {"date": date, "batch_id": batch_id, **sectors_data}
        out_path = results_dir / f"{date}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(  # NOTE: uses print()
            f"[ok]   {date} -> {out_path} ({len(sectors_data.get('sectors', []))} sector(s))"
        )
        ok_count += 1

    return ok_count, failed_count
