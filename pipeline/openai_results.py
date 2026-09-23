"""
openai_results.py
Shared plumbing for structured-output chat completions, used by both the track
and the artist enrichment paths (batch and direct).

  validate_result_item()   Batch-API-shaped result item -> validated Pydantic model
  call_chat_completions()  direct (non-batch) calls, wrapped in the same item shape

Both transports produce the same item shape:
    {"custom_id": str, "error": dict | None,
     "response": {"status_code": int, "body": <ChatCompletion dict>} | None}
so one validator serves both, and a failure is always a ResultError with a
human-readable reason (logged per item by the CLIs, never fatal for a run).

Invariant: no module-level side effects; no client construction here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


class ResultError(Exception):
    """One result item could not be turned into a valid model instance."""


def validate_result_item(item: dict, model_cls: type[M]) -> M:
    """Check transport/HTTP/refusal/truncation, then validate content against model_cls.

    Raises:
        ResultError: item-level error, non-200 HTTP, unexpected structure, model
            refusal, finish_reason=length, or JSON/schema mismatch.
    """
    if item.get("error"):
        raise ResultError(f"batch item error: {item['error']}")

    response = item.get("response") or {}
    status = response.get("status_code")
    if status != 200:
        raise ResultError(f"HTTP {status}")

    try:
        choice = response["body"]["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ResultError(f"unexpected response structure: {exc!r}") from exc

    if message.get("refusal"):
        raise ResultError(f"model refused: {message['refusal']}")
    if choice.get("finish_reason") == "length":
        raise ResultError("output truncated (finish_reason=length)")

    try:
        return model_cls.model_validate_json(message.get("content") or "")
    except ValidationError as exc:
        raise ResultError(f"response does not match schema: {exc.errors()[:3]}") from exc


def completion_to_result_item(custom_id: str, completion) -> dict:
    """Wrap a ChatCompletion in the Batch API result-item shape."""
    return {"custom_id": custom_id, "error": None,
            "response": {"status_code": 200, "body": completion.model_dump()}}


def error_to_result_item(custom_id: str, exc: Exception) -> dict:
    return {"custom_id": custom_id, "response": None,
            "error": {"type": type(exc).__name__, "message": str(exc)}}


def call_chat_completions(client, bodies: dict[str, dict], workers: int = 4) -> list[dict]:
    """Call chat.completions.create(**body) per custom_id with a thread pool.

    Exceptions are captured per item (error_to_result_item), never raised, so
    one bad request never loses the rest of the run. Order follows `bodies`.
    """
    def call(cid: str) -> dict:
        try:
            return completion_to_result_item(cid, client.chat.completions.create(**bodies[cid]))
        except Exception as exc:  # noqa: BLE001 — per-item failure is data, not a crash
            return error_to_result_item(cid, exc)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(call, list(bodies)))
