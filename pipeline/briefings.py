"""Live briefing generation utilities.

Extracted from daily_briefing.py.

Covers the data layer only — no printing or sys.exit calls.
The CLI script (daily_briefing.py) owns all console output and exit behaviour.

Public API:
  KEYWORD_TO_SECTORS  — configurable keyword → sector mapping list
  infer_sectors       — topic label → relevant sector names
  rag_summary         — RAG narrative summary for a spiking topic label
  sector_crosscheck   — sector sentiment snapshots relevant to a topic label
  build_briefing      — full structured briefing dict for a date (no I/O side effects)
  load_precomputed    — load a pre-computed briefing JSON; returns None if absent
  save_briefing       — atomically write a briefing dict to data/briefings/{date}.json
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from cluster_topics import get_emerging_topics
from constants import BRIEFINGS_DIR, TOPIC_TRENDS_FILE

from .query_sector import get_time_series, list_sectors

log = logging.getLogger(__name__)

# ── Sector keyword map ────────────────────────────────────────────────────────
# Maps lowercase regex patterns that might appear in a topic label → sector names.
# Extend as needed; first match wins.

KEYWORD_TO_SECTORS: list[tuple[str, list[str]]] = [
    ("fed|rate|inflation|treasury|bond|yield",
     ["Finance", "Banking & Credit Services", "Real Estate"]),
    ("oil|energy|gas|crude|opec",
     ["Energy"]),
    ("tariff|trade|china|import|export|supply chain",
     ["Industrial Goods", "Technology Services", "Consumer Goods"]),
    ("ai|nvidia|chip|semiconductor",
     ["Technology Services", "Semiconductors & Electronics"]),
    ("drug|pharma|fda|biotech",
     ["Pharmaceuticals & Biotechnology"]),
    ("bank|credit|loan|mortgage",
     ["Banking & Credit Services", "Finance"]),
    ("retail|consumer|spending",
     ["Consumer Goods", "Retail & E-Commerce"]),
    ("auto|vehicle|ev|tesla",
     ["Automotive"]),
]


# ── Sector inference ──────────────────────────────────────────────────────────

def infer_sectors(label: str) -> list[str]:
    """Return a deduplicated list of sector names relevant to a topic label.

    Matches against KEYWORD_TO_SECTORS patterns; first match per sector wins.
    Returns [] for empty or non-string labels.

    Args:
        label: Human-readable topic label string.

    Returns:
        Ordered list of valid sector names (no duplicates).
    """
    if not label or not isinstance(label, str):
        return []
    label_lower = label.lower()
    valid = set(list_sectors())
    seen: list[str] = []
    for pattern, sectors in KEYWORD_TO_SECTORS:
        if re.search(pattern, label_lower):
            for s in sectors:
                if s in valid and s not in seen:
                    seen.append(s)
    return seen


# ── RAG summary ───────────────────────────────────────────────────────────────

def rag_summary(label: str) -> dict:
    """Query the RAG pipeline for a narrative summary of a spiking topic.

    Uses hybrid_rag.ask() with strategy='expand' for query augmentation.
    Fails gracefully — never raises; returns an error string in 'answer' instead.

    Args:
        label: Human-readable topic label used as the retrieval query.

    Returns:
        {"answer": str, "sources": list[dict]}
        or {"answer": "<error message>", "sources": []} on failure.
    """
    try:
        from hybrid_rag import ask
    except ImportError as exc:
        return {"answer": f"hybrid_rag import failed: {exc}", "sources": []}

    query = (
        f"What is happening with {label}? "
        f"Summarise the key developments and market implications in 3–4 sentences."
    )
    try:
        result = ask(query, strategy="expand", k_semantic=10, k_bm25=8)
        return {"answer": result["answer"], "sources": result["sources"][:4]}
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"RAG error: {exc}", "sources": []}


# ── Sector cross-check ────────────────────────────────────────────────────────

def sector_crosscheck(label: str, lookback_days: int = 14) -> list[dict]:
    """Return sector sentiment snapshots relevant to a topic label.

    Infers candidate sectors via infer_sectors(), then queries each for its
    recent sentiment trend.

    Args:
        label:        Topic label string used to infer relevant sectors.
        lookback_days: Rolling window for get_time_series (default: 14).

    Returns:
        List of dicts, one per matched sector, with keys:
            sector, trend_direction, trend_delta, mean_sentiment_score,
            data_age_days.
        Empty list when no sectors match or no data is available.
    """
    sectors = infer_sectors(label)
    results = []
    for sector in sectors:
        try:
            ts = get_time_series(sector, lookback_days=lookback_days)
            results.append({
                "sector":               sector,
                "trend_direction":      ts["trend_direction"],
                "trend_delta":          round(ts["trend_delta"], 3),
                "mean_sentiment_score": round(ts["mean_sentiment_score"], 3),
                "data_age_days":        ts.get("data_age_days", None),
            })
        except LookupError:
            pass  # no sector data yet — skip silently
        except Exception as exc:  # noqa: BLE001
            log.warning("sector crosscheck failed for %s: %s", sector, exc)
    return results


# ── Build briefing ────────────────────────────────────────────────────────────

def build_briefing(
    run_date: date,
    top_n: int = 5,
    use_rag: bool = True,
    trends_path: Path = TOPIC_TRENDS_FILE,
) -> dict:
    """Build a structured briefing dict for run_date.

    Pure data function — no printing, no sys.exit.
    The caller is responsible for displaying the result and handling exit codes.

    Args:
        run_date:    Date to generate the briefing for.
        top_n:       Maximum number of spikes to include (default: 5).
        use_rag:     Whether to query the RAG for narrative summaries (default: True).
        trends_path: Path to topic_trends.tsv (injectable for testing).

    Returns:
        {
            "date":     str,
            "n_spikes": int,
            "spikes":   list[{topic_id, label, spike_ratio, article_count,
                               rag_answer, rag_sources, sectors}]
        }

    Raises:
        FileNotFoundError: If trends_path does not exist.
    """
    if not Path(trends_path).exists():
        raise FileNotFoundError(
            f"{trends_path} not found. Run: python cluster_topics.py to generate it."
        )

    trends_df = pd.read_csv(trends_path, sep="\t")
    log.info(
        "Loaded %d rows from topic_trends.tsv (%d dates).",
        len(trends_df),
        trends_df["date"].nunique(),
    )

    spikes = get_emerging_topics(run_date, trends_df)
    if not spikes:
        return {"date": str(run_date), "n_spikes": 0, "spikes": []}

    spikes = spikes[:top_n]
    log.info("Found %d spike(s) — fetching context.", len(spikes))

    rag_results:    dict[str, dict]       = {}
    sector_results: dict[str, list[dict]] = {}

    for spike in spikes:
        tid   = spike["topic_id"]
        label = (spike["label"] if isinstance(spike["label"], str) else "") or ""

        if use_rag and label:
            log.info("RAG: %s", label[:60])
            rag_results[tid] = rag_summary(label)

        sector_results[tid] = sector_crosscheck(label)

    output_spikes = []
    for spike in spikes:
        tid = spike["topic_id"]
        rag = rag_results.get(tid, {})
        output_spikes.append({
            **spike,
            "rag_answer":  rag.get("answer", ""),
            "rag_sources": rag.get("sources", []),
            "sectors":     sector_results.get(tid, []),
        })

    return {
        "date":     str(run_date),
        "n_spikes": len(spikes),
        "spikes":   output_spikes,
    }


# ── Precomputed briefing I/O ──────────────────────────────────────────────────

def load_precomputed(
    run_date: date,
    briefings_dir: Path = BRIEFINGS_DIR,
) -> dict | None:
    """Load a pre-computed briefing JSON if it exists.

    Args:
        run_date:      Date to look up.
        briefings_dir: Directory containing {date}.json files (injectable).

    Returns:
        The briefing dict if the file exists, None otherwise.
    """
    path = Path(briefings_dir) / f"{run_date}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_briefing(
    briefing: dict,
    briefings_dir: Path = BRIEFINGS_DIR,
) -> Path:
    """Write a briefing dict to briefings_dir/{date}.json atomically.

    Args:
        briefing:      Briefing dict with a 'date' key (YYYY-MM-DD string).
        briefings_dir: Destination directory (injectable for testing).

    Returns:
        Path of the written file.
    """
    out_dir = Path(briefings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{briefing['date']}.json"
    path.write_text(json.dumps(briefing, indent=2, default=str), encoding="utf-8")
    return path
