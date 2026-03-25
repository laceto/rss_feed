"""
daily_briefing.py

Morning insight script: surfaces spiking narratives from topic clustering,
queries the RAG for a plain-language summary of each spike, and cross-checks
against sector sentiment to flag where narrative and price signal agree.

Usage:
    python daily_briefing.py                    # briefing for today
    python daily_briefing.py --date 2026-03-21  # briefing for a specific date
    python daily_briefing.py --top 5            # limit to top N spikes (default: 5)
    python daily_briefing.py --no-rag           # skip RAG summaries (faster)

Output:
    Printed briefing + optional data/briefings/{date}.json for downstream use.

Invariants:
    - Never writes to topic_trends.tsv or any pipeline output file.
    - RAG is read-only; no embeddings are created.
    - Missing trends file or zero spikes exits cleanly with a message.

Failure modes:
    - OPENAI_API_KEY missing: --no-rag still works; RAG mode exits with clear message.
    - topic_trends.tsv absent: exits with instructions to run cluster_topics.py.
    - Sector data absent: sector block is skipped, rest of briefing still prints.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from cluster_topics import get_emerging_topics
from constants import TOPIC_TRENDS_FILE
from query_sector import get_time_series, list_sectors

BRIEFINGS_DIR = PROJECT_ROOT / "data" / "briefings"

logging.basicConfig(
    level=logging.WARNING,          # suppress INFO from hybrid_rag / langchain
    format="%(levelname)s: %(message)s",
)

# ── Sector keyword map ─────────────────────────────────────────────────────────
# Maps lowercase keywords that might appear in a topic label → sector names.
# Extend as needed; first match wins.

_KEYWORD_TO_SECTORS: list[tuple[str, list[str]]] = [
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


def _infer_sectors(label: str) -> list[str]:
    """Return a deduplicated list of sector names relevant to a topic label."""
    import re
    if not label or not isinstance(label, str):
        return []
    label_lower = label.lower()
    valid = set(list_sectors())
    seen: list[str] = []
    for pattern, sectors in _KEYWORD_TO_SECTORS:
        if re.search(pattern, label_lower):
            for s in sectors:
                if s in valid and s not in seen:
                    seen.append(s)
    return seen


# ── RAG summary ───────────────────────────────────────────────────────────────

def _rag_summary(label: str, spike_ratio: float) -> dict:
    """Query the RAG for a narrative summary of a spiking topic.

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

def _sector_crosscheck(label: str, lookback_days: int = 14) -> list[dict]:
    """Return sector sentiment snapshots relevant to a topic label.

    Returns a list of dicts, one per sector, with keys:
        sector, trend_direction, trend_delta, mean_sentiment_score, data_age_days
    """
    sectors = _infer_sectors(label)
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
            pass   # no sector data yet — skip silently
        except Exception as exc:  # noqa: BLE001
            logging.warning("sector crosscheck failed for %s: %s", sector, exc)
    return results


# ── Formatting ────────────────────────────────────────────────────────────────

_TREND_EMOJI = {"improving": "+", "deteriorating": "-", "stable": "="}
_DIRECTION_COLOR = {"improving": "\033[32m", "deteriorating": "\033[31m", "stable": "\033[33m"}
_RESET = "\033[0m"


def _fmt_trend(direction: str, delta: float) -> str:
    arrow = _TREND_EMOJI.get(direction, "?")
    return f"[{arrow}] {direction}  (delta {delta:+.3f})"


def _print_briefing(
    run_date: date,
    spikes: list[dict],
    rag_results: dict[str, dict],
    sector_results: dict[str, list[dict]],
) -> None:
    sep = "-" * 72

    print(f"\n{'=' * 72}")
    print(f"  DAILY BRIEFING  {run_date}  ({len(spikes)} spikes)")
    print(f"{'=' * 72}")

    if not spikes:
        print("  No spike signals today. Check back after more history accumulates.")
        print(f"  (spike detection requires >=3 prior days per topic)\n")
        return

    for i, spike in enumerate(spikes, 1):
        tid   = spike["topic_id"]
        label = (spike["label"] if isinstance(spike["label"], str) else "") or "(unlabeled)"
        ratio = spike["spike_ratio"]
        count = spike["article_count"]

        print(f"\n{sep}")
        print(f"  #{i}  {label}")
        print(f"       spike={ratio:.2f}x   articles={count}")
        print(sep)

        # RAG summary
        rag = rag_results.get(tid, {})
        if rag.get("answer"):
            print(f"\n  NARRATIVE")
            # Word-wrap at 68 chars
            words = rag["answer"].split()
            line = "    "
            for word in words:
                if len(line) + len(word) + 1 > 70:
                    print(line)
                    line = "    " + word
                else:
                    line += (" " if line != "    " else "") + word
            if line.strip():
                print(line)

            if rag.get("sources"):
                print(f"\n  SOURCES")
                for s in rag["sources"]:
                    d = s.get("date", "")
                    t = s.get("title", "")[:65]
                    print(f"    [{d}] {t}")

        # Sector cross-check
        sectors = sector_results.get(tid, [])
        if sectors:
            print(f"\n  SECTOR SIGNAL")
            for sc in sectors:
                trend_str = _fmt_trend(sc["trend_direction"], sc["trend_delta"])
                score     = sc["mean_sentiment_score"]
                print(f"    {sc['sector']:<38} {trend_str}  score={score:+.3f}")

    print(f"\n{'=' * 72}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_briefing(
    run_date: date,
    top_n: int = 5,
    use_rag: bool = True,
) -> dict:
    """Build a structured briefing dict for run_date.

    Returns:
        {
            "date":     str,
            "n_spikes": int,
            "spikes":   list[{topic_id, label, spike_ratio, article_count,
                               rag_answer, rag_sources, sectors}]
        }
    """
    # ── Load trends ──────────────────────────────────────────────────────────
    trends_path = Path(TOPIC_TRENDS_FILE)
    if not trends_path.exists():
        print(
            f"ERROR: {trends_path} not found.\n"
            "Run:  python cluster_topics.py   to generate it."
        )
        sys.exit(1)

    trends_df = pd.read_csv(trends_path, sep="\t")
    print(f"Loaded {len(trends_df)} rows from topic_trends.tsv "
          f"({trends_df['date'].nunique()} dates)")

    # ── Spikes ───────────────────────────────────────────────────────────────
    spikes = get_emerging_topics(run_date, trends_df)
    if not spikes:
        _print_briefing(run_date, [], {}, {})
        return {"date": str(run_date), "n_spikes": 0, "spikes": []}

    spikes = spikes[:top_n]
    print(f"Found {len(spikes)} spike(s) — fetching context...\n")

    # ── RAG + sector (per spike) ──────────────────────────────────────────────
    rag_results:    dict[str, dict]       = {}
    sector_results: dict[str, list[dict]] = {}

    for spike in spikes:
        tid   = spike["topic_id"]
        label = (spike["label"] if isinstance(spike["label"], str) else "") or ""

        if use_rag and label:
            print(f"  RAG: {label[:60]} ...")
            rag_results[tid] = _rag_summary(label, spike["spike_ratio"])

        sector_results[tid] = _sector_crosscheck(label)

    # ── Print ─────────────────────────────────────────────────────────────────
    _print_briefing(run_date, spikes, rag_results, sector_results)

    # ── Structured output ────────────────────────────────────────────────────
    output_spikes = []
    for spike in spikes:
        tid  = spike["topic_id"]
        rag  = rag_results.get(tid, {})
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


def _save(briefing: dict) -> Path:
    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = BRIEFINGS_DIR / f"{briefing['date']}.json"
    path.write_text(json.dumps(briefing, indent=2, default=str), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily topic-spike briefing")
    parser.add_argument(
        "--date", default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top spikes to show (default: 5)",
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Skip RAG summaries (faster, no API calls)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save briefing JSON to data/briefings/{date}.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    run_date: date
    if args.date:
        run_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        run_date = date.today()

    briefing = build_briefing(
        run_date=run_date,
        top_n=args.top,
        use_rag=not args.no_rag,
    )

    if args.save and briefing["n_spikes"] > 0:
        path = _save(briefing)
        print(f"Saved: {path}")
