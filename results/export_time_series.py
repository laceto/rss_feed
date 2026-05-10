"""
export_time_series.py
Pre-compute and persist the rolling 90-day sentiment time series for all
sectors and all entities. Run this after read_sector_results.py has built
(or refreshed) data/sector_summary.tsv.

Reads:   data/sector_summary.tsv
Writes:  data/sector_sentiment_pivot.tsv   (wide: date × 19 sectors)
         data/entity_sentiment_ts.tsv      (long: date × entity × sector)

Usage:
    python export_time_series.py

External callers — R / pandas read-back:

    import pandas as pd
    pivot     = pd.read_csv("data/sector_sentiment_pivot.tsv", sep="\\t",
                            index_col=0, parse_dates=True)
    entity_ts = pd.read_csv("data/entity_sentiment_ts.tsv",   sep="\\t",
                            parse_dates=["date"])
    anthropic = entity_ts[entity_ts["entity"] == "Anthropic"]

Debugging:
    - RuntimeError from export_sector_pivot  -> sector_summary.tsv is missing
      or has no data in the last EXPORT_LOOKBACK_DAYS days. Run
      read_sector_results.py first to regenerate it.
    - RuntimeError from export_entity_ts     -> same root cause; also check
      that at least one TSV row has a non-empty entities column.
    - The script exits with code 1 on any error so CI fails loudly rather
      than silently writing empty/stale files.
"""

import sys
import pandas as pd

from constants import EXPORT_LOOKBACK_DAYS
from query_sector import export_sector_pivot
from query_entity import export_entity_ts


def main() -> None:
    print(f"=== Export Time Series (last {EXPORT_LOOKBACK_DAYS} days) ===")

    # ── Sector pivot ─────────────────────────────────────────────────────────
    path = export_sector_pivot()
    pivot = pd.read_csv(path, sep="\t", index_col=0, parse_dates=True)
    print(
        f"Sector pivot:  {len(pivot)} dates "
        f"x {len(pivot.columns)} sectors  ->  {path}"
    )

    # ── Entity time series ────────────────────────────────────────────────────
    path = export_entity_ts()
    ets = pd.read_csv(path, sep="\t")
    print(
        f"Entity TS:     {len(ets)} rows, "
        f"{ets['entity'].nunique()} unique entities  ->  {path}"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
