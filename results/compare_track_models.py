"""
compare_track_models.py
Field-by-field diff of two track-metadata stores (e.g. default model vs a
candidate model written with `enrich_tracks_direct.py --output-tag <tag>`).

Outputs (both derived, safe to regenerate):
  <out>.csv  long format: track_id, artist, title, field, baseline, candidate, changed
  <out>.md   review report: per-field change counts, known/unknown agreement matrix,
             and per-track side-by-side of identity, release facts and review_blurb

Only tracks present in BOTH stores are compared. String values are compared
case/whitespace-insensitively; lists order-insensitively.

Usage:
    python results/compare_track_models.py \
        --baseline data/track_metadata.jsonl \
        --candidate data/track_metadata_gpt-5.jsonl \
        --out data/track_model_diff_gpt-5

Exit codes: 0 = written, 1 = an input store is missing or there is no overlap.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import track_metadata as tm

COMPARE_FIELDS: list[str] = list(tm.TrackMetadata.model_fields)
_SIDE_BY_SIDE = ["description_basis", "artists", "title", "mix_name", "record_label",
                 "catalog_number", "release_year", "primary_genre", "review_blurb"]


def _norm(value):
    if isinstance(value, str):
        return " ".join(value.split()).lower()
    if isinstance(value, list):
        return sorted(repr(_norm(v)) for v in value)
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


def _render(value) -> str:
    return tm._cell(value)


def diff_records(baseline: dict[str, dict], candidate: dict[str, dict],
                 fields: list[str] = COMPARE_FIELDS) -> list[dict]:
    """Long-format diff rows for tracks present in both stores (sorted by track_id)."""
    rows = []
    for tid in sorted(set(baseline) & set(candidate)):
        b, c = baseline[tid], candidate[tid]
        bm, cm = b.get("metadata") or {}, c.get("metadata") or {}
        for f in fields:
            rows.append({
                "track_id": tid, "artist": b.get("input_artist", ""), "title": b.get("input_title", ""),
                "field": f, "baseline": _render(bm.get(f)), "candidate": _render(cm.get(f)),
                "changed": _norm(bm.get(f)) != _norm(cm.get(f)),
            })
    return rows


def _model_of(records: dict[str, dict]) -> str:
    return ", ".join(sorted({r.get("model", "?") for r in records.values()}))


def render_markdown(baseline: dict, candidate: dict, rows: list[dict], names: tuple[str, str]) -> str:
    common = sorted(set(baseline) & set(candidate))
    bname, cname = names
    changed = Counter(r["field"] for r in rows if r["changed"])
    basis = Counter((baseline[t]["metadata"].get("description_basis"),
                     candidate[t]["metadata"].get("description_basis")) for t in common)

    out = [f"# Track metadata model diff", "",
           f"- baseline: `{bname}` (model {_model_of(baseline)})",
           f"- candidate: `{cname}` (model {_model_of(candidate)})",
           f"- tracks compared: {len(common)}", "",
           "## Known / unknown agreement", "",
           "| baseline \\ candidate | known track | unknown |", "|---|---|---|"]
    for bb in ("known track", "unknown"):
        out.append(f"| {bb} | {basis[(bb, 'known track')]} | {basis[(bb, 'unknown')]} |")
    out += ["", "## Fields changed (tracks)", "", "| field | changed |", "|---|---|"]
    out += [f"| {f} | {changed.get(f, 0)} |" for f in COMPARE_FIELDS]
    out += ["", "## Per track", ""]
    by_track: dict[str, dict] = {}
    for r in rows:
        by_track.setdefault(r["track_id"], {})[r["field"]] = r
    for tid in common:
        fr = by_track[tid]
        rec = baseline[tid]
        out += [f"### {rec.get('input_artist')} | {rec.get('input_title')}", "",
                f"| field | {bname} | {cname} |", "|---|---|---|"]
        for f in _SIDE_BY_SIDE:
            r = fr[f]
            mark = " **≠**" if r["changed"] else ""
            b = r["baseline"].replace("|", "\\|") or "—"
            c = r["candidate"].replace("|", "\\|") or "—"
            out.append(f"| {f}{mark} | {b} | {c} |")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Diff two track-metadata stores")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output path stem (.csv and .md are added)")
    args = p.parse_args(argv)

    for path in (args.baseline, args.candidate):
        if not path.exists():
            print(f"[error] store not found: {path}")
            return 1
    baseline, candidate = tm.load_records(args.baseline), tm.load_records(args.candidate)
    rows = diff_records(baseline, candidate)
    if not rows:
        print("[error] no tracks in common between the two stores.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = args.out.with_suffix(".csv"), args.out.with_suffix(".md")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md_path.write_text(render_markdown(baseline, candidate, rows,
                                       (args.baseline.stem, args.candidate.stem)), encoding="utf-8")

    n = len(set(baseline) & set(candidate))
    print(f"Compared {n} track(s); {sum(r['changed'] for r in rows)} field difference(s).")
    print(f"  -> {csv_path}\n  -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
