import rich
from query_entity import (
    get_entity_snapshot, get_entity_time_series, list_entities,
    get_all_entities_ts, export_entity_ts,
)
from query_sector import (
    get_snapshot, get_time_series, list_sectors,
    get_all_sectors_pivot, export_sector_pivot,
)

# ── 1. Discover all known entities ───────────────────────────────────────────
names = list_entities()
print(f"{len(names)} entities found")
print("First 10:", names[:10])

# ── 2. Snapshot — case-insensitive lookup ─────────────────────────────────────
# Pick the first entity in the list as a live example
if names:
    example = 'Anthropic'
    snap = get_entity_snapshot(example)
    print(f"\nSnapshot for '{snap['entity']}':")
    rich.print(snap)

# ── 3. Time series — trend over the last 90 days ─────────────────────────────
if names:
    ts = get_entity_time_series(example, lookback_days=90, include_articles=False)
    print(f"\nTime series for '{ts['entity']}' ({ts['lookback_days']}d):")
    rich.print({k: v for k, v in ts.items() if k != "time_series"})   # summary
    print(f"  {ts['n_observations']} observations | trend: {ts['trend_direction']}")

# ── 4. Error handling ─────────────────────────────────────────────────────────
try:
    get_entity_snapshot("doesnotexist_xyz")
except LookupError as e:
    print(f"\nExpected LookupError: {e}")

# ── 5. Sector — discover valid names ─────────────────────────────────────────
sectors = list_sectors()
print(f"\n{len(sectors)} sectors: {sectors}")

# ── 6. Sector snapshot ────────────────────────────────────────────────────────
snap_s = get_snapshot("Finance")
print(f"\nSector snapshot for '{snap_s['sector']}':")
rich.print(snap_s)

# ── 7. Sector time series ─────────────────────────────────────────────────────
ts_s = get_time_series("Finance", lookback_days=30, include_articles=False)
print(f"\nSector time series for '{ts_s['sector']}' ({ts_s['lookback_days']}d):")
rich.print({k: v for k, v in ts_s.items() if k != "time_series"})
print(f"  {ts_s['n_observations']} observations | trend: {ts_s['trend_direction']}")

# ── 8. Bulk sector pivot ───────────────────────────────────────────────────────
print("\n=== Section 8: Sector Pivot ===")
pivot = get_all_sectors_pivot(lookback_days=90)
print(f"Sector pivot shape: {pivot.shape}")        # expect (dates, 19)
assert pivot.shape[1] == 19, f"Expected 19 sector columns, got {pivot.shape[1]}"
assert list(pivot.columns) == sorted(pivot.columns), "Columns are not alphabetically sorted"
print(pivot.tail(5))
# Write to file
path = export_sector_pivot()
print(f"Written to: {path}")

# ── 9. Bulk entity time series ─────────────────────────────────────────────────
print("\n=== Section 9: Entity Time Series ===")
ets = get_all_entities_ts(lookback_days=90)
print(f"Entity TS: {len(ets)} rows, {ets['entity'].nunique()} unique entities")
expected_cols = ["date", "entity", "sector", "sentiment", "sentiment_score", "news_category"]
missing = [c for c in expected_cols if c not in ets.columns]
assert not missing, f"Missing expected columns: {missing}"
anthropic = ets[ets["entity"] == "Anthropic"]
print(f"Anthropic rows: {len(anthropic)}")
rich.print(anthropic.head(5))
# Write to file
path = export_entity_ts()
print(f"Written to: {path}")