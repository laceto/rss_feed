# Data Schemas

## SectorAnalysis (Pydantic)

Defined in `create_batch_files_v2.py`. `SectorName` in `constants.py` is the single source of truth.

| Field | Type |
|---|---|
| `entities` | `list[str]` — named companies/orgs |
| `sector` | `SectorName` — 19-value Literal from `constants.py` |
| `sentiment` | `Literal["positive", "neutral", "negative"]` |
| `news_category` | `Literal["earnings","M&A","regulation","macro","appointments","products","markets","other"]` |
| `extraction_status` | `Literal["ok", "partial"]` |

`MultiSectorAnalysis` wraps `list[SectorAnalysis]` (1–8 sectors per day).

`_make_openai_strict()` converts the Pydantic schema to OpenAI strict JSON schema format (adds `additionalProperties: false` + `required[]` recursively).

## Briefing Output (`data/briefings/{date}.json`)

```json
{
  "date": "YYYY-MM-DD",
  "n_spikes": 3,
  "spikes": [
    {
      "topic_id": "uuid",
      "label": "Fed Rate Decision",
      "spike_ratio": 2.5,
      "article_count": 45,
      "rag_answer": "The Federal Reserve...",
      "rag_sources": [{"title": "", "date": "", "link": "", "snippet": "", "guid": ""}],
      "sectors": [{"sector": "", "trend_direction": "", "trend_delta": 0.0, "mean_sentiment_score": 0.0}]
    }
  ]
}
```

Compatible with `daily_briefing.py`'s `build_briefing()` output — same schema.

`custom_id` convention: `"briefing-YYYY-MM-DD-{topic_id[:8]}"`

## SQLite Schema (`data/sector_results.db`)

```sql
sector_analyses  (id, date, sector, sentiment, sentiment_score,
                  news_category, extraction_status, batch_id)
sector_entities  (id, analysis_id FK → sector_analyses.id, entity)
```

- `date` is the **filename stem** (`2026-03-12.json` → `"2026-03-12"`), not the JSON body field
- `sentiment_score` is denormalized (1/0/−1) from `SENTIMENT_SCORE` in `constants.py`
- Malformed JSON files are logged to stderr and skipped; build continues
- Indices on `date`, `sector`, `date+sector`, `lower(entity)`
- Full rebuild on every run (< 1 s), written atomically via `.db.tmp` + `os.replace()`

## Feed TSV (`output/feeds{YYYY-MM-DD}.txt`)

Tab-separated. Columns: `title, description, link, guid, type, id, sponsored, pubDate`

## Bulk Export Files

| File | Format | Description |
|---|---|---|
| `data/sector_sentiment_pivot.tsv` | wide (date × 19 sectors) | mean `sentiment_score`; NaN = no data |
| `data/entity_sentiment_ts.tsv` | long (date × entity × sector) | one row per mention |
| `data/sector_results.db` | SQLite | lossless, normalized |
| `data/topic_trends.tsv` | append-only TSV | date × topic_id × topic_label × article_count × sentiment_score |
| `data/topic_centroids.json` | JSON | topic_id → {label, centroid, first_seen, last_seen} |
| `data/topic_labels.json` | JSON | topic_id → label string (LLM cache) |
| `data/topic_clusters/{date}.json` | JSON array | article → topic_id mapping for the 45-day window |
| `data/briefings/{date}.json` | JSON | one per date: n_spikes + spikes list |

- TSV rolling window controlled by `EXPORT_LOOKBACK_DAYS = 90` in `constants.py`
- Topic clustering window: `CLUSTER_WINDOW_DAYS = 45` (separate constant, also in `constants.py`)
- SQLite contains all dates regardless of rolling window
