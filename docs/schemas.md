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

## TrackMetadata (`data/track_metadata.jsonl`)

Source of truth for track enrichment; defined in `pipeline/track_metadata.py`.
One JSON record per line, sorted by `track_id`, first-write-wins per `track_id`:

```json
{"track_id": "3f2a...", "input_artist": "(unknown)", "input_title": "Marco Carola - Weekend",
 "play_count": 1, "batch_id": "batch_...", "model": "gpt-4.1",
 "metadata": { ...TrackMetadata... }}
```

- `track_id` = sha1(lower/whitespace-normalised `"artist|title"`)[:12] — idempotency key
- `metadata` fields: `identified`, `confidence` (high/medium/low), `artists`, `title`, `mix_name`,
  `remixers`, `featured_artists`, `record_label`, `catalog_number`, `release_title`,
  `release_type`, `release_year`, `formats`, `primary_genre` (fixed Literal),
  `subgenres`, `bpm_estimate`, `musical_key`, `energy`, `mood_tags`, `dj_set_role`,
  sound description: `groove`, `percussion`, `bassline`, `vocals`, `melodic_elements`, `texture`,
  `emotional_character`, `dancefloor_effect`, `review_blurb` (press-style prose),
  `description_basis` (`known track` | `unknown`), `similar_artists`, `description`, `artist_details[]`
  (`name`, `real_name`, `aliases`, `country`, `city`, `active_since`, `associated_labels`,
  `associated_acts`, `short_bio`), `parsing_notes`
- Facts are LLM recall, not a lookup: treat `confidence="low"` / `identified=false` rows as unverified.
- **No inference** (enforced in code by `enforce_no_inference()`, not just the prompt):
  `identified=false` <=> `description_basis="unknown"`; unknown tracks have all descriptive
  fields null/[] (genre, subgenres, BPM, key, energy, mood, set role, similar artists, sound
  fields, review_blurb) and `description = "Track not known: no reliable information available."`.
  Tag-parsed fields (artists, title, mix, remixers, label/catalog from the tag) and
  `artist_details` are kept.

`data/track_metadata.csv` is a derived flat view (regenerated on every write):
list-of-string fields joined with `"; "`, `artist_details` as a JSON string, null as empty cell.
To re-enrich tracks: `enrich/enrich_tracks_direct.py --refresh` (workflow input `refresh`);
only successful re-asks overwrite, failures keep the old record.
Records written before a schema change lack the newer fields (empty CSV cells) until refreshed.
