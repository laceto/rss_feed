# Topic Clustering

`cluster_topics.py` — unsupervised narrative discovery layered on top of the FAISS vectorstore. Runs daily after `build_sector_db.py` in CI. Does NOT re-embed — all vectors come from the existing store.

## How It Works

1. Load the 45-day rolling window of article vectors from FAISS
2. PCA(50) for dimensionality reduction (`random_state=42`, deterministic)
3. HDBSCAN → ~19 clusters, ~60% noise (expected for news data)
4. Centroid-cosine matching (threshold 0.85) assigns stable `topic_id` across daily runs
5. New clusters get LLM labels via `gpt-4o-mini`; cached labels reused for matched topics
6. `compute_topic_sentiment` joins cluster assignments → `sector_summary.tsv` (pure join, no API)
7. Appended to `data/topic_trends.tsv` (append-only)

## Exceptions

- `ClusteringAborted` — raised when `noise_ratio > CLUSTER_MAX_NOISE_RATIO` (0.90) or `n_clusters < CLUSTER_MIN_CLUSTERS` (3). CI exit code 2 = warning, not failure.
- `DuplicateDateError` — raised by `append_trends` if the target date already exists. Delete the rows manually and re-run.

## Topic Continuity Invariant

`topic_id` (UUID) is assigned once and never changes for the same narrative across runs. Labels may be updated but IDs are stable. Prior centroids stored in `data/topic_centroids.json`.

## Empirical Parameters

| Constant | Value | Rationale |
|---|---|---|
| `CLUSTER_WINDOW_DAYS` | 45 | ~1,700 articles; below 30d HDBSCAN collapses to 2 clusters |
| `CLUSTER_MIN_SIZE` | 10 | ~19 clusters at median size 18; above 20 too few clusters |
| `CLUSTER_MIN_SAMPLES` | 3 | balanced noise/cluster tradeoff |
| `CLUSTER_MAX_NOISE_RATIO` | 0.90 | 60–70% noise is expected baseline for news data |
| `CLUSTER_MIN_CLUSTERS` | 3 | abort on truly degenerate runs only |
| `CLUSTER_SELECTION_METHOD` | `"leaf"` | `"eom"` over-merges to ~3 clusters on this corpus |

All of these live in `constants.py` — never hardcode.

## Public API

See `docs/api-reference.md` for the full API.
