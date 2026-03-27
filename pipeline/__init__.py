"""pipeline — reusable utilities for the rss_feed financial news analysis pipeline.

Modules:
    batch_briefings  — briefing batch collection (kitai.batch): parsing, saving
    batch_collect    — sector batch collection (OpenAI API): polling, download, routing
    batch_sector     — sector batch task building and submission
    hf_io            — HuggingFace Dataset I/O (feeds + analysis datasets)
    query_entity     — entity query API: snapshot, time-series, bulk export
    query_sector     — sector query API: snapshot, time-series, bulk pivot
    sector_io        — sector result file reading and SQLite database building
    sentiment_charts — sentiment visualisation (heatmap, trend lines, distribution)
    topic_charts     — topic visualisation (6 static charts + animated GIF)
    vectorstore_io   — FAISS vectorstore build/update: registry, documents, embedding batch

cluster_topics is not re-exported here — it is already a top-level module
registered in pyproject.toml and should be imported directly:
    from cluster_topics import run, get_emerging_topics, ...
"""

from .batch_briefings import (
    check_briefings_batch_status,
    collect_briefing_results,
    parse_briefing_custom_id,
    read_spike_metadata,
    save_briefings,
)
from .batch_collect import (
    TERMINAL_STATES,
    check_batch_status,
    date_from_custom_id,
    download_results,
    parse_sectors,
    read_pending_batch_id,
    save_batch_results,
)
from .batch_sector import (
    MAX_CHARS,
    STRICT_SCHEMA,
    SYSTEM_PROMPT,
    MultiSectorAnalysis,
    SectorAnalysis,
    build_batch_tasks,
    build_daily_contents,
    make_openai_strict,
    persist_batch_id,
    submit_batch,
)
from .hf_io import (
    create_hf_dataset_repo,
    load_feed_for_date,
    load_feeds_from_files,
    load_feeds_from_hf,
    load_tsv,
    push_df_to_hub,
    push_feeds_to_hub,
    push_incremental,
)
from .query_entity import (
    export_entity_ts,
    get_all_entities_ts,
    get_entity_snapshot,
    get_entity_time_series,
    list_entities,
)
from .query_sector import (
    export_sector_pivot,
    get_all_sectors_pivot,
    get_snapshot,
    get_time_series,
    list_sectors,
    load_summary,
)
from .sector_io import (
    build_sector_dataframe,
    build_sector_db,
    insert_sector_date,
    load_sector_json,
    load_sector_results,
)
from .sentiment_charts import (
    MIN_DATA_POINTS,
    ROLLING_WINDOW,
    SENTIMENT_COLORS,
    SENTIMENT_SCORE,
    chart_distribution,
    chart_heatmap,
    chart_trends,
    load_sentiment_data,
)
from .topic_charts import (
    load_trends,
    pick_top_topics,
    plot_frequency_ts,
    plot_sentiment_delta,
    plot_sentiment_heatmap,
    plot_signal_scatter,
    plot_signal_scatter_animation,
    plot_spike_heatmap,
    plot_topic_timeline,
)
from .vectorstore_io import (
    EMBED_MODEL,
    POLL_INTERVAL,
    REGISTRY_COLUMNS,
    align_pairs_to_docs,
    assign_ids,
    build_documents,
    find_new_articles,
    init_vectorstore,
    load_feed_articles,
    load_registry,
    run_embedding_batch,
    save_registry,
    update_vectorstore,
)

__all__ = [
    # batch_briefings
    "check_briefings_batch_status",
    "collect_briefing_results",
    "parse_briefing_custom_id",
    "read_spike_metadata",
    "save_briefings",
    # batch_collect
    "TERMINAL_STATES",
    "check_batch_status",
    "date_from_custom_id",
    "download_results",
    "parse_sectors",
    "read_pending_batch_id",
    "save_batch_results",
    # batch_sector
    "MAX_CHARS",
    "STRICT_SCHEMA",
    "SYSTEM_PROMPT",
    "MultiSectorAnalysis",
    "SectorAnalysis",
    "build_batch_tasks",
    "build_daily_contents",
    "make_openai_strict",
    "persist_batch_id",
    "submit_batch",
    # hf_io
    "create_hf_dataset_repo",
    "load_feed_for_date",
    "load_feeds_from_files",
    "load_feeds_from_hf",
    "load_tsv",
    "push_df_to_hub",
    "push_feeds_to_hub",
    "push_incremental",
    # query_entity
    "export_entity_ts",
    "get_all_entities_ts",
    "get_entity_snapshot",
    "get_entity_time_series",
    "list_entities",
    # query_sector
    "export_sector_pivot",
    "get_all_sectors_pivot",
    "get_snapshot",
    "get_time_series",
    "list_sectors",
    "load_summary",
    # sector_io
    "build_sector_dataframe",
    "build_sector_db",
    "insert_sector_date",
    "load_sector_json",
    "load_sector_results",
    # sentiment_charts
    "MIN_DATA_POINTS",
    "ROLLING_WINDOW",
    "SENTIMENT_COLORS",
    "SENTIMENT_SCORE",
    "chart_distribution",
    "chart_heatmap",
    "chart_trends",
    "load_sentiment_data",
    # topic_charts
    "load_trends",
    "pick_top_topics",
    "plot_frequency_ts",
    "plot_sentiment_delta",
    "plot_sentiment_heatmap",
    "plot_signal_scatter",
    "plot_signal_scatter_animation",
    "plot_spike_heatmap",
    "plot_topic_timeline",
    # vectorstore_io
    "EMBED_MODEL",
    "POLL_INTERVAL",
    "REGISTRY_COLUMNS",
    "align_pairs_to_docs",
    "assign_ids",
    "build_documents",
    "find_new_articles",
    "init_vectorstore",
    "load_feed_articles",
    "load_registry",
    "run_embedding_batch",
    "save_registry",
    "update_vectorstore",
]
