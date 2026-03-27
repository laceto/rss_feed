"""pipeline — reusable utilities for the rss_feed financial news analysis pipeline.

This package exposes a flat import surface via lazy attribute loading, so
top-level scripts can import narrow submodules such as pipeline.backfill or
pipeline.topic_clustering without paying the dependency cost of unrelated
modules during package bootstrap.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, str] = {
    # backfill
    "briefing_dates": ".backfill",
    "clustered_dates": ".backfill",
    "phase1_cluster": ".backfill",
    "phase2_briefing": ".backfill",
    "trading_dates": ".backfill",
    # batch_briefings
    "check_briefings_batch_status": ".batch_briefings",
    "collect_briefing_results": ".batch_briefings",
    "parse_briefing_custom_id": ".batch_briefings",
    "read_spike_metadata": ".batch_briefings",
    "save_briefings": ".batch_briefings",
    # batch_collect
    "TERMINAL_STATES": ".batch_collect",
    "check_batch_status": ".batch_collect",
    "date_from_custom_id": ".batch_collect",
    "download_results": ".batch_collect",
    "parse_sectors": ".batch_collect",
    "read_pending_batch_id": ".batch_collect",
    "save_batch_results": ".batch_collect",
    # batch_sector
    "MAX_CHARS": ".batch_sector",
    "STRICT_SCHEMA": ".batch_sector",
    "SYSTEM_PROMPT": ".batch_sector",
    "MultiSectorAnalysis": ".batch_sector",
    "SectorAnalysis": ".batch_sector",
    "build_batch_tasks": ".batch_sector",
    "build_daily_contents": ".batch_sector",
    "make_openai_strict": ".batch_sector",
    "persist_batch_id": ".batch_sector",
    "submit_batch": ".batch_sector",
    # briefing_batch_submit
    "MAX_CONTEXT_CHARS": ".briefing_batch_submit",
    "TOP_N_DEFAULT": ".briefing_batch_submit",
    "build_briefing_batch_tasks": ".briefing_batch_submit",
    "build_briefing_custom_id": ".briefing_batch_submit",
    "build_prompt": ".briefing_batch_submit",
    "retrieve_docs_for_label": ".briefing_batch_submit",
    "run_briefing_batch_submission": ".briefing_batch_submit",
    # briefing_batch_collect
    "run_briefing_batch_collection": ".briefing_batch_collect",
    # briefings
    "KEYWORD_TO_SECTORS": ".briefings",
    "build_briefing": ".briefings",
    "infer_sectors": ".briefings",
    "load_precomputed": ".briefings",
    "rag_summary": ".briefings",
    "save_briefing": ".briefings",
    "sector_crosscheck": ".briefings",
    # hf_io
    "create_hf_dataset_repo": ".hf_io",
    "load_feed_for_date": ".hf_io",
    "load_feeds_from_files": ".hf_io",
    "load_feeds_from_hf": ".hf_io",
    "load_tsv": ".hf_io",
    "push_df_to_hub": ".hf_io",
    "push_feeds_to_hub": ".hf_io",
    "push_incremental": ".hf_io",
    # query_entity
    "export_entity_ts": ".query_entity",
    "get_all_entities_ts": ".query_entity",
    "get_entity_snapshot": ".query_entity",
    "get_entity_time_series": ".query_entity",
    "list_entities": ".query_entity",
    # query_sector
    "export_sector_pivot": ".query_sector",
    "get_all_sectors_pivot": ".query_sector",
    "get_snapshot": ".query_sector",
    "get_time_series": ".query_sector",
    "list_sectors": ".query_sector",
    "load_summary": ".query_sector",
    # sector_db
    "run_sector_db_build": ".sector_db",
    # sector_batch_collect
    "run_sector_batch_collection": ".sector_batch_collect",
    # sector_io
    "build_sector_dataframe": ".sector_io",
    "build_sector_db": ".sector_io",
    "insert_sector_date": ".sector_io",
    "load_sector_json": ".sector_io",
    "load_sector_results": ".sector_io",
    # sector_summary
    "run_sector_summary_build": ".sector_summary",
    # sentiment_charts
    "MIN_DATA_POINTS": ".sentiment_charts",
    "ROLLING_WINDOW": ".sentiment_charts",
    "SENTIMENT_COLORS": ".sentiment_charts",
    "SENTIMENT_SCORE": ".sentiment_charts",
    "chart_distribution": ".sentiment_charts",
    "chart_heatmap": ".sentiment_charts",
    "chart_trends": ".sentiment_charts",
    "load_sentiment_data": ".sentiment_charts",
    # sentiment_visualization
    "run_sentiment_visualizations": ".sentiment_visualization",
    # topic_clustering
    "parse_cluster_topics_args": ".topic_clustering",
    "print_cluster_summary": ".topic_clustering",
    "run_cluster_topics_cli": ".topic_clustering",
    # topic_charts
    "load_trends": ".topic_charts",
    "pick_top_topics": ".topic_charts",
    "plot_frequency_ts": ".topic_charts",
    "plot_sentiment_delta": ".topic_charts",
    "plot_sentiment_heatmap": ".topic_charts",
    "plot_signal_scatter": ".topic_charts",
    "plot_signal_scatter_animation": ".topic_charts",
    "plot_spike_heatmap": ".topic_charts",
    "plot_topic_timeline": ".topic_charts",
    # topic_visualization
    "DEFAULT_ANIMATION_FPS": ".topic_visualization",
    "DEFAULT_LOOKBACK_DAYS": ".topic_visualization",
    "DEFAULT_TOP_N": ".topic_visualization",
    "parse_topic_visualization_args": ".topic_visualization",
    "run_topic_visualizations": ".topic_visualization",
    # vectorstore_io
    "EMBED_MODEL": ".vectorstore_io",
    "POLL_INTERVAL": ".vectorstore_io",
    "REGISTRY_COLUMNS": ".vectorstore_io",
    "align_pairs_to_docs": ".vectorstore_io",
    "assign_ids": ".vectorstore_io",
    "build_documents": ".vectorstore_io",
    "find_new_articles": ".vectorstore_io",
    "init_vectorstore": ".vectorstore_io",
    "load_feed_articles": ".vectorstore_io",
    "load_registry": ".vectorstore_io",
    "run_embedding_batch": ".vectorstore_io",
    "save_registry": ".vectorstore_io",
    "update_vectorstore": ".vectorstore_io",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
