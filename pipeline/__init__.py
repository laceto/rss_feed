# pipeline — shared library for the rss_feed analysis pipeline.
#
# Public API surface (import from here or from the submodule directly):
#   from pipeline.constants import SECTOR_TAXONOMY, VECTORSTORE_DIR, ...
#   from pipeline.hybrid_rag import ask
#   from pipeline.cluster_topics import run, get_emerging_topics, ClusteringAborted
#   from pipeline.query_sector import get_snapshot, get_time_series
#   from pipeline.query_entity import get_entity_snapshot, get_entity_time_series
