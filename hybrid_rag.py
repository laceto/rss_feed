# Backward-compat shim. New code: from pipeline.hybrid_rag import ask
from pipeline.hybrid_rag import *  # noqa: F401, F403
from pipeline.hybrid_rag import ask  # explicit for type checkers
