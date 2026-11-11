"""Graph-grounded RAG for utility asset-document networks.

Pipeline: corpus/ -> ingest (regex/heuristic extraction) -> graph.json ->
query (numpy tf-idf + 1-hop graph expansion + provenance-tracked answer)
or check-consistency (temporal-validity conflicts).
"""

__version__ = "1.0.0"

from . import consistency, ingest, retriever, tfidf  # noqa: F401
