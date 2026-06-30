"""Lazy retrieval service for FarmLink RAG."""

from __future__ import annotations

from typing import Any, Dict, List

from core.config import active_qdrant_endpoints, raw_qdrant_endpoints

_retriever: Any = None
_endpoints_cache: Dict[str, Dict[str, str]] | None = None


def reset_retriever_cache() -> None:
    """Reset module-level cache, mainly for tests."""
    global _retriever, _endpoints_cache
    _retriever = None
    _endpoints_cache = None


def get_retriever():
    """
    Initialise MultiQdrantRetriever only on first use.
    This preserves the previous cold-start optimisation.
    """
    global _retriever, _endpoints_cache
    if _retriever is not None:
        return _retriever

    from retrievers.multi_qdrant_retriever import MultiQdrantRetriever

    if _endpoints_cache is None:
        _endpoints_cache = active_qdrant_endpoints(raw_qdrant_endpoints())

    _retriever = MultiQdrantRetriever(_endpoints_cache or {})
    return _retriever


def available_domains() -> List[str]:
    """Return available Qdrant collections plus the aggregate 'all' domain."""
    try:
        retriever = get_retriever()
        domain_list = list(getattr(retriever, "available_collections", []))
    except Exception:
        domain_list = []

    if domain_list:
        return domain_list + ["all"]
    return ["all"]
