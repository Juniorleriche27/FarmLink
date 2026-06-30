"""Factory helpers for Qdrant clients."""

from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient


def create_qdrant_client(url: str, api_key: str) -> Optional[QdrantClient]:
    """Create a Qdrant client when both credentials are provided."""
    url = (url or "").strip()
    api_key = (api_key or "").strip()
    if not url or not api_key:
        return None
    return QdrantClient(url=url, api_key=api_key)
