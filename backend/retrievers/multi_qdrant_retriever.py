"""Helpers for querying multiple Qdrant collections."""
import logging
from typing import Dict, List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


class MultiQdrantRetriever:
    def __init__(self, endpoints: Dict[str, Dict]):
        """Initialise a retriever from a mapping of collection -> endpoint config."""
        self.model = SentenceTransformer(EMB_NAME)
        self.clients: Dict[str, QdrantClient] = {}

        for collection, cfg in (endpoints or {}).items():
            cfg = cfg or {}
            url = (cfg.get("url") or "").strip()
            api_key = (cfg.get("api_key") or "").strip()
            if not url or not api_key:
                continue
            try:
                self.clients[collection] = QdrantClient(url=url, api_key=api_key)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Qdrant client init failed for %s: %s", collection, exc)

        if not self.clients:
            logger.warning("MultiQdrantRetriever initialised with no active Qdrant endpoints.")

    @property
    def available_collections(self) -> List[str]:
        return list(self.clients.keys())

    def search(self, query: str, top_k: int = 4, domain: str = "all") -> List[Dict]:
        if not self.clients:
            return []

        vector = self.model.encode(query).tolist()
        results: List[Dict] = []

        if domain in self.clients:
            collections = [domain]
        else:
            collections = list(self.clients.keys())

        for collection in collections:
            client = self.clients.get(collection)
            if client is None:
                continue
            try:
                hits = client.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=top_k,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Qdrant search failed for %s: %s", collection, exc)
                continue

            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    {
                        "collection": collection,
                        "score": hit.score,
                        "text": payload.get("text", ""),
                        "source": payload.get("source", ""),
                        "title": payload.get("title", ""),
                        "domain": payload.get("domain", ""),
                    }
                )

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
