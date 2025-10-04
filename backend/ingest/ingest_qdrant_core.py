from datetime import datetime
import uuid
from typing import Iterable, Dict

from qdrant_client.http import models as qm

EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMB_NAME)


def ensure_collection(client, collection: str, dim: int = 384):
    try:
        client.get_collection(collection)
    except Exception:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )


def ingest_documents(
    client,
    collection: str,
    docs: Iterable[Dict[str, str]],
    domain: str,
    batch_size: int = 64,
) -> int:
    model = _embedder()
    ensure_collection(client, collection)
    now = datetime.utcnow().isoformat()
    batch = []
    total = 0

    for doc in docs:
        batch.append(doc)
        if len(batch) >= batch_size:
            total += _upsert_batch(client, collection, batch, model, now, domain)
            batch = []

    if batch:
        total += _upsert_batch(client, collection, batch, model, now, domain)

    return total


def _upsert_batch(client, collection, batch, model, timestamp, domain):
    texts = [doc["text"] for doc in batch]
    vectors = model.encode(texts).tolist()

    points = []
    for doc, vector in zip(batch, vectors):
        payload = dict(doc)
        payload.setdefault("domain", domain)
        payload["created_at"] = timestamp
        pid = str(uuid.uuid4())
        points.append(qm.PointStruct(id=pid, vector=vector, payload=payload))

    client.upsert(collection_name=collection, points=points)
    return len(points)
