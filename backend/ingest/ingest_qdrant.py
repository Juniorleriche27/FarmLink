import argparse
import os

from qdrant_client import QdrantClient

try:
    from ingest.chunkers import build_chunks, load_docs_from_folder
    from ingest.ingest_qdrant_core import ingest_documents
except ImportError:  # pragma: no cover - direct script execution from backend/ingest
    from chunkers import build_chunks, load_docs_from_folder
    from ingest_qdrant_core import ingest_documents

try:  # load .env for local runs
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass

_COLLECTION_SUFFIXES = {
    "farmlink_sols": "SOL",
    "farmlink_marche": "MARCHE",
    "farmlink_cultures": "CULT",
    "farmlink_eau": "EAU",
    "farmlink_meca": "MECA",
}


def _get_qdrant_env(collection: str):
    if collection not in _COLLECTION_SUFFIXES:
        raise SystemExit(f"Collection inconnue: {collection}")

    base_url = (os.getenv("QDRANT_URL") or "").strip()
    base_key = (os.getenv("QDRANT_API_KEY") or "").strip()

    suffix = _COLLECTION_SUFFIXES[collection]
    url = (os.getenv(f"QDRANT_{suffix}_URL") or base_url).strip()
    key = (os.getenv(f"QDRANT_{suffix}_KEY") or base_key).strip()
    return url, key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Chemin des documents (raw)")
    parser.add_argument(
        "--collection",
        required=True,
        choices=list(_COLLECTION_SUFFIXES.keys()),
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Nom logique du domaine (sols|marche|cultures|eau|meca)",
    )
    args = parser.parse_args()

    url, key = _get_qdrant_env(args.collection)
    if not url or not key:
        raise SystemExit(
            f"Qdrant URL/KEY manquants pour {args.collection} (verifie ton .env)."
        )

    client = QdrantClient(url=url, api_key=key)
    docs = load_docs_from_folder(args.folder, domain=args.domain)
    chunks = build_chunks(docs, chunk_size=1200, overlap=200, domain=args.domain)
    inserted = ingest_documents(client, args.collection, chunks, domain=args.domain)
    print(f"Ingestion OK: {inserted} chunks -> {args.collection}")


if __name__ == "__main__":
    main()
