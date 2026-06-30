"""HTTP route handlers for FarmLink API."""

from __future__ import annotations

from fastapi import APIRouter

from schemas.query import QueryIn
from services.query_service import handle_query
from services.retrieval import available_domains

router = APIRouter()


@router.get("/", include_in_schema=False)
def root():
    return {
        "name": "FarmLink API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "domains": "/domains",
        "query": {"path": "/query", "method": "POST"},
    }


@router.get("/health")
def health():
    return {"ok": True}


@router.get("/domains")
def domains():
    return {"domains": available_domains()}


@router.post("/query")
def query(q: QueryIn):
    return handle_query(q)
