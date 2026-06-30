"""Pydantic schemas for query endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class QueryIn(BaseModel):
    question: str
    domain: str = "all"
    top_k: int = 4
    temperature: float = 0.2
