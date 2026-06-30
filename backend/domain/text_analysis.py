"""Domain-level text utilities used by FarmLink services."""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Dict, List, Optional, Set
from unicodedata import normalize

from core.config import DOMAIN_KEYWORDS

_WORD_RE = re.compile(r"[a-z0-9]{3,}")


def normalize_text(value: str) -> str:
    """Return a lowercase ASCII-only version of the input."""
    normalized = normalize("NFKD", (value or ""))
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def tokenize(value: str) -> List[str]:
    """Tokenize text for simple keyword coverage checks."""
    return _WORD_RE.findall(normalize_text(value))


def missing_keywords(question: str, contexts: List[Dict], cutoff: float = 0.82) -> List[str]:
    """Identify question keywords that are absent from retrieved contexts."""
    query_tokens = tokenize(question)
    if not query_tokens or not contexts:
        return query_tokens if query_tokens and not contexts else []

    context_tokens: Set[str] = set()
    for ctx in contexts:
        context_tokens.update(tokenize(ctx.get("text", "")))
        context_tokens.update(tokenize(ctx.get("title", "")))

    if not context_tokens:
        return query_tokens

    vocab = list(context_tokens)
    missing: List[str] = []
    for token in query_tokens:
        if token in context_tokens:
            continue
        if get_close_matches(token, vocab, n=1, cutoff=cutoff):
            continue
        missing.append(token)
    return missing


def infer_domain(question: str) -> Optional[str]:
    """Infer the best FarmLink collection from a natural-language question."""
    tokens = set(tokenize(question))
    if not tokens:
        return None

    best_domain: Optional[str] = None
    best_score = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = len(tokens & keywords)
        if score > best_score:
            best_domain = domain
            best_score = score
    return best_domain if best_score else None
