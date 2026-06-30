"""Application service for FarmLink question answering."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import HTTPException

from core.config import (
    DOMAIN_LABELS,
    GREETINGS,
    MAX_SOURCES,
    SPECIALTY_PATTERNS,
)
from domain.text_analysis import infer_domain, missing_keywords, tokenize
from llm.generator import generate_answer
from schemas.query import QueryIn
from services.retrieval import get_retriever


def short_sources(contexts: List[Dict], limit: int = MAX_SOURCES) -> List[str]:
    """Return at most `limit` human-readable context titles."""
    out = []
    for context in contexts[:limit]:
        title = context.get("title") or "Document"
        out.append(str(title))
    return out


def handle_query(q: QueryIn) -> Dict:
    """Run FarmLink's greeting, specialty and RAG flows."""
    retriever = get_retriever()

    available = set(getattr(retriever, "available_collections", []))
    if q.domain != "all" and q.domain not in available:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{q.domain}'")

    question_clean = q.question.strip().lower()

    if question_clean in GREETINGS or question_clean.rstrip("!?.") in GREETINGS:
        return {
            "answer": (
                "Bonjour ! Je suis FarmLink, ton copilote agricole. "
                "N'hésite pas à me poser une question sur les sols, les cultures, "
                "l'irrigation, la mécanisation ou les politiques agricoles."
            ),
            "contexts": [],
        }

    if any(pattern in question_clean for pattern in SPECIALTY_PATTERNS):
        return {"answer": specialty_answer(q.domain), "contexts": []}

    search_domain = q.domain
    inferred_domain = None
    if q.domain == "all":
        inferred_domain = infer_domain(q.question)
        if inferred_domain and inferred_domain in available:
            search_domain = inferred_domain

    contexts = retriever.search(q.question, top_k=q.top_k, domain=search_domain)

    missing = missing_keywords(q.question, contexts)
    question_tokens = tokenize(q.question)
    contexts_for_prompt = contexts
    if contexts and question_tokens and len(missing) == len(question_tokens):
        contexts_for_prompt = []
        contexts = []

    effective_domain = search_domain if search_domain != "all" else (inferred_domain or q.domain)
    domain_label = (
        DOMAIN_LABELS.get(effective_domain)
        if effective_domain and effective_domain != "all"
        else None
    )
    prompt = build_prompt(
        q.question,
        contexts_for_prompt,
        missing_keywords=missing,
        domain_label=domain_label,
    )
    answer = generate_answer(prompt, temperature=q.temperature)

    if q.domain == "all" and inferred_domain and search_domain == inferred_domain:
        label = DOMAIN_LABELS.get(inferred_domain)
        if label:
            answer = f"**Domaine ciblé : {label}.**\n\n" + answer

    import re as _re

    if not _re.search(r"(?i)\bsources?\s*:", answer):
        titles = short_sources(contexts_for_prompt, MAX_SOURCES)
        if titles:
            answer = (
                f"{answer.rstrip()}\n\nSources (contexte FarmLink):\n"
                + "\n".join(f"- {title}" for title in titles)
            )

    return {"answer": answer, "contexts": contexts}


def specialty_answer(domain: str) -> str:
    """Return FarmLink's meta answer about covered domains."""
    if domain != "all":
        active = DOMAIN_LABELS.get(domain, domain)
        return (
            "Je suis FarmLink, assistant RAG agricole.\n"
            f"Actuellement, je suis **réglé sur** : **{active}**.\n"
            "Je réponds uniquement aux questions liées à ce domaine."
        )

    return (
        "Je suis FarmLink, assistant RAG agricole. Domaines couverts :\n"
        "• Sols & fertilisation\n• Cultures vivrières\n• Irrigation & eau\n"
        "• Mécanisation & innovation\n• Politiques & marchés\n\n"
        "Choisis un domaine ou pose ta question."
    )


def build_prompt(
    question: str,
    contexts: List[Dict],
    missing_keywords: Optional[List[str]] = None,
    domain_label: Optional[str] = None,
) -> str:
    """Build the strict prompt used by the existing RAG flow."""
    missing = sorted(set(missing_keywords or []))
    if missing:
        keywords = ", ".join(missing)
        guidance = (
            "IMPORTANT : le CONTEXTE fourni ne couvre pas certains mots clés de la question : "
            f"{keywords}. Si l'information manque dans le CONTEXTE, dis-le explicitement, "
            "invite à reformuler ou propose de cibler un autre domaine.\n\n"
        )
    else:
        guidance = ""

    domain_text = f"dans le domaine **{domain_label}**" if domain_label else "en agriculture"
    guardrails = (
        "RÈGLES:\n"
        "1) Utilise UNIQUEMENT le CONTEXTE fourni. Si une info manque, dis-le explicitement.\n"
        "2) N'invente aucun mélange (pas de 40%/60%), ne mentionne aucune source externe.\n"
        "3) Si la question est hors du domaine actif, refuse poliment et propose des exemples pertinents.\n"
        "4) Structure: Résumé express → Analyse structurée → Recommandations (si utiles) → Ouverture (facultatif).\n"
        "5) Termine par 'Sources' listant au plus 3 titres EXACTS du CONTEXTE (pas d'URL, pas d'année si absente).\n"
    )

    if not contexts:
        return (
            f"Tu es FarmLink, assistant RAG {domain_text}.\n"
            f"{guidance}"
            + guardrails
            + "Aucune information n'est disponible dans le CONTEXTE. "
            "Réponds: indique que le contexte ne couvre pas la question et propose une reformulation précise."
        )

    context_block = "\n\n".join(
        f"- {context['text']}\n(source: {context.get('title', 'Document')} | {context.get('source', 'Corpus FarmLink')})"
        for context in contexts
        if context.get("text")
    )

    return (
        f"Tu es FarmLink, assistant RAG {domain_text}.\n"
        f"{guidance}"
        + guardrails
        + f"Question: {question}\n\n"
        "CONTEXTE:\n"
        f"{context_block}\n\n"
        "Maintenant, produis la réponse en respectant strictement les règles."
    )
