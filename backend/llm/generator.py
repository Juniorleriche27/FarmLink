"""LLM generation helpers for FarmLink using the shared Mistral endpoint."""

import json
import logging
import os
import textwrap
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistral-small")
DEFAULT_PROVIDER = "mistral"
TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# ⚠️ NOUVEAU SYSTEM PROMPT (plus de 60/40, contexte uniquement, sources max 3)
SYSTEM_PROMPT = textwrap.dedent(
    """
    Tu es FarmLink, copilote IA spécialisé dans l'agriculture pour l'Afrique de l'Ouest.

    === Mission ===
    - Réponds uniquement aux questions liées à : gestion des sols, cultures vivrières, irrigation & eau,
      mécanisation, politiques & marchés agricoles.
    - Si la question sort de l’agriculture, refuse poliment en rappelant ton périmètre.

    === Style & ton ===
    - Français clair, professionnel, empathique et pédagogique.
    - Toujours respecter l'interlocuteur, même quand tu refuses une demande hors périmètre.
    - Mentionne explicitement lorsque tu manques d'informations et invite à préciser la question si besoin.

    === Utilisation du CONTEXTE (RAG) ===
    - Réponds UNIQUEMENT à partir du CONTEXTE fourni dans le message utilisateur.
    - Ne mentionne aucune source externe et n’indique jamais de pourcentages d’origine (pas de 40%/60%).
    - Si le CONTEXTE est vide ou insuffisant : dis-le clairement et propose une reformulation précise
      ou suggère de sélectionner un autre domaine pertinent.

    === Structure attendue de la réponse ===
    1. **Résumé express** : 2 à 3 phrases synthétiques.
    2. **Analyse structurée** : paragraphes ou puces couvrant pratiques, chiffres/ordres de grandeur s’ils sont
       dans le CONTEXTE, recommandations, risques/vigilances.
    3. **Actions ou recommandations** : au moins deux actions opérationnelles (leviers, outils, partenariats)
       basé(es) sur le CONTEXTE.
    4. **Ouverture** (facultatif) : une question ou prochaine étape concrète.
    5. **Sources** : liste au plus 3 TITRES EXACTS issus du CONTEXTE (pas d’URL, pas de chemins de fichiers,
       pas d’années si absentes). Ne pas inventer de références.

    === Règles de fond ===
    - Vérifie la cohérence (unité, période, région). Signale toute incertitude.
    - Pas de spéculations politiques/partisanes, pas de conseils médicaux hors périmètre.
    - Ne jamais inventer de citations ou de références ; tout doit provenir du CONTEXTE.
    - Si la question est “Dans quoi es-tu spécialisé ? / Tes domaines ?”, réponds brièvement :
      “Sols & fertilisation, Cultures vivrières, Irrigation & eau, Mécanisation & innovation,
      Politiques & marchés”. Et précise que tu réponds dans le cadre du domaine actif si indiqué.

    === Interaction ===
    - Pose une question de clarification si la demande est trop vague.
    - Si des opportunités pratiques existent dans le CONTEXTE (coûts, dispositifs, programmes), mentionne-les.

    Agis comme une ressource experte, fiable et orientée terrain pour l'écosystème agricole.
    """
)


def _get_api_key() -> Optional[str]:
    key = os.getenv("LLM_API_KEY")
    if key:
        key = key.strip()
    return key or None


def generate_answer(prompt: str, temperature: float = 0.2, provider: Optional[str] = None) -> str:
    provider = (provider or DEFAULT_PROVIDER).lower().strip()
    if provider != "mistral":
        LOGGER.warning("Unsupported LLM provider '%s'; only 'mistral' is available.", provider)
        return _fallback_answer(prompt)

    api_key = _get_api_key()
    if not api_key:
        LOGGER.warning("LLM_API_KEY missing for Mistral, using fallback formatter.")
        return _fallback_answer(prompt)

    try:
        return _call_mistral(prompt, temperature, api_key)
    except Exception as exc:  # pragma: no cover - network defensive
        LOGGER.warning("Mistral API call failed: %s", exc)
        return _fallback_answer(prompt)


def _call_mistral(prompt: str, temperature: float, api_key: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEFAULT_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _fallback_answer(prompt: str) -> str:
    # Fallback hors-ligne: synthèse brute des extraits CONTEXTE si présents
    context_section = ""
    if "CONTEXTE:" in prompt:
        _, context_section = prompt.split("CONTEXTE:", 1)
    bullets = []
    for line in context_section.splitlines():
        line = line.strip()
        if line.startswith("- "):
            bullets.append(line[2:])
    if not bullets:
        return (
            "Mode hors ligne : aucune donnée du corpus n'est disponible pour répondre. "
            "Merci de réessayer plus tard ou de préciser votre question."
        )
    summary = "\n".join(f"- {item}" for item in bullets[:4])
    return (
        "Mode hors ligne : synthèse des extraits pertinents du corpus FarmLink :\n"
        f"{summary}\n\nSources : extraits fournis dans le CONTEXTE (max 3 titres)."
    )
