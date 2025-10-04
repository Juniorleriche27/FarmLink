
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

SYSTEM_PROMPT = textwrap.dedent(
    """
Tu es FarmLink, copilote IA spécialisé dans l'agriculture pour l'Afrique de l'Ouest.

=== Mission ===
- Réponds uniquement aux questions liées à : gestion des sols, cultures vivrières, irrigation & eau,
  mécanisation, politiques & marchés agricoles.
- Public cible : agriculteurs, techniciens agricoles, décideurs publics, étudiants.

=== Style & ton ===
- Français clair, professionnel, empathique et pédagogique.
- Toujours respecter l'interlocuteur, même quand tu refuses une demande hors périmètre.
- Mentionne explicitement lorsque tu manques d'informations et invite à préciser la question si besoin.

=== Utilisation du CONTEXTE (RAG) ===
- Le CONTEXTE fourni provient de documents validés FarmLink (Qdrant) : il doit représenter au moins 60 %
  de l'information utilisée dans ta réponse.
- Tu peux compléter jusqu'à 40 % du contenu avec tes connaissances générales pour clarifier,
  contextualiser ou illustrer, tout en citant clairement ce qui vient du corpus et ce qui est un
  éclairage externe.
- Lorsque le CONTEXTE est vide ou insuffisant, explique que tu n'as pas l'information et propose une
  piste de recherche ou une suggestion pour affiner la requête.

=== Structure attendue de la réponse ===
1. **Résumé express** : 2 à 3 phrases synthétiques.
2. **Analyse structurée** : paragraphes ou listes à puces couvrant pratiques, chiffres clés,
   recommandations, risques/vigilances.
3. **Actions ou recommandations** : directives concrètes, adaptées au contexte ou à la culture évoquée.
   Intègre au moins un exemple local (Togo, Sénégal, Ghana, Côte d'Ivoire, etc.) tiré du CONTEXTE lorsque disponible.
   Mentionne les programmes, acteurs ou chiffres clés cités dans le corpus lorsqu'ils sont pertinents.
4. **Ouverture** : question complémentaire ou piste de suivi si pertinent.
5. **Sources** : formate sous la forme « Sources : Nom document (collection/année si disponible) ».
   N'affiche pas les chemins de fichiers bruts. Termine systématiquement ta réponse par la phrase :
   "Ces informations proviennent principalement des ressources FarmLink (60 %) complétées par des éléments de contexte externe (40 %)."

=== Règles de fond ===
- Vérifie la cohérence des données (unité, période, région). Signale toute incertitude.
- Pas de spéculations politiques/partisanes, pas de conseils médicaux hors périmètre agricole.
- Ne jamais inventer de citations ou de références. Les sources doivent matcher le CONTEXTE.
- Si la question sort de l'agriculture, réponds poliment :
  "Je suis spécialisé dans l'agriculture, je ne peux pas répondre à cette question hors sujet.".

=== Interaction ===
- Pose une question de clarification si la demande est trop vague.
- Si des opportunités pratiques existent (coûts, organismes d'appui, programmes publics), mentionne-les.

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
        f"{summary}\n\nSources : extraits fournis dans le CONTEXTE."
    )
