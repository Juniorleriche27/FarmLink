"""Configuration constants for the FarmLink backend."""

from __future__ import annotations

import os
from typing import Dict

APP_TITLE = "FarmLink API"
DEFAULT_DOMAIN = "all"
MAX_SOURCES = 3

GREETINGS = {
    "salut",
    "bonjour",
    "bonsoir",
    "hello",
    "hi",
    "coucou",
    "bjr",
    "bon matin",
    "bonsoir farm",
    "hey",
}

SPECIALTY_PATTERNS = [
    "spécialisé",
    "specialise",
    "specialisé",
    "domaines",
    "compétences",
    "tu fais quoi",
    "tu aides sur quoi",
    "c'est quoi ton domaine",
    "tes domaines",
    "dans quoi es-tu spécialisé",
    "specialite",
    "spécialité",
]

DOMAIN_LABELS = {
    "all": "Tous domaines",
    "farmlink_sols": "Sols & fertilisation",
    "farmlink_cultures": "Cultures vivrières",
    "farmlink_eau": "Irrigation & eau",
    "farmlink_meca": "Mécanisation & innovation",
    "farmlink_marche": "Politiques & marchés",
}

DOMAIN_KEYWORDS = {
    "farmlink_sols": {
        "sol",
        "sols",
        "fertilite",
        "fertilisation",
        "amendement",
        "compost",
        "humus",
        "erosion",
        "ph",
        "matiere",
        "microbiologie",
        "terre",
        "nutriment",
    },
    "farmlink_cultures": {
        "culture",
        "cultures",
        "mais",
        "riz",
        "cacao",
        "coton",
        "sorgho",
        "arachide",
        "manioc",
        "banane",
        "rendement",
        "semence",
        "production",
        "recolte",
    },
    "farmlink_eau": {
        "irrigation",
        "irriguer",
        "goutte",
        "eau",
        "hydrique",
        "arrosage",
        "drainage",
        "barrage",
        "forage",
        "pluvial",
        "pluie",
        "canal",
    },
    "farmlink_meca": {
        "mecanisation",
        "mechanisation",
        "machinisme",
        "tracteur",
        "tractor",
        "moissonneuse",
        "equipement",
        "equipements",
        "motorisation",
        "semis",
        "batteuse",
        "outil",
        "machine",
    },
    "farmlink_marche": {
        "marche",
        "marches",
        "prix",
        "politique",
        "politiques",
        "subvention",
        "subventions",
        "commerce",
        "commercialisation",
        "chaine",
        "valeur",
        "credit",
        "financement",
        "investissement",
        "market",
    },
}

COLLECTION_SUFFIXES = {
    "farmlink_sols": "SOL",
    "farmlink_marche": "MARCHE",
    "farmlink_cultures": "CULT",
    "farmlink_eau": "EAU",
    "farmlink_meca": "MECA",
}


def raw_qdrant_endpoints() -> Dict[str, Dict[str, str]]:
    """Read Qdrant endpoint settings without validating availability."""
    base_url = (os.getenv("QDRANT_URL") or "").strip()
    base_key = (os.getenv("QDRANT_API_KEY") or "").strip()
    active_only = {
        name.strip()
        for name in (os.getenv("QDRANT_ACTIVE_COLLECTIONS") or "").split(",")
        if name.strip()
    }

    endpoints: Dict[str, Dict[str, str]] = {}
    for collection, suffix in COLLECTION_SUFFIXES.items():
        if active_only and collection not in active_only:
            continue
        url = (os.getenv(f"QDRANT_{suffix}_URL") or base_url).strip()
        api_key = (os.getenv(f"QDRANT_{suffix}_KEY") or base_key).strip()
        endpoints[collection] = {"url": url, "api_key": api_key}
    return endpoints


def active_qdrant_endpoints(raw: Dict[str, Dict[str, str]] | None = None) -> Dict[str, Dict[str, str]]:
    """Keep only configured Qdrant endpoints."""
    active: Dict[str, Dict[str, str]] = {}
    for name, cfg in (raw or raw_qdrant_endpoints()).items():
        url = (cfg.get("url") or "").strip()
        api_key = (cfg.get("api_key") or "").strip()
        if url and api_key:
            active[name] = {"url": url, "api_key": api_key}
    return active
