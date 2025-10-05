import os
import re
from difflib import get_close_matches
from unicodedata import normalize
from typing import Any, Dict, List, Set, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ⚠️ on n'importe PAS le retriever ici (trop lourd) → import lazy plus bas
# from retrievers.multi_qdrant_retriever import MultiQdrantRetriever
from llm.generator import generate_answer  # OK (léger)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = FastAPI(title="FarmLink API")

# CORS: autorise le front (tu pourras restreindre l'origine plus tard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # mets l’URL de ton Streamlit si tu veux restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GREETINGS = {
    'salut', 'bonjour', 'bonsoir', 'hello', 'hi', 'coucou',
    'bjr', 'bon matin', 'bonsoir farm', 'hey'
}

# Détection question “Tu es spécialisé en quoi ?”
SPECIALTY_PATTERNS = [
    "spécialisé", "specialise", "specialisé", "domaines", "compétences",
    "tu fais quoi", "tu aides sur quoi", "c'est quoi ton domaine",
    "tes domaines", "dans quoi es-tu spécialisé", "specialite", "spécialité",
]

DOMAIN_LABELS = {
    "all": "Tous domaines",
    "farmlink_sols": "Sols & fertilisation",
    "farmlink_cultures": "Cultures vivrières",
    "farmlink_eau": "Irrigation & eau",
    "farmlink_meca": "Mécanisation & innovation",
    "farmlink_marche": "Politiques & marchés",
}

_DOMAIN_KEYWORDS = {
    "farmlink_sols": {
        "sol", "sols", "fertilite", "fertilisation", "amendement", "compost", "humus", "erosion",
        "ph", "matiere", "microbiologie", "terre", "nutriment"
    },
    "farmlink_cultures": {
        "culture", "cultures", "mais", "riz", "cacao", "coton", "sorgho", "arachide", "manioc",
        "banane", "rendement", "semence", "production", "recolte"
    },
    "farmlink_eau": {
        "irrigation", "irriguer", "goutte", "eau", "hydrique", "arrosage", "drainage", "barrage",
        "forage", "pluvial", "pluie", "canal"
    },
    "farmlink_meca": {
        "mecanisation", "mechanisation", "machinisme", "tracteur", "tractor", "moissonneuse",
        "equipement", "equipements", "motorisation", "semis", "batteuse", "outil", "machine"
    },
    "farmlink_marche": {
        "marche", "marches", "prix", "politique", "politiques", "subvention", "subventions",
        "commerce", "commercialisation", "chaine", "valeur", "credit", "financement",
        "investissement", "market"
    },
}

_COLLECTION_SUFFIXES = {
    "farmlink_sols": "SOL",
    "farmlink_marche": "MARCHE",
    "farmlink_cultures": "CULT",
    "farmlink_eau": "EAU",
    "farmlink_meca": "MECA",
}

# Limite d’affichage des sources dans la réponse
MAX_SOURCES = 3

_WORD_RE = re.compile(r"[a-z0-9]{3,}")

def _normalize_text(value: str) -> str:
    """Return a lowercase ASCII-only version of the input."""
    normalized = normalize("NFKD", (value or ""))
    return normalized.encode("ascii", "ignore").decode("ascii").lower()

def _tokenize(value: str) -> List[str]:
    """Tokenize text for simple keyword coverage checks."""
    return _WORD_RE.findall(_normalize_text(value))

def _missing_keywords(question: str, contexts: List[Dict], cutoff: float = 0.82) -> List[str]:
    """Identify question keywords that are absent from retrieved contexts."""
    query_tokens = _tokenize(question)
    if not query_tokens or not contexts:
        return query_tokens if query_tokens and not contexts else []

    context_tokens: Set[str] = set()
    for ctx in contexts:
        context_tokens.update(_tokenize(ctx.get("text", "")))
        context_tokens.update(_tokenize(ctx.get("title", "")))

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

def _infer_domain(question: str) -> Optional[str]:
    """Heuristique simple pour deviner le domaine si l'utilisateur n'en choisit pas."""
    tokens = set(_tokenize(question))
    if not tokens:
        return None
    best_domain: Optional[str] = None
    best_score = 0
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = len(tokens & keywords)
        if score > best_score:
            best_domain = domain
            best_score = score
    return best_domain if best_score else None

def _raw_endpoints() -> Dict[str, Dict[str, str]]:
    base_url = (os.getenv("QDRANT_URL") or "").strip()
    base_key = (os.getenv("QDRANT_API_KEY") or "").strip()
    active_only = {
        name.strip()
        for name in (os.getenv("QDRANT_ACTIVE_COLLECTIONS") or "").split(",")
        if name.strip()
    }

    endpoints: Dict[str, Dict[str, str]] = {}
    for collection, suffix in _COLLECTION_SUFFIXES.items():
        if active_only and collection not in active_only:
            continue
        url_env = f"QDRANT_{suffix}_URL"
        key_env = f"QDRANT_{suffix}_KEY"
        url = (os.getenv(url_env) or base_url).strip()
        api_key = (os.getenv(key_env) or base_key).strip()
        endpoints[collection] = {"url": url, "api_key": api_key}
    return endpoints

def _filter_endpoints(raw: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    active: Dict[str, Dict[str, str]] = {}
    for name, cfg in raw.items():
        url = (cfg.get("url") or "").strip()
        api_key = (cfg.get("api_key") or "").strip()
        if not url or not api_key:
            continue
        active[name] = {"url": url, "api_key": api_key}
    return active

# ===== Lazy init du retriever =====
_retriever: Any = None
_endpoints_cache: Dict[str, Dict[str, str]] | None = None

def get_retriever():
    """
    Initialise le MultiQdrantRetriever au premier appel seulement.
    Évite un cold start trop long.
    """
    global _retriever, _endpoints_cache
    if _retriever is not None:
        return _retriever

    # import LOURD ici, pas au module
    from retrievers.multi_qdrant_retriever import MultiQdrantRetriever

    if _endpoints_cache is None:
        _endpoints_cache = _filter_endpoints(_raw_endpoints())

    _retriever = MultiQdrantRetriever(_endpoints_cache or {})
    return _retriever

# ===== Modèles =====
class QueryIn(BaseModel):
    question: str
    domain: str = "all"
    top_k: int = 4
    temperature: float = 0.2

# ===== Utils =====
def _short_sources(contexts: List[Dict], limit: int = MAX_SOURCES) -> List[str]:
    out = []
    for c in contexts[:limit]:
        title = c.get("title") or "Document"
        out.append(str(title))
    return out

# ===== Endpoints =====
@app.get("/", include_in_schema=False)
def root():
    return {
        "name": "FarmLink API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
        "domains": "/domains",
        "query": {"path": "/query", "method": "POST"},
    }

@app.get("/health")
def health():
    # Ne déclenche pas le chargement du modèle → réponse instantanée
    return {"ok": True}

@app.get("/domains")
def domains():
    # essaie d'utiliser le retriever, mais si endpoints vides renvoie quand même "all"
    try:
        r = get_retriever()
        domain_list = getattr(r, "available_collections", [])
    except Exception:
        domain_list = []
    if domain_list:
        domain_list = domain_list + ["all"]
    else:
        domain_list = ["all"]
    return {"domains": domain_list}

@app.post("/query")
def query(q: QueryIn):
    retriever = get_retriever()  # le modèle est (lazily) chargé ici

    available = set(getattr(retriever, "available_collections", []))
    if q.domain != "all" and q.domain not in available:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{q.domain}'")

    question_clean = q.question.strip().lower()

    # 1) Salutations ?
    if question_clean in GREETINGS or question_clean.rstrip('!?.') in GREETINGS:
        return {
            "answer": (
                "Bonjour ! Je suis FarmLink, ton copilote agricole. "
                "N'hésite pas à me poser une question sur les sols, les cultures, "
                "l'irrigation, la mécanisation ou les politiques agricoles."
            ),
            "contexts": []
        }

    # 2) “Tu es spécialisé en quoi ?” → réponse meta, sans RAG
    if any(p in question_clean for p in SPECIALTY_PATTERNS):
        if q.domain != "all":
            actif = DOMAIN_LABELS.get(q.domain, q.domain)
            answer = (
                f"Je suis FarmLink, assistant RAG agricole.\n"
                f"Actuellement, je suis **réglé sur** : **{actif}**.\n"
                "Je réponds uniquement aux questions liées à ce domaine."
            )
        else:
            answer = (
                "Je suis FarmLink, assistant RAG agricole. Domaines couverts :\n"
                "• Sols & fertilisation\n• Cultures vivrières\n• Irrigation & eau\n"
                "• Mécanisation & innovation\n• Politiques & marchés\n\n"
                "Choisis un domaine ou pose ta question."
            )
        return {"answer": answer, "contexts": []}

    # 3) RAG normal
    search_domain = q.domain
    inferred_domain = None
    if q.domain == "all":
        inferred_domain = _infer_domain(q.question)
        if inferred_domain and inferred_domain in available:
            search_domain = inferred_domain

    contexts = retriever.search(q.question, top_k=q.top_k, domain=search_domain)

    # Heuristique: si aucun mot de la question ne se retrouve dans le contexte → vide
    missing_keywords = _missing_keywords(q.question, contexts)
    question_tokens = _tokenize(q.question)
    contexts_for_prompt = contexts
    if contexts and question_tokens and len(missing_keywords) == len(question_tokens):
        contexts_for_prompt = []
        contexts = []

    effective_domain = search_domain if search_domain != "all" else (inferred_domain or q.domain)
    domain_label = DOMAIN_LABELS.get(effective_domain) if effective_domain and effective_domain != "all" else None
    prompt = build_prompt(
        q.question,
        contexts_for_prompt,
        missing_keywords=missing_keywords,
        domain_label=domain_label,
    )
    answer = generate_answer(prompt, temperature=q.temperature)

    if q.domain == "all" and inferred_domain and search_domain == inferred_domain:
        label = DOMAIN_LABELS.get(inferred_domain)
        if label:
            answer = f"**Domaine ciblé : {label}.**\n\n" + answer

    # 4) Si le modèle “oublie” la section sources, on ajoute (max 3 titres)
    import re as _re
    if not _re.search(r"(?i)\bsources?\s*:", answer):
        titles = _short_sources(contexts_for_prompt, MAX_SOURCES)
        if titles:
            answer = f"{answer.rstrip()}\n\nSources (contexte FarmLink):\n" + \
                     "\n".join(f"- {t}" for t in titles)

    return {"answer": answer, "contexts": contexts}

def build_prompt(
    question: str,
    contexts: List[Dict],
    missing_keywords: Optional[List[str]] = None,
    domain_label: Optional[str] = None,
) -> str:
    """Prompt strict : contexte uniquement, max 3 sources, refus hors périmètre si besoin."""
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

    domaine_txt = f"dans le domaine **{domain_label}**" if domain_label else "en agriculture"
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
            f"Tu es FarmLink, assistant RAG {domaine_txt}.\n"
            f"{guidance}"
            + guardrails +
            "Aucune information n'est disponible dans le CONTEXTE. "
            "Réponds: indique que le contexte ne couvre pas la question et propose une reformulation précise."
        )

    ctx_block = "\n\n".join(
        f"- {c['text']}\n(source: {c.get('title','Document')} | {c.get('source','Corpus FarmLink')})"
        for c in contexts
        if c.get('text')
    )

    return (
        f"Tu es FarmLink, assistant RAG {domaine_txt}.\n"
        f"{guidance}"
        + guardrails +
        f"Question: {question}\n\n"
        "CONTEXTE:\n"
        f"{ctx_block}\n\n"
        "Maintenant, produis la réponse en respectant strictement les règles."
    )
