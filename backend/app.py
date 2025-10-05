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

_COLLECTION_SUFFIXES = {
    "farmlink_sols": "SOL",
    "farmlink_marche": "MARCHE",
    "farmlink_cultures": "CULT",
    "farmlink_eau": "EAU",
    "farmlink_meca": "MECA",
}

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
    if question_clean in GREETINGS or question_clean.rstrip('!?.') in GREETINGS:
        return {
            "answer": (
                "Bonjour ! Je suis FarmLink, ton copilote agricole. "
                "N'hésite pas à me poser une question sur les sols, les cultures, "
                "l'irrigation, la mécanisation ou les politiques agricoles."
            ),
            "contexts": []
        }

    contexts = retriever.search(q.question, top_k=q.top_k, domain=q.domain)
    missing_keywords = _missing_keywords(q.question, contexts)
    question_tokens = _tokenize(q.question)
    contexts_for_prompt = contexts
    if contexts and question_tokens and len(missing_keywords) == len(question_tokens):
        contexts_for_prompt = []
        contexts = []
    prompt = build_prompt(q.question, contexts_for_prompt, missing_keywords)
    answer = generate_answer(prompt, temperature=q.temperature)
    return {"answer": answer, "contexts": contexts}

def build_prompt(question: str, contexts: List[Dict], missing_keywords: Optional[List[str]] = None) -> str:
    """Build the final prompt for the LLM, optionally flagging missing keywords."""
    missing = sorted(set(missing_keywords or []))
    if missing:
        keywords = ", ".join(missing)
        guidance = (
            "IMPORTANT : le CONTEXTE fourni ne couvre pas les mots clés suivants issus de la question : "
            f"{keywords}. Si tu n'as pas assez d'informations provenant du corpus, "
            "dis-le explicitement, invite à reformuler ou suggère de cibler un autre domaine.\n\n"
        )
    else:
        guidance = ""

    if not contexts:
        return (
            "Tu es FarmLink, assistant RAG specialise en agriculture.\n"
            f"{guidance}"
            "Aucune information n'est disponible. Invite l'utilisateur à reformuler."
        )

    ctx_block = "\n\n".join(
        f"- {c['text']}\n(source: {c['title']} | {c['source']})" for c in contexts
    )

    return (
        "Tu es FarmLink, assistant RAG specialise en agriculture.\n"
        f"{guidance}"
        "Réponds uniquement avec les informations du CONTEXTE. "
        "Si ce n'est pas couvert, dis-le clairement.\n"
        f"Question: {question}\n\n"
        "CONTEXTE:\n"
        f"{ctx_block}\n\n"
        "Réponse détaillée, claire et structurée avec une courte synthèse finale et les sources citées en fin de message."
    )

