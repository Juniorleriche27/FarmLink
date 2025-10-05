import os
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import de ton moteur RAG et LLM
from retrievers.multi_qdrant_retriever import MultiQdrantRetriever
from llm.generator import generate_answer

app = FastAPI(title="FarmLink API")

# Autoriser ton front Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tu pourras remplacer * par ton URL Streamlit plus tard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- Configuration Qdrant -----------

_COLLECTION_SUFFIXES = {
    "farmlink_sols": "SOL",
    "farmlink_marche": "MARCHE",
    "farmlink_cultures": "CULT",
    "farmlink_eau": "EAU",
    "farmlink_meca": "MECA",
}

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

ENDPOINTS = _filter_endpoints(_raw_endpoints())
retriever = MultiQdrantRetriever(ENDPOINTS)

# ----------- Modèle de requête -----------

class QueryIn(BaseModel):
    question: str
    domain: str = "all"
    top_k: int = 4
    temperature: float = 0.2

# ----------- Endpoints -----------

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/domains")
def domains():
    domain_list = retriever.available_collections
    if domain_list:
        domain_list = domain_list + ["all"]
    return {"domains": domain_list}

@app.post("/query")
def query(q: QueryIn):
    available = set(retriever.available_collections)
    if q.domain != "all" and q.domain not in available:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{q.domain}'")

    greetings = {
        'salut', 'bonjour', 'bonsoir', 'hello', 'hi', 'coucou',
        'bjr', 'bon matin', 'bonsoir farm', 'hey'
    }

    question_clean = q.question.strip().lower()
    if question_clean in greetings or question_clean.rstrip('!?.') in greetings:
        return {
            "answer": (
                "Bonjour ! Je suis FarmLink, ton copilote agricole. "
                "N'hésite pas à me poser une question sur les sols, les cultures, "
                "l'irrigation, la mécanisation ou les politiques agricoles."
            ),
            "contexts": []
        }

    contexts = retriever.search(q.question, top_k=q.top_k, domain=q.domain)

    prompt = build_prompt(q.question, contexts)
    answer = generate_answer(prompt, temperature=q.temperature)
    return {"answer": answer, "contexts": contexts}


def build_prompt(question: str, contexts: List[Dict]) -> str:
    if not contexts:
        return (
            "Tu es FarmLink, assistant RAG specialise en agriculture.\n"
            "Aucune information n'est disponible. Invite l'utilisateur à reformuler."
        )
    ctx_block = "\n\n".join(
        f"- {c['text']}\n(source: {c['title']} | {c['source']})" for c in contexts
    )
    return (
        "Tu es FarmLink, assistant RAG specialise en agriculture.\n"
        "Réponds uniquement avec les informations du CONTEXTE. "
        "Si ce n'est pas couvert, dis-le clairement.\n"
        f"Question: {question}\n\n"
        "CONTEXTE:\n"
        f"{ctx_block}\n\n"
        "Réponse détaillée, claire et structurée avec une courte synthèse finale et les sources citées en fin de message."
    )
