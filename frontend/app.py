"""Streamlit front-end for FarmLink Copilot."""
import os, time, requests
from html import escape
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Dict, List
import streamlit as st

st.set_page_config(page_title="FarmLink Copilot", page_icon="🌾", layout="wide")

# ⚠️ Utilise les variables d’environnement sur Render
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# === ton CSS (inchangé) ===
CUSTOM_CSS = """
<style>
:root {
    --bg-gradient: linear-gradient(180deg, #f8fbff 0%, #f1f9f3 45%, #ffffff 100%);
    --assistant-bg: #ffffff;
    --assistant-border: #d7e7dd;
    --user-bg: linear-gradient(135deg, #27c082 0%, #4de8b7 100%);
    --user-text: #ffffff;
    --shell-bg: rgba(255, 255, 255, 0.92);
    --border-soft: rgba(22, 95, 66, 0.12);
    --accent: #1f9e6a;
    --text-main: #113824;
}
html, body, [class*="block-container"] {
    background: var(--bg-gradient) !important;
    color: var(--text-main);
}
.stAppDeployButton, footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# === Fonctions d’appel API ===
@st.cache_data(show_spinner=False)
def fetch_domains(url: str) -> List[str]:
    try:
        resp = requests.get(f"{url}/domains", timeout=10)
        resp.raise_for_status()
        return resp.json().get("domains", []) or ["all"]
    except Exception:
        return ["all"]

@st.cache_data(show_spinner=False)
def check_health(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        r.raise_for_status()
        return r.json().get("ok", False)
    except Exception:
        return False

# === Session state ===
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Bonjour ! Je suis FarmLink, copilote agricole. Pose-moi une question.",
    }]

if "contexts" not in st.session_state:
    st.session_state.contexts = []

# === Interface principale ===
st.title("🌾 FarmLink Copilot — Assistant RAG agricole africain")
st.write("Pose une question sur les sols, cultures, irrigation ou mécanisation.")

domain = st.selectbox("Domaine", fetch_domains(API_URL))
top_k = st.slider("Nombre de documents", 2, 10, 4)
temperature = st.slider("Créativité", 0.0, 1.0, 0.2, 0.05)

user_input = st.text_input("Ta question ici 👇")

if st.button("Envoyer"):
    if user_input.strip():
        payload = {"question": user_input, "domain": domain, "top_k": top_k, "temperature": temperature}
        with st.spinner("Réflexion..."):
            try:
                resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": data.get("answer", "Pas de réponse.")})
                st.session_state.contexts = data.get("contexts", [])
            except requests.RequestException as e:
                st.error(f"Erreur API : {e}")
            except ValueError:
                st.error("Réponse API non valide (JSON attendu).")

for m in st.session_state.messages:
    st.markdown(f"**{m['role'].capitalize()} :** {m['content']}")

if st.session_state.contexts:
    st.markdown("### Sources :")
    for c in st.session_state.contexts:
        st.write(f"- {c.get('title')} ({c.get('source')})")
