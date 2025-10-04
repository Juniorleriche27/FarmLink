
"""Streamlit front-end for FarmLink Copilot."""

from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List
import time

import requests
import streamlit as st

st.set_page_config(page_title="FarmLink Copilot", page_icon="🌾", layout="wide")

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

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

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(242, 252, 247, 0.95) 0%, rgba(232, 244, 255, 0.95) 100%);
    border-right: 1px solid rgba(31, 158, 106, 0.18);
    padding-top: 1.6rem;
    color: var(--text-main);
}

.sidebar-header {
    text-align: center;
    margin-bottom: 1.6rem;
}

.sidebar-header .logo-circle {
    width: 84px;
    height: 84px;
    margin: 0 auto 0.7rem auto;
    border-radius: 50%;
    background: rgba(31, 158, 106, 0.12);
    border: 1px solid rgba(31, 158, 106, 0.36);
    display: grid;
    place-items: center;
    font-size: 2.2rem;
}

.sidebar-header h2 {
    margin: 0;
    font-size: 1.28rem;
    letter-spacing: 0.05em;
}

.sidebar-section {
    background: rgba(255,255,255,0.92);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1.2rem 1.25rem;
    margin-bottom: 1.2rem;
}

.sidebar-history button {
    width: 100%;
    text-align: left;
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(31, 158, 106, 0.15);
    color: var(--text-main);
    padding: 0.55rem 0.8rem;
    border-radius: 9px;
    margin-bottom: 0.55rem;
    font-size: 0.92rem;
    line-height: 1.2;
}

.sidebar-history button:hover {
    background: rgba(31, 158, 106, 0.16);
    border-color: rgba(31, 158, 106, 0.4);
}

.chat-container {
    max-width: 960px;
    margin: 0 auto;
    padding-bottom: 4.5rem;
}

.chat-shell {
    background: var(--shell-bg);
    border: 1px solid var(--border-soft);
    border-radius: 22px;
    padding: 1.4rem 1.6rem 1.6rem 1.6rem;
    min-height: 60vh;
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
    box-shadow: 0 15px 45px rgba(13, 51, 34, 0.08);
}

.chat-shell form { margin-top: auto; }

.chat-message {
    display: flex;
    gap: 0.9rem;
    margin-bottom: 1rem;
    color: var(--text-main);
}

.chat-message .avatar {
    width: 38px;
    height: 38px;
    background: rgba(31, 158, 106, 0.16);
    display: grid;
    place-items: center;
    border-radius: 50%;
    font-size: 1.1rem;
}

.chat-message.assistant .bubble {
    background: var(--assistant-bg);
    border: 1px solid var(--assistant-border);
    color: var(--text-main);
}

.chat-message.user {
    flex-direction: row-reverse;
}

.chat-message.user .avatar {
    background: rgba(31, 158, 106, 0.25);
    color: #ffffff;
}

.chat-message.user .bubble {
    background: var(--user-bg);
    color: var(--user-text);
    border: 1px solid rgba(31, 158, 106, 0.25);
}

.chat-message .bubble {
    padding: 1rem 1.25rem;
    border-radius: 18px;
    line-height: 1.55;
    font-size: 0.98rem;
    max-width: 100%;
    word-break: break-word;
}

.prompt-bar {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(31, 158, 106, 0.18);
    border-radius: 999px;
    padding: 0.45rem 0.7rem;
    box-shadow: 0 18px 40px rgba(20, 84, 56, 0.15);
}

.prompt-bar .stColumn { padding: 0 !important; }

.prompt-icon {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: 1px solid rgba(31,158,106,0.25);
    background: rgba(31,158,106,0.12);
    display: grid;
    place-items: center;
    font-size: 1.05rem;
}

.prompt-bar .stTextInput > div > div > input {
    background: transparent !important;
    color: var(--text-main) !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 1rem !important;
    padding: 0.2rem 0 !important;
    height: 38px !important;
}

.prompt-bar .stButton button {
    width: 42px; height: 42px;
    border-radius: 50%;
    background: linear-gradient(120deg, #1f9e6a, #3fce93);
    color: #ffffff;
    border: none;
    font-size: 1.1rem;
}

.sources-card {
    background: rgba(242, 249, 244, 0.92);
    border: 1px solid rgba(31, 158, 106, 0.18);
    border-radius: 16px;
    padding: 1rem 1.4rem;
    margin-top: 0.8rem;
}

.sources-card h4 { margin-top: 0; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_domains(url: str) -> List[str]:
    try:
        resp = requests.get(f"{url}/domains", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("domains", [])
    except requests.RequestException:
        return ["all"]

@st.cache_data(show_spinner=False)
def check_health(url: str) -> bool:
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json().get("ok", False)
    except requests.RequestException:
        return False

DOMAIN_LABELS = {
    "all": "Tous domaines",
    "farmlink_sols": "Sols & fertilisation",
    "farmlink_cultures": "Cultures vivrières",
    "farmlink_eau": "Irrigation & eau",
    "farmlink_meca": "Mécanisation & innovation",
    "farmlink_marche": "Politiques & marchés",
}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour ! Je suis FarmLink, copilote agricole. Pose-moi une question sur les sols, "
                "les cultures, l'irrigation, la mécanisation ou les politiques agricoles : je m'appuie "
                "sur nos dossiers pour t'aider."
            ),
            "animate": False,
        }
    ]
if "contexts" not in st.session_state:
    st.session_state.contexts = []
if "history" not in st.session_state:
    st.session_state.history = []
if "animation_states" not in st.session_state:
    st.session_state.animation_states = {}

available_domains = fetch_domains(API_URL) or ["all"]

if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = available_domains[0]

health_ok = check_health(API_URL)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="logo-circle">🌾</div>
            <h2>FarmLink</h2>
            <p style="margin:0;font-size:0.9rem;color:rgba(30,70,52,0.7);">Co-pilote agricole</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown("### Réglages")
        domain_choice = st.selectbox(
            "Collection FarmLink",
            options=available_domains,
            index=available_domains.index(st.session_state.selected_domain)
            if st.session_state.selected_domain in available_domains
            else 0,
            format_func=lambda x: DOMAIN_LABELS.get(x, x),
        )
        st.session_state.selected_domain = domain_choice
        top_k = st.slider(
            "Documents (top-k)",
            min_value=2,
            max_value=10,
            value=4,
            help="Nombre de passages issus de Qdrant",
        )
        temperature = st.slider(
            "Créativité (température)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )
        st.markdown("---")
        status_text = "🟢 en ligne" if health_ok else "🔴 hors ligne"
        st.markdown(f"<small>API : {status_text}</small>", unsafe_allow_html=True)
        st.markdown(f"<small>Endpoint&nbsp;: <code>{escape(API_URL)}</code></small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Historique", unsafe_allow_html=True)
    if st.button("➕ Nouveau chat", use_container_width=True):
        if len(st.session_state.messages) > 1:
            first_user = next(
                (m["content"] for m in st.session_state.messages if m["role"] == "user"),
                "Conversation précédente",
            )
            summary = first_user[:80] + ("…" if len(first_user) > 80 else "")
            st.session_state.history.insert(
                0,
                {
                    "title": summary,
                    "timestamp": datetime.now().strftime("%d/%m %H:%M"),
                    "messages": deepcopy(st.session_state.messages),
                },
            )
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Nouvelle session ouverte. Pose-moi une question agricole !",
                "animate": False,
            }
        ]
        st.session_state.contexts = []
        st.session_state.animation_states = {}
        st.rerun()
    st.markdown('<div class="sidebar-history">', unsafe_allow_html=True)
    for idx, thread in enumerate(st.session_state.history):
        label = thread.get("title", f"Chat {idx + 1}")
        timestamp = thread.get("timestamp", "")
        btn_label = f"{label} ({timestamp})" if timestamp else label
        if st.button(btn_label, key=f"history-{idx}", use_container_width=True):
            st.session_state.messages = deepcopy(thread["messages"])
            st.session_state.contexts = []
            st.session_state.animation_states = {}
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:1.4rem 0 0.6rem 0;">
        <div>
            <h1 style="margin-bottom:0.3rem;">FarmLink Copilot</h1>
            <p style="margin:0;color:rgba(30,70,52,0.7);">Assistant RAG pour les filières agricoles africaines.</p>
        </div>
        <div class="status-pill">🌾 Domaine actif : {DOMAIN_LABELS.get(st.session_state.selected_domain, st.session_state.selected_domain)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

needs_rerun = False

def render_message(msg: Dict[str, str], idx: int) -> None:
    global needs_rerun
    role = msg.get("role", "assistant")
    avatar = "🌾" if role == "assistant" else "👤"
    cls = "assistant" if role == "assistant" else "user"
    key = f"msg-{idx}"
    content = msg.get("content", "")
    target_len = len(content)
    animate = msg.get("animate", False) and role == "assistant"
    current_len = st.session_state.animation_states.get(key, target_len if not animate else 0)
    if animate:
        step = max(1, target_len // 60)
        current_len = min(current_len + step, target_len)
        st.session_state.animation_states[key] = current_len
        displayed = content[:current_len]
        if current_len < target_len:
            needs_rerun = True
        else:
            msg["animate"] = False
    else:
        displayed = content
        st.session_state.animation_states[key] = target_len
    html = escape(displayed).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="chat-message {cls}">
            <div class="avatar">{avatar}</div>
            <div class="bubble">{html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for index, message in enumerate(st.session_state.messages):
    render_message(message, index)

if st.session_state.contexts:
    st.markdown('<div class="sources-card">', unsafe_allow_html=True)
    st.markdown("<h4>Sources mobilisées</h4>", unsafe_allow_html=True)
    for ctx in st.session_state.contexts:
        title = escape(ctx.get("title", "Document"))
        display_src = ctx.get("display_source")
        if display_src:
            source_label = escape(display_src)
        else:
            source_raw = ctx.get("source", "")
            source_label = escape(Path(source_raw).name) if source_raw else "Corpus FarmLink"
        score = ctx.get("score")
        score_txt = f" — score {score:.3f}" if isinstance(score, (int, float)) else ""
        st.markdown(f"- **{title}** — {source_label}{score_txt}")
    st.markdown('</div>', unsafe_allow_html=True)

with st.form("chat-input", clear_on_submit=True):
    st.markdown('<div class="prompt-bar">', unsafe_allow_html=True)
    icon_col, input_col, send_col = st.columns([0.09, 0.82, 0.09])
    with icon_col:
        st.markdown('<div class="prompt-icon">＋</div>', unsafe_allow_html=True)
    with input_col:
        question = st.text_input(
            "Question agricole",
            placeholder="Posez votre question...",
            label_visibility="collapsed",
        )
    with send_col:
        submitted = st.form_submit_button("➤")
    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    user_message = question.strip()
    if not user_message:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_message, "animate": False})

    payload = {
        "question": user_message,
        "domain": st.session_state.selected_domain,
        "top_k": top_k,
        "temperature": temperature,
    }

    with st.spinner("Analyse du corpus FarmLink…"):
        try:
            response = requests.post(f"{API_URL}/query", json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "") or "Je n'ai pas pu générer de réponse."
            st.session_state.messages.append({"role": "assistant", "content": answer, "animate": True})
            st.session_state.contexts = data.get("contexts", [])
            st.session_state.animation_states = {}
        except requests.RequestException as exc:
            error_msg = f"Je n'arrive pas à contacter l'API FarmLink ({exc})."
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "animate": False})
            st.session_state.contexts = []

    st.rerun()

if needs_rerun:
    time.sleep(0.03)
    st.rerun()
