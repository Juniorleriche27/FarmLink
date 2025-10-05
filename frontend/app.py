"""Streamlit front-end for FarmLink Copilot."""
import os
from copy import deepcopy
from html import escape
from typing import Dict, List

import requests
import streamlit as st

st.set_page_config(page_title="FarmLink Copilot", page_icon="🌾", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

DEFAULT_GREETING = {
    "role": "assistant",
    "content": (
        "Bonjour ! Je suis FarmLink, ton copilote agricole. "
        "Pose-moi une question sur les sols, les cultures, l'irrigation, "
        "la mécanisation ou les politiques publiques."
    ),
}


def inject_styles() -> None:
    """Inject custom CSS so the UI matches the FarmLink branding."""
    st.markdown(
        """
        <style>
        :root {
            --gradient-bg: linear-gradient(180deg, #f4fbff 0%, #f1f9f3 45%, #ffffff 100%);
            --sidebar-gradient: linear-gradient(180deg, rgba(216, 238, 231, 0.95) 0%, rgba(240, 250, 245, 0.98) 100%);
            --assistant-bg: rgba(250, 255, 252, 0.9);
            --assistant-border: rgba(31, 158, 106, 0.18);
            --user-bg: linear-gradient(135deg, #22b473 0%, #3edb9f 100%);
            --user-text: #ffffff;
            --section-bg: rgba(247, 255, 251, 0.9);
            --section-border: rgba(31, 158, 106, 0.22);
            --accent: #1f9e6a;
            --accent-dark: #0f5a3a;
            --muted: #587060;
            --text-main: #113824;
            --border-soft: rgba(23, 102, 70, 0.14);
            --card-shadow: 0 18px 44px rgba(15, 68, 48, 0.08);
            --rounded-xl: 20px;
        }

        html, body, [data-testid="stApp"] {
            background: var(--gradient-bg);
            color: var(--text-main);
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
        }

        header[data-testid="stHeader"], div[data-testid="stToolbar"], footer {
            display: none !important;
        }

        main .block-container {
            padding-top: 2rem;
            max-width: 72rem;
        }

        section[data-testid="stSidebar"] {
            background: var(--sidebar-gradient);
            color: var(--text-main);
            border-right: 1px solid rgba(17, 75, 51, 0.08);
        }

        section[data-testid="stSidebar"] > div {
            padding-left: 1.4rem;
            padding-right: 1.4rem;
        }

        .sidebar-card {
            margin-top: 1.2rem;
            padding: 1.8rem 1.4rem;
            background: rgba(255, 255, 255, 0.5);
            border-radius: var(--rounded-xl);
            border: 1px solid rgba(255, 255, 255, 0.55);
            box-shadow: 0 16px 30px rgba(26, 95, 68, 0.12);
            text-align: center;
        }

        .sidebar-avatar {
            width: 76px;
            height: 76px;
            border-radius: 50%;
            margin: 0 auto 0.9rem auto;
            display: grid;
            place-items: center;
            background: radial-gradient(circle at 30% 30%, #ffffff 0%, rgba(210, 240, 229, 0.9) 70%);
            font-size: 32px;
        }

        .sidebar-title {
            font-size: 1.32rem;
            font-weight: 800;
            letter-spacing: 0.3px;
        }

        .sidebar-subtitle {
            font-size: 0.92rem;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }

        .sidebar-caption {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            border: 1px solid rgba(17, 98, 64, 0.18);
            background: rgba(255, 255, 255, 0.7);
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--accent-dark);
        }

        .health-badge {
            margin-top: 0.9rem;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
        }

        .health-badge .dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
            background: currentColor;
        }

        .health-ok {
            color: #0f6b45;
            border: 1px solid rgba(15, 115, 72, 0.3);
            background: rgba(37, 160, 104, 0.12);
        }

        .health-ko {
            color: #b52727;
            border: 1px solid rgba(181, 39, 39, 0.28);
            background: rgba(249, 101, 101, 0.14);
        }

        .sidebar-section {
            margin-top: 1.8rem;
            margin-bottom: 0.6rem;
        }

        .sidebar-section h4 {
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }

        .sidebar-reset button {
            width: 100%;
            border-radius: 999px;
            font-weight: 600;
            border: 1px solid rgba(20, 102, 70, 0.2);
            background: rgba(255, 255, 255, 0.35);
            color: var(--accent-dark);
        }

        .sidebar-reset button:hover {
            border-color: rgba(20, 102, 70, 0.32);
            background: rgba(255, 255, 255, 0.55);
        }

        .stSlider label, .stSelectbox label {
            font-weight: 600;
            color: var(--accent-dark);
        }

        .stSlider > div > div > div[data-baseweb="slider"] {
            padding: 0.4rem 0.1rem 1.5rem 0.1rem;
        }

        .hero-card {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--border-soft);
            border-radius: 28px;
            padding: 1.8rem 2.2rem;
            box-shadow: var(--card-shadow);
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--accent);
            background: rgba(31, 158, 106, 0.14);
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
        }

        .hero-card h1 {
            margin-top: 1rem;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0.4px;
        }

        .hero-card p {
            margin-top: 0.8rem;
            font-size: 1rem;
            color: var(--muted);
        }

        [data-testid="stChatMessage"] {
            border-radius: 22px;
            padding: 1.3rem 1.4rem;
            margin-top: 1.1rem;
            border: 1px solid var(--assistant-border);
            background: var(--assistant-bg);
            box-shadow: 0 14px 32px rgba(15, 68, 48, 0.08);
        }

        [data-testid="stChatMessage"].stChatMessageUser {
            background: var(--user-bg);
            border-color: transparent;
            color: var(--user-text);
            box-shadow: 0 14px 28px rgba(50, 160, 115, 0.28);
        }

        [data-testid="stChatMessage"].stChatMessageUser .stMarkdown p,
        [data-testid="stChatMessage"].stChatMessageUser .stMarkdown li {
            color: var(--user-text);
        }

        [data-testid="stChatMessage"] .stMarkdown p {
            font-size: 1rem;
            line-height: 1.6;
        }

        .section-title {
            margin-top: 2.2rem;
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--accent-dark);
        }

        .source-card {
            margin-top: 0.9rem;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--section-border);
            border-radius: 20px;
            padding: 1rem 1.2rem;
            box-shadow: 0 10px 26px rgba(17, 68, 46, 0.1);
        }

        .source-head {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.3rem;
        }

        .source-index {
            font-weight: 700;
            font-size: 0.85rem;
            color: rgba(17, 86, 56, 0.65);
            background: rgba(31, 158, 106, 0.15);
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
        }

        .source-title {
            font-weight: 700;
            font-size: 1rem;
        }

        .source-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            font-size: 0.82rem;
            color: var(--muted);
            margin-bottom: 0.5rem;
        }

        .source-score {
            font-weight: 700;
            color: var(--accent);
        }

        .source-snippet {
            margin: 0;
            font-size: 0.92rem;
            line-height: 1.5;
            color: var(--muted);
        }

        form[data-testid="stForm"] {
            margin-top: 2.4rem;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            padding: 1.6rem 1.8rem 1.4rem 1.8rem;
            box-shadow: 0 16px 40px rgba(15, 68, 48, 0.08);
        }

        form[data-testid="stForm"] label {
            font-weight: 700;
            color: var(--accent-dark);
            font-size: 1rem;
        }

        form[data-testid="stForm"] textarea {
            border-radius: 18px !important;
            border: 1px solid rgba(31, 158, 106, 0.22) !important;
            background: rgba(255, 255, 255, 0.95) !important;
        }

        form[data-testid="stForm"] textarea:focus {
            border-color: rgba(31, 158, 106, 0.45) !important;
            box-shadow: 0 0 0 0.2rem rgba(31, 158, 106, 0.16) !important;
        }

        form[data-testid="stForm"] button[kind="primary"] {
            border-radius: 999px;
            font-weight: 700;
            padding: 0.55rem 1.1rem;
            background: var(--accent) !important;
            border: 1px solid var(--accent) !important;
        }

        form[data-testid="stForm"] button[kind="primary"]:hover {
            background: #1a8e5f !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=600)
def fetch_domains(url: str) -> List[str]:
    """Return the list of available domains exposed by the API."""
    try:
        response = requests.get(f"{url}/domains", timeout=10)
        response.raise_for_status()
        domains = response.json().get("domains") or []
        return [d for d in domains if isinstance(d, str) and d.strip()]
    except requests.RequestException:
        return []


@st.cache_data(show_spinner=False, ttl=60)
def check_health(url: str) -> bool:
    """Ping the backend health endpoint."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        response.raise_for_status()
        return bool(response.json().get("ok"))
    except requests.RequestException:
        return False


def reset_conversation() -> None:
    """Reset the chat history to the default greeting."""
    st.session_state.messages = [deepcopy(DEFAULT_GREETING)]
    st.session_state.contexts = []


def domain_label(value: str) -> str:
    """Human readable label for a domain option."""
    if value == "all":
        return "Tous domaines"
    pretty = value.replace("_", " ")
    return pretty.capitalize()


def render_source_card(ctx: Dict, index: int) -> str:
    """Build the HTML card for a retrieved context."""
    title = ctx.get("title") or ctx.get("source") or f"Document {index}"
    origin = ctx.get("source") or ctx.get("collection") or "Corpus FarmLink"
    snippet = ctx.get("text") or ctx.get("chunk") or ""
    snippet = snippet.replace("\n", " ").strip()
    if len(snippet) > 220:
        snippet = snippet[:217].rsplit(" ", 1)[0] + "…"
    score = ctx.get("score")
    if isinstance(score, (int, float)):
        score_str = f"{score:.3f}"
    else:
        score_str = "—"
    return (
        "<div class=\"source-card\">"
        f"  <div class=\"source-head\"><span class=\"source-index\">{index:02d}</span>"
        f"  <span class=\"source-title\">{escape(title)}</span></div>"
        f"  <div class=\"source-meta\"><span class=\"source-score\">score {score_str}</span>"
        f"  <span>{escape(origin)}</span></div>"
        f"  <p class=\"source-snippet\">{escape(snippet)}</p>"
        "</div>"
    )


def ensure_state() -> None:
    """Make sure session state keys exist."""
    if "messages" not in st.session_state:
        reset_conversation()
    if "contexts" not in st.session_state:
        st.session_state.contexts = []


def main() -> None:
    inject_styles()
    ensure_state()

    api_online = check_health(API_URL)

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-avatar">🌾</div>
                <div class="sidebar-title">FarmLink</div>
                <div class="sidebar-subtitle">Co-pilote agricole</div>
                <div class="sidebar-caption">Synthèse des connaissances africaines</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        badge_class = "health-ok" if api_online else "health-ko"
        badge_label = "API connectée" if api_online else "API indisponible"
        st.markdown(
            f"<div class='health-badge {badge_class}'><span class='dot'></span>{badge_label}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sidebar-section'><h4>Réglages</h4></div>", unsafe_allow_html=True)

        options = fetch_domains(API_URL)
        if "all" not in options:
            options.insert(0, "all")
        if not options:
            options = ["all"]

        stored_domain = st.session_state.get("selected_domain", options[0])
        if stored_domain not in options:
            stored_domain = options[0]
        domain = st.selectbox(
            "Collection FarmLink",
            options,
            index=options.index(stored_domain),
            format_func=domain_label,
        )
        st.session_state.selected_domain = domain

        top_k = st.slider(
            "Documents (top-k)",
            min_value=2,
            max_value=10,
            value=int(st.session_state.get("top_k_value", 4)),
        )
        st.session_state.top_k_value = top_k

        temperature = st.slider(
            "Créativité (température)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("temperature_value", 0.2)),
            step=0.05,
        )
        st.session_state.temperature_value = temperature

        st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
        if st.button("🧹 Nouvelle conversation", use_container_width=True, key="reset_chat"):
            reset_conversation()
            st.experimental_rerun()

    st.markdown(
        """
        <div class="hero-card">
            <span class="hero-chip">Assistant RAG africain</span>
            <h1>FarmLink répond à vos questions agricoles en contexte</h1>
            <p>Choisissez un domaine, ajustez la profondeur documentaire et obtenez une synthèse actionnable des ressources FarmLink.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_placeholder = st.container()
    sources_placeholder = st.container()

    with st.form("question_form", clear_on_submit=True):
        prompt = st.text_area(
            "Pose ta question",
            placeholder="Exemple : Comment améliorer l'irrigation des exploitations de cacao au Ghana ?",
            height=140,
        )
        submitted = st.form_submit_button("Interroger FarmLink", use_container_width=True)

    if submitted:
        question = (prompt or "").strip()
        if not question:
            st.warning("Merci de saisir une question avant d'envoyer.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            payload = {
                "question": question,
                "domain": st.session_state.selected_domain,
                "top_k": int(st.session_state.top_k_value),
                "temperature": float(st.session_state.temperature_value),
            }
            try:
                with st.spinner("Analyse des ressources FarmLink…"):
                    response = requests.post(
                        f"{API_URL}/query",
                        json=payload,
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()
            except requests.Timeout:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "La requête a expiré. Merci de réessayer dans quelques instants.",
                    }
                )
                st.session_state.contexts = []
            except requests.RequestException as exc:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Impossible d'interroger l'API FarmLink : {exc}.",
                    }
                )
                st.session_state.contexts = []
            except ValueError:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "La réponse du serveur n'est pas au format JSON attendu.",
                    }
                )
                st.session_state.contexts = []
            else:
                answer = data.get("answer") or "Pas de réponse disponible pour le moment."
                st.session_state.messages.append({"role": "assistant", "content": answer})
                contexts = data.get("contexts")
                st.session_state.contexts = contexts if isinstance(contexts, list) else []

    with chat_placeholder:
        for message in st.session_state.messages:
            avatar = "🌿" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    with sources_placeholder:
        if st.session_state.contexts:
            st.markdown("<div class='section-title'>Sources mobilisées</div>", unsafe_allow_html=True)
            for idx, ctx in enumerate(st.session_state.contexts, start=1):
                st.markdown(render_source_card(ctx, idx), unsafe_allow_html=True)



if __name__ == "__main__":
    main()


