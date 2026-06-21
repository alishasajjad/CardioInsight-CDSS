"""
CardioInsight CDSS — Streamlit Application
Run: streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from frontend.views.about_page import render_about_page
from frontend.views.admin_page import render_admin_page
from frontend.views.ai_assistant_page import render_ai_assistant_page
from frontend.views.analytics_page import render_analytics_page
from frontend.views.auth_page import logout_button, render_auth_gate
from frontend.views.history_page import render_history_page
from frontend.views.home_page import render_home_page
from frontend.views.prediction_page import render_prediction_page
from frontend.ui.session import init_session
from frontend.ui.theme import inject_theme
from backend.database.database import init_db
from backend.ensemble.ensemble import load_all_models
from backend.logging_config import setup_logging

setup_logging()

NAV_ITEMS = [
    "Home",
    "Risk Assessment",
    "Health Assistant",
    "History",
    "Analytics",
    "Admin",
    "About",
]

PAGE_RENDERERS = {
    "Home": render_home_page,
    "Risk Assessment": render_prediction_page,
    "Health Assistant": render_ai_assistant_page,
    "History": render_history_page,
    "Analytics": render_analytics_page,
    "Admin": render_admin_page,
    "About": render_about_page,
}

st.set_page_config(
    page_title="CardioInsight CDSS",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading clinical models...")
def load_system():
    init_db()
    return load_all_models()


def main() -> None:
    inject_theme()
    init_session()

    if not render_auth_gate():
        return

    system = load_system()
    if not system.get("models"):
        st.error("Models not found. Run `python machine_learning/run_pipeline.py` first.")
        st.stop()

    meta = system.get("metadata", {})
    user = st.session_state.user

    # Sidebar
    st.sidebar.markdown("## CardioInsight")
    st.sidebar.caption(f"Signed in as **{user['username']}**")

    default_nav = st.session_state.pop("_nav", "Home")
    default_idx = NAV_ITEMS.index(default_nav) if default_nav in NAV_ITEMS else 0
    nav = st.sidebar.radio("Menu", NAV_ITEMS, index=default_idx, label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.metric("Deploy model", meta.get("deployment_model", "XGBoost"))
    st.sidebar.metric("ROC-AUC", f"{meta.get('best_metrics', {}).get('roc_auc', 0):.3f}")
    if st.session_state.get("prediction_context"):
        st.sidebar.success("Patient context active")
    logout_button()

    PAGE_RENDERERS[nav](system)


if __name__ == "__main__":
    main()
