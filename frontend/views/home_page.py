"""Dashboard home page."""
from __future__ import annotations

import streamlit as st

from frontend.ui.components import info_card, stat_tile, welcome_banner
from frontend.ui.theme import disclaimer_box, page_header
from backend.database.database import get_user_prediction_count


def render_home_page(system: dict) -> None:
    meta = system.get("metadata", {})
    user = st.session_state.user
    best = meta.get("best_metrics", {})

    welcome_banner(
        "CardioInsight Clinical Decision Support",
        "AI-powered heart disease risk assessment with explainability and clinical guidance",
        user["username"],
    )
    disclaimer_box()

    count = get_user_prediction_count(user["id"])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_tile("Your assessments", str(count), "Saved to history")
    with c2:
        stat_tile("Deploy model", meta.get("deployment_model", "XGBoost"), "Production default")
    with c3:
        stat_tile("ROC-AUC", f"{best.get('roc_auc', 0):.3f}", "Hold-out benchmark")
    with c4:
        stat_tile("Accuracy", f"{best.get('accuracy', 0):.1%}", "Clinical cohort")

    st.markdown("<br>", unsafe_allow_html=True)
    page_header("Quick Actions", "Navigate to core clinical workflows")

    a1, a2, a3 = st.columns(3)
    with a1:
        info_card(
            "Risk Assessment",
            "Run ensemble ML analysis with SHAP explainability, clinical recommendations, and PDF report generation.",
            "🩺",
        )
        if st.button("Start assessment", key="go_assess", use_container_width=True):
            st.session_state["_nav"] = "Risk Assessment"
            st.rerun()
    with a2:
        info_card(
            "Health Assistant",
            "Ask questions grounded in medical knowledge retrieval and your latest assessment context.",
            "💬",
        )
        if st.button("Open assistant", key="go_ai", use_container_width=True):
            st.session_state["_nav"] = "Health Assistant"
            st.rerun()
    with a3:
        info_card(
            "Assessment History",
            "Review past evaluations, reload sessions, and download PDF medical reports.",
            "📋",
        )
        if st.button("View history", key="go_hist", use_container_width=True):
            st.session_state["_nav"] = "History"
            st.rerun()

    if st.session_state.get("prediction_context"):
        ctx = st.session_state.prediction_context
        st.success(
            f"Active session: {ctx.get('risk_probability', 0)*100:.1f}% risk — "
            "open Health Assistant for follow-up questions."
        )
