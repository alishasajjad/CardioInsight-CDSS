"""Professional About page for CardioInsight CDSS."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from frontend.ui.components import info_card, section_title
from frontend.ui.theme import disclaimer_box, page_header
from backend.config import KNOWLEDGE_BASE_DIR, PROJECT_ROOT
from backend.database.database import get_connection


def _file_mtime(path: Path) -> str:
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%B %d, %Y")
    return "N/A"


def render_about_page(system: dict) -> None:
    meta = system.get("metadata", {})
    page_header("About CardioInsight", "Clinical Decision Support System — product information")
    disclaimer_box()

    section_title("About the System")
    info_card(
        "CardioInsight CDSS",
        "CardioInsight is an AI-powered Clinical Decision Support System for educational heart disease "
        "risk assessment. It combines ensemble machine learning, SHAP explainability, rule-based clinical "
        "recommendations, PDF reporting, secure user history, and a RAG-powered health assistant.",
        "🏥",
    )

    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "Mission Statement",
            "To demonstrate how responsible AI can support cardiovascular risk awareness through "
            "transparent models, explainable predictions, and educational clinical guidance — never "
            "replacing qualified medical professionals.",
            "🎯",
        )
        info_card(
            "Clinical Decision Support Purpose",
            "Provide structured risk stratification, feature attribution, and evidence-informed "
            "recommendations to help users understand cardiovascular risk factors in an educational context.",
            "📊",
        )
    with c2:
        info_card(
            "Project Overview",
            "Built on four clinically aligned angiographic datasets with Random Forest, XGBoost, and "
            "ANN models. Ensemble predictions use ROC-AUC–weighted averaging with XGBoost as the default "
            "deployment model.",
            "📁",
        )
        info_card(
            "Development Status",
            "Production-ready educational demonstration. Suitable for academic presentation, project "
            "exhibition, and portfolio showcase.",
            "✅",
        )

    section_title("AI & Machine Learning")
    t1, t2, t3 = st.columns(3)
    with t1:
        info_card("ML Models", "Random Forest · XGBoost (deployed) · Artificial Neural Network (MLP)", "🤖")
    with t2:
        info_card("Explainable AI", "SHAP TreeExplainer for per-patient feature attribution on deployment model", "🔍")
    with t3:
        info_card("RAG Assistant", "FAISS vector search + Groq LLM with source-grounded medical knowledge", "💡")

    section_title("Version Information")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.metric("Build version", "1.0.0")
    with v2:
        st.metric("Release date", "June 2025")
    with v3:
        st.metric("Last modified", _file_mtime(PROJECT_ROOT / "app" / "streamlit_app.py"))
    with v4:
        st.metric("Deploy model", meta.get("deployment_model", "XGBoost"))

    section_title("System Statistics")
    try:
        with get_connection() as conn:
            users = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
            preds = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Registered users", users)
        with s2:
            st.metric("Total assessments", preds)
        with s3:
            kb_docs = len(list(KNOWLEDGE_BASE_DIR.glob("*.md"))) if KNOWLEDGE_BASE_DIR.exists() else 0
            st.metric("Knowledge documents", kb_docs)
    except Exception:
        st.caption("Statistics unavailable.")

    section_title("Author & Contact")
    info_card(
        "Development Team",
        "CardioInsight CDSS — Academic / Portfolio Project. "
        "Contact: alisha@gmail.com.",
        "👥",
    )

    section_title("Legal")
    info_card(
        "Disclaimer",
        "This system is for educational purposes only. It is not a medical device and does not provide "
        "diagnosis, treatment, or emergency care. Always consult qualified healthcare professionals.",
        "⚠️",
    )
