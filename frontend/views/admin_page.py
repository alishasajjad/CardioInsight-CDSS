"""Administration — database and knowledge base maintenance."""
from __future__ import annotations

import streamlit as st

from frontend.ui.components import info_card, section_title
from frontend.ui.theme import page_header
from backend.config import DB_PATH, FAISS_INDEX_PATH, KNOWLEDGE_BASE_DIR
from backend.database.database import get_connection, init_db
from backend.rag.rag import build_vector_store, collect_documents


def render_admin_page(system: dict) -> None:  # noqa: ARG001
    page_header("Administration", "System maintenance and knowledge base tools")
    init_db()

    section_title("Database")
    info_card("SQLite database", f"Active database: `{DB_PATH.name}` — user accounts, predictions, reports, chat.", "🗄️")
    try:
        with get_connection() as conn:
            users = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
            preds = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
        c1, c2 = st.columns(2)
        c1.metric("Users", users)
        c2.metric("Predictions", preds)
    except Exception as exc:
        st.error(f"Database read error: {exc}")

    section_title("Knowledge Base")
    chunks = len(collect_documents())
    info_card(
        "RAG documents",
        f"{chunks} chunks available from `{KNOWLEDGE_BASE_DIR.name}/`. "
        "Add .md files and rebuild the index.",
        "📚",
    )

    if st.button("Rebuild vector index", type="primary", use_container_width=True):
        with st.spinner("Embedding documents — this may take a minute..."):
            try:
                result = build_vector_store(force=True)
                st.success(f"Index rebuilt: {result}")
            except Exception as exc:
                st.error(f"Rebuild failed: {exc}")

    if FAISS_INDEX_PATH.exists():
        st.caption(f"FAISS index: {FAISS_INDEX_PATH.name}")
