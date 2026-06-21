"""RAG-powered health assistant."""
from __future__ import annotations

import streamlit as st

from frontend.ui.components import empty_state
from frontend.ui.theme import disclaimer_box, page_header
from backend.database.database import get_chat_history, save_chat_message
from backend.llm.groq_assistant import DEFAULT_MODEL, GROQ_MODELS, get_api_key
from backend.rag.rag import rag_chat, rag_initial_explanation


def render_ai_assistant_page(system: dict) -> None:  # noqa: ARG001
    page_header("Health Assistant", "Knowledge-grounded answers powered by Groq and medical retrieval")
    disclaimer_box()

    if st.session_state.groq_model is None:
        st.session_state.groq_model = DEFAULT_MODEL

    ctx = st.session_state.get("prediction_context")
    if ctx is None:
        empty_state(
            "No assessment loaded",
            "Complete a Risk Assessment or load a record from History to enable personalized guidance.",
            "Go to Risk Assessment →",
        )
        return

    with st.expander("Assistant settings", expanded=not get_api_key()):
        key_in = st.text_input(
            "Groq API key", value=get_api_key() or "", type="password",
            help="Kept only for your current session — never written to disk or shared with other users.",
        )
        if key_in:
            # Per-session only (do NOT write to os.environ — that would leak across all sessions).
            st.session_state["groq_api_key"] = key_in
        st.session_state.groq_model = st.selectbox(
            "Model", list(GROQ_MODELS.keys()), format_func=lambda k: GROQ_MODELS[k],
            index=list(GROQ_MODELS.keys()).index(st.session_state.groq_model)
            if st.session_state.groq_model in GROQ_MODELS else 0,
        )

    if not get_api_key():
        st.warning("Enter your Groq API key above to enable the assistant.")
        return

    user = st.session_state.user
    pid = st.session_state.get("last_prediction_id")

    if not st.session_state.chat_messages and pid:
        st.session_state.chat_messages = get_chat_history(user["id"], pid)

    if not st.session_state.get("auto_explained"):
        with st.spinner("Preparing personalized insights..."):
            try:
                reply, sources = rag_initial_explanation(ctx, model=st.session_state.groq_model)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply, "sources": sources})
                save_chat_message(user["id"], "assistant", reply, pid, sources)
            except Exception:
                st.error("Could not generate an initial explanation. You can still ask questions below.")
            finally:
                # Mark as attempted regardless, to avoid re-running on every rerun (retry storm).
                st.session_state.auto_explained = True

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.auto_explained = False
            st.rerun()

    if prompt := st.chat_input("Ask about your results or heart health..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        save_chat_message(user["id"], "user", prompt, pid)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages[:-1]]
        with st.spinner("Searching knowledge base..."):
            try:
                reply, sources = rag_chat(prompt, ctx, history, model=st.session_state.groq_model)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply, "sources": sources})
                save_chat_message(user["id"], "assistant", reply, pid, sources)
            except Exception:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": "I could not complete that request right now. Please try again.",
                    "sources": [],
                })
        st.rerun()
