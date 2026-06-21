"""Session state initialization."""
from __future__ import annotations

import streamlit as st


def init_session() -> None:
    defaults = {
        "auth_token": None,
        "user": None,
        "prediction_context": None,
        "chat_messages": [],
        "auto_explained": False,
        "last_prediction_id": None,
        "last_ensemble": None,
        "last_pdf_path": None,
        "groq_model": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
