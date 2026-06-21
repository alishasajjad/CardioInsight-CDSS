"""Authentication — login, register, logout."""
from __future__ import annotations

import streamlit as st

from frontend.ui.components import info_card
from frontend.ui.theme import disclaimer_box, inject_theme, page_header
from backend.auth.auth import login_user, logout_user, register_user, resolve_session
from backend.database.database import init_db


def render_auth_gate() -> bool:
    """Return True when user is authenticated."""
    inject_theme()
    init_db()

    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None
    if "user" not in st.session_state:
        st.session_state.user = None

    user = resolve_session(st.session_state.auth_token)
    if user:
        st.session_state.user = user
        return True

    # Unauthenticated: hide the sidebar for a clean, centered sign-in screen.
    st.markdown(
        "<style>section[data-testid='stSidebar'],[data-testid='collapsedControl']"
        "{display:none!important;}</style>",
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        page_header("CardioInsight CDSS", "Clinical Decision Support — secure sign in")
        info_card(
            "Secure clinical workspace",
            "Sign in to run risk assessments, view history, download PDF reports, and use the AI health assistant.",
            "🔐",
        )
        disclaimer_box()
        tab_login, tab_reg = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                    token, user, err = login_user(username, password)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.auth_token = token
                        st.session_state.user = user
                        st.rerun()

        with tab_reg:
            with st.form("register"):
                new_user = st.text_input("Username", key="reg_u", placeholder="Choose a username (min 3 chars)")
                email = st.text_input("Email", key="reg_e", placeholder="you@example.com")
                pw = st.text_input("Password", type="password", key="reg_p", placeholder="Min 6 characters", help="At least 6 characters.")
                pw2 = st.text_input("Confirm password", type="password", key="reg_p2", placeholder="Re-enter password")
                if st.form_submit_button("Register", use_container_width=True):
                    if pw != pw2:
                        st.error("Passwords do not match.")
                    else:
                        _, err = register_user(new_user, email, pw)
                        if err:
                            st.error(err)
                        else:
                            st.success("Account created. Please sign in.")

    return False


def logout_button() -> None:
    if st.sidebar.button("Sign Out", use_container_width=True):
        logout_user(st.session_state.get("auth_token"))
        for key in (
            "auth_token", "user", "prediction_context", "chat_messages",
            "auto_explained", "last_prediction_id", "last_ensemble", "last_pdf_path", "_nav",
        ):
            st.session_state.pop(key, None)
        st.session_state.chat_messages = []
        st.rerun()
