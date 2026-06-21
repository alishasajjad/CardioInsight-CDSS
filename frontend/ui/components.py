"""Reusable UI components for CardioInsight CDSS."""
from __future__ import annotations

import streamlit as st

from frontend.ui.theme import COLORS


def info_card(title: str, body: str, icon: str = "ℹ️") -> None:
    st.markdown(
        f"""
        <div class="ci-card">
            <div class="ci-card-icon">{icon}</div>
            <div class="ci-card-title">{title}</div>
            <div class="ci-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_tile(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""
        <div class="ci-stat">
            <div class="ci-stat-label">{label}</div>
            <div class="ci-stat-value">{value}</div>
            <div class="ci-stat-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "info") -> None:
    colors = {
        "info": COLORS["primary"],
        "success": COLORS["success"],
        "warning": COLORS["warning"],
        "danger": COLORS["accent"],
    }
    color = colors.get(kind, COLORS["primary"])
    st.markdown(
        f'<span class="ci-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">{text}</span>',
        unsafe_allow_html=True,
    )


def welcome_banner(title: str, subtitle: str, user: str) -> None:
    st.markdown(
        f"""
        <div class="ci-banner">
            <div class="ci-banner-title">{title}</div>
            <div class="ci-banner-sub">{subtitle}</div>
            <div class="ci-banner-user">Welcome back, <strong>{user}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, message: str, action: str = "") -> None:
    st.markdown(
        f"""
        <div class="ci-empty">
            <div class="ci-empty-title">{title}</div>
            <div class="ci-empty-msg">{message}</div>
            <div class="ci-empty-action">{action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="ci-section-title">{text}</div>', unsafe_allow_html=True)
