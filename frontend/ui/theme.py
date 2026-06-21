"""Healthcare UI theme for CardioInsight CDSS — Refined Clinical Blue.

Light, high-contrast design system. Pinned light base lives in
``.streamlit/config.toml``; this module layers the brand styling, larger type,
and accessible widget contrast on top.
"""
from __future__ import annotations

import streamlit as st

COLORS = {
    "primary": "#0B6E99",        # marine blue — primary actions
    "primary_dark": "#0F2A43",   # deep navy — headings
    "accent": "#B91C1C",         # red — danger / high risk
    "success": "#15803D",        # green — low risk / healthy
    "warning": "#B45309",        # amber — caution
    "bg": "#F1F5F9",             # page canvas (slate-100)
    "card": "#FFFFFF",           # surface
    "text": "#1E293B",           # body text (slate-800, ~13:1 on white)
    "muted": "#475569",          # secondary text (slate-600, ~7:1 on white)
    "border": "#D8E0EA",         # hairline borders
    "sidebar": "#0A2540",        # dark navy sidebar
    "gradient_start": "#0B6E99",
    "gradient_end": "#0F2A43",
}

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: {COLORS['text']};
}}
html, body {{ font-size: 16px; }}
/* Force light rendering of native form controls even if the browser/OS is dark. */
:root, html, body, .stApp, section.main, [data-testid="stMain"] {{ color-scheme: light; }}

/* App canvas + main-content text (robust even if base theme is overridden) */
[data-testid="stAppViewContainer"], .stApp {{
    background: {COLORS['bg']};
}}
[data-testid="stHeader"] {{ background: transparent; }}
section.main, [data-testid="stMain"] {{ color: {COLORS['text']}; font-size: 1rem; line-height: 1.6; }}
section.main p, section.main li, [data-testid="stMain"] p, [data-testid="stMain"] li {{
    color: {COLORS['text']};
}}

.main .block-container {{
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    max-width: 1180px;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

.hero-title {{
    font-size: 2rem;
    font-weight: 700;
    color: {COLORS['primary_dark']};
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
}}
.hero-subtitle {{
    color: {COLORS['muted']};
    font-size: 1.02rem;
    margin: 0 0 1.25rem 0;
    line-height: 1.55;
}}

.ci-banner {{
    background: linear-gradient(135deg, {COLORS['gradient_start']} 0%, {COLORS['gradient_end']} 100%);
    color: white;
    padding: 1.75rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(11, 110, 153, 0.25);
}}
.ci-banner-title {{ font-size: 1.7rem; font-weight: 700; margin-bottom: 0.35rem; }}
.ci-banner-sub {{ opacity: 0.95; font-size: 1rem; }}
.ci-banner-user {{ margin-top: 0.75rem; font-size: 0.9rem; opacity: 0.9; }}

.ci-card {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(15, 42, 67, 0.06);
    transition: box-shadow 0.2s ease;
}}
.ci-card:hover {{ box-shadow: 0 4px 16px rgba(15, 42, 67, 0.10); }}
.ci-card-icon {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
.ci-card-title {{ font-weight: 600; font-size: 1.05rem; color: {COLORS['primary_dark']}; margin-bottom: 0.4rem; }}
.ci-card-body {{ color: {COLORS['muted']}; font-size: 0.95rem; line-height: 1.6; }}

.ci-stat {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(15, 42, 67, 0.05);
}}
.ci-stat-label {{ font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em; color: {COLORS['muted']}; font-weight: 700; }}
.ci-stat-value {{ font-size: 1.6rem; font-weight: 700; color: {COLORS['primary_dark']}; margin: 0.25rem 0; }}
.ci-stat-hint {{ font-size: 0.8rem; color: {COLORS['muted']}; }}

.ci-badge {{
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.35rem;
}}

.ci-section-title {{
    font-size: 1.15rem;
    font-weight: 600;
    color: {COLORS['primary_dark']};
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid {COLORS['border']};
}}

.ci-empty {{
    background: #F8FAFC;
    border: 1px dashed #B7C3D2;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}}
.ci-empty-title {{ font-weight: 600; color: {COLORS['text']}; margin-bottom: 0.5rem; font-size: 1.02rem; }}
.ci-empty-msg {{ color: {COLORS['muted']}; font-size: 0.95rem; }}
.ci-empty-action {{ color: {COLORS['primary']}; font-size: 0.9rem; margin-top: 0.5rem; }}

.disclaimer-banner {{
    background: linear-gradient(90deg, #FFFBEB 0%, #FEF3C7 100%);
    border-left: 4px solid {COLORS['warning']};
    padding: 0.9rem 1.1rem;
    border-radius: 10px;
    font-size: 0.9rem;
    color: #7C2D12;
    margin: 0.75rem 0 1.25rem 0;
    line-height: 1.5;
}}

/* ── Sidebar (intentionally dark navy with light text) ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {COLORS['sidebar']} 0%, #051525 100%);
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] label {{
    color: #E8EEF3 !important;
}}
section[data-testid="stSidebar"] .stMetric label {{ color: #9FB2C6 !important; }}
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {{ color: white !important; }}

/* ── Forms & widget labels: high-contrast on light surfaces ── */
div[data-testid="stForm"] {{
    background: {COLORS['card']};
    padding: 1.25rem 1.35rem;
    border-radius: 14px;
    border: 1px solid {COLORS['border']};
    box-shadow: 0 2px 8px rgba(15, 42, 67, 0.05);
}}
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stRadio"] > label,
[data-testid="stDateInput"] label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {{
    color: {COLORS['text']} !important;
    font-weight: 600;
    font-size: 0.92rem;
    opacity: 1;
}}

/* Text / password / number / select inputs (robust across Streamlit versions) */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
[data-testid="stTextInputRootElement"],
div[data-baseweb="select"] > div {{
    background: #FFFFFF !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important;
}}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within {{
    border-color: {COLORS['primary']} !important;
    box-shadow: 0 0 0 3px rgba(11, 110, 153, 0.18) !important;
}}
.stTextInput input, .stNumberInput input, .stDateInput input,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {{
    color: {COLORS['text']} !important;
    -webkit-text-fill-color: {COLORS['text']} !important;
    background-color: #FFFFFF !important;
}}
[data-baseweb="select"] div {{ color: {COLORS['text']} !important; }}
.main input::placeholder {{ color: #94A3B8 !important; opacity: 1; }}

/* Tabs */
[data-baseweb="tab-list"] {{ gap: 0.5rem; border-bottom: 1px solid {COLORS['border']}; }}
[data-baseweb="tab"] {{ color: {COLORS['muted']}; font-weight: 600; }}
[data-baseweb="tab"][aria-selected="true"] {{ color: {COLORS['primary']}; }}
[data-baseweb="tab-highlight"] {{ background-color: {COLORS['primary']} !important; }}

/* Buttons */
.stButton > button {{ border-radius: 10px; font-weight: 600; }}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
    border: none;
    color: #FFFFFF;
}}
.stButton > button[kind="secondary"] {{
    background: #FFFFFF;
    border: 1px solid {COLORS['border']};
    color: {COLORS['primary_dark']};
}}
.stButton > button:focus-visible {{
    outline: 3px solid rgba(11, 110, 153, 0.35);
    outline-offset: 2px;
}}

/* Form-submit + download buttons (Sign In, Register, Run Assessment, Download) */
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {{
    background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px;
    font-weight: 600;
}}
[data-testid="stFormSubmitButton"] button p,
[data-testid="stDownloadButton"] button p {{ color: #FFFFFF !important; }}
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {{ filter: brightness(1.06); }}

/* Sidebar Sign Out button: readable on the dark navy sidebar */
section[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255, 255, 255, 0.10) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.30) !important;
    border-radius: 10px;
    font-weight: 600;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255, 255, 255, 0.18) !important;
}}

/* Metric + progress accents */
[data-testid="stMetricValue"] {{ color: {COLORS['primary_dark']}; }}
.stProgress > div > div > div > div {{ background-color: {COLORS['primary']}; }}

/* ════ Streamlit 1.58 widget overrides — force LIGHT regardless of theme/browser ════ */
/* Text / number / textarea fields */
[data-testid="stTextInputRootElement"],
[data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stNumberInputContainer"],
[data-testid="stTextArea"] [data-baseweb="base-input"] {{
    background-color: #FFFFFF !important;
    border: 1px solid {COLORS['border']} !important;
}}
[data-testid="stTextInput"] input,
[data-testid="stNumberInputField"],
[data-testid="stTextArea"] textarea {{
    background-color: #FFFFFF !important;
    color: {COLORS['text']} !important;
    -webkit-text-fill-color: {COLORS['text']} !important;
}}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stNumberInputField"]::placeholder,
[data-testid="stTextArea"] textarea::placeholder {{
    color: #94A3B8 !important; -webkit-text-fill-color: #94A3B8 !important;
}}

/* Chat input bar + messages */
[data-testid="stChatInput"] {{
    background-color: #FFFFFF !important;
    border: 1px solid {COLORS['border']} !important;
}}
[data-testid="stChatInputTextArea"] {{
    background-color: #FFFFFF !important;
    color: {COLORS['text']} !important;
    -webkit-text-fill-color: {COLORS['text']} !important;
}}
[data-testid="stChatInputTextArea"]::placeholder {{ color: #94A3B8 !important; -webkit-text-fill-color: #94A3B8 !important; }}
[data-testid="stChatMessage"] {{ background-color: #FFFFFF !important; border: 1px solid {COLORS['border']} !important; }}
[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] * {{ color: {COLORS['text']} !important; }}

/* Bottom bar that holds the chat input — match the page canvas (kill the dark band) */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div {{
    background-color: {COLORS['bg']} !important;
}}
/* Chat send (arrow) button → solid blue with a small, centered white arrow */
[data-testid="stChatInputSubmitButton"] {{
    background-color: {COLORS['primary']} !important;
    border: none !important;
    border-radius: 8px !important;
}}
[data-testid="stChatInputSubmitButton"]:hover {{ filter: brightness(1.08); }}
[data-testid="stChatInputSubmitButton"]:disabled {{ opacity: 0.45 !important; }}
[data-testid="stChatInputSubmitButton"] svg {{
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
    width: 18px !important;
    height: 18px !important;
}}

/* Buttons (1.58 uses stBaseButton-* testids, NOT kind= attributes) */
[data-testid^="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {{
    background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
[data-testid^="stBaseButton-primary"] *,
[data-testid="stFormSubmitButton"] button *,
[data-testid="stDownloadButton"] button * {{ color: #FFFFFF !important; }}
[data-testid^="stBaseButton-secondary"] {{
    background: #FFFFFF !important;
    color: {COLORS['primary_dark']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
[data-testid^="stBaseButton-secondary"] * {{ color: {COLORS['primary_dark']} !important; }}

/* Sidebar buttons (Sign Out) — readable on the dark navy sidebar */
section[data-testid="stSidebar"] [data-testid^="stBaseButton"] {{
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.32) !important;
}}
section[data-testid="stSidebar"] [data-testid^="stBaseButton"],
section[data-testid="stSidebar"] [data-testid^="stBaseButton"] * {{ color: #FFFFFF !important; }}
</style>
"""


def inject_theme() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<p class="hero-title">{title}</p>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="hero-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def disclaimer_box(text: str | None = None) -> None:
    from backend.config import MEDICAL_DISCLAIMER
    st.markdown(f'<div class="disclaimer-banner">{text or MEDICAL_DISCLAIMER}</div>', unsafe_allow_html=True)
