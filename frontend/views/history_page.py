"""Prediction history with date filtering."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

from frontend.views.prediction_utils import inputs_to_human_readable
from frontend.ui.components import badge, empty_state
from frontend.ui.theme import page_header
from backend.database.database import get_prediction, get_predictions, get_report_for_prediction


def _parse_json_field(record: dict, field: str) -> dict | list:
    parsed = record.get(field.replace("_json", ""))
    if parsed is not None:
        return parsed
    raw = record.get(field)
    if raw:
        return json.loads(raw)
    return {} if "recommendation" in field or "ensemble" in field else []


def render_history_page(system: dict) -> None:  # noqa: ARG001
    page_header("Assessment History", "Review and reopen prior clinical evaluations")
    user = st.session_state.user

    c1, c2 = st.columns(2)
    with c1:
        date_from = st.date_input("From", value=date.today() - timedelta(days=90))
    with c2:
        date_to = st.date_input("To", value=date.today())

    preds = get_predictions(user["id"], str(date_from), str(date_to))
    st.metric("Records in range", len(preds))

    if not preds:
        empty_state(
            "No assessments found",
            "Try widening the date range or run a new Risk Assessment.",
            "Risk Assessment →",
        )
        return

    for p in preds:
        try:
            ens = json.loads(p["ensemble_json"]) if p.get("ensemble_json") else {}
        except (ValueError, TypeError):
            st.caption(f"Assessment #{p.get('id', '?')} could not be read (corrupt record) — skipped.")
            continue
        prob = ens.get("ensemble_probability", p.get("ensemble_probability") or 0)
        label = "Disease" if p.get("ensemble_prediction") == 1 else "No disease"
        tier = p.get("risk_level", "n/a").upper()
        with st.expander(f"Assessment #{p['id']} · {p['created_at'][:16]} UTC · {prob*100:.1f}% risk"):
            badge(label, "danger" if p.get("ensemble_prediction") == 1 else "success")
            badge(f"Tier: {tier}", "warning")
            ca, cb = st.columns(2)
            with ca:
                if st.button("Load into session", key=f"load_{p['id']}", use_container_width=True):
                    full = get_prediction(p["id"], user["id"])
                    if full:
                        inputs = _parse_json_field(full, "inputs_json")
                        ens_full = _parse_json_field(full, "ensemble_json")
                        st.session_state.prediction_context = {
                            "prediction": full.get("ensemble_prediction"),
                            "risk_probability": full.get("ensemble_probability"),
                            "confidence": ens_full.get("confidence", 0.5) if isinstance(ens_full, dict) else 0.5,
                            "model_name": "Ensemble",
                            "ensemble": ens_full,
                            "human_readable_inputs": inputs_to_human_readable(inputs),
                            "raw_inputs": inputs,
                            "shap_contributors": _parse_json_field(full, "shap_json"),
                            "recommendations": _parse_json_field(full, "recommendations_json"),
                            "ai_explanation": full.get("ai_explanation", ""),
                        }
                        st.session_state.last_prediction_id = p["id"]
                        st.session_state.chat_messages = []
                        st.session_state.auto_explained = False
                        st.success("Session loaded. Open Health Assistant to continue.")
                        st.rerun()
            with cb:
                report = get_report_for_prediction(p["id"], user["id"])
                if report:
                    try:
                        pdf_file = Path(report["pdf_path"])
                        if pdf_file.exists():
                            with open(pdf_file, "rb") as f:
                                st.download_button(
                                    "Download PDF",
                                    f.read(),
                                    file_name=pdf_file.name,
                                    mime="application/pdf",
                                    key=f"pdf_{p['id']}",
                                    use_container_width=True,
                                )
                        else:
                            st.caption("Report file not found.")
                    except OSError:
                        st.caption("Report file could not be opened.")
