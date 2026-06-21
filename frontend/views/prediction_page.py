"""Risk assessment — ensemble prediction, SHAP, recommendations, PDF."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from frontend.views.prediction_utils import (
    CP_LABELS, ECG_LABELS, SLOPE_LABELS, THAL_LABELS,
    build_features, build_prediction_context, inputs_to_human_readable, shap_for_instance,
)
from frontend.ui.components import badge, section_title, stat_tile
from frontend.ui.theme import disclaimer_box, page_header
from backend.config import FEATURE_COLUMNS
from backend.database.database import save_prediction, save_report
from backend.llm.groq_assistant import get_api_key
from backend.logging_config import get_logger
from backend.reporting.pdf_report import generate_pdf_report
from backend.rag.rag import rag_initial_explanation
from backend.recommendations.recommendations import generate_recommendations
from backend.ensemble.ensemble import ensemble_predict

logger = get_logger(__name__)


def render_prediction_page(system: dict) -> None:
    page_header("Risk Assessment", "Ensemble analysis with explainability and clinical guidance")
    disclaimer_box()

    section_title("Patient Clinical Profile")

    with st.form("clinical_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age (years)", 29, 77, 54)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            chest_pain_type = st.selectbox("Chest pain", list(range(4)), format_func=lambda i: CP_LABELS[i])
            resting_bp = st.slider("Resting BP (mmHg)", 94, 200, 128, help="Resting systolic blood pressure in mmHg.")
            cholesterol = st.slider("Cholesterol (mg/dL)", 126, 564, 246, help="Serum cholesterol in mg/dL.")
        with c2:
            fasting_blood_sugar = st.selectbox("Fasting BS > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            resting_ecg = st.selectbox("Resting ECG", list(range(3)), format_func=lambda i: ECG_LABELS[i])
            max_heart_rate = st.slider("Max heart rate", 71, 202, 149, help="Maximum heart rate achieved (bpm).")
            exercise_angina = st.selectbox("Exercise angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.2, 1.0, 0.1, help="ST-segment depression induced by exercise (mm).")
        with c3:
            st_slope = st.selectbox("ST slope", list(range(3)), format_func=lambda i: SLOPE_LABELS[i], help="Slope of the peak exercise ST segment.")
            major_vessels = st.slider("Major vessels (0–3)", 0, 3, 0, help="Number of major vessels coloured by fluoroscopy.")
            thalassemia = st.selectbox("Thalassemia", list(range(3)), format_func=lambda i: THAL_LABELS[i])

        submitted = st.form_submit_button("Run Assessment", type="primary", use_container_width=True)

    inputs = {
        "age": age, "sex": sex, "chest_pain_type": chest_pain_type,
        "resting_bp": resting_bp, "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar, "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate, "exercise_angina": exercise_angina,
        "oldpeak": oldpeak, "st_slope": st_slope,
        "major_vessels": major_vessels, "thalassemia": thalassemia,
    }

    if not submitted:
        return

    pdf_path = None
    try:
        with st.spinner("Running ensemble models and building your report..."):
            X = build_features(inputs)
            ensemble = ensemble_predict(X, system)
            deploy_name = system.get("metadata", {}).get("deployment_model", "XGBoost")
            deploy_model = system["models"].get(deploy_name) or system.get("deploy_model")
            shap_result = shap_for_instance(deploy_model, deploy_name, X) if deploy_model else None
            shap_vals = shap_result[0] if shap_result else None
            shap_list = build_prediction_context(inputs, ensemble, shap_vals)["shap_contributors"]
            rec = generate_recommendations(ensemble["ensemble_probability"], inputs, shap_list)

            ai_text = ""
            if get_api_key():
                try:
                    ctx_tmp = build_prediction_context(inputs, ensemble, shap_vals, rec)
                    ai_text, _ = rag_initial_explanation(ctx_tmp)
                except Exception as exc:
                    logger.warning("AI narrative generation failed: %s", exc)
                    ai_text = "AI narrative unavailable — check your Groq API key or try again."
            else:
                ai_text = "Configure GROQ_API_KEY for AI-generated narrative."

            ctx = build_prediction_context(inputs, ensemble, shap_vals, rec, ai_text)
            st.session_state.prediction_context = ctx
            st.session_state.last_ensemble = ensemble
            st.session_state.chat_messages = []
            st.session_state.auto_explained = False

            user = st.session_state.user
            pid = save_prediction(user["id"], inputs, ensemble, shap_list, rec, ai_text, rec["risk_level"])
            st.session_state.last_prediction_id = pid

            # PDF + report are best-effort: a failure here must not lose the saved prediction.
            try:
                pdf_path = generate_pdf_report(
                    f"PRED-{pid}", inputs_to_human_readable(inputs), ensemble,
                    shap_list, rec, ai_text, user["username"],
                )
                save_report(pid, user["id"], str(pdf_path))
                st.session_state.last_pdf_path = str(pdf_path)
            except Exception as pdf_exc:
                logger.warning("PDF/report generation failed for prediction %s: %s", pid, pdf_exc)
                st.warning("Assessment saved, but the PDF report could not be generated.")
    except Exception:
        logger.exception("Risk assessment failed")
        st.error("Sorry — the assessment could not be completed. Please try again.")
        st.stop()

    _display_results(ensemble, rec, shap_result, deploy_name, pdf_path)


def _display_results(ensemble, rec, shap_result, deploy_name, pdf_path) -> None:
    ep, pred = ensemble["ensemble_probability"], ensemble["ensemble_prediction"]
    section_title("Assessment Results")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        stat_tile("Result", "Disease indicated" if pred == 1 else "No disease indicated", "Ensemble output")
    with m2:
        stat_tile("Ensemble risk", f"{ep * 100:.1f}%", "Disease probability")
    with m3:
        stat_tile("Confidence", f"{ensemble['confidence'] * 100:.1f}%", "Model agreement")
    with m4:
        stat_tile("Risk tier", rec["tier"].upper(), rec["risk_level"])
    badge("High priority" if pred == 1 else "Low risk", "danger" if pred == 1 else "success")
    st.progress(min(ep, 1.0), text="Estimated disease probability")

    tab_models, tab_rec, tab_shap = st.tabs(["Model comparison", "Recommendations", "Explainability"])

    with tab_models:
        ind = ensemble["individual"]
        df_probs = pd.DataFrame([
            {"Model": k, "Probability": f"{v['probability']*100:.1f}%", "Weight": v["weight"]}
            for k, v in ind.items()
        ])
        st.dataframe(df_probs, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(list(ind.keys()), [v["probability"] for v in ind.values()], color=["#0B6E99", "#B91C1C", "#6C3483"])
        ax.axhline(0.5, linestyle="--", color="#475569", alpha=0.7)
        ax.axhline(ep, color="#15803D", label=f"Ensemble {ep:.2f}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with tab_rec:
        st.info(rec["summary"])
        for action in rec["actions"]:
            st.markdown(f"- {action}")
        with st.expander("Suggested clinical follow-up"):
            for fu in rec.get("clinical_follow_up", []):
                st.markdown(f"- {fu}")

    with tab_shap:
        if shap_result:
            sv, base = shap_result
            contrib = pd.DataFrame({"Feature": FEATURE_COLUMNS, "SHAP": sv}).sort_values(
                "SHAP", key=abs, ascending=True
            ).tail(10)
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.barh(contrib["Feature"], contrib["SHAP"], color=["#B91C1C" if v > 0 else "#15803D" for v in contrib["SHAP"]])
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.set_title(f"SHAP — {deploy_name} (base={base:.3f})")
            st.pyplot(fig2)
            plt.close()
        else:
            st.caption("SHAP unavailable for this model type.")

    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF report", f, file_name=Path(pdf_path).name, mime="application/pdf", use_container_width=True)
    st.success("Assessment saved. Open Health Assistant for guided follow-up questions.")
