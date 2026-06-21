"""Model performance analytics."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from frontend.ui.components import section_title
from frontend.ui.theme import page_header
from backend.config import FIGURES_DIR
from backend.ensemble.ensemble import compute_ensemble_weights


def render_analytics_page(system: dict) -> None:
    page_header("Model Analytics", "Training benchmarks and ensemble configuration")
    meta = system.get("metadata", {})
    metrics = meta.get("metrics", [])

    if metrics:
        section_title("Hold-out Test Performance")
        df = pd.DataFrame(metrics).set_index("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        st.dataframe(df.style.format("{:.3f}").highlight_max(axis=0), use_container_width=True)

    weights = compute_ensemble_weights(meta)
    wdf = pd.DataFrame({"Model": list(weights.keys()), "Weight": list(weights.values())})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(wdf["Model"], wdf["Weight"], color=["#0B5FA5", "#E74C3C", "#6C3483"])
    ax.set_title("Ensemble weights (ROC-AUC normalized)")
    ax.set_ylabel("Weight")
    st.pyplot(fig)
    plt.close()

    st.caption(f"Deployment model: **{meta.get('deployment_model', 'XGBoost')}**")

    section_title("Training Visualizations")
    for title, fname in [("ROC curves", "roc_curves.png"), ("Confusion matrices", "confusion_matrices.png"), ("SHAP summary", "shap_beeswarm.png")]:
        path = FIGURES_DIR / fname
        if path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path), use_container_width=True)
