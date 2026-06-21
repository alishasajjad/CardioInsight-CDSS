"""Exploratory data analysis for the clinical cohort."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from machine_learning.config import FEATURE_COLUMNS, FIGURES_DIR, METRICS_DIR, UNIFIED_DATASET_PATH

sns.set_theme(style="whitegrid")


def run_eda() -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(UNIFIED_DATASET_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df["target"].value_counts().plot(kind="bar", ax=axes[0], color=["#27ae60", "#c0392b"])
    axes[0].set_xticklabels(["No Disease", "Disease"], rotation=0)
    axes[0].set_title("Class Distribution")
    df.groupby("source_dataset")["target"].mean().sort_values().plot(kind="barh", ax=axes[1], color="#2980b9")
    axes[1].set_title("Disease Rate by Source")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_overview.png", bbox_inches="tight")
    plt.close()

    cols = ["age", "resting_bp", "cholesterol", "max_heart_rate", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df, x=col, hue="target", kde=True, ax=ax, stat="density", common_norm=False)
    plt.suptitle("Clinical Feature Distributions")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_distributions.png", bbox_inches="tight")
    plt.close()

    corr_cols = [c for c in FEATURE_COLUMNS if c in df.columns] + ["target"]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlation.png", bbox_inches="tight")
    plt.close()

    insights = {
        "total_samples": int(len(df)),
        "disease_rate": float(df["target"].mean()),
        "sources": df["source_dataset"].value_counts().to_dict(),
        "mean_age": float(df["age"].mean()),
        "mean_cholesterol": float(df["cholesterol"].mean()),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "eda_insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")
    return insights
