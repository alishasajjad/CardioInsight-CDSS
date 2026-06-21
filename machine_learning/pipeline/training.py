"""
Model training, evaluation, SHAP, and deployment artifacts (RF, XGBoost, ANN).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from machine_learning.config import (
    ARTIFACTS_PATH,
    BEST_MODEL_PATH,
    CV_FOLDS,
    DEFAULT_DEPLOYMENT_MODEL,
    DEFAULT_MODEL_PATH,
    DEPLOYMENT_DIR,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METADATA_PATH,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    UNIFIED_DATASET_PATH,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def _shap_positive_class_2d(values) -> np.ndarray:
    """Normalize TreeExplainer output to 2D (n_samples, n_features), positive class.

    Handles list[class] (older SHAP), plain 2D arrays, and the 3D arrays
    (n_samples, n_features, n_classes) emitted by newer SHAP / XGBoost.
    """
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _expected_value_scalar(base) -> float:
    arr = np.asarray(base, dtype=float).ravel()
    return float(arr[-1]) if arr.size else 0.0


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(UNIFIED_DATASET_PATH)
    return df[FEATURE_COLUMNS], df["target"]


def _evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_prob": y_prob,
    }


def tune_random_forest(X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [200, 300, 500],
            "max_depth": [6, 10, 14, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        n_iter=30,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"    RF CV AUC={search.best_score_:.4f} params={search.best_params_}")
    return search.best_estimator_


def tune_xgboost(X_train, y_train):
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale,
            n_jobs=-1,
        ),
        {
            "n_estimators": [150, 250, 400],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 2, 5],
        },
        n_iter=35,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"    XGB CV AUC={search.best_score_:.4f} params={search.best_params_}")
    return search.best_estimator_


def tune_ann(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            MLPClassifier(
                activation="relu",
                solver="adam",
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_STATE,
                max_iter=500,
            ),
        ),
    ])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe,
        {
            "clf__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
            "clf__alpha": [0.0001, 0.001, 0.01],
            "clf__learning_rate_init": [0.0005, 0.001, 0.005],
        },
        n_iter=25,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"    ANN CV AUC={search.best_score_:.4f} params={search.best_params_}")
    return search.best_estimator_


def _feature_importance(model, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf", model), "coefs_"):
        imp = np.abs(model.named_steps["clf"].coefs_[0]).mean(axis=0)
    else:
        return {}
    return {f: float(v) for f, v in zip(feature_names, imp)}


def compute_shap(model, model_name: str, X_sample: pd.DataFrame) -> dict:
    """Tree SHAP for RF/XGB; coefficient fallback for ANN."""
    if model_name in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
        values = _shap_positive_class_2d(explainer.shap_values(X_sample.values))
        base = _expected_value_scalar(explainer.expected_value)
    else:
        return {"note": "ANN uses coefficient-based importance in dashboard"}
    mean_abs = np.abs(values).mean(axis=0)
    return {
        "shap_values_sample": values[:100].tolist(),
        "base_value": base,
        "mean_abs_shap": {f: float(v) for f, v in zip(X_sample.columns, mean_abs)},
        "feature_names": list(X_sample.columns),
    }


def plot_results(all_metrics: list[dict], y_test) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m in zip(axes, all_metrics):
        sns.heatmap(np.array(m["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(m["model"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices — Clinical Cohort")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", bbox_inches="tight")
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in all_metrics:
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        ax.plot(fpr, tpr, label=f"{m['model']} (AUC={m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", bbox_inches="tight")
    plt.close()

    # Metrics bar chart
    names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    df_m = pd.DataFrame(all_metrics).set_index("model")[names]
    fig, ax = plt.subplots(figsize=(10, 5))
    df_m.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Metrics")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_comparison.png", bbox_inches="tight")
    plt.close()


def plot_shap_summary(model, model_name: str, X_sample: pd.DataFrame) -> None:
    if model_name not in ("Random Forest", "XGBoost"):
        return
    explainer = shap.TreeExplainer(model)
    values = _shap_positive_class_2d(explainer.shap_values(X_sample.values))
    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", bbox_inches="tight", dpi=120)
    plt.close()
    plt.figure(figsize=(10, 5))
    shap.summary_plot(values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", bbox_inches="tight", dpi=120)
    plt.close()


def plot_feature_importance(importance: dict[str, float], title: str) -> None:
    if not importance:
        return
    s = pd.Series(importance).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    s.plot(kind="barh", ax=ax, color="#c0392b")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close()


def train_and_deploy() -> dict:
    """Full training pipeline; deploy XGBoost if best, else overall best."""
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    fitted = {}
    all_metrics = []

    for name, tuner in [
        ("Random Forest", tune_random_forest),
        ("XGBoost", tune_xgboost),
        ("ANN", tune_ann),
    ]:
        print(f"  Training {name}...")
        model = tuner(X_train, y_train)
        fitted[name] = model
        metrics = _evaluate(name, model, X_test, y_test)
        all_metrics.append(metrics)
        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    # Select best by ROC-AUC; prefer XGBoost on tie within 0.005
    best = max(all_metrics, key=lambda m: m["roc_auc"])
    xgb_m = next(m for m in all_metrics if m["model"] == "XGBoost")
    if xgb_m["roc_auc"] >= best["roc_auc"] - 0.005:
        deploy_name = DEFAULT_DEPLOYMENT_MODEL
        deploy_metrics = xgb_m
    else:
        deploy_name = best["model"]
        deploy_metrics = best

    deploy_model = fitted[deploy_name]
    joblib.dump(deploy_model, BEST_MODEL_PATH)
    joblib.dump(fitted["XGBoost"], DEFAULT_MODEL_PATH)

    importance = _feature_importance(deploy_model, FEATURE_COLUMNS)
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
    shap_data = compute_shap(deploy_model, deploy_name, shap_sample)
    plot_shap_summary(deploy_model, deploy_name, shap_sample)
    plot_feature_importance(importance, f"Feature Importance — {deploy_name}")
    plot_results(all_metrics, y_test)

    # Clean metrics for JSON (remove y_prob)
    metrics_json = [{k: v for k, v in m.items() if k != "y_prob"} for m in all_metrics]
    # ROC-AUC–normalized ensemble weights (consumed at inference via metadata).
    _aucs = {m["model"]: max(m["roc_auc"], 0.01) for m in metrics_json}
    _total = sum(_aucs.values())
    ensemble_weights = {k: v / _total for k, v in _aucs.items()}

    metadata = {
        "deployment_model": deploy_name,
        "xgboost_is_default": deploy_name == "XGBoost",
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics_json,
        "ensemble_weights": ensemble_weights,
        "best_metrics": {k: deploy_metrics[k] for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]},
        "clinical_datasets_only": True,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(METRICS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    artifacts = {
        "feature_importance": importance,
        "shap": shap_data,
        "imputer_medians": None,
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)

    print(f"\n  Deployed model: {deploy_name}")
    print(f"  ROC-AUC: {deploy_metrics['roc_auc']:.4f} | Accuracy: {deploy_metrics['accuracy']:.4f}")
    return metadata
