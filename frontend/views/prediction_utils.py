"""Shared prediction helpers — ensemble-aware context."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from backend.config import FEATURE_COLUMNS

CP_LABELS = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"]
ECG_LABELS = ["Normal", "LV Hypertrophy", "ST-T Abnormality"]
SLOPE_LABELS = ["Upsloping", "Flat", "Downsloping"]
THAL_LABELS = ["Normal", "Fixed Defect", "Reversible Defect"]


def inputs_to_human_readable(inputs: dict) -> dict[str, str]:
    return {
        "Age": f"{inputs['age']} years",
        "Sex": "Male" if inputs["sex"] == 1 else "Female",
        "Chest pain type": CP_LABELS[inputs["chest_pain_type"]],
        "Resting blood pressure": f"{inputs['resting_bp']} mmHg",
        "Cholesterol": f"{inputs['cholesterol']} mg/dl",
        "Fasting blood sugar > 120": "Yes" if inputs["fasting_blood_sugar"] else "No",
        "Resting ECG": ECG_LABELS[inputs["resting_ecg"]],
        "Max heart rate": f"{inputs['max_heart_rate']} bpm",
        "Exercise angina": "Yes" if inputs["exercise_angina"] else "No",
        "ST depression (oldpeak)": str(inputs["oldpeak"]),
        "ST slope": SLOPE_LABELS[inputs["st_slope"]],
        "Major vessels (0-3)": str(inputs["major_vessels"]),
        "Thalassemia": THAL_LABELS[inputs["thalassemia"]],
    }


def build_features(inputs: dict) -> pd.DataFrame:
    age = inputs["age"]
    row = {**inputs}
    row["chol_bp_ratio"] = inputs["cholesterol"] / max(inputs["resting_bp"], 1)
    row["age_group"] = 0 if age <= 40 else 1 if age <= 55 else 2 if age <= 65 else 3
    row["hr_reserve"] = max(0, 220 - age - inputs["max_heart_rate"])
    row["vessel_thal_score"] = inputs["major_vessels"] + inputs["thalassemia"]
    return pd.DataFrame([row])[FEATURE_COLUMNS]


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


def shap_for_instance(model, model_name: str, X: pd.DataFrame) -> tuple[np.ndarray, float] | None:
    # Tree SHAP only (RF/XGBoost expose feature_importances_; the ANN pipeline does not).
    if not hasattr(model, "feature_importances_"):
        return None
    try:
        explainer = shap.TreeExplainer(model)
        vals = _shap_positive_class_2d(explainer.shap_values(X.values))
        base = _expected_value_scalar(explainer.expected_value)
        return vals[0], base
    except Exception:
        return None


def shap_contributors_from_values(shap_values: np.ndarray | None, top_n: int = 8) -> list[dict]:
    if shap_values is None:
        return []
    pairs = sorted(zip(FEATURE_COLUMNS, shap_values), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return [{"feature": f, "shap": float(v)} for f, v in pairs]


def build_prediction_context(
    inputs: dict,
    ensemble: dict,
    shap_values: np.ndarray | None,
    recommendations: dict | None = None,
    ai_explanation: str = "",
) -> dict:
    deploy = ensemble.get("deployment_model", "XGBoost")
    return {
        "prediction": ensemble["ensemble_prediction"],
        "risk_probability": ensemble["ensemble_probability"],
        "confidence": ensemble["confidence"],
        "model_name": f"Ensemble ({deploy} primary)",
        "ensemble": ensemble,
        "human_readable_inputs": inputs_to_human_readable(inputs),
        "raw_inputs": inputs,
        "shap_contributors": shap_contributors_from_values(shap_values),
        "recommendations": recommendations or {},
        "ai_explanation": ai_explanation,
    }
