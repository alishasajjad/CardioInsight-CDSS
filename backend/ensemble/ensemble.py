"""Ensemble prediction — RF, XGBoost, ANN weighted averaging."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.config import (
    ANN_MODEL_PATH,
    ARTIFACTS_PATH,
    BEST_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    ENSEMBLE_WEIGHTS,
    METADATA_PATH,
    MODELS_DIR,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
)
from backend.logging_config import get_logger

logger = get_logger(__name__)


def _load_model(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def load_all_models() -> dict:
    """Load RF, XGBoost, ANN, and deployment metadata."""
    models = {
        "Random Forest": _load_model(RF_MODEL_PATH),
        "XGBoost": _load_model(XGB_MODEL_PATH) or _load_model(DEFAULT_MODEL_PATH),
        "ANN": _load_model(ANN_MODEL_PATH),
    }
    meta = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    artifacts = joblib.load(ARTIFACTS_PATH) if ARTIFACTS_PATH.exists() else {}
    deploy = _load_model(BEST_MODEL_PATH)
    return {
        "models": {k: v for k, v in models.items() if v is not None},
        "metadata": meta,
        "artifacts": artifacts,
        "deploy_model": deploy,
    }


def compute_ensemble_weights(metadata: dict | None = None) -> dict[str, float]:
    """ROC-AUC–normalized weights from training metadata or config defaults."""
    if metadata and metadata.get("ensemble_weights"):
        return metadata["ensemble_weights"]
    if metadata and metadata.get("metrics"):
        aucs = {m["model"]: max(m["roc_auc"], 0.01) for m in metadata["metrics"]}
        total = sum(aucs.values())
        return {k: v / total for k, v in aucs.items()}
    total = sum(ENSEMBLE_WEIGHTS.values())
    return {k: v / total for k, v in ENSEMBLE_WEIGHTS.items()}


def predict_single(model, X: pd.DataFrame) -> tuple[float, int]:
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"{type(model).__name__} has no predict_proba")
    prob = float(model.predict_proba(X)[0, 1])
    return prob, int(prob >= 0.5)


def ensemble_predict(X: pd.DataFrame, system: dict | None = None) -> dict:
    """
    Run all models and return ensemble result with per-model breakdown.
    """
    if system is None:
        system = load_all_models()
    models = system["models"]
    metadata = system.get("metadata", {})

    # Enforce the feature contract recorded at training time. This is the runtime
    # guard against the two config files drifting apart (feature order/columns).
    expected = metadata.get("feature_columns")
    if expected and list(X.columns) != list(expected):
        raise ValueError(
            "Input feature columns do not match the trained model contract. "
            "Re-align FEATURE_COLUMNS or retrain the models."
        )

    weights = compute_ensemble_weights(metadata)

    individual = {}
    weighted_sum = 0.0
    weight_used = 0.0

    for name, model in models.items():
        try:
            prob, pred = predict_single(model, X)
        except Exception as exc:
            logger.warning("Model '%s' could not predict and was skipped: %s", name, exc)
            continue
        w = weights.get(name, 1 / max(len(models), 1))
        individual[name] = {
            "probability": prob,
            "prediction": pred,
            "weight": round(w, 4),
            "weighted_contribution": round(prob * w, 4),
        }
        weighted_sum += prob * w
        weight_used += w

    if not individual or weight_used == 0:
        raise RuntimeError(
            "No usable models are available to produce a prediction. "
            "Run `python machine_learning/run_pipeline.py` to (re)train and publish models."
        )

    ensemble_prob = weighted_sum / weight_used
    ensemble_pred = int(ensemble_prob >= 0.5)
    confidence = float(max(ensemble_prob, 1 - ensemble_prob))

    # Agreement boosts confidence slightly
    preds = [v["prediction"] for v in individual.values()]
    agreement = sum(p == ensemble_pred for p in preds) / max(len(preds), 1)
    adjusted_confidence = min(0.99, confidence * (0.85 + 0.15 * agreement))

    deploy = metadata.get("deployment_model", "XGBoost")
    rationale = (
        f"Ensemble combines {', '.join(individual.keys())} using ROC-AUC–weighted averaging. "
        f"Final risk {ensemble_prob:.1%} from weighted votes. "
        f"Primary deployment model remains **{deploy}** (best single-model performer). "
        f"Model agreement: {agreement:.0%}."
    )

    return {
        "individual": individual,
        "ensemble_probability": ensemble_prob,
        "ensemble_prediction": ensemble_pred,
        "confidence": adjusted_confidence,
        "weights": weights,
        "model_agreement": agreement,
        "deployment_model": deploy,
        "rationale": rationale,
    }
