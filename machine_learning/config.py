"""Configuration for the CardioInsight machine-learning pipeline.

Self-contained: datasets, trained models, and outputs all live under
``machine_learning/``. After training, ``run_pipeline.py`` PUBLISHES a copy of
the deployed model bundle into ``backend/models`` so the app has its own copy.

NOTE: the feature contract and risk thresholds are intentionally mirrored in
``backend/config.py``. If you change them here, change them there too.
"""
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ML_ROOT.parent

# ── Data ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = ML_ROOT / "datasets" / "raw"
PROCESSED_DATA_DIR = ML_ROOT / "datasets" / "processed"
CLINICAL_DATASETS = [
    "heart_cleveland_upload",
    "Heart_disease_statlog",
    "heart_disease_uci",
    "heart_statlog_cleveland_hungary_final",
]
UNIFIED_DATASET_PATH = PROCESSED_DATA_DIR / "clinical_heart_disease.csv"
DATASET_SUMMARY_PATH = PROCESSED_DATA_DIR / "clinical_dataset_summary.json"

STANDARD_COLUMNS = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
    "oldpeak", "st_slope", "major_vessels", "thalassemia", "bmi", "target", "source_dataset",
]

CLINICAL_FEATURES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
    "oldpeak", "st_slope", "major_vessels", "thalassemia",
]
ENGINEERED_FEATURES = ["chol_bp_ratio", "age_group", "hr_reserve", "vessel_thal_score"]
FEATURE_COLUMNS = CLINICAL_FEATURES + ENGINEERED_FEATURES

# ── Models (source of truth, produced by training) ───────────────────────────
MODELS_DIR = ML_ROOT / "models"
DEPLOYMENT_DIR = MODELS_DIR / "deployment"
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.joblib"
ANN_MODEL_PATH = MODELS_DIR / "ann.joblib"
BEST_MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"
DEFAULT_MODEL_PATH = DEPLOYMENT_DIR / "xgboost_model.joblib"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
ARTIFACTS_PATH = DEPLOYMENT_DIR / "artifacts.joblib"

DEFAULT_DEPLOYMENT_MODEL = "XGBoost"
ENSEMBLE_WEIGHTS = {"Random Forest": 0.33, "XGBoost": 0.40, "ANN": 0.27}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ── Outputs ──────────────────────────────────────────────────────────────────
OUTPUTS_DIR = ML_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# ── Documentation (shared at project root) ───────────────────────────────────
DOCS_DIR = PROJECT_ROOT / "docs"

# ── Risk tiers (must match backend/config.py) ────────────────────────────────
RISK_LOW_THRESHOLD = 0.35
RISK_MEDIUM_THRESHOLD = 0.55
