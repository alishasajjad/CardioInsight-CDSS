"""Runtime configuration for the CardioInsight backend (development) services.

This is the configuration the live app uses. Model artifacts are read from the
LOCAL ``backend/models`` copy (published here by the training pipeline) — the
backend never reaches into the machine_learning/ folder at runtime.

NOTE: the feature contract (CLINICAL_FEATURES / ENGINEERED_FEATURES /
FEATURE_COLUMNS) and the risk thresholds are intentionally mirrored from
``machine_learning/config.py``. If you change them in one place, change both.
"""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# ── Model artifacts (LOCAL copy used by the app) ─────────────────────────────
MODELS_DIR = BACKEND_DIR / "models"
DEPLOYMENT_DIR = MODELS_DIR / "deployment"
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.joblib"
ANN_MODEL_PATH = MODELS_DIR / "ann.joblib"
BEST_MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"
DEFAULT_MODEL_PATH = DEPLOYMENT_DIR / "xgboost_model.joblib"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
ARTIFACTS_PATH = DEPLOYMENT_DIR / "artifacts.joblib"

DEFAULT_DEPLOYMENT_MODEL = "XGBoost"
# Fallback weights only; the live weights come from model_metadata.json.
ENSEMBLE_WEIGHTS = {"Random Forest": 0.33, "XGBoost": 0.40, "ANN": 0.27}

# ── Feature contract (must match machine_learning/config.py) ─────────────────
CLINICAL_FEATURES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
    "oldpeak", "st_slope", "major_vessels", "thalassemia",
]
ENGINEERED_FEATURES = ["chol_bp_ratio", "age_group", "hr_reserve", "vessel_thal_score"]
FEATURE_COLUMNS = CLINICAL_FEATURES + ENGINEERED_FEATURES

# ── Application storage (runtime, shared at project root) ────────────────────
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "clinical_cds.db"
PDF_REPORTS_DIR = DATA_DIR / "reports"
KNOWLEDGE_BASE_DIR = BACKEND_DIR / "knowledge_base"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
FAISS_META_PATH = VECTOR_STORE_DIR / "chunks.json"

# Training charts published into the backend for the Analytics page.
FIGURES_DIR = BACKEND_DIR / "assets" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# ── Risk tiers (must match machine_learning/config.py) ───────────────────────
RISK_LOW_THRESHOLD = 0.35
RISK_MEDIUM_THRESHOLD = 0.55

# ── RAG ──────────────────────────────────────────────────────────────────────
RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 80
RAG_TOP_K = 4

MEDICAL_DISCLAIMER = (
    "This system provides educational clinical decision support only. "
    "It is not a medical device, diagnosis, or substitute for professional healthcare."
)
