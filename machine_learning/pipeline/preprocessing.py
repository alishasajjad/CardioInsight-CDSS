"""
Clinical data pipeline — merges four angiography-aligned datasets.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.impute import SimpleImputer

from machine_learning.config import (
    DATASET_SUMMARY_PATH,
    FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    UNIFIED_DATASET_PATH,
)
from machine_learning.pipeline.loaders import (
    clip_outliers,
    load_heart_csv,
    load_heart_disease_uci,
    load_heart_statlog_hungary,
    remove_duplicates,
)
from machine_learning.logging_config import get_logger

logger = get_logger(__name__)

CLINICAL_LOADERS = [
    (RAW_DATA_DIR / "heart_cleveland_upload.csv", lambda p: load_heart_csv(p, "condition")),
    (RAW_DATA_DIR / "Heart_disease_statlog.csv", lambda p: load_heart_csv(p, "target")),
    (RAW_DATA_DIR / "heart_disease_uci.csv", load_heart_disease_uci),
    (RAW_DATA_DIR / "heart_statlog_cleveland_hungary_final.csv", load_heart_statlog_hungary),
]


def engineer_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived clinical features."""
    df = df.copy()
    df["chol_bp_ratio"] = df["cholesterol"] / df["resting_bp"].replace(0, pd.NA)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 40, 55, 65, 120], labels=[0, 1, 2, 3], include_lowest=True
    ).astype(float)
    df["hr_reserve"] = (220 - df["age"] - df["max_heart_rate"]).clip(lower=0)
    df["vessel_thal_score"] = df["major_vessels"].fillna(0) + df["thalassemia"].fillna(0)
    return df


def build_clinical_dataset() -> pd.DataFrame:
    """Load, merge, clean, and persist the clinical unified dataset."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    summary = []

    for path, loader in CLINICAL_LOADERS:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset missing: {path}")
        frame = loader(path)
        frames.append(frame)
        summary.append({
            "dataset": path.stem,
            "rows": len(frame),
            "target_positive_rate": round(float(frame["target"].mean()), 4),
        })
        logger.info("Loaded %s (%d rows)", path.name, len(frame))

    df = pd.concat(frames, ignore_index=True)
    df = clip_outliers(df)
    df = remove_duplicates(df)
    df = engineer_clinical_features(df)

    imputer = SimpleImputer(strategy="median")
    df[FEATURE_COLUMNS] = imputer.fit_transform(df[FEATURE_COLUMNS])
    df.to_csv(UNIFIED_DATASET_PATH, index=False)

    meta = {
        "datasets": summary,
        "total_rows": len(df),
        "target_rate": round(float(df["target"].mean()), 4),
        "feature_columns": FEATURE_COLUMNS,
    }
    DATASET_SUMMARY_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved clinical dataset: %s (%d rows)", UNIFIED_DATASET_PATH, len(df))
    return df
