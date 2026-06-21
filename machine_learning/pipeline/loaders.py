"""Clinical dataset loaders — Cleveland, Statlog, UCI harmonization."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from machine_learning.config import STANDARD_COLUMNS

CP_MAP = {
    "typical angina": 0,
    "atypical angina": 1,
    "non-anginal": 2,
    "asymptomatic": 3,
}
RESTECG_MAP = {"normal": 0, "lv hypertrophy": 1, "st-t abnormality": 2}
SLOPE_MAP = {"upsloping": 0, "flat": 1, "downsloping": 2}
THAL_MAP = {"normal": 0, "fixed defect": 1, "reversable defect": 2, "reversible defect": 2}


def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize schema and binarize target."""
    df = df.copy()
    df["source_dataset"] = source
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[STANDARD_COLUMNS]
    for col in STANDARD_COLUMNS:
        if col != "source_dataset":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = (df["target"] > 0).astype(int)
    return df


def load_heart_csv(path: Path, target_col: str = "target") -> pd.DataFrame:
    """Load Cleveland-style heart CSV files."""
    df = pd.read_csv(path)
    rename = {
        "cp": "chest_pain_type",
        "trestbps": "resting_bp",
        "chol": "cholesterol",
        "fbs": "fasting_blood_sugar",
        "restecg": "resting_ecg",
        "thalach": "max_heart_rate",
        "exang": "exercise_angina",
        "slope": "st_slope",
        "ca": "major_vessels",
        "thal": "thalassemia",
        target_col: "target",
        "condition": "target",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return _finalize(df, path.stem)


def load_heart_statlog_hungary(path: Path) -> pd.DataFrame:
    """Load Statlog Cleveland/Hungary dataset."""
    df = pd.read_csv(path)
    rename = {
        "chest pain type": "chest_pain_type",
        "resting bp s": "resting_bp",
        "cholesterol": "cholesterol",
        "fasting blood sugar": "fasting_blood_sugar",
        "resting ecg": "resting_ecg",
        "max heart rate": "max_heart_rate",
        "exercise angina": "exercise_angina",
        "ST slope": "st_slope",
    }
    return _finalize(df.rename(columns=rename), path.stem)


def load_heart_disease_uci(path: Path) -> pd.DataFrame:
    """Load UCI heart disease dataset with string categoricals."""
    df = pd.read_csv(path)

    def map_cat(series: pd.Series, mapping: dict) -> pd.Series:
        return series.astype(str).str.strip().str.lower().map(
            {k.lower(): v for k, v in mapping.items()}
        )

    out = pd.DataFrame()
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["sex"] = df["sex"].map({"Male": 1, "Female": 0, "male": 1, "female": 0})
    out["chest_pain_type"] = map_cat(df["cp"], CP_MAP)
    out["resting_bp"] = pd.to_numeric(df["trestbps"], errors="coerce")
    out["cholesterol"] = pd.to_numeric(df["chol"], errors="coerce")
    out["fasting_blood_sugar"] = df["fbs"].map(
        {True: 1, False: 0, "TRUE": 1, "FALSE": 0}
    )
    out["resting_ecg"] = map_cat(df["restecg"], RESTECG_MAP)
    out["max_heart_rate"] = pd.to_numeric(df["thalch"], errors="coerce")
    out["exercise_angina"] = df["exang"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})
    out["oldpeak"] = pd.to_numeric(df["oldpeak"], errors="coerce")
    out["st_slope"] = map_cat(df["slope"], SLOPE_MAP)
    out["major_vessels"] = pd.to_numeric(df["ca"], errors="coerce")
    out["thalassemia"] = map_cat(df["thal"], THAL_MAP)
    out["bmi"] = np.nan
    out["target"] = pd.to_numeric(df["num"], errors="coerce")
    return _finalize(out, path.stem)


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip values outside physiological ranges."""
    df = df.copy()
    rules = {
        "age": (1, 120),
        "resting_bp": (50, 250),
        "cholesterol": (80, 600),
        "max_heart_rate": (40, 220),
        "oldpeak": (0, 10),
    }
    for col, (lo, hi) in rules.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records on core clinical fields."""
    subset = ["age", "sex", "resting_bp", "cholesterol", "max_heart_rate", "target"]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    removed = before - len(df)
    if removed:
        from machine_learning.logging_config import get_logger
        get_logger(__name__).info("Removed %d duplicate rows", removed)
    return df
