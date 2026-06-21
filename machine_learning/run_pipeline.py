#!/usr/bin/env python
"""CardioInsight — machine-learning pipeline (training & deployment).

Builds the unified clinical dataset, runs EDA, trains and selects the ensemble
models, then PUBLISHES a copy of the deployed model bundle (and the training
charts) into the backend so the running app has its own local copy — the
backend never reads models out of the machine_learning/ folder.

Run:  python machine_learning/run_pipeline.py
Then: streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from machine_learning.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("pipeline")


def _publish_to_backend() -> None:
    """Copy the trained model bundle + charts into the backend (development) copy."""
    from machine_learning import config as ml
    from backend import config as be

    be.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ml.MODELS_DIR, be.MODELS_DIR, dirs_exist_ok=True)
    logger.info("Published model bundle -> %s", be.MODELS_DIR)

    be.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for png in ml.FIGURES_DIR.glob("*.png"):
        shutil.copy2(png, be.FIGURES_DIR / png.name)
    logger.info("Published training charts -> %s", be.FIGURES_DIR)


def main() -> None:
    logger.info("Starting clinical CDSS pipeline")

    from machine_learning.pipeline.preprocessing import build_clinical_dataset
    from machine_learning.pipeline.eda import run_eda
    from machine_learning.pipeline.training import train_and_deploy
    from machine_learning.pipeline.report import write_training_summary

    build_clinical_dataset()
    insights = run_eda()
    meta = train_and_deploy()
    write_training_summary(insights, meta)

    # Publish trained artifacts into the backend (development) copy.
    _publish_to_backend()

    # Backend runtime setup (best-effort; the app also initializes the DB on start).
    try:
        from backend.database.database import init_db
        init_db()
        from backend.rag.rag import build_vector_store
        logger.info("Vector store: %s", build_vector_store(force=False))
    except Exception as exc:
        logger.warning("Backend setup step skipped: %s", exc)

    logger.info("Pipeline complete — run: streamlit run frontend/streamlit_app.py")


if __name__ == "__main__":
    main()
