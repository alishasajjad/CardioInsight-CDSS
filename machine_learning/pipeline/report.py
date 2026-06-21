"""Generate pipeline documentation after training."""
from __future__ import annotations

import json
from datetime import datetime

from machine_learning.config import DATASET_SUMMARY_PATH, DOCS_DIR, METADATA_PATH


def write_training_summary(eda_insights: dict, training_meta: dict) -> None:
    """Write machine-readable training summary for the app and docs."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.utcnow().isoformat(),
        "eda": eda_insights,
        "training": training_meta,
    }
    (DOCS_DIR / "training_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
