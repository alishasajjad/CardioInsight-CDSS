#!/usr/bin/env python
"""Initialize SQLite database schema."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.database.database import init_db
from backend.config import DB_PATH

if __name__ == "__main__":
    init_db()
    print(f"Database initialized: {DB_PATH}")
