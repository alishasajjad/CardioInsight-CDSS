#!/usr/bin/env python
"""Rebuild FAISS vector store from knowledge_base/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.rag.rag import build_vector_store

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Rebuild even if index exists")
    args = p.parse_args()
    result = build_vector_store(force=args.force)
    print(result)
