"""RAG pipeline — FAISS retrieval with cached embeddings and index."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import (
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    KNOWLEDGE_BASE_DIR,
    MEDICAL_DISCLAIMER,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_EMBEDDING_MODEL,
    RAG_TOP_K,
    VECTOR_STORE_DIR,
)
from backend.llm.groq_assistant import DEFAULT_MODEL, chat_completion, format_patient_context, get_api_key
from backend.logging_config import get_logger

logger = get_logger(__name__)

RAG_SYSTEM_PROMPT = """You are an AI Health Assistant in an educational Clinical Decision Support System.

RULES:
1. Answer using RETRIEVED MEDICAL KNOWLEDGE and PATIENT CONTEXT when provided.
2. Use the retrieved knowledge to inform your answer, but do NOT list filenames, sources, or citations in your reply.
3. Clearly state guidance is EDUCATIONAL ONLY — not diagnosis or treatment.
4. Never prescribe medications. Refer emergencies to local emergency services.
5. Distinguish ML model output from established medical guidelines.
6. If retrieved context lacks the answer, say so and give general educational guidance.
"""


def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)
    return ""


def chunk_text(text: str, source: str) -> list[dict[str, str]]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[dict[str, str]] = []
    start, idx = 0, 0
    while start < len(text):
        chunk = text[start : start + RAG_CHUNK_SIZE]
        chunks.append({"id": f"{source}_{idx}", "source": source, "text": chunk})
        start += RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP
        idx += 1
    return chunks


def collect_documents() -> list[dict[str, str]]:
    """Collect ONLY medical knowledge-base documents (never project/dev docs)."""
    docs: list[dict[str, str]] = []
    for folder in (KNOWLEDGE_BASE_DIR,):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".txt", ".md", ".json"):
                rel = str(path.relative_to(folder))
                try:
                    text = _read_document(path)
                except Exception as exc:  # malformed/non-UTF8 file — skip, don't crash
                    logger.warning("Skipping unreadable knowledge-base file %s: %s", rel, exc)
                    continue
                for chunk in chunk_text(text, rel):
                    docs.append(chunk)
    return docs


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(RAG_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _load_index_bundle() -> tuple[Any, list[dict]] | None:
    try:
        import faiss
        if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
            return None
        with open(FAISS_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        return index, meta["chunks"]
    except Exception as exc:  # corrupt index/meta, or faiss missing
        logger.warning("Could not load FAISS index (RAG disabled): %s", exc)
        return None


def build_vector_store(force: bool = False) -> dict[str, Any]:
    """Build or load FAISS index from knowledge base."""
    import faiss

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists() and not force:
        with open(FAISS_META_PATH, encoding="utf-8") as f:
            return {"status": "exists", "chunks": json.load(f).get("count", 0)}

    chunks = collect_documents()
    if not chunks:
        logger.warning("No documents found in knowledge_base/")
        return {"status": "empty", "chunks": 0}

    embedder = _get_embedder()
    embeddings = np.array(
        embedder.encode([c["text"] for c in chunks], show_progress_bar=False, normalize_embeddings=True),
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    FAISS_META_PATH.write_text(
        json.dumps({"count": len(chunks), "chunks": chunks, "model": RAG_EMBEDDING_MODEL}),
        encoding="utf-8",
    )
    _load_index_bundle.cache_clear()
    logger.info("Built FAISS index with %d chunks", len(chunks))
    return {"status": "built", "chunks": len(chunks)}


def retrieve(query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    """Return top-k knowledge chunks. Degrades to [] (LLM-only) on any RAG failure."""
    try:
        bundle = _load_index_bundle()
        if bundle is None:
            build_vector_store()
            bundle = _load_index_bundle()
        if bundle is None:
            return []

        index, chunks = bundle
        if not chunks:
            return []
        q_emb = _get_embedder().encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(q_emb, min(top_k, len(chunks)))
        return [
            {"source": chunks[i]["source"], "text": chunks[i]["text"], "score": float(s)}
            for s, i in zip(scores[0], indices[0])
            if 0 <= i < len(chunks)
        ]
    except Exception as exc:  # missing embedder/network/faiss — never crash the assistant
        logger.warning("RAG retrieval unavailable, continuing without context: %s", exc)
        return []


def format_retrieved_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No retrieved documents."
    lines = ["=== RETRIEVED MEDICAL KNOWLEDGE ==="]
    for i, c in enumerate(chunks, 1):
        lines += [f"[{i}] Source: {c['source']}", c["text"][:800], ""]
    return "\n".join(lines)


def rag_chat(
    user_message: str,
    patient_ctx: dict | None,
    history: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
) -> tuple[str, list[dict]]:
    if not get_api_key():
        raise ValueError("GROQ_API_KEY not configured.")

    chunks = retrieve(user_message)
    patient_block = format_patient_context(patient_ctx) if patient_ctx else "No patient context."
    messages: list[dict[str, str]] = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"{patient_block}\n\n{format_retrieved_context(chunks)}"},
        {"role": "assistant", "content": "Ready to provide educational, source-grounded guidance."},
    ]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    reply = chat_completion(messages, model=model, max_tokens=1200)
    reply += f"\n\n---\n_{MEDICAL_DISCLAIMER}_"
    return reply, chunks


def rag_initial_explanation(patient_ctx: dict, model: str = DEFAULT_MODEL) -> tuple[str, list[dict]]:
    query = "heart disease risk factors prevention cardiovascular guidelines blood pressure cholesterol"
    chunks = retrieve(query)
    prompt = (
        f"{format_patient_context(patient_ctx)}\n\n{format_retrieved_context(chunks)}\n\n"
        "Provide: Prediction Summary, Key Risk Factors, Guideline-Informed Tips (cite sources), Disclaimer."
    )
    reply = chat_completion(
        [{"role": "system", "content": RAG_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        model=model,
    )
    return reply, chunks
