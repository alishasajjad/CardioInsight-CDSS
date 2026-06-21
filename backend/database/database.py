"""SQLite database layer for users, predictions, reports, and chat."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import DB_PATH, DATA_DIR

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    inputs_json TEXT NOT NULL,
    ensemble_json TEXT NOT NULL,
    shap_json TEXT,
    recommendations_json TEXT,
    ai_explanation TEXT,
    risk_level TEXT,
    ensemble_probability REAL,
    ensemble_prediction INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    pdf_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    prediction_id INTEGER,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_messages(user_id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_connection(db_path: Path | None = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")        # tolerate concurrent readers/writers
    conn.execute("PRAGMA busy_timeout = 30000")      # wait up to 30s on a locked DB
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))


def create_user(username: str, email: str, password_hash: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username.strip().lower(), email.strip().lower(), password_hash, utc_now()),
        )
        return int(cur.lastrowid)


def get_user_by_username(username: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip().lower(),)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_session(user_id: int, token: str, expires_at: str | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (user_id, token, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (user_id, token, expires_at, utc_now()),
        )


def get_session(token: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT s.*, u.username, u.email FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ? AND (s.expires_at IS NULL OR s.expires_at > ?)
            """,
            (token, utc_now()),
        ).fetchone()
        return dict(row) if row else None


def delete_session(token: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def save_prediction(
    user_id: int,
    inputs: dict,
    ensemble: dict,
    shap: list | None,
    recommendations: dict,
    ai_explanation: str,
    risk_level: str,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                user_id, inputs_json, ensemble_json, shap_json,
                recommendations_json, ai_explanation, risk_level,
                ensemble_probability, ensemble_prediction, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                json.dumps(inputs),
                json.dumps(ensemble),
                json.dumps(shap or []),
                json.dumps(recommendations),
                ai_explanation,
                risk_level,
                ensemble.get("ensemble_probability"),
                ensemble.get("ensemble_prediction"),
                utc_now(),
            ),
        )
        return int(cur.lastrowid)


def save_report(prediction_id: int, user_id: int, pdf_path: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO reports (prediction_id, user_id, pdf_path, created_at) VALUES (?, ?, ?, ?)",
            (prediction_id, user_id, pdf_path, utc_now()),
        )
        return int(cur.lastrowid)


def get_user_prediction_count(user_id: int) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM predictions WHERE user_id = ?", (user_id,)
        ).fetchone()
        return int(row["cnt"]) if row else 0


def get_predictions(
    user_id: int,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
) -> list[dict]:
    query = "SELECT * FROM predictions WHERE user_id = ?"
    params: list[Any] = [user_id]
    if date_from:
        query += " AND date(created_at) >= date(?)"
        params.append(date_from)
    if date_to:
        query += " AND date(created_at) <= date(?)"
        params.append(date_to)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_prediction(prediction_id: int, user_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, user_id),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for key in ("inputs_json", "ensemble_json", "shap_json", "recommendations_json"):
            if d.get(key):
                d[key.replace("_json", "")] = json.loads(d[key])
        return d


def get_report_for_prediction(prediction_id: int, user_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM reports WHERE prediction_id = ? AND user_id = ? ORDER BY id DESC LIMIT 1",
            (prediction_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def save_chat_message(
    user_id: int,
    role: str,
    content: str,
    prediction_id: int | None = None,
    sources: list | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (user_id, prediction_id, role, content, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, prediction_id, role, content, json.dumps(sources or []), utc_now()),
        )


def get_chat_history(user_id: int, prediction_id: int | None = None, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        if prediction_id:
            rows = conn.execute(
                """
                SELECT role, content, sources_json, created_at FROM chat_messages
                WHERE user_id = ? AND prediction_id = ?
                ORDER BY id ASC LIMIT ?
                """,
                (user_id, prediction_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT role, content, sources_json, created_at FROM chat_messages
                WHERE user_id = ? ORDER BY id DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        out = []
        for r in rows:
            out.append({
                "role": r["role"],
                "content": r["content"],
                "sources": json.loads(r["sources_json"] or "[]"),
            })
        return out
