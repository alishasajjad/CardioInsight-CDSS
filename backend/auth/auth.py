"""User authentication — bcrypt hashing and session tokens."""
from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt

from backend.database.database import (
    create_session,
    create_user,
    delete_session,
    get_session,
    get_user_by_username,
    init_db,
)

EMAIL_RE = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
SESSION_TTL_HOURS = 24 * 7  # sessions expire after 7 days


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def validate_registration(username: str, email: str, password: str) -> str | None:
    if len(username.strip()) < 3:
        return "Username must be at least 3 characters."
    if not EMAIL_RE.match(email.strip()):
        return "Invalid email address."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    if get_user_by_username(username):
        return "Username already exists."
    return None


def register_user(username: str, email: str, password: str) -> tuple[int | None, str | None]:
    init_db()
    err = validate_registration(username, email, password)
    if err:
        return None, err
    try:
        uid = create_user(username, email, hash_password(password))
        return uid, None
    except Exception as e:
        if "UNIQUE" in str(e):
            return None, "Email or username already registered."
        return None, str(e)


def login_user(username: str, password: str) -> tuple[str | None, dict | None, str | None]:
    init_db()
    user = get_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        return None, None, "Invalid username or password."
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
    create_session(user["id"], token, expires)
    return token, {"id": user["id"], "username": user["username"], "email": user["email"]}, None


def logout_user(token: str) -> None:
    if token:
        delete_session(token)


def resolve_session(token: str | None) -> dict | None:
    if not token:
        return None
    session = get_session(token)
    if not session:
        return None
    return {"id": session["user_id"], "username": session["username"], "email": session["email"], "token": token}
