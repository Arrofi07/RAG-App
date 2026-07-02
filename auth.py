# auth.py
#
# Authentication for the Study-in-Germany AI Advisor.
#
# DESIGN DECISIONS (recorded so they're easy to revisit)
# ───────────────────────────────────────────────────────
# 1. Password hashing: bcrypt via `passlib`. bcrypt is intentionally slow
#    (work factor 12 ≈ 250ms/hash on M2) — this is the point. It makes
#    brute-force attacks expensive even if the DB leaks. We do NOT use
#    plain SHA-256 or MD5, which are too fast for password storage.
#
# 2. Token format: JWT (JSON Web Token) signed with HS256 using a secret
#    key from .env. Tokens are stateless — the server doesn't store them,
#    so there's no token table to maintain. Downside: tokens can't be
#    individually revoked before expiry. Acceptable for a 30-day window
#    on a student tool; if you need revocation, add a blocklist table later.
#
# 3. Token lifetime: 30 days (per your choice). The expiry is verified on
#    every request — a tampered or expired token is rejected with 401.
#
# 4. No email verification (yet): registration stores the email but doesn't
#    send a verification link. You said you'll add SMTP later. The `is_verified`
#    flag is stored in the DB so you can gate features on it when ready.
#
# 5. Assumption recorded: the Streamlit frontend stores the JWT in
#    st.session_state (not a cookie). This means it's lost on hard page
#    refresh. This is the simplest approach for Streamlit and acceptable
#    for a student tool. Browser localStorage would survive refresh but
#    requires a custom Streamlit component.
#
# DEPENDENCIES ADDED TO pyproject.toml
# ─────────────────────────────────────
#   passlib[bcrypt]>=1.7.4
#   python-jose[cryptography]>=3.3.0

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

log = logging.getLogger(__name__)

# ── Password hashing ──────────────────────────────────────────────────────────

# CryptContext handles algorithm upgrades gracefully — if you ever switch
# from bcrypt to argon2, existing hashes keep working (deprecated=["auto"]
# means old hashes are re-hashed on next successful login).
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Hash a plain-text password using bcrypt. Store the result, never the plain text."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    return _pwd_context.verify(plain, hashed)


# ── JWT configuration ─────────────────────────────────────────────────────────

# SECRET_KEY signs the JWT. Must be random and secret — never commit it.
# Generate a good one with: python -c "import secrets; print(secrets.token_hex(32))"
# Startup will raise clearly if it's not set.
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
ALGORITHM  = "HS256"
TOKEN_DAYS = int(os.getenv("JWT_TOKEN_DAYS", "30"))


def _require_secret() -> str:
    """Raise at call time (not import time) if JWT_SECRET_KEY is missing."""
    if not SECRET_KEY:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set in .env. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return SECRET_KEY


def create_access_token(user_id: str, email: str) -> str:
    """
    Issue a signed JWT containing the user's ID and email.

    The token encodes:
      sub  — subject (user_id, the primary lookup key)
      email — stored for display without a DB round-trip
      exp  — expiry timestamp (now + TOKEN_DAYS days)
      iat  — issued-at timestamp (useful for audit logs)

    Returns the encoded JWT string to send to the client.
    """
    now    = datetime.now(timezone.utc)
    expiry = now + timedelta(days=TOKEN_DAYS)

    payload = {
        "sub":   user_id,
        "email": email,
        "exp":   expiry,
        "iat":   now,
    }

    return jwt.encode(payload, _require_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT. Returns the payload dict on success.

    Raises HTTPException 401 on:
      - Invalid signature (token was tampered)
      - Expired token (exp is in the past)
      - Missing required claims (sub)
      - Any other JWT error
    """
    try:
        payload = jwt.decode(token, _require_secret(), algorithms=[ALGORITHM])

        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing subject claim.")

        return payload

    except JWTError as exc:
        # JWTError covers ExpiredSignatureError, JWTClaimsError, etc.
        # We give a generic message to avoid leaking which check failed.
        log.warning("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Token is invalid or has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency — use this to protect any endpoint ────────────────────
#
# Usage in main.py:
#   from auth import get_current_user
#
#   @app.get("/protected")
#   def protected(current_user: dict = Depends(get_current_user)):
#       return {"user_id": current_user["sub"]}

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> dict:
    """
    FastAPI dependency that extracts and validates the Bearer token.

    Returns the decoded JWT payload (contains "sub" = user_id, "email").
    Raises 401 if no token is provided or the token is invalid/expired.

    Mark an endpoint as requiring auth by adding:
        current_user: dict = Depends(get_current_user)
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> str:
    """
    Convenience dependency — returns just the user_id string.
    Use when you only need the ID, not the full payload.
    """
    return get_current_user(credentials)["sub"]


# ── Password strength validation ──────────────────────────────────────────────
# Simple rules — enough for a student tool without being annoying.

def validate_password_strength(password: str) -> Optional[str]:
    """
    Return an error message string if the password is too weak, else None.
    Call this before hashing — we never store a hash of a bad password.
    """
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if password.isdigit():
        return "Password cannot be all numbers."
    if password.lower() == password and not any(c.isdigit() for c in password):
        return "Password must contain at least one number or uppercase letter."
    return None  # password is acceptable