# auth.py
#
# Authentication for the Study-in-Germany AI Advisor.
#
# DEPENDENCY CHANGE (passlib removed)
# ─────────────────────────────────────
# passlib[bcrypt] was removed because it conflicts with bcrypt >= 4.x.
# passlib tries to read bcrypt.__about__.__version__ which no longer exists
# in bcrypt 4.x, causing it to fall through to a broken code path that
# crashes even on short passwords (the "72 bytes" error is a red herring —
# passlib's backend detection is what actually fails).
#
# We now use the `bcrypt` package directly. It's simpler, actively maintained,
# and has no compatibility issues. API is almost identical.
#
# pyproject.toml change:
#   REMOVE: "passlib[bcrypt]>=1.7.4"
#   KEEP:   "bcrypt>=4.0.0"           (likely already installed transitively)
#
# All other design decisions are unchanged — see original comments below.

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

log = logging.getLogger(__name__)

# ── Password hashing ──────────────────────────────────────────────────────────
#
# bcrypt is intentionally slow (work factor 12 ≈ 250ms/hash on M2).
# This makes brute-force attacks expensive even if the DB leaks.
#
# bcrypt has a hard 72-byte input limit. We SHA-256 the password first so
# any length password is safely reduced to 32 bytes before bcrypt sees it.
# This is a well-known pattern (called "pre-hashing") — it's safe because
# SHA-256 is a one-way function and bcrypt still provides the salt and
# work-factor protection.

import hashlib

def _prepare(plain: str) -> bytes:
    """
    SHA-256 the password before bcrypt so we never hit the 72-byte limit.
    Returns raw bytes suitable for bcrypt.hashpw().
    """
    return hashlib.sha256(plain.encode("utf-8")).digest()


def hash_password(plain: str) -> str:
    """Hash a plain-text password. Returns a UTF-8 string for DB storage."""
    hashed = bcrypt.hashpw(_prepare(plain), bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Return True if plain matches the stored hash.
    Constant-time comparison — safe against timing attacks.
    """
    try:
        return bcrypt.checkpw(_prepare(plain), hashed.encode("utf-8"))
    except Exception:
        # Malformed hash (e.g. NULL from DB for pre-auth users) → always False
        return False


# ── JWT configuration ─────────────────────────────────────────────────────────

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
ALGORITHM  = "HS256"
TOKEN_DAYS = int(os.getenv("JWT_TOKEN_DAYS", "30"))


def _require_secret() -> str:
    if not SECRET_KEY:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set in .env. "
            "Generate one: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return SECRET_KEY


def create_access_token(user_id: str, email: str) -> str:
    """Issue a signed JWT. Returns the encoded token string."""
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
    Decode and validate a JWT. Returns payload dict on success.
    Raises HTTP 401 on invalid signature, expiry, or missing claims.
    """
    try:
        payload = jwt.decode(token, _require_secret(), algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Token missing subject claim.")
        return payload
    except JWTError as exc:
        log.warning("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Token is invalid or has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency ────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> dict:
    """
    FastAPI dependency — validates Bearer token and returns the JWT payload.
    Add to any endpoint: current_user: dict = Depends(get_current_user)
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
    """Convenience dependency — returns just the user_id string."""
    return get_current_user(credentials)["sub"]


# ── Password strength validation ──────────────────────────────────────────────

def validate_password_strength(password: str) -> Optional[str]:
    """Return an error string if too weak, else None."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if password.isdigit():
        return "Password cannot be all numbers."
    if password.lower() == password and not any(c.isdigit() for c in password):
        return "Password must contain at least one number or uppercase letter."
    return None