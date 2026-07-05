# tests/test_auth.py
#
# Tests for auth.py — password hashing, JWT lifecycle, strength validation.
#
# These tests need NO mocks because auth.py is pure logic (bcrypt + JWT).
# They run offline and require only passlib and python-jose installed.

import os
import pytest

os.environ["JWT_SECRET_KEY"] = "test-secret-for-auth-tests-32ch"


# ─────────────────────────────────────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordHashing:

    def test_hash_is_not_plaintext(self):
        from auth import hash_password
        hashed = hash_password("mypassword123")
        assert hashed != "mypassword123"
        assert len(hashed) > 20

    def test_verify_correct_password(self):
        from auth import hash_password, verify_password
        hashed = hash_password("correct-horse-battery")
        assert verify_password("correct-horse-battery", hashed) is True

    def test_verify_wrong_password(self):
        from auth import hash_password, verify_password
        hashed = hash_password("correct-horse-battery")
        assert verify_password("wrong-password", hashed) is False

    def test_two_hashes_of_same_password_differ(self):
        """bcrypt uses a random salt — same input → different hash every time."""
        from auth import hash_password
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2

    def test_both_hashes_verify_correctly(self):
        from auth import hash_password, verify_password
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert verify_password("same-password", h1) is True
        assert verify_password("same-password", h2) is True


# ─────────────────────────────────────────────────────────────────────────────
# JWT creation and decoding
# ─────────────────────────────────────────────────────────────────────────────

class TestJWT:

    def test_token_is_non_empty_string(self):
        from auth import create_access_token
        token = create_access_token("user-123", "user@example.com")
        assert isinstance(token, str)
        assert len(token) > 50

    def test_decode_recovers_user_id(self):
        from auth import create_access_token, decode_token
        token = create_access_token("user-abc", "user@example.com")
        payload = decode_token(token)
        assert payload["sub"] == "user-abc"

    def test_decode_recovers_email(self):
        from auth import create_access_token, decode_token
        token = create_access_token("user-abc", "student@example.com")
        payload = decode_token(token)
        assert payload["email"] == "student@example.com"

    def test_tampered_token_raises_401(self):
        from auth import create_access_token, decode_token
        from fastapi import HTTPException
        token = create_access_token("user-abc", "user@example.com")
        bad_token = token[:-4] + "XXXX"
        with pytest.raises(HTTPException) as exc_info:
            decode_token(bad_token)
        assert exc_info.value.status_code == 401

    def test_expired_token_raises_401(self):
        """Forge a token that expired 1 second ago."""
        from auth import _require_secret, ALGORITHM
        from jose import jwt
        from datetime import datetime, timezone, timedelta
        from fastapi import HTTPException

        payload = {
            "sub":   "user-expired",
            "email": "x@y.com",
            "exp":   datetime.now(timezone.utc) - timedelta(seconds=1),
            "iat":   datetime.now(timezone.utc) - timedelta(days=1),
        }
        expired_token = jwt.encode(payload, _require_secret(), algorithm=ALGORITHM)

        from auth import decode_token
        with pytest.raises(HTTPException) as exc_info:
            decode_token(expired_token)
        assert exc_info.value.status_code == 401

    def test_token_contains_exp_and_iat(self):
        from auth import create_access_token, decode_token
        token = create_access_token("u1", "u@e.com")
        payload = decode_token(token)
        assert "exp" in payload
        assert "iat" in payload


# ─────────────────────────────────────────────────────────────────────────────
# Password strength validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordStrength:

    def test_strong_password_returns_none(self):
        from auth import validate_password_strength
        assert validate_password_strength("Secure123!") is None

    def test_too_short_returns_error(self):
        from auth import validate_password_strength
        result = validate_password_strength("abc12")
        assert result is not None
        assert "8" in result

    def test_all_digits_returns_error(self):
        from auth import validate_password_strength
        result = validate_password_strength("12345678")
        assert result is not None

    def test_all_lowercase_no_digits_returns_error(self):
        from auth import validate_password_strength
        result = validate_password_strength("alllowercase")
        assert result is not None

    def test_lowercase_with_digit_is_acceptable(self):
        from auth import validate_password_strength
        assert validate_password_strength("lowercase1") is None

    def test_exactly_8_chars_is_acceptable(self):
        from auth import validate_password_strength
        assert validate_password_strength("Abcdef1!") is None