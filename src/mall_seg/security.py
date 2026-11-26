"""Security primitives: scrypt password hashing, pinned HS256 JWT tokens, and RBAC helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Literal

RoleName = Literal["admin", "analyst", "viewer"]
VALID_ROLES: tuple[RoleName, ...] = ("admin", "analyst", "viewer")

# In-memory ephemeral fallback when JWT_SECRET is not provided in environment.
_EPHEMERAL_JWT_SECRET = secrets.token_hex(32)


def get_jwt_secret() -> bytes:
    """Resolve JWT signing secret from environment or ephemeral in-memory secret."""
    configured = os.getenv("JWT_SECRET", "").strip()
    secret = configured if configured else _EPHEMERAL_JWT_SECRET
    return secret.encode("utf-8")


def hash_password(password: str) -> str:
    """Hash a plaintext password using salted scrypt (N=16384, r=8, p=1)."""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1, dklen=32)
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
    dk_b64 = base64.urlsafe_b64encode(derived).decode("ascii")
    return f"scrypt$16384$8$1${salt_b64}${dk_b64}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored scrypt hash using constant-time comparison."""
    try:
        scheme, n_str, r_str, p_str, salt_b64, dk_b64 = stored_hash.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(dk_b64.encode("ascii"))
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n_str),
            r=int(r_str),
            p=int(p_str),
            dklen=len(expected),
        )
        return hmac.compare_digest(derived, expected)
    except Exception:
        return False


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * ((4 - len(segment) % 4) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def create_access_token(
    *,
    user_id: int,
    email: str,
    role: str,
    expires_in_seconds: int = 1800,
) -> str:
    """Mint a short-lived HS256 JWT access token with pinned header algorithm."""
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    signing_input = (
        f"{_b64url_encode(json.dumps(header, separators=(',', ':')).encode('utf-8'))}."
        f"{_b64url_encode(json.dumps(payload, separators=(',', ':')).encode('utf-8'))}"
    )
    sig = hmac.new(get_jwt_secret(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(sig)}"


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Verify HS256 signature, algorithm pinning, and expiration cleanly."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        if not isinstance(header, dict) or header.get("alg") != "HS256":
            return None
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(get_jwt_secret(), signing_input, hashlib.sha256).digest()
        provided_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, provided_sig):
            return None
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("type") != "access":
            return None
        exp = payload.get("exp")
        if not isinstance(exp, int | float) or int(time.time()) >= int(exp):
            return None
        return payload
    except Exception:
        return None


def generate_refresh_token() -> str:
    """Generate a high-entropy opaque refresh token."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    """Hash a refresh token deterministically via SHA-256 before DB storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
