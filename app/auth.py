import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from app.config import settings


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def hash_password(password: str, rounds: int = 200_000) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), rounds)
    return f"pbkdf2_sha256${rounds}${salt}${dk.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, rounds, salt, digest = encoded.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), int(rounds))
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:
        return False


def create_access_token(user_id: str, role: str) -> str:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + int(settings.auth_token_exp_minutes) * 60,
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_b64 = _b64url_encode(raw)
    sig = hmac.new(settings.auth_secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def decode_access_token(token: str) -> dict[str, Any]:
    if not token or "." not in token:
        raise ValueError("Invalid token")
    payload_b64, sig_b64 = token.split(".", 1)
    expected = hmac.new(settings.auth_secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    got = _b64url_decode(sig_b64)
    if not hmac.compare_digest(expected, got):
        raise ValueError("Invalid token signature")
    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    now = int(time.time())
    if int(payload.get("exp", 0)) < now:
        raise ValueError("Token expired")
    if not payload.get("sub") or not payload.get("role"):
        raise ValueError("Invalid token payload")
    return payload

