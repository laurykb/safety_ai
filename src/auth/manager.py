"""Auth, roles, anti-bot."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

try:
    import bcrypt
    import yaml
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Chemins par défaut
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
USERS_FILE = CONFIG_DIR / "users.yaml"

# Session (rate limiting, lockout)
_login_attempts: dict[str, list[float]] = {}
_lockout_until: dict[str, float] = {}


def _get_users_path() -> Path:
    path = os.environ.get("USERS_CONFIG_PATH")
    if path:
        return Path(path)
    return USERS_FILE


def load_users() -> dict[str, Any]:
    if not AUTH_AVAILABLE:
        return {}
    path = _get_users_path()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_captcha(expected: int, user_answer: str | int | float | None) -> bool:
    if user_answer is None:
        return False
    try:
        val = user_answer if isinstance(user_answer, (int, float)) else str(user_answer).strip()
        if val == "":
            return False
        return int(float(val)) == expected
    except (ValueError, TypeError):
        return False


def _rate_limit(username: str, max_attempts: int = 5, lockout_min: int = 15) -> bool:
    now = time.time()
    if username in _lockout_until and now < _lockout_until[username]:
        return True
    if username in _lockout_until and now >= _lockout_until[username]:
        del _lockout_until[username]
        _login_attempts[username] = []
    attempts = _login_attempts.get(username, [])
    cutoff = now - 3600
    attempts = [t for t in attempts if t > cutoff]
    if len(attempts) >= max_attempts:
        _lockout_until[username] = now + lockout_min * 60
        _login_attempts[username] = []
        return True
    return False


def authenticate(
    username: str,
    password: str,
    captcha_ok: bool = True,
) -> tuple[bool, str | None]:
    """Authentifie username/password. Retourne (ok, role) ou (False, msg_erreur)."""
    if not AUTH_AVAILABLE:
        return False, "bcrypt ou pyyaml manquant"
    cfg = load_users()
    users = cfg.get("users", {})
    bot_cfg = cfg.get("bot_check", {})
    max_attempts = bot_cfg.get("max_login_attempts", 5)
    lockout_min = bot_cfg.get("lockout_minutes", 15)
    if _rate_limit(username, max_attempts, lockout_min):
        return False, f"Trop de tentatives. Réessayez dans {lockout_min} min."
    if not captcha_ok:
        _login_attempts.setdefault(username, []).append(time.time())
        return False, "Vérification anti-robot échouée"
    u = users.get(username)
    if not u:
        _login_attempts.setdefault(username, []).append(time.time())
        return False, "Identifiants incorrects"
    h = u.get("password_hash")
    if not h or not bcrypt.checkpw(password.encode("utf-8"), h.encode("utf-8")):
        _login_attempts.setdefault(username, []).append(time.time())
        return False, "Identifiants incorrects"
    _login_attempts[username] = []
    return True, u.get("role", "user")


def get_user_role(session: dict) -> str | None:
    return session.get("user_role") if isinstance(session, dict) else None


def is_admin(session: dict) -> bool:
    return get_user_role(session) == "admin"


def make_captcha() -> tuple[int, int, str]:
    a, b = random.randint(1, 10), random.randint(1, 10)
    return a, b, f"{a} + {b}"
