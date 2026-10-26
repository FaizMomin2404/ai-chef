"""Runtime configuration for AI Chef.

Defaults preserve the original local Windows experience. Deployment settings
are opt-in through environment variables, documented in .env.example.
"""

from __future__ import annotations

import os
from typing import List

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env_file() -> None:
    """Load a local .env without adding a dependency.

    Hosting platforms supply real environment variables; this only fills values
    that are not already set. The file is git-ignored and never served.
    """
    path = os.path.join(_BASE_DIR, ".env")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_env_file()


def _bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on", "y")


def _int(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _mode(name: str, default: str, allowed: set) -> str:
    value = (os.environ.get(name) or default).strip().lower()
    return value if value in allowed else default


# --- speech ---------------------------------------------------------------
# auto: use local engines when they load, otherwise let the browser do it.
STT_MODE = _mode("STT_MODE", "auto", {"auto", "local", "browser"})
TTS_MODE = _mode("TTS_MODE", "auto", {"auto", "server", "browser"})
TTS_ENABLED = _bool("TTS_ENABLED", True)
PIPER_DIR = os.environ.get("PIPER_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper_models"))
PIPER_VOICE = os.environ.get("PIPER_VOICE", "en_US-amy-medium")

# --- sessions -------------------------------------------------------------
# Multi-user creates isolated session/timer state per WebSocket. Enable it on
# any shared/public deployment; keep it off locally for session resume.
MULTI_USER = _bool("MULTI_USER", False)
SESSION_PERSISTENCE = _bool("SESSION_PERSISTENCE", not MULTI_USER)
MAX_CONNECTIONS = max(1, _int("MAX_CONNECTIONS", 50))
MAX_MESSAGE_CHARS = max(80, _int("MAX_MESSAGE_CHARS", 600))

# --- HTTP -----------------------------------------------------------------
HOST = (os.environ.get("HOST") or "127.0.0.1").strip()
PORT = _int("PORT", 8000)
DEPLOYMENT_MODE = _bool("DEPLOYMENT_MODE", bool(os.environ.get("RENDER")))

_origins = (os.environ.get("ALLOWED_ORIGINS") or "").strip()
if _origins:
    ALLOWED_ORIGINS: List[str] = [o.strip() for o in _origins.split(",") if o.strip()]
else:
    # Same-origin requests need no CORS entry. These are development origins.
    ALLOWED_ORIGINS = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
