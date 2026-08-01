"""Runtime configuration for Helyx.

Secrets are read from the pre-existing ``config/.env`` at the repo root and are
never copied, logged, or echoed. Everything in this module that reports on a
secret reports a boolean or a length, never a value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# helyx/src/helyx/config.py -> helyx/src/helyx -> helyx/src -> helyx -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / "config" / ".env"
VAR_DIR = Path(__file__).resolve().parents[2] / "var"

_LOADED = False


def load_env() -> None:
    """Load the shared .env exactly once. Idempotent."""
    global _LOADED
    if not _LOADED:
        load_dotenv(ENV_PATH)
        _LOADED = True


# --- LLM gateway -----------------------------------------------------------
# Empirically established: the Lambda proxy in OPENAI_BASE_URL 404s on every
# route (GET/POST, /models and /chat/completions). The live, working path is
# LiveKit's inference gateway, authenticated with a LiveKit JWT carrying
# InferenceGrants. See llm.py.
LIVEKIT_GATEWAY = "https://agent-gateway.livekit.cloud/v1"

DEFAULT_MODEL = "openai/gpt-5.6"
FALLBACK_MODEL = "openai/gpt-5.4"


@dataclass(frozen=True)
class Settings:
    """Typed view of the environment. No secret is ever rendered by __repr__."""

    model: str
    fallback_model: str
    gateway_url: str
    a1mobile_base: str
    a1mobile_from: str
    callback_email: str
    callback_number: str
    dashboard_port: int
    public_base_url: str

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Settings(model={self.model!r}, fallback_model={self.fallback_model!r}, "
            f"dashboard_port={self.dashboard_port})"
        )


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_env()
    return Settings(
        model=os.getenv("HELYX_MODEL", DEFAULT_MODEL),
        fallback_model=os.getenv("HELYX_FALLBACK_MODEL", FALLBACK_MODEL),
        gateway_url=os.getenv("HELYX_GATEWAY_URL", LIVEKIT_GATEWAY),
        a1mobile_base=os.getenv("A1MOBILE_BASE_URL", "https://hack.a1mobile.com/api"),
        a1mobile_from=os.getenv("A1MOBILE_FROM_NUMBER", ""),
        callback_email=os.getenv("CALLBACK_EMAIL", ""),
        callback_number=os.getenv("CALLBACK_NUMBER", ""),
        dashboard_port=int(os.getenv("HELYX_PORT", "8123")),
        public_base_url=os.getenv("WEBHOOK_BASE_URL", ""),
    )


def secret(name: str) -> str:
    """Fetch a secret by env name. Returns '' when absent."""
    load_env()
    return os.getenv(name, "") or ""


def credential_report() -> dict[str, dict[str, object]]:
    """Presence-only report for the dashboard. Never includes values."""
    names = [
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "A1MOBILE_TEAM_KEY",
        "OPENAI_API_KEY",
        "HELYX_SMTP_PASSWORD",
        "HELYX_IMAP_PASSWORD",
    ]
    return {n: {"present": bool(secret(n)), "length": len(secret(n))} for n in names}
