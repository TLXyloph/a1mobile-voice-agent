"""Empirically determine which model ids answer, through which endpoint.

Reports only what actually responded. Nothing here is assumed.

    .venv/bin/python helyx/scripts/probe_models.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helyx.config import secret, settings  # noqa: E402
from helyx.llm import probe_models  # noqa: E402

CANDIDATES = [
    "openai/gpt-5.6",
    "gpt-5.6",
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-4o",
]


def probe_lambda() -> None:
    """The base URL configured in config/.env, for the record."""
    base = (os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    if not base:
        print("OPENAI_BASE_URL is unset")
        return
    root = base.rsplit("/openai/v1", 1)[0]
    key = secret("OPENAI_API_KEY")
    print(f"\nLambda proxy from OPENAI_BASE_URL (host redacted): .../{base.split('/')[-3:][0]}")
    for path, method in [
        ("/openai/v1/models", "GET"),
        ("/v1/models", "GET"),
        ("/openai/v1/chat/completions", "POST"),
        ("/v1/chat/completions", "POST"),
    ]:
        url = root + path
        data = (
            json.dumps(
                {"model": "openai/gpt-5.4", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()
            if method == "POST"
            else None
        )
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                print(f"  {method:4s} {path:32s} -> {r.status}")
        except urllib.error.HTTPError as e:
            print(f"  {method:4s} {path:32s} -> {e.code} {e.read()[:80].decode('utf-8','replace')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {method:4s} {path:32s} -> {type(e).__name__}")


def main() -> int:
    cfg = settings()
    print(f"Helyx model      : {cfg.model}")
    print(f"Helyx fallback   : {cfg.fallback_model}")
    print(f"Gateway          : {cfg.gateway_url}")
    print(f"LIVEKIT creds set: {bool(secret('LIVEKIT_API_KEY') and secret('LIVEKIT_API_SECRET'))}")

    print("\nLiveKit inference gateway:")
    results = probe_models(CANDIDATES)
    for model, r in results.items():
        if r.get("ok"):
            print(f"  OK   {model:22s} -> served as {r['served_model']!r}, replied {r['reply']!r}")
        else:
            print(f"  FAIL {model:22s} -> {r.get('error')} {str(r.get('body', ''))[:80]}")

    probe_lambda()

    working = [m for m, r in results.items() if r.get("ok")]
    print(f"\n{len(working)}/{len(CANDIDATES)} model ids answered on the LiveKit gateway.")
    return 0 if cfg.model in working or cfg.fallback_model in working else 1


if __name__ == "__main__":
    raise SystemExit(main())
