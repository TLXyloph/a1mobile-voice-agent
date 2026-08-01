"""Check every credential and model is actually live.

Run this the moment you sit down, and again after you swap in a1mobile's kit.
It makes real calls against each provider - a key that is present but revoked
looks identical to a working one until you try it, and finding that out during
judging is expensive.

    uv run scripts/preflight.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "config" / ".env")
load_dotenv(ROOT / ".env")

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def line(ok: bool | None, label: str, detail: str = "") -> None:
    mark = {True: f"{GREEN}PASS{RESET}", False: f"{RED}FAIL{RESET}", None: f"{YELLOW}SKIP{RESET}"}[ok]
    print(f"  [{mark}] {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


async def check_openai() -> bool | None:
    if not os.getenv("OPENAI_API_KEY"):
        line(None, "OpenAI", "OPENAI_API_KEY unset")
        return None
    try:
        from openai import AsyncOpenAI

        models = await AsyncOpenAI().models.list()
        has_realtime = any("realtime" in m.id for m in models.data)
        line(True, "OpenAI", f"{len(models.data)} models"
             + ("" if has_realtime else "  (no realtime model visible!)"))
        return True
    except Exception as exc:  # noqa: BLE001
        line(False, "OpenAI", str(exc)[:90])
        return False


async def check_anthropic() -> bool | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        line(None, "Anthropic", "ANTHROPIC_API_KEY unset (Claude Code sub != API key)")
        return None
    try:
        from anthropic import AsyncAnthropic

        await AsyncAnthropic().messages.create(
            model="claude-sonnet-4-5", max_tokens=4, messages=[{"role": "user", "content": "hi"}]
        )
        line(True, "Anthropic", "key accepted")
        return True
    except Exception as exc:  # noqa: BLE001
        line(False, "Anthropic", str(exc)[:90])
        return False


async def check_livekit() -> bool | None:
    missing = [k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
               if not os.getenv(k)]
    if missing:
        line(None, "LiveKit", f"unset: {', '.join(missing)}  (run: lk cloud auth)")
        return None
    try:
        from livekit import api

        lkapi = api.LiveKitAPI()
        try:
            rooms = await lkapi.room.list_rooms(api.ListRoomsRequest())
            trunk = os.getenv("LIVEKIT_SIP_TRUNK_ID")
            line(True, "LiveKit", f"{len(rooms.rooms)} rooms"
                 + (f", trunk {trunk}" if trunk else ", NO SIP TRUNK - cannot call out"))
            return True
        finally:
            await lkapi.aclose()
    except Exception as exc:  # noqa: BLE001
        line(False, "LiveKit", str(exc)[:90])
        return False


async def check_telephony() -> bool | None:
    from src.tools.telephony import get_provider

    provider = get_provider()
    ok, detail = await provider.preflight()
    line(ok, f"Telephony ({provider.name})", detail)
    return ok  # None = deferred, and `failures` below only counts False


def check_models() -> bool:
    """Confirm the offline models really are cached, not just installed."""
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
        repos = {r.repo_id for r in cache.repos}
    except Exception:  # noqa: BLE001
        repos = set()

    ok = True
    whisper = any("whisper" in r for r in repos)
    turn = any("turn-detector" in r or "livekit" in r for r in repos)
    line(whisper, "Whisper (local transcription)", "cached" if whisper else "run scripts/warm_models.py")
    line(turn, "Turn detector", "cached" if turn else "run scripts/warm_models.py")
    ok = whisper and turn

    try:
        from livekit.plugins import silero  # noqa: F401
        line(True, "Silero VAD", "importable")
    except Exception as exc:  # noqa: BLE001
        line(False, "Silero VAD", str(exc)[:60])
        ok = False
    return ok


def check_verification_setup() -> bool:
    """The verification path is worth more points than the call. Check it hard."""
    ok = True
    cb = os.getenv("CALLBACK_NUMBER")
    line(bool(cb), "Callback number", cb or "CALLBACK_NUMBER unset - nothing can be verified!")
    ok &= bool(cb)

    hook = os.getenv("WEBHOOK_BASE_URL")
    line(bool(hook), "Webhook base URL", hook or "unset - run: ngrok http 8080")
    return ok


def check_tools() -> None:
    import shutil

    for tool, why in [
        ("lk", "LiveKit CLI"),
        ("ngrok", "webhook tunnel"),
        ("ffmpeg", "audio conversion"),
        ("sox", "audio inspection"),
    ]:
        path = shutil.which(tool)
        line(bool(path), f"{tool} ({why})", path or "missing")


async def main() -> int:
    print("\n=== CREDENTIALS ===")
    results = [
        await check_openai(),
        await check_anthropic(),
        await check_livekit(),
        await check_telephony(),
    ]

    print("\n=== OFFLINE MODELS ===")
    models_ok = check_models()

    print("\n=== VERIFICATION PATH ===")
    verify_ok = check_verification_setup()

    print("\n=== CLI TOOLS ===")
    check_tools()

    failures = [r for r in results if r is False]
    print()
    if failures:
        print(f"{RED}{len(failures)} credential check(s) FAILED - fix before calling.{RESET}")
        return 1
    if not models_ok:
        print(f"{YELLOW}Models not fully cached - run scripts/warm_models.py{RESET}")
        return 1
    if not verify_ok:
        print(f"{YELLOW}Callable, but nothing can be VERIFIED yet. Fix the verification path.{RESET}")
        return 0
    print(f"{GREEN}Ready.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
