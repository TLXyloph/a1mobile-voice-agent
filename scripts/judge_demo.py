"""Hand the phone to a judge: verify their number, then call it.

a1mobile's sandbox only permits calling numbers that have passed an OTP check,
so a judge cannot simply be dialed. This wraps the whole dance - send code,
confirm code, place call - into one prompt-driven flow that takes about a
minute, so the demo is "what's your number?" rather than four curl commands
typed under pressure.

    .venv/bin/python scripts/judge_demo.py            # prompts for the number
    .venv/bin/python scripts/judge_demo.py +14155550142
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "config" / ".env")

import os  # noqa: E402

API = os.getenv("A1MOBILE_BASE_URL", "https://hack.a1mobile.com/api")
KEY = os.getenv("A1MOBILE_TEAM_KEY", "")
H = {"X-Team-Key": KEY, "Content-Type": "application/json"}

G, R, Y, DIM, X = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def normalise(raw: str) -> str | None:
    """Accept what a human says out loud; return E.164 or None."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if raw.strip().startswith("+") and 8 <= len(digits) <= 15:
        return f"+{digits}"
    return None


def already_verified(phone: str) -> bool:
    try:
        r = httpx.get(f"{API}/verified-numbers", headers=H, timeout=15)
        return phone in (r.json().get("verified_numbers") or [])
    except Exception:  # noqa: BLE001
        return False


def send_code(phone: str) -> bool:
    r = httpx.post(f"{API}/verified-numbers", headers=H, json={"phone": phone}, timeout=25)
    ok = r.status_code < 300
    print(f"  {'sent' if ok else 'FAILED'}: {r.text[:160]}")
    return ok


def confirm(phone: str, code: str) -> bool:
    r = httpx.post(f"{API}/verified-numbers/confirm", headers=H,
                   json={"phone": phone, "code": code}, timeout=25)
    ok = r.status_code < 300 and r.json().get("verified") is True
    print(f"  {'verified' if ok else 'FAILED'}: {r.text[:160]}")
    return ok


def place(phone: str) -> int:
    room = f"judge-{phone.lstrip('+')}"
    return subprocess.call(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/place_call.py"),
         phone, "--room", room],
        cwd=ROOT,
    )


def main() -> int:
    if not KEY:
        print(f"{R}A1MOBILE_TEAM_KEY is unset.{X}")
        return 1

    ap = argparse.ArgumentParser()
    ap.add_argument("phone", nargs="?")
    args = ap.parse_args()

    raw = args.phone or input(f"\n{Y}Judge's mobile number:{X} ")
    phone = normalise(raw)
    if phone is None:
        print(f"{R}Could not read '{raw}' as a phone number.{X}")
        return 1
    print(f"  number: {phone}")

    if already_verified(phone):
        print(f"  {G}already verified - skipping OTP{X}")
    else:
        print(f"\n{Y}Sending a 5-digit code to their phone...{X}")
        if not send_code(phone):
            return 1
        code = input(f"\n{Y}Ask them to read the code, type it here:{X} ").strip()
        code = re.sub(r"\D", "", code)
        if not confirm(phone, code):
            print(f"{R}Verification failed. Re-run to send a fresh code.{X}")
            return 1

    print(f"\n{Y}Calling...{X}  {DIM}(the worker must be running: "
          f"python -m src.agents.run_call dev){X}")
    rc = place(phone)
    if rc == 0:
        print(f"\n{G}Connected.{X} Watch the board at http://127.0.0.1:8130")
        print(f"{DIM}The receipt lands in evidence/ when the call ends.{X}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
