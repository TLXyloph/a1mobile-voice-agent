"""Keep the judge link alive.

Cloudflare quick tunnels die - the process drops, the URL 404s, and the link a
judge is holding stops working with no warning. This watches every 20 seconds
and, when the tunnel is gone, brings up a new one, writes the new URL to
config/.env, and re-points a1mobile's webhook at it.

The URL necessarily changes on a restart (quick tunnels are randomly named), so
the current one is always written to evidence/JUDGE_LINK.txt. Check that file
before handing the link to anyone.

    .venv/bin/python scripts/keep_demo_up.py        # foreground, ctrl-C to stop
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "config" / ".env")

import os  # noqa: E402

ENV = ROOT / "config" / ".env"
LINK = ROOT / "evidence" / "JUDGE_LINK.txt"
CF_LOG = Path("/tmp/cf_watchdog.log")
CHECK_EVERY = 20
G, R, Y, X = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def current_url() -> str:
    return os.getenv("WEBHOOK_BASE_URL", "").rstrip("/")


def alive(url: str) -> bool:
    if not url:
        return False
    try:
        return httpx.get(f"{url}/demo", timeout=12).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def local_ok() -> bool:
    """Is the thing behind the tunnel even running? Restarting the tunnel when
    uvicorn is down just produces a working tunnel to a dead port."""
    try:
        return httpx.get("http://127.0.0.1:8080/demo", timeout=5).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def save(url: str) -> None:
    text = ENV.read_text()
    if re.search(r"^WEBHOOK_BASE_URL=", text, re.M):
        text = re.sub(r"^WEBHOOK_BASE_URL=.*$", f"WEBHOOK_BASE_URL={url}", text, flags=re.M)
    else:
        text += f"\nWEBHOOK_BASE_URL={url}\n"
    ENV.write_text(text)
    os.environ["WEBHOOK_BASE_URL"] = url
    LINK.parent.mkdir(parents=True, exist_ok=True)
    LINK.write_text(f"{url}/demo\n\nupdated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def repoint(url: str) -> None:
    key = os.getenv("A1MOBILE_TEAM_KEY", "")
    if not key:
        return
    try:
        httpx.post(f"{os.getenv('A1MOBILE_BASE_URL', 'https://hack.a1mobile.com/api')}"
                   "/numbers/point",
                   headers={"X-Team-Key": key, "Content-Type": "application/json"},
                   json={"webhook_url": f"{url}/texml/inbound"}, timeout=20)
        print(f"  {stamp()} re-pointed a1mobile webhook")
    except Exception as exc:  # noqa: BLE001
        print(f"  {stamp()} {R}could not re-point a1mobile: {exc}{X}")


def restart_tunnel() -> str | None:
    subprocess.run(["pkill", "-f", "cloudflared tunnel"], capture_output=True)
    time.sleep(2)
    CF_LOG.write_text("")
    subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8080",
         "--protocol", "http2", "--no-autoupdate"],
        stdout=CF_LOG.open("a"), stderr=subprocess.STDOUT,
    )
    for _ in range(30):
        time.sleep(2)
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", CF_LOG.read_text())
        if m:
            url = m.group(0)
            # Registered is not the same as routable; wait for a real 200.
            for _ in range(15):
                if alive(url):
                    return url
                time.sleep(2)
            return url
    return None


def main() -> int:
    print(f"{Y}Watching the judge link. Ctrl-C to stop.{X}")
    url = current_url()
    if alive(url):
        save(url)
        print(f"  {stamp()} {G}up{X}  {url}/demo")

    while True:
        try:
            url = current_url()
            if alive(url):
                time.sleep(CHECK_EVERY)
                continue

            if not local_ok():
                print(f"  {stamp()} {R}port 8080 is down - fix uvicorn first, "
                      f"not the tunnel{X}")
                time.sleep(CHECK_EVERY)
                continue

            print(f"  {stamp()} {R}tunnel down - restarting{X}")
            new = restart_tunnel()
            if not new:
                print(f"  {stamp()} {R}could not bring up a tunnel; retrying{X}")
                time.sleep(CHECK_EVERY)
                continue

            save(new)
            repoint(new)
            print(f"  {stamp()} {G}NEW LINK{X}  {new}/demo")
            print(f"  {Y}>>> the URL changed - re-share it <<<{X}")
        except KeyboardInterrupt:
            print("\nstopped")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"  {stamp()} watchdog error: {exc}")
            time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    raise SystemExit(main())
