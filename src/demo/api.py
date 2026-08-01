"""Judge-facing demo: see the agent's constraints, then make it call you.

Served from the public tunnel so a judge only needs a link. The same endpoints
back a static Vercel page if one is deployed later - hence the permissive CORS
and the fact that nothing here reads from the request origin.

The constraints shown are read from the live config the agent will actually use,
not hardcoded. A demo page that displays a floor the agent does not enforce is
the same category of lie this whole project exists to prevent.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
router = APIRouter()

A1_API = os.getenv("A1MOBILE_BASE_URL", "https://hack.a1mobile.com/api")


def _hdrs() -> dict[str, str]:
    return {"X-Team-Key": os.getenv("A1MOBILE_TEAM_KEY", ""),
            "Content-Type": "application/json"}


def _cors(payload: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    })


def normalise(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if (raw or "").strip().startswith("+") and 8 <= len(digits) <= 15:
        return f"+{digits}"
    return None


def constraints() -> dict[str, Any]:
    """Read the real limits the agent will run under."""
    from src.agents.run_call import build_call_session

    s = build_call_session()
    qty = 200
    return {
        "business": os.getenv("BUSINESS_NAME", "Rosewater Bakehouse"),
        "persona": "Sam, calling on behalf of the bakery",
        "offer": s.campaign.offer,
        "unit": s.ledger.unit,
        "capacity_total": s.ledger.total,
        "capacity_available": s.ledger.available(),
        "unit_cost": float(s.costs.unit_cost),
        "sample_qty": qty,
        "floor": float(s.costs.floor_price(qty)),
        "target": float(s.costs.target_price(qty)),
        "min_margin_pct": float(s.costs.min_margin_pct),
        "target_margin_pct": float(s.costs.target_margin_pct or 0),
        "max_discount_pct": s.campaign.envelope.max_discount_pct,
        "max_concessions": 1,
        "max_declines": 2,
        "escalation": False,
    }


@router.get("/demo/constraints")
async def get_constraints() -> JSONResponse:
    try:
        return _cors(constraints())
    except Exception as exc:  # noqa: BLE001
        return _cors({"error": str(exc)}, 500)


@router.options("/demo/{rest:path}")
async def preflight(rest: str) -> JSONResponse:
    return _cors({"ok": True})


@router.post("/demo/verify")
async def verify(request: Request) -> JSONResponse:
    """Send the OTP. a1mobile only dials numbers that have passed this."""
    body = await request.json()
    phone = normalise(body.get("phone", ""))
    if not phone:
        return _cors({"ok": False, "error": "That doesn't look like a phone number."}, 400)

    async with httpx.AsyncClient(timeout=25) as c:
        try:
            existing = (await c.get(f"{A1_API}/verified-numbers", headers=_hdrs())).json()
            if phone in (existing.get("verified_numbers") or []):
                return _cors({"ok": True, "phone": phone, "already_verified": True})
        except Exception:  # noqa: BLE001
            pass
        r = await c.post(f"{A1_API}/verified-numbers", headers=_hdrs(), json={"phone": phone})
        if r.status_code >= 300:
            return _cors({"ok": False, "error": r.text[:200]}, 400)
    return _cors({"ok": True, "phone": phone, "already_verified": False})


@router.post("/demo/confirm")
async def confirm(request: Request) -> JSONResponse:
    body = await request.json()
    phone = normalise(body.get("phone", ""))
    code = re.sub(r"\D", "", body.get("code", ""))
    if not phone or not code:
        return _cors({"ok": False, "error": "Need both a number and the code."}, 400)

    async with httpx.AsyncClient(timeout=25) as c:
        r = await c.post(f"{A1_API}/verified-numbers/confirm", headers=_hdrs(),
                         json={"phone": phone, "code": code})
    ok = r.status_code < 300 and r.json().get("verified") is True
    return _cors({"ok": ok, "error": None if ok else "That code didn't match."},
                 200 if ok else 400)


@router.post("/demo/call")
async def call(request: Request) -> JSONResponse:
    """Place the call. Fires and returns - the page polls for state."""
    body = await request.json()
    phone = normalise(body.get("phone", ""))
    if not phone:
        return _cors({"ok": False, "error": "Bad number."}, 400)

    room = f"judge-{phone.lstrip('+')}"
    try:
        subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/place_call.py"),
             phone, "--room", room],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # noqa: BLE001
        return _cors({"ok": False, "error": str(exc)}, 500)
    return _cors({"ok": True, "phone": phone, "room": room})


@router.get("/demo/receipts")
async def receipts() -> JSONResponse:
    """Latest receipts, so the judge watches the verdict rather than hearing it."""
    import json

    out = []
    files = sorted((ROOT / "evidence").glob("receipt_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[:40]:
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not d.get("claims"):
            continue
        out.append({
            "id": d["id"], "headline": d["headline"], "task": d.get("task", ""),
            "claims": [{
                "description": c["description"],
                "verdict": c["verdict"],
                "expected": c.get("expected_side_effect", ""),
                "evidence": [{"channel": e["channel"], "summary": e["summary"][:120],
                              "independent": e.get("independent", False)}
                             for e in c.get("evidence", [])],
            } for c in d["claims"]],
        })
        if len(out) >= 6:
            break
    return _cors({"receipts": out})


@router.get("/demo")
async def page() -> HTMLResponse:
    html = (Path(__file__).parent / "index.html").read_text()
    return HTMLResponse(html)
