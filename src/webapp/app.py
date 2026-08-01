"""The intake web app.

    .venv/bin/python -m uvicorn src.webapp.app:app --port 8100

One page. A business owner describes a job in plain language on the left, the
brief fills in on the right, and when it is full the Launch button dials a real
phone. The panel is not decoration: it is the only way the owner sees what the
agent actually understood before it starts talking to a stranger on their
behalf.

Three properties this file is responsible for:

* **Launch is refused on an incomplete spec.** `TaskSpec.can_launch` is checked
  server-side and returns 409 with the blockers. The button being disabled in
  the browser is a courtesy; this is the rule.

* **No escalation, ever.** `TaskSpec.to_call_session()` passes
  `allow_escalation=False`, and nothing in the UI implies a human can be reached
  mid-call. The limits the owner typed here are the whole mandate.

* **The receipt is not ours to write.** `/api/call/{room}` reports what it can
  observe - whether the phone answered, and whichever receipt the worker
  process wrote into `evidence/`. It never synthesises a result from the fact
  that a call was placed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / "config" / ".env")

from src.webapp import intake
from src.webapp.dialer import Dialer, default_dialer
from src.webapp.intake import Conversation, LiveKitResponder, Responder
from src.webapp.spec import FIELDS, GROUPS, TaskSpec

logger = logging.getLogger("webapp")

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
EVIDENCE = ROOT / "evidence"

#: Where a launched brief is written so the call worker can pick it up.
#: `src/agents/run_call.py` currently configures itself from environment
#: variables, so this file plus `worker_env` in the launch response is the
#: handoff: one is machine-readable, the other is copy-pasteable right now.
HANDOFF = ROOT / "config" / "webapp.json"

app = FastAPI(title="Expeditor intake", docs_url=None, redoc_url=None)

app.state.sessions: dict[str, Conversation] = {}
app.state.launches: dict[str, dict[str, Any]] = {}
app.state.responder: Responder = LiveKitResponder()
app.state.dialer: Dialer = default_dialer()
app.state.handoff_path: Path = HANDOFF


def use_responder(replacement: Responder) -> Responder:
    """Point the intake at a different model. The seam tests pull on."""
    app.state.responder = replacement
    return replacement


def use_dialer(replacement: Dialer) -> Dialer:
    """Point launch at a different dialer. The seam that keeps tests silent."""
    app.state.dialer = replacement
    return replacement


def use_handoff(path: Path) -> Path:
    """Write the launched brief somewhere else. Tests point this at a tmpdir so
    a test run cannot overwrite a real operator's live handoff file."""
    app.state.handoff_path = path
    return path


def reset() -> None:
    app.state.sessions = {}
    app.state.launches = {}
    app.state.handoff_path = HANDOFF


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX.read_text())


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "sessions": len(app.state.sessions),
        "dialer": type(app.state.dialer).__name__,
        "model": intake.MODEL,
    }


# ---------------------------------------------------------------------------
# intake
# ---------------------------------------------------------------------------


@app.get("/api/fields")
async def field_registry() -> JSONResponse:
    """The brief's shape, so the panel and the interview cannot drift apart."""
    return JSONResponse(
        {
            "groups": [{"id": gid, "label": label} for gid, label in GROUPS],
            "fields": [
                {"id": f.id, "label": f.label, "group": f.group,
                 "kinds": list(f.kinds), "question": f.question}
                for f in FIELDS
            ],
        }
    )


@app.post("/api/session")
async def new_session() -> JSONResponse:
    convo = intake.start()
    app.state.sessions[convo.id] = convo
    return JSONResponse(convo.to_dict())


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
    convo = app.state.sessions.get(session_id)
    if convo is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    return JSONResponse(convo.to_dict())


@app.post("/api/message")
async def message(body: dict[str, Any]) -> JSONResponse:
    convo = app.state.sessions.get(str(body.get("session_id", "")))
    if convo is None:
        return JSONResponse({"error": "no such session"}, status_code=404)

    text = str(body.get("text", ""))
    if not text.strip():
        return JSONResponse({"error": "empty message"}, status_code=400)

    result = await intake.advance(convo, text, app.state.responder)
    return JSONResponse({**convo.to_dict(), "changed": result.changed,
                         "used_model": result.used_model})


@app.post("/api/spec")
async def edit_spec(body: dict[str, Any]) -> JSONResponse:
    """Direct edit of the brief, for correcting what the agent misheard.

    The panel is read-mostly, but an operator who can see a wrong number and
    cannot fix it will fix it by re-explaining it to a chatbot, which is worse.
    """
    convo = app.state.sessions.get(str(body.get("session_id", "")))
    if convo is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    changed = convo.spec.apply(body.get("patch") or {})
    return JSONResponse({**convo.to_dict(), "changed": changed})


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


@app.post("/launch")
async def launch(body: dict[str, Any]) -> JSONResponse:
    convo = app.state.sessions.get(str(body.get("session_id", "")))
    if convo is None:
        return JSONResponse({"error": "no such session"}, status_code=404)

    if convo.launched_room:
        # The disabled button is a courtesy; this is the rule. A double-click,
        # a refreshed tab or a retried fetch must not ring a second stranger.
        return JSONResponse(
            {"error": "this brief has already been launched",
             "room": convo.launched_room,
             "blockers": [f"call {convo.launched_room} is already placed"]},
            status_code=409,
        )

    spec = convo.spec
    if not spec.can_launch:
        # 409, not 400: the request is well-formed, the brief is not ready. The
        # blockers are the same strings the panel shows, so the operator does
        # not get one explanation in the UI and a different one from the API.
        return JSONResponse(
            {"error": "spec is not ready to launch", "blockers": spec.blockers(),
             "missing": spec.missing_fields()},
            status_code=409,
        )

    target = _pick_target(spec, body.get("target"))
    if target is None:
        return JSONResponse(
            {"error": "no dialable target", "blockers": spec.blockers()},
            status_code=409,
        )

    # Building the session here does two jobs: it hands the worker a validated
    # campaign, and it fails loudly *before* a phone rings if the conversion
    # produced something the engine would reject.
    session = spec.to_call_session()
    campaign_problems = session.campaign.problems()
    if campaign_problems:
        return JSONResponse(
            {"error": "brief does not convert to a valid campaign",
             "blockers": campaign_problems},
            status_code=409,
        )

    room = _room_name(target.name or target.phone)
    handoff = {
        "room": room,
        "launched_at": time.time(),
        "to": target.phone,
        "spec": spec.to_dict(),
        "campaign": session.campaign.to_dict(),
        "costs": None if spec.to_cost_model() is None else spec.to_cost_model().to_dict(),
        "allow_escalation": session.allow_escalation,
        "worker_env": _worker_env(spec),
    }
    _write_handoff(handoff)

    result = await app.state.dialer.dial(
        to_number=target.phone, room=room, metadata=json.dumps(handoff["spec"])
    )

    record = {
        **handoff,
        "dial": result.to_dict(),
        "target": target.to_dict(),
        "receipt": None,
    }
    app.state.launches[room] = record
    convo.launched_room = room
    convo.say(
        "assistant",
        f"Calling {target.name or target.phone} now. Nothing is confirmed until "
        f"written confirmation reaches {spec.confirm_to} - I will show the "
        "receipt when the call ends.",
    )

    return JSONResponse(
        {"room": room, "dial": result.to_dict(), "target": target.to_dict(),
         "campaign": session.campaign.to_dict(), "worker_env": handoff["worker_env"]},
        status_code=200 if result.ok else 502,
    )


@app.get("/api/call/{room}")
async def call_status(room: str) -> JSONResponse:
    record = app.state.launches.get(room)
    if record is None:
        return JSONResponse({"error": "no such call"}, status_code=404)
    record["receipt"] = _find_receipt(record)
    return JSONResponse(record)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pick_target(spec: TaskSpec, wanted: Any) -> Any:
    dialable = spec.dialable_targets
    if not dialable:
        return None
    if wanted:
        key = str(wanted).strip().lower()
        for t in dialable:
            if key in (t.name.strip().lower(), t.phone):
                return t
    return dialable[0]


def _room_name(seed: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", seed.lower()).strip("-")[:28] or "call"
    return f"call-{slug}-{int(time.time()) % 100000}"


def _worker_env(spec: TaskSpec) -> dict[str, str]:
    """Exactly the variables `src/agents/run_call.py` reads today.

    The campaign this app derives is not one of the three preset keys, so
    CAMPAIGN is deliberately absent: the full brief travels as dispatch
    metadata and in `config/webapp.json`. These cover the parts the worker can
    already consume without a code change.
    """
    e = spec.economics
    env = {
        "BUSINESS_NAME": spec.business_name,
        "ERRAND_TASK": spec.objective,
        "CAPACITY_TOTAL": str(spec.capacity_total or spec.max_qty or 1),
        "CAPACITY_UNIT": spec.unit_label or "units",
    }
    if e is not None and e.is_complete:
        env |= {
            "COST_MATERIALS": str(e.materials_per_unit),
            "COST_LABOR": str(e.labor_per_unit),
            "COST_TRANSPORT": str(e.transport_per_unit),
            "MIN_MARGIN_PCT": str(e.min_margin_pct),
            "TARGET_MARGIN_PCT": str(
                e.target_margin_pct if e.target_margin_pct is not None
                else e.min_margin_pct
            ),
        }
    return env


def _write_handoff(payload: dict[str, Any]) -> None:
    path: Path = app.state.handoff_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"active": payload}, indent=2, default=str))
    except OSError as exc:  # a full disk must not block a call
        logger.warning("could not write %s: %s", path, exc)


def _find_receipt(record: dict[str, Any]) -> dict[str, Any] | None:
    """The worker's receipt for this call, if it has written one yet.

    Matched on task text and start time, never fabricated. A call with no
    receipt file reports None, which the UI renders as "no receipt yet" - the
    honest state, and the one that distinguishes a call still running from a
    call that lied.
    """
    task = (record.get("spec") or {}).get("objective") or ""
    launched = float(record.get("launched_at") or 0)
    if not EVIDENCE.is_dir():
        return None
    best: dict[str, Any] | None = None
    for path in EVIDENCE.glob("receipt_*.json"):
        try:
            if path.stat().st_mtime + 5 < launched:
                continue
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if task and data.get("task") != task:
            continue
        if best is None or str(data.get("started_at", "")) > str(best.get("started_at", "")):
            best = data
    return best


if os.getenv("WEBAPP_DEBUG") == "1":  # pragma: no cover - dev convenience only
    logging.basicConfig(level=logging.INFO)
