"""The unified operations board.

    .venv/bin/python -m uvicorn src.opsboard.app:app --port 8130

One screen for a projector: where the live call is in the flow graph, every
refusal the guardrails issued, every receipt on disk with its verdict, and the
two numbers the whole pitch rests on — booked versus proven.

Routes
------
``GET  /``              the board
``GET  /api/state``     poll target: a state key plus per-panel key + HTML
``GET  /api/data``      the same snapshot as raw data, for anything else
``GET  /healthz``       liveness plus the headline totals
``POST /api/call``      push live call fields (JSON body)
``POST /api/refusal``   push one guardrail refusal (JSON body)
``POST /api/reset``     clear the live call and the ledger

The board is a view and holds no invariants. Verdicts are re-derived from
`src.verify.receipts.INDEPENDENT_CHANNELS` on every read and the derivation is
one-way pessimistic, so there is no route here — including the POSTs — that can
turn an unproven claim green.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.opsboard import render
from src.opsboard.registry import OPS, OpsRegistry
from src.opsboard.state import EVIDENCE_DIR, build

HERE = Path(__file__).parent

app = FastAPI(title="Expeditor — ops board", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

# Bound at import, not in a lifespan hook: an app whose behaviour depends on
# whether someone ran its startup is an app with two behaviours, and the one a
# test exercises would not be the one on the projector.
app.state.registry = OPS
app.state.evidence = EVIDENCE_DIR


def registry() -> OpsRegistry:
    return app.state.registry


def use_registry(replacement: OpsRegistry) -> OpsRegistry:
    """The seam tests pull on."""
    app.state.registry = replacement
    return replacement


def use_evidence(directory: Path | str) -> Path:
    app.state.evidence = Path(directory)
    return app.state.evidence


def snapshot() -> dict[str, Any]:
    return build(app.state.evidence, app.state.registry)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(render.page_html(snapshot()))


@app.get("/api/state")
def api_state() -> JSONResponse:
    """Everything the page needs to patch itself, and nothing that changes on
    its own. No wall clock goes into any key, so an idle board returns the same
    `state_key` forever and the DOM is never touched."""
    s = snapshot()
    return JSONResponse(
        {
            "state_key": s["state_key"],
            "panels": {
                name: {"key": s["keys"][name], "html": render.panel(name, data)}
                for name, data in s["panels"].items()
            },
            "totals": s["panels"]["metric"],
        }
    )


@app.get("/api/data")
def api_data() -> JSONResponse:
    return JSONResponse(snapshot())


@app.post("/api/call")
def api_call(body: dict[str, Any] = Body(default={})) -> JSONResponse:
    """Patch the live call. `{"end": true}` closes it, `{"start": true}` opens
    a fresh one and clears the ledger."""
    reg = registry()
    fields = {k: v for k, v in body.items() if k not in ("start", "end")}
    if body.get("start"):
        reg.start_call(
            business=str(fields.pop("business", "")),
            phone=str(fields.pop("phone", "")),
            task=str(fields.pop("task", "")),
        )
    if fields:
        reg.update_call(**fields)
    if body.get("end"):
        reg.end_call()
    return JSONResponse({"ok": True, "call": reg.snapshot()["call"]})


@app.post("/api/refusal")
def api_refusal(body: dict[str, Any] = Body(default={})) -> JSONResponse:
    """Either `{"raw": "BLOCKED. ..."}` to have the gate's own words phrased for
    the projector, or `{"headline": "refused 600 units - capacity is 400"}`."""
    reg = registry()
    kind = str(body.get("kind", "gate"))
    phase = body.get("phase")
    if raw := body.get("raw"):
        r = reg.gate_refusal(str(raw), phase=phase, kind=kind)
    elif headline := body.get("headline"):
        r = reg.refusal(
            str(headline), detail=str(body.get("detail", "")), kind=kind, phase=phase
        )
    else:
        return JSONResponse({"ok": False, "error": "need raw or headline"}, status_code=400)
    return JSONResponse({"ok": True, "refusal": r.to_dict() if r else None})


@app.post("/api/reset")
def api_reset() -> JSONResponse:
    registry().reset()
    return JSONResponse({"ok": True})


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    s = snapshot()
    return {"ok": True, "state_key": s["state_key"], "totals": s["panels"]["metric"]}


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------


def seed(reg: OpsRegistry | None = None) -> OpsRegistry:
    """The $74 call, mid-flight. Labelled FIXTURE DATA on screen for as long as
    it is held — an unlabelled fixture on a projector is a fabricated success
    wearing a better suit."""
    reg = reg if reg is not None else registry()
    reg.start_call(
        business="Golden Crumb Bakery",
        phone="+1 415 555 0142",
        task="Sell 200 pastries into a Friday morning event",
    )
    reg.update_call(
        fixture=True,
        phase="negotiating",
        units=200,
        unit_label="pastries",
        capacity_total=400,
        capacity_held=200,
        quote=400.0,
        floor=385.72,
        budget=280.0,
    )
    reg.gate_refusal(
        "BLOCKED. Before reserving 600, confirm whether 600 is the number of ITEMS "
        "or the number of PEOPLE. If they gave a headcount, ask how many items each.",
        phase="discovery",
    )
    reg.refusal(
        "refused 600 units — capacity is 400",
        detail="CapacityLedger: 400 available, 200 already held; 600 would oversell by 400.",
        kind="capacity",
        phase="discovery",
    )
    reg.gate_refusal(
        "BLOCKED - too early to quote. Still needed: what they pay now (ask before "
        "you name a price). Do not name a price yet.",
        phase="discovery",
    )
    reg.gate_refusal(
        "BLOCKED. They already offered 385.00; quoting 74.00 discards 311.00. "
        "Offer 385.00 or more.",
        phase="qualified",
    )
    reg.refusal(
        "refused below floor $385.72",
        detail="CostModel.validate_quote: 280.00 clamped to floor 385.72 for 200 units.",
        kind="pricing",
        phase="negotiating",
    )
    return reg


if os.environ.get("OPSBOARD_SEED", "0") == "1":
    seed(OPS)
