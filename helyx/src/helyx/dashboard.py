"""Operator dashboard: live state, intake, call control, and the SMS webhook.

Served on ``HELYX_PORT`` (default 8123 -- 8080 and 8095 belong to other
processes and are left alone).

The dashboard is deliberately blunt about provenance. Every proposal is
rendered with its status *and the reason for that status*, and agent-sourced
evidence is visually separated from independent evidence. An operator glancing
at it should never have to wonder whether "confirmed" means the agent said so.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import credential_report, settings
from .email_loop import run_email_loop
from .intake import IntakeAgent
from .llm import LLMClient, probe_models
from .negotiator import Negotiation, Negotiator
from .sms import SMSError, confirmation_request, normalise_inbound, send_sms
from .store import STORE

logger = logging.getLogger("helyx.dashboard")

TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"

app = FastAPI(title="Helyx", docs_url=None, redoc_url=None)

_llm = LLMClient()
_intake_agent = IntakeAgent(_llm)
_negotiator = Negotiator(_llm)


# --- request models (validation at the boundary) ---------------------------


class IntakeTurn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class SupplierTurn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class HumanReview(BaseModel):
    body: str = Field(min_length=1, max_length=2000)
    supports: bool = True


class SMSConfirm(BaseModel):
    to: str = Field(min_length=2, max_length=32)


# --- pages -----------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(TEMPLATE.read_text())


@app.get("/api/state")
async def state() -> JSONResponse:
    return JSONResponse(STORE.snapshot())


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "port": settings().dashboard_port,
            "model": settings().model,
            "fallback_model": settings().fallback_model,
            "gateway": settings().gateway_url,
            "credentials": credential_report(),
        }
    )


@app.get("/api/stream")
async def stream(request: Request) -> StreamingResponse:
    """Server-sent events. The dashboard never polls."""
    queue = STORE.subscribe()

    async def gen() -> Any:
        try:
            yield f"data: {json.dumps({'event': None, 'state': STORE.snapshot()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {message}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            STORE.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- intake ----------------------------------------------------------------


@app.post("/api/intake")
async def intake(turn: IntakeTurn) -> JSONResponse:
    await asyncio.to_thread(_intake_agent.turn, STORE.intake, turn.text)
    STORE.emit(
        "intake.turn",
        text=turn.text[:280],
        missing=STORE.intake.missing,
        ready=STORE.intake.ready,
    )
    return JSONResponse(STORE.intake.to_dict())


@app.post("/api/intake/reset")
async def intake_reset() -> JSONResponse:
    from .intake import IntakeSession

    STORE.intake = IntakeSession()
    STORE.emit("intake.reset")
    return JSONResponse(STORE.intake.to_dict())


# --- call control ----------------------------------------------------------


@app.post("/api/call/start")
async def call_start() -> JSONResponse:
    """Blocked unless intake validates. `ready` is computed, not asserted."""
    if not STORE.intake.ready:
        return JSONResponse(
            {
                "error": "intake incomplete",
                "missing": STORE.intake.missing,
                "validation_error": STORE.intake.validation_error(),
            },
            status_code=422,
        )
    mandate = STORE.intake.mandate()
    STORE.negotiation = Negotiation(mandate=mandate)
    STORE.call_active = True
    opening = _negotiator.opening_line(STORE.negotiation)
    STORE.emit(
        "call.started",
        counterparty=mandate.counterparty_name,
        ladder=STORE.negotiation.guard.ladder.schedule(),
        opening_line=opening,
    )
    return JSONResponse({"opening_line": opening, "state": STORE.snapshot()})


@app.post("/api/call/say")
async def call_say(turn: SupplierTurn) -> JSONResponse:
    """Feed one supplier utterance into the negotiation."""
    neg = STORE.negotiation
    if neg is None:
        return JSONResponse({"error": "no active negotiation"}, status_code=409)
    if neg.finished:
        return JSONResponse(
            {"error": f"negotiation already {neg.outcome.value}"}, status_code=409
        )

    result = await asyncio.to_thread(_negotiator.turn, neg, turn.text)
    STORE.emit("call.turn", **result.to_dict())
    if neg.finished:
        STORE.call_active = False
        STORE.emit("call.outcome", outcome=neg.outcome.value)
    return JSONResponse({"turn": result.to_dict(), "state": STORE.snapshot()})


@app.post("/api/call/end")
async def call_end() -> JSONResponse:
    """Hang up, then run the email loop: check inbound, then report to operator."""
    STORE.call_active = False
    STORE.emit("call.ended")
    record = await asyncio.to_thread(run_email_loop, STORE)
    receipt = STORE.write_receipt()
    return JSONResponse(
        {"email": record, "receipt": str(receipt), "state": STORE.snapshot()}
    )


# --- independent channels --------------------------------------------------


@app.post("/api/sms/confirm")
async def sms_confirm(req: SMSConfirm) -> JSONResponse:
    """Text the supplier asking them to restate the terms in writing.

    This is what opens the independent channel: their reply, not our message,
    is what can confirm anything.
    """
    neg = STORE.negotiation
    proposal = STORE.proposals[-1] if STORE.proposals else None
    if neg is None or proposal is None:
        return JSONResponse({"error": "nothing proposed yet"}, status_code=409)

    body = confirmation_request(
        proposal.terms.quantity,
        proposal.terms.item,
        proposal.terms.unit_price_cents,
        proposal.terms.fulfilment_date,
    )
    try:
        result = await asyncio.to_thread(send_sms, req.to, body)
    except SMSError as exc:
        STORE.emit("sms.error", detail=str(exc)[:300])
        return JSONResponse({"error": str(exc)}, status_code=502)

    STORE.sms_log.append({"direction": "out", "to": req.to, "text": body, **result.to_dict()})
    STORE.emit("sms.outbound", to=req.to, **result.to_dict())
    return JSONResponse({"result": result.to_dict(), "body": body})


@app.post("/webhooks/sms")
async def sms_webhook(request: Request) -> JSONResponse:
    """a1mobile inbound SMS. This is where UNCONFIRMED can become CONFIRMED."""
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "payload must be an object"}, status_code=400)

    try:
        msg = normalise_inbound(payload)
    except SMSError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    result = STORE.attach_inbound_sms(msg)
    return JSONResponse({"ok": True, **result})


@app.post("/api/evidence/human")
async def human_evidence(review: HumanReview) -> JSONResponse:
    confirmed = STORE.add_human_review(review.body, review.supports)
    return JSONResponse({"confirmed": confirmed, "state": STORE.snapshot()})


# --- models ----------------------------------------------------------------


@app.get("/api/models")
async def models() -> JSONResponse:
    """Live probe of which model ids actually answer. No claims, just results."""
    cfg = settings()
    candidates = [cfg.model, cfg.fallback_model]
    report = await asyncio.to_thread(probe_models, candidates)
    STORE.model_report = {
        "configured": cfg.model,
        "fallback": cfg.fallback_model,
        "gateway": cfg.gateway_url,
        "results": report,
    }
    STORE.emit("models.probed", **STORE.model_report)
    return JSONResponse(STORE.model_report)


@app.post("/api/reset")
async def reset() -> JSONResponse:
    from .store import HelyxStore

    globals()["STORE"].__init__()  # type: ignore[misc]
    STORE.emit("session.reset")
    return JSONResponse(STORE.snapshot())


def main() -> None:  # pragma: no cover - entrypoint
    import uvicorn

    port = settings().dashboard_port
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
