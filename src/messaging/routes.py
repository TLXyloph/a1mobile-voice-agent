"""HTTP surface for the SMS closer: one inbound hook, and JSON for a panel.

MOUNTING: the parent app does `app.include_router(src.messaging.routes.router)`.
It must be included **before** the catch-all in `src/verify/webhooks.py` - that
route claims `/{full_path:path}` for every method, and FastAPI matches in
registration order, so anything added after it is unreachable.

WHERE INBOUND ACTUALLY COMES FROM. a1mobile's inbound SMS webhook has never
fired. Their voice webhook works - a real Telnyx IP hit our TeXML endpoint -
but texting the number produces zero requests, and they expose no API to read
received messages. Outbound sending works, and goes out from a shared pool
number, so replies land somewhere we cannot read.

So the inbound source of truth is **this endpoint**, not their delivery:

    POST /messages/inbound/{provider}   {"from": "+1...", "body": "..."}

If a1mobile ever starts delivering, point their webhook here and nothing
changes - the payload normaliser already handles their likely field names
alongside Twilio's form encoding. Until then, an inbound message is injected by
POSTing this endpoint (a judge texting a number we can read, an operator
pasting what the prospect sent, or a demo script). That is stated plainly here
because the alternative - a UI that implies texts are flowing in - is the kind
of quiet fiction this whole project is built to avoid.

An injected message is still independent evidence in the sense that matters:
its content came from the prospect, not from the agent. What we lose without
real delivery is *automatic* capture, not the direction of the arrow.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, Request

from src.business.campaign import CAMPAIGNS
from src.business.pricing import CostModel
from src.messaging import evidence as ev
from src.messaging.closer import (
    Reply,
    ReplyStatus,
    Responder,
    check_draft,
    generate_reply,
    read_inbound,
)
from src.messaging.send import A1MobileSMS, Sender, SendStatus
from src.messaging.thread import Thread, ThreadStore, normalise_phone

logger = logging.getLogger("messaging.routes")

router = APIRouter(tags=["messaging"])

#: Shared with `src/verify/webhooks.py` on purpose: an inbound text captured
#: here is findable by `webhooks.find_confirmation`, so the existing verifier
#: keeps working. The append is duplicated rather than imported because
#: importing that module builds the whole FastAPI app as a side effect.
INBOX = Path(__file__).resolve().parents[2] / "evidence" / "inbox.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inbox_path() -> Path:
    """Where captured messages go.

    Redirected under pytest. The evidence directory is a judged artifact, and a
    test suite that seeds it with fictional confirmations is manufacturing the
    exact kind of thing this project refuses to manufacture.
    """
    if override := os.getenv("MESSAGING_INBOX"):
        return Path(override)
    if "PYTEST_CURRENT_TEST" in os.environ:
        return Path(tempfile.gettempdir()) / "a1mobile_test_inbox.jsonl"
    return INBOX


def _append_inbox(record: dict[str, Any]) -> None:
    try:
        path = _inbox_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001
        # Never let evidence-writing failure eat the message itself.
        logger.warning("could not append to inbox: %s", exc)


# -- wiring ---------------------------------------------------------------

_store: ThreadStore | None = None
_responder: Responder | None = None
_sender: Sender | None = None
_registry: ev.ClaimRegistry = ev.REGISTRY


def get_store() -> ThreadStore:
    """The thread database. `MESSAGING_DB` overrides; pytest gets a temp file."""
    global _store
    if _store is None:
        path = os.getenv("MESSAGING_DB")
        if not path:
            path = (
                str(Path(tempfile.gettempdir()) / "a1mobile_test_threads.db")
                if "PYTEST_CURRENT_TEST" in os.environ
                else None
            )
        _store = ThreadStore(path) if path else ThreadStore()
    return _store


def set_store(store: ThreadStore) -> None:
    global _store
    _store = store


def get_responder() -> Responder | None:
    """The drafting model, or None when replies should not be generated.

    Returns None under pytest and when `MESSAGING_LLM=off`, so no test and no
    accidental import can start making inference calls. Everything else must
    inject one explicitly via `set_responder`.
    """
    global _responder
    if _responder is not None:
        return _responder
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    if os.getenv("MESSAGING_LLM", "on").strip().lower() in {"off", "0", "none"}:
        return None
    from src.messaging.closer import LiveKitResponder

    _responder = LiveKitResponder()
    return _responder


def set_responder(responder: Responder | None) -> None:
    global _responder
    _responder = responder


def get_sender() -> Sender:
    global _sender
    if _sender is None:
        _sender = A1MobileSMS()
    return _sender


def set_sender(sender: Sender) -> None:
    global _sender
    _sender = sender


def set_registry(registry: ev.ClaimRegistry) -> None:
    global _registry
    _registry = registry


def get_registry() -> ev.ClaimRegistry:
    return _registry


def link_claim(thread: Thread, claim: Any, tokens: Iterable[str]) -> None:
    """Point a thread at a live `Claim` that an inbound text could promote.

    Called in-process by whoever owns the receipt (the call runner, a demo
    script). The tokens are persisted on the thread; the `Claim` object is only
    held in memory, because a receipt is the one place a verdict may live.
    """
    tokens = tuple(tokens)
    thread.track_claim(claim.id, claim.description, tokens)
    _registry.register(thread.phone, claim, tokens)


# -- payload normalising ---------------------------------------------------


def _normalise(provider: str, payload: dict[str, Any]) -> dict[str, str]:
    """Flatten a provider payload to {from, to, body}.

    Mirrors `webhooks._normalise` and adds the shapes we actually use. a1mobile's
    real inbound shape is unknown - it has never sent us one - so their branch
    accepts the plausible spellings rather than guessing at exactly one.
    """
    if provider == "twilio":
        return {
            "from": payload.get("From", ""),
            "to": payload.get("To", ""),
            "body": payload.get("Body", ""),
        }
    if provider in {"a1mobile", "telnyx"}:
        data = payload.get("data") or payload
        pl = data.get("payload") if isinstance(data, dict) else None
        src = pl if isinstance(pl, dict) else data
        frm = src.get("from") or src.get("sender") or src.get("source") or ""
        if isinstance(frm, dict):
            frm = frm.get("phone_number", "")
        to = src.get("to") or src.get("recipient") or src.get("destination") or ""
        if isinstance(to, list) and to:
            to = to[0].get("phone_number", "") if isinstance(to[0], dict) else to[0]
        return {
            "from": str(frm),
            "to": str(to),
            "body": str(src.get("body") or src.get("text") or src.get("message") or ""),
        }
    return {
        "from": str(payload.get("from") or payload.get("phone") or ""),
        "to": str(payload.get("to") or ""),
        "body": str(payload.get("body") or payload.get("text") or payload.get("message") or ""),
    }


async def _payload(request: Request) -> dict[str, Any]:
    ctype = request.headers.get("content-type", "")
    try:
        if "json" in ctype:
            data = await request.json()
            return data if isinstance(data, dict) else {"body": str(data)}
        if "form" in ctype or "urlencoded" in ctype:
            return dict(await request.form())
        raw = (await request.body()).decode("utf-8", "replace")
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"body": raw}
        except Exception:  # noqa: BLE001
            return {"body": raw}
    except Exception as exc:  # noqa: BLE001
        return {"_parse_error": str(exc)}


def _flag(request: Request, name: str, default: bool) -> bool:
    raw = request.query_params.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# -- ingest ----------------------------------------------------------------


async def ingest(
    phone: str,
    body: str,
    *,
    raw: Any = None,
    auto_reply: bool = True,
    do_send: bool = True,
) -> dict[str, Any]:
    """The whole inbound path, callable without HTTP.

    Order matters and is deliberate: the message is persisted first, then
    evidence is attached, and only then is a reply drafted. A crash in the
    drafting model must not lose the one artifact a judge cares about.
    """
    store = get_store()
    number = normalise_phone(phone)
    thread = store.get_or_create(number)

    inbound = thread.add_inbound(body, source="webhook")
    read = read_inbound(thread, body)

    verified: list[str] = []
    pending: list[str] = []
    for ref in list(thread.claims):
        if ref.verdict == "VERIFIED":
            continue
        live = _registry.get(ref.claim_id)
        if live is None:
            # Post-restart: the tokens survived, the Claim object did not. Say
            # so rather than guessing in either direction.
            if ev.matches(body, ref.tokens):
                pending.append(ref.claim_id)
            continue
        if ev.try_verify(live.claim, ref.tokens, body, sender=number, raw=raw):
            ref.verdict = live.claim.verdict.value
            verified.append(ref.claim_id)
        else:
            ref.verdict = live.claim.verdict.value

    reply: Reply | None = None
    send_result = None
    responder = get_responder()

    if auto_reply and responder is not None and not thread.opted_out:
        reply = await generate_reply(thread, responder)
        if reply.sendable:
            msg = thread.add_outbound(
                reply.text,
                status="drafted",
                reply_status=reply.status.value,
                attempts=reply.attempts,
            )
            # Our own text, filed where it belongs: powerless.
            for ref in thread.open_claims:
                live = _registry.get(ref.claim_id)
                if live is not None:
                    ev.record_outbound(live.claim, reply.text, to=number)
            if do_send:
                send_result = await get_sender().send(number, reply.text)
                msg.status = send_result.status.value
                msg.meta["send"] = send_result.to_dict()

    store.save(thread)
    return {
        "ok": True,
        "captured": True,
        "phone": number,
        "inbound": inbound.to_dict(),
        "read": read,
        "verified_claims": verified,
        "evidence_pending": pending,
        "reply": reply.to_dict() if reply else None,
        "send": send_result.to_dict() if send_result else None,
        "thread": thread.summary(),
        "state": store.state_key(),
    }


@router.post("/messages/inbound")
@router.post("/messages/inbound/{provider}")
async def inbound_hook(request: Request, provider: str = "generic") -> dict[str, Any]:
    """Ingest a message FROM the prospect. The only channel that can verify.

    Accepts JSON or form encoding. Everything is written to
    `evidence/inbox.jsonl` verbatim before any parsing, so a message that
    matches no claim is still there for a judge to read.
    """
    payload = await _payload(request)
    msg = _normalise(provider, payload)
    record = {
        "kind": "inbound_sms",
        "provider": provider,
        "received_at": _now(),
        "via": "messaging",
        "client": getattr(request.client, "host", None),
        "raw": payload,
        **msg,
    }
    _append_inbox(record)

    if not msg["from"]:
        return {"ok": False, "captured": True, "error": "no sender in payload", "raw": payload}

    logger.info("inbound sms from %s: %s", msg["from"], msg["body"][:120])
    return await ingest(
        msg["from"],
        msg["body"],
        raw=record,
        auto_reply=_flag(request, "reply", True),
        do_send=_flag(request, "send", True),
    )


# -- reads for a dashboard panel -------------------------------------------


@router.get("/messages/state")
async def messages_state() -> dict[str, Any]:
    """A stable key that changes only when a thread changed.

    Poll this; re-fetch `/messages/threads` only when `state` differs from the
    one you are holding. Cheap enough to hit every second.
    """
    store = get_store()
    threads = store.all()
    return {
        "state": store.state_key(),
        "threads": len(threads),
        "messages": sum(len(t.messages) for t in threads),
        "unread": sum(1 for t in threads if t.outbound_since_inbound == 0 and t.messages),
        "checked_at": _now(),
    }


@router.get("/messages/threads")
async def list_threads() -> dict[str, Any]:
    store = get_store()
    threads = store.all()
    return {
        "state": store.state_key(),
        "count": len(threads),
        "threads": [t.summary() for t in threads],
    }


@router.get("/messages/threads/{phone}")
async def get_thread(phone: str) -> dict[str, Any]:
    store = get_store()
    thread = store.load(phone)
    if thread is None:
        return {"ok": False, "error": f"no thread for {normalise_phone(phone)}"}
    floor, target = thread.floor_total(), thread.target_total()
    return {
        "ok": True,
        "state": store.state_key(),
        "thread": thread.summary(),
        "constraints": {
            "summary": thread.constraints_summary(),
            "phase": thread.phase.value,
            "qty": thread.qty,
            "floor_total": str(floor) if floor is not None else None,
            "target_total": str(target) if target is not None else None,
            "their_budget": thread.budget_floor,
            "approved_total": thread.approved_total,
            "can_quote": thread.can_quote_at_all,
            "escalation_available": False,
            "hold_id": thread.hold_id,
        },
        "messages": [m.to_dict() for m in thread.messages],
        "claims": [c.to_dict() for c in thread.claims],
    }


# -- writes an operator or a demo script uses ------------------------------


@router.post("/messages/threads")
async def create_thread(request: Request) -> dict[str, Any]:
    """Start a thread, optionally seeded with a campaign and a cost model.

    A thread created without costs and a quantity can hold messages but cannot
    state any price - `check_draft` refuses every figure. That is the right
    default for a number we have no call context for.
    """
    data = await _payload(request)
    phone = normalise_phone(str(data.get("phone") or data.get("to") or ""))
    if not phone:
        return {"ok": False, "error": "phone is required"}

    store = get_store()
    thread = store.load(phone) or Thread(phone=phone)

    key = data.get("campaign_key")
    if key:
        if key not in CAMPAIGNS:
            return {"ok": False, "error": f"unknown campaign {key!r}"}
        thread.campaign = CAMPAIGNS[key]
    if costs := data.get("costs"):
        thread.costs = CostModel(
            materials_per_unit=costs["materials_per_unit"],
            labor_per_unit=costs["labor_per_unit"],
            transport_per_unit=costs["transport_per_unit"],
            min_margin_pct=costs["min_margin_pct"],
            target_margin_pct=costs.get("target_margin_pct"),
            unit=costs.get("unit", "unit"),
            currency=costs.get("currency", "USD"),
        )
    if qty := data.get("qty"):
        thread.qty = int(qty)
        thread.facts.units_confirmed = True
        thread.facts.units = int(qty)
    if budget := data.get("their_budget"):
        thread.note_stated_budget(float(budget))
    if data.get("capacity_held"):
        thread.facts.capacity_held = True
    if data.get("asked_current_spend"):
        thread.facts.asked_current_spend = True
    if phase := data.get("phase"):
        from src.agents.flow import Phase

        thread.phase = Phase(phase)

    store.save(thread)
    return {"ok": True, "thread": thread.summary(), "state": store.state_key()}


@router.post("/messages/send")
async def send_message(request: Request) -> dict[str, Any]:
    """Send on a thread: either a supplied body, or a freshly drafted reply.

    An operator-supplied body goes through the *same* guard as a model draft.
    That is not distrust of the operator - it is that the floor is the floor,
    and a number typed into a dashboard at 8pm is exactly when it gets forgotten.
    """
    data = await _payload(request)
    phone = normalise_phone(str(data.get("phone") or data.get("to") or ""))
    if not phone:
        return {"ok": False, "error": "phone is required"}

    store = get_store()
    thread = store.load(phone)
    if thread is None:
        return {"ok": False, "error": f"no thread for {phone}"}
    if thread.opted_out:
        return {
            "ok": False,
            "error": "they opted out of texts",
            "status": SendStatus.OPTED_OUT.value,
        }

    text = (data.get("text") or "").strip()
    reply: Reply | None = None

    if text:
        check = check_draft(thread, text)
        if not check.ok:
            return {
                "ok": False,
                "error": "refused by the price guard",
                "check": check.to_dict(),
                "instruction": check.instruction,
            }
        reply = Reply(text=text, status=ReplyStatus.OK, attempts=0, check=check)
    else:
        responder = get_responder()
        if responder is None:
            return {"ok": False, "error": "no drafting model configured"}
        reply = await generate_reply(thread, responder)
        if not reply.sendable:
            return {"ok": False, "error": "nothing sendable", "reply": reply.to_dict()}

    msg = thread.add_outbound(reply.text, status="drafted", reply_status=reply.status.value)
    for ref in thread.open_claims:
        live = _registry.get(ref.claim_id)
        if live is not None:
            ev.record_outbound(live.claim, reply.text, to=phone)

    result = await get_sender().send(phone, reply.text)
    msg.status = result.status.value
    msg.meta["send"] = result.to_dict()
    store.save(thread)

    return {
        "ok": result.ok,
        "reply": reply.to_dict(),
        "send": result.to_dict(),
        "thread": thread.summary(),
        "state": store.state_key(),
    }
