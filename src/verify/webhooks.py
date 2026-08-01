"""Inbound webhook listener - the channel the agent cannot forge.

This is where UNVERIFIED becomes VERIFIED. It accepts confirmations that arrive
from the outside world (SMS from the business, provider callbacks) and matches
them against open claims.

Provider payload shapes differ, so `/sms/<provider>` normalises Twilio's form
encoding and a generic JSON shape. Add a1mobile's shape at kickoff - it is one
branch in `_normalise`.

Run:
    uv run uvicorn src.verify.webhooks:app --port 8080
    ngrok http 8080          # point the provider's SMS webhook at the URL
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

from src.verify.receipts import Channel, Evidence

logger = logging.getLogger("verify.webhooks")

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "evidence"
INBOX = EVIDENCE_DIR / "inbox.jsonl"

app = FastAPI(title="a1mobile verification listener")

# Vapi tool + event webhooks share this server so one tunnel serves everything.
# Imported lazily-ish but at module scope so the routes exist before the
# catch-all below claims every path.
try:
    from src.agents.vapi_bridge import router as _vapi_router

    app.include_router(_vapi_router)
except Exception as _exc:  # noqa: BLE001
    logging.getLogger("verify.webhooks").warning("vapi bridge unavailable: %s", _exc)

# Judge-facing demo page + API. Served from the public tunnel so a judge needs
# nothing but a link.
try:
    from src.demo.api import router as _demo_router

    app.include_router(_demo_router)
except Exception as _exc:  # noqa: BLE001
    logging.getLogger("verify.webhooks").warning("demo page unavailable: %s", _exc)

# SMS closer. Must be included ahead of the catch-all below: that route claims
# /{full_path:path} for every method, and FastAPI matches in registration order,
# so anything mounted after it is unreachable.
try:
    from src.messaging.routes import router as _messaging_router

    app.include_router(_messaging_router)
except Exception as _exc:  # noqa: BLE001
    logging.getLogger("verify.webhooks").warning("messaging unavailable: %s", _exc)

# TeXML responder. a1mobile is Telnyx underneath, and Telnyx plays "an
# application error has occurred" to the caller unless the webhook returns
# valid TeXML - which JSON is not.
try:
    from src.verify.texml import router as _texml_router

    app.include_router(_texml_router)
except Exception as _exc:  # noqa: BLE001
    logging.getLogger("verify.webhooks").warning("texml unavailable: %s", _exc)


def _normalise(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten a provider payload to {from, to, body}."""
    if provider == "twilio":
        return {
            "from": payload.get("From", ""),
            "to": payload.get("To", ""),
            "body": payload.get("Body", ""),
        }
    if provider == "a1mobile":
        # TODO(kickoff): confirm against their docs. These are the most common
        # field names; verify before relying on them.
        return {
            "from": payload.get("from") or payload.get("sender", ""),
            "to": payload.get("to") or payload.get("recipient", ""),
            "body": payload.get("body") or payload.get("text") or payload.get("message", ""),
        }
    return {
        "from": payload.get("from", ""),
        "to": payload.get("to", ""),
        "body": payload.get("body", ""),
    }


def _append(record: dict[str, Any]) -> None:
    INBOX.parent.mkdir(parents=True, exist_ok=True)
    with INBOX.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


@app.post("/sms/{provider}")
async def inbound_sms(provider: str, request: Request) -> dict[str, Any]:
    """Capture an inbound SMS as independent evidence.

    Everything is persisted verbatim before any parsing, so a message that fails
    to match a claim is still available to a judge.
    """
    ctype = request.headers.get("content-type", "")
    if "json" in ctype:
        payload = await request.json()
    else:
        payload = dict(await request.form())

    msg = _normalise(provider, payload)
    record = {
        "kind": "inbound_sms",
        "provider": provider,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "raw": payload,
        **msg,
    }
    _append(record)
    logger.info("inbound SMS from %s: %s", msg["from"], msg["body"][:120])
    return {"ok": True, "captured": True}


@app.get("/inbox")
async def read_inbox(contains: str | None = None) -> dict[str, Any]:
    """Read captured messages, optionally filtered. Used by the verifier."""
    if not INBOX.exists():
        return {"messages": []}
    messages = [json.loads(line) for line in INBOX.read_text().splitlines() if line]
    if contains:
        needle = contains.lower()
        messages = [m for m in messages if needle in str(m.get("body", "")).lower()]
    return {"messages": messages}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/voice")
async def voice_webhook(request: Request) -> dict[str, Any]:
    """a1mobile's voice webhook target.

    We answer calls over SIP rather than through this hook, so its job here is
    purely evidential: it records that a call event happened, from a source we
    do not control.
    """
    payload = await _any_payload(request)
    _append({
        "kind": "voice_event",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "raw": payload,
    })
    logger.info("voice webhook: %s", str(payload)[:200])
    return {"ok": True}


@app.websocket("/voice")
@app.websocket("/voice/{rest:path}")
async def voice_stream(websocket: WebSocket, rest: str = "") -> None:
    """Media-stream endpoint for a1mobile's webhook transport.

    Their docs say they "stream the call to you (works with Pipecat)", which
    means a WebSocket carrying audio frames rather than an HTTP callback. We do
    not have their frame format, so this deliberately does not try to answer the
    call - it accepts the socket and records exactly what arrives.

    That is the point: one real connection tells us the protocol (JSON events vs
    raw PCM, sample rate, encoding, how the call id is carried), which is
    strictly better than guessing at it. Refusing the upgrade would throw away
    the only source of that information.
    """
    await websocket.accept()
    peer = getattr(websocket.client, "host", None)
    logger.warning("WEBSOCKET CONNECTED on /voice/%s from %s", rest, peer)
    _append({
        "kind": "ws_open",
        "path": "/voice/" + rest,
        "client": peer,
        "headers": dict(websocket.headers),
        "received_at": datetime.now(timezone.utc).isoformat(),
    })

    frames = 0
    try:
        while True:
            msg = await websocket.receive()
            frames += 1
            # Record the first few frames in full, then only count. Audio is
            # continuous; the protocol is knowable from the opening exchange.
            if frames <= 8:
                text, data = msg.get("text"), msg.get("bytes")
                _append({
                    "kind": "ws_frame",
                    "n": frames,
                    "type": "text" if text is not None else "binary",
                    "text": (text or "")[:1000],
                    "bytes_len": len(data) if data else 0,
                    "bytes_head": data[:64].hex() if data else None,
                    "received_at": datetime.now(timezone.utc).isoformat(),
                })
                logger.warning("WS frame %d: %s", frames,
                               (text or f"<{len(data or b'')} bytes>")[:200])
            if msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("ws error after %d frames: %s", frames, exc)
    finally:
        _append({
            "kind": "ws_close", "frames": frames,
            "received_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.warning("WEBSOCKET CLOSED after %d frames", frames)


async def _any_payload(request: Request) -> Any:
    """Best-effort body parse. Never raise - an unparseable webhook is still
    evidence that something arrived, and losing it to a 500 is worse than
    storing it as text."""
    ctype = request.headers.get("content-type", "")
    try:
        if "json" in ctype:
            return await request.json()
        if "form" in ctype or "urlencoded" in ctype:
            return dict(await request.form())
        raw = await request.body()
        return raw.decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return {"_parse_error": str(exc)}


# Registered last so it never shadows a specific route above. Its purpose is
# discovery: we do not know a1mobile's inbound-SMS path or payload shape, and
# a 404 would throw away the one message that tells us.
#
# Every method, not just POST. A provider that delivers via GET (query string)
# or PUT would otherwise get a 405 and we would record nothing but an access-log
# line - the exact failure mode that makes "we received no SMS" ambiguous
# between "none was sent" and "we dropped it".
@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def catch_all(full_path: str, request: Request) -> dict[str, Any]:
    payload = await _any_payload(request)

    # Local health checks are not evidence. The catch-all exists to discover an
    # unknown provider payload; with several services polling this port it
    # otherwise writes hundreds of empty rows an hour and buries the one real
    # message we are waiting for.
    peer = getattr(request.client, "host", None)
    if peer in ("127.0.0.1", "::1") and not payload:
        return {"ok": True, "captured": False, "note": "local no-op ignored"}

    record = {
        "kind": "unrouted",
        "method": request.method,
        "path": "/" + full_path,
        "query": dict(request.query_params),
        "received_at": datetime.now(timezone.utc).isoformat(),
        # All headers, not a whitelist. We do not know what a1mobile sends, and
        # the whole point of this route is to find out. Client IP is recorded
        # separately because it is what distinguishes a real provider callback
        # from our own probes.
        "client": getattr(request.client, "host", None),
        "headers": dict(request.headers),
        "raw": payload,
    }
    _append(record)
    logger.warning(
        "UNROUTED %s /%s params=%s: %s",
        request.method, full_path, dict(request.query_params), str(payload)[:300],
    )
    return {"ok": True, "captured": True, "note": f"logged unrouted /{full_path}"}


# -- matching ------------------------------------------------------------


def find_confirmation(
    expected_tokens: list[str], *, since: str | None = None
) -> dict[str, Any] | None:
    """Find an inbound message containing every expected token.

    Requiring ALL tokens is deliberate. A booking claim for "4 people at 7pm"
    should not be satisfied by a message that merely says "confirmed" - that is
    how a wrong booking gets scored as a right one.
    """
    if not INBOX.exists():
        return None

    for line in INBOX.read_text().splitlines():
        if not line:
            continue
        msg = json.loads(line)
        if since and msg.get("received_at", "") < since:
            continue
        body = str(msg.get("body", "")).lower()
        if all(re.search(re.escape(t.lower()), body) for t in expected_tokens):
            return msg
    return None


def verify_claim_via_sms(claim: Any, expected_tokens: list[str]) -> bool:
    """Promote a claim if a matching inbound message exists.

    Returns True only on a real match; attaches contradicting evidence never -
    absence of a text is not proof of failure, only absence of proof.
    """
    match = find_confirmation(expected_tokens, since=claim.created_at)
    if match is None:
        logger.info("no SMS matching %s for claim %s", expected_tokens, claim.id)
        return False

    claim.attach_evidence(
        Evidence(
            channel=Channel.INBOUND_SMS,
            summary=f"SMS from {match.get('from')}: {match.get('body')}",
            raw=match,
        )
    )
    logger.info("claim %s VERIFIED via inbound SMS", claim.id)
    return True
