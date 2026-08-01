"""TeXML responder for a1mobile inbound calls.

a1mobile runs on Telnyx, and a Telnyx number whose webhook returns anything
other than valid TeXML makes the caller hear "Sorry, an application error has
occurred". That is exactly the symptom we saw, so the previous JSON responses
were either not arriving at all or being rejected as malformed.

This module answers with real TeXML. It also serves as a decisive experiment:
if the caller now hears speech instead of the error, the webhook IS reaching us
and our request logging was blind to it. If they still hear the error, nothing
is arriving and the fault is entirely upstream at a1mobile.

Every inbound hit is recorded as evidence either way - an inbound call is a
side effect we did not manufacture, which makes it exactly the kind of thing a
receipt is allowed to rest on.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape

from fastapi import APIRouter, Request, Response

logger = logging.getLogger("verify.texml")

router = APIRouter()

XML = "application/xml"


def _texml(body: str) -> Response:
    return Response(
        content=f'<?xml version="1.0" encoding="UTF-8"?>\n<Response>{body}</Response>',
        media_type=XML,
    )


def say(text: str, *, voice: str = "Polly.Joanna-Neural") -> str:
    return f"<Say voice=\"{voice}\">{escape(text)}</Say>"


def stream_to(ws_url: str) -> str:
    """Bidirectional media stream, the Pipecat-style path a1mobile describes."""
    return f'<Connect><Stream url="{escape(ws_url)}" /></Connect>'


def _record(kind: str, request: Request, payload: Any) -> None:
    from src.verify.webhooks import _append

    _append({
        "kind": kind,
        "method": request.method,
        "path": str(request.url.path),
        "query": dict(request.query_params),
        "client": getattr(request.client, "host", None),
        "headers": dict(request.headers),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "raw": payload,
    })


async def _payload(request: Request) -> Any:
    from src.verify.webhooks import _any_payload

    return await _any_payload(request)


@router.api_route("/texml", methods=["GET", "POST"])
@router.api_route("/texml/voice", methods=["GET", "POST"])
async def texml_voice(request: Request) -> Response:
    """Answer an inbound call with speech.

    Deliberately minimal. Getting audible speech out of a real inbound call is
    the milestone; bridging that call into the agent is a separate step and
    should not be attempted until this proves the webhook fires at all.
    """
    payload = await _payload(request)
    _record("inbound_call_texml", request, payload)
    logger.warning("TeXML inbound call: %s", str(payload)[:300])

    return _texml(
        say(
            "Thanks for calling Rosewater Bakehouse. Your call reached our "
            "automated assistant. This confirms the inbound webhook is working."
        )
        + "<Pause length=\"1\"/>"
        + say("Goodbye.")
    )


@router.api_route("/texml/status", methods=["GET", "POST"])
async def texml_status(request: Request) -> Response:
    """Call status callbacks. Recorded, never acted on."""
    _record("call_status", request, await _payload(request))
    return _texml("")


@router.api_route("/texml/sms", methods=["GET", "POST"])
async def texml_sms(request: Request) -> Response:
    """Inbound SMS, if a1mobile ever routes it here.

    Normalised into the same inbox the verifier reads, so a message arriving by
    this path verifies claims identically to one arriving by any other.
    """
    payload = await _payload(request)
    _record("inbound_sms_texml", request, payload)

    if isinstance(payload, dict):
        from src.verify.webhooks import _append, _normalise

        msg = _normalise("a1mobile", payload)
        if msg.get("body"):
            _append({
                "kind": "inbound_sms",
                "provider": "a1mobile-texml",
                "received_at": datetime.now(timezone.utc).isoformat(),
                "raw": payload,
                **msg,
            })
            logger.warning("INBOUND SMS via TeXML from %s: %s",
                           msg["from"], msg["body"][:120])
    return _texml("")


#: Fields that only ever appear on a messaging webhook. Telnyx sends voice and
#: SMS callbacks to whatever URL the number is pointed at, and a1mobile exposes
#: exactly one `webhook_url` for both - so a single endpoint has to tell them
#: apart and answer with the right document type. Replying to an SMS with voice
#: TeXML is how you get "an application error has occurred" on the next call.
_SMS_FIELDS = ("Body", "body", "MessageSid", "messageSid", "message_id", "text",
               "SmsSid", "SmsStatus", "NumMedia")


def looks_like_sms(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(k in payload for k in _SMS_FIELDS)


@router.api_route("/texml/inbound", methods=["GET", "POST"])
async def texml_inbound(request: Request) -> Response:
    """One endpoint for both voice and SMS. Point a1mobile here.

    Branching on the payload rather than the path means we cannot be wrong about
    which kind of event arrived, which matters because we do not control - and
    cannot inspect - how a1mobile routes the two.
    """
    payload = await _payload(request)

    if looks_like_sms(payload):
        _record("inbound_sms_texml", request, payload)
        from src.verify.webhooks import _append, _normalise

        msg = _normalise("a1mobile", payload) if isinstance(payload, dict) else {}
        _append({
            "kind": "inbound_sms",
            "provider": "a1mobile-inbound",
            "received_at": datetime.now(timezone.utc).isoformat(),
            "raw": payload,
            **msg,
        })
        logger.warning("INBOUND SMS from %s: %s",
                       msg.get("from"), str(msg.get("body"))[:160])
        # Empty messaging response: acknowledged, no auto-reply. The SMS closer
        # owns replying, so answering here too would double-send.
        return _texml("")

    _record("inbound_call_texml", request, payload)
    logger.warning("INBOUND CALL: %s", str(payload)[:200])
    return _texml(
        say("Thanks for calling Rosewater Bakehouse. Your call reached our "
            "automated assistant.") + '<Pause length="1"/>' + say("Goodbye.")
    )
