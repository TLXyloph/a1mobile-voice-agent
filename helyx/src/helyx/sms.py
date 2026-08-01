"""a1mobile SMS rails -- the independent confirmation channel.

Verified against the live API tonight:

* Outbound: ``POST https://hack.a1mobile.com/api/sms`` with header
  ``x-team-key``, body ``{"to": "+1...", "body": "..."}``. The message field is
  ``body``; ``text`` is rejected with a 422. Success returns
  ``{"sent": true, "message_id": ..., "from": "+19378608348"}``.
* Webhook registration: ``POST /api/sms/webhook`` with
  ``{"sms_webhook_url": "..."}``.
* Inbound delivery payload is lowercase:
  ``{"type": "message.received", "from", "to", "text", "media_urls", "telnyx_id"}``.

Note ``A1MOBILE_API_KEY`` is empty in the environment; ``A1MOBILE_TEAM_KEY`` is
the credential that works.

Why this matters: the supplier texting our number is a channel the agent cannot
write to. That asymmetry is what makes a confirmation worth anything.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import secret, settings

logger = logging.getLogger("helyx.sms")


class SMSError(RuntimeError):
    pass


@dataclass(frozen=True)
class SendResult:
    sent: bool
    message_id: str
    from_number: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sent": self.sent,
            "message_id": self.message_id,
            "from": self.from_number,
        }


@dataclass(frozen=True)
class InboundSMS:
    """Normalised inbound message. Tolerates casing drift in the payload."""

    from_number: str
    to_number: str
    text: str
    provider_id: str
    event_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_number,
            "to": self.to_number,
            "text": self.text,
            "provider_id": self.provider_id,
            "type": self.event_type,
        }


def _headers() -> dict[str, str]:
    key = secret("A1MOBILE_TEAM_KEY")
    if not key:
        raise SMSError("A1MOBILE_TEAM_KEY is not set")
    return {"x-team-key": key, "Content-Type": "application/json"}


def send_sms(to: str, body: str, timeout: float = 30.0) -> SendResult:
    """Send an SMS via a1mobile. The field is `body`, not `text`."""
    to = (to or "").strip()
    if not to.startswith("+"):
        raise SMSError(f"recipient must be E.164, got {to!r}")
    if not (body or "").strip():
        raise SMSError("message body must be non-empty")

    url = settings().a1mobile_base.rstrip("/") + "/sms"
    req = urllib.request.Request(
        url,
        data=json.dumps({"to": to, "body": body}).encode(),
        method="POST",
        headers=_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise SMSError(f"a1mobile returned HTTP {exc.code}: {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SMSError(f"a1mobile request failed: {exc!r}") from exc

    return SendResult(
        sent=bool(data.get("sent")),
        message_id=str(data.get("message_id", "")),
        from_number=str(data.get("from", "")),
        raw=data,
    )


def register_webhook(public_url: str, timeout: float = 30.0) -> dict[str, Any]:
    """Point a1mobile's inbound SMS webhook at our public URL."""
    if not public_url.startswith("http"):
        raise SMSError(f"public_url must be absolute, got {public_url!r}")
    url = settings().a1mobile_base.rstrip("/") + "/sms/webhook"
    req = urllib.request.Request(
        url,
        data=json.dumps({"sms_webhook_url": public_url}).encode(),
        method="POST",
        headers=_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise SMSError(f"a1mobile returned HTTP {exc.code}: {detail}") from exc


def normalise_inbound(payload: dict[str, Any]) -> InboundSMS:
    """Read a1mobile's inbound payload, tolerating key-casing differences.

    Their documented shape is lowercase, but a webhook that silently changes
    case must not silently stop verifying things, so every key is looked up
    case-insensitively with common aliases.
    """
    if not isinstance(payload, dict):
        raise SMSError("inbound payload must be a JSON object")
    lowered = {str(k).lower(): v for k, v in payload.items()}

    def pick(*names: str) -> str:
        for n in names:
            v = lowered.get(n)
            if isinstance(v, dict):  # some providers nest {"phone_number": ...}
                v = v.get("phone_number") or v.get("number")
            if v not in (None, ""):
                return str(v)
        return ""

    return InboundSMS(
        from_number=pick("from", "from_number", "source", "msisdn"),
        to_number=pick("to", "to_number", "destination"),
        text=pick("text", "body", "message"),
        provider_id=pick("telnyx_id", "message_id", "id", "sid"),
        event_type=pick("type", "event_type") or "message.received",
    )


def confirmation_request(quantity: int, item: str, unit_price_cents: int, when: str) -> str:
    """The text we send asking the supplier to put the terms in writing.

    It restates the numbers so that a bare "yes" reply cannot confirm anything:
    the reply is matched against the terms, and we want the supplier's own
    message to carry the figures back to us.
    """
    return (
        f"Helyx here, following up on our call. To confirm in writing: "
        f"{quantity} {item} at ${unit_price_cents / 100:.2f} per unit, for {when}. "
        f"Please reply with those details to confirm. Nothing is booked until you do."
    )
