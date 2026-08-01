"""Outbound SMS over a1mobile, with the two failure modes made loud.

`POST https://hack.a1mobile.com/api/sms` with an `X-Team-Key` header works. Two
things about it shape this module:

1. **They only deliver to OTP-verified numbers.** A send to anything else is
   rejected. That rejection is the single most likely reason a demo goes quiet,
   so it gets its own status (`UNVERIFIED_NUMBER`) with a message that says what
   to do about it, rather than being flattened into a generic error and logged
   at debug level where nobody will see it.

2. **It sends from a shared pool number.** Replies go to that pool, and we
   cannot read them - there is no inbound webhook and no API to list received
   messages. So a successful send is *not* evidence of anything, and nothing in
   this module ever touches a `Claim`. See `src/messaging/routes.py` for where
   inbound actually comes from.

Dry run is ON by default, everywhere, and forced on under pytest. Turning it
off is an explicit act (`MESSAGING_DRY_RUN=0`), because the failure mode of the
opposite default is texting a stranger during development.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from src.messaging.thread import normalise_phone

logger = logging.getLogger("messaging.send")

DEFAULT_BASE_URL = "https://hack.a1mobile.com"
SMS_PATH = "/api/sms"

#: Their field name for the message text is not documented. Overridable so a
#: 400 at 8pm is an env var, not a code change.
BODY_FIELD = os.getenv("A1MOBILE_SMS_BODY_FIELD", "message")

#: Response fragments that mean "that number is not OTP-verified". Matched
#: case-insensitively against the body, alongside a 401/403 status check.
_VERIFICATION_HINTS = (
    "verif",           # "not verified", "verification required"
    "otp",
    "whitelist",
    "allowlist",
    "not allowed",
    "unauthorized recipient",
    "recipient not",
)


class SendStatus(str, Enum):
    SENT = "sent"
    """The provider accepted it. Not proof it arrived, and never proof of a deal."""

    DRY_RUN = "dry_run"
    """Nothing left the process. The default."""

    UNVERIFIED_NUMBER = "unverified_number"
    """a1mobile refused the destination. Get the number OTP-verified first."""

    NOT_CONFIGURED = "not_configured"
    """No A1MOBILE_TEAM_KEY. Nothing was attempted."""

    OPTED_OUT = "opted_out"
    """They texted STOP. We do not send, at all, ever."""

    INVALID_NUMBER = "invalid_number"
    ERROR = "error"


@dataclass(frozen=True)
class SendResult:
    """What happened, in a form a dashboard can render without interpreting."""

    ok: bool
    status: SendStatus
    to: str
    body: str
    detail: str = ""
    http_status: int | None = None
    raw: Any = None

    @property
    def needs_operator(self) -> bool:
        """True when a human has to do something before this can ever work."""
        return self.status in (
            SendStatus.UNVERIFIED_NUMBER,
            SendStatus.NOT_CONFIGURED,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status.value,
            "to": self.to,
            "body": self.body,
            "detail": self.detail,
            "http_status": self.http_status,
            "needs_operator": self.needs_operator,
        }


class Sender(Protocol):
    async def send(self, to: str, body: str) -> SendResult: ...


def _dry_run_default() -> bool:
    """Dry run unless someone explicitly turned it off, and always under pytest.

    The pytest check is not belt-and-braces paranoia: a test that accidentally
    constructs a real sender with a real key would text a real stranger, and no
    assertion in the suite would catch it.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    return os.getenv("MESSAGING_DRY_RUN", "1").strip().lower() not in {"0", "false", "no"}


def _verified_numbers_from_env() -> set[str]:
    raw = os.getenv("A1MOBILE_VERIFIED_NUMBERS", "")
    return {normalise_phone(n) for n in raw.split(",") if n.strip()}


@dataclass
class A1MobileSMS:
    """Outbound only. Knows nothing about claims, threads or verdicts."""

    team_key: str | None = None
    base_url: str | None = None
    dry_run: bool | None = None
    verified_numbers: set[str] | None = None
    """Locally known OTP-verified destinations. When non-empty, anything else is
    refused before a request is made - a fast, offline version of the rejection
    a1mobile would send back anyway, with a clearer message."""

    timeout: float = 15.0
    sent: list[dict[str, Any]] = field(default_factory=list)
    """Everything this sender was asked to send, dry run included. The demo
    reads this; so does the test that proves dry run sends nothing."""

    def __post_init__(self) -> None:
        if self.team_key is None:
            self.team_key = os.getenv("A1MOBILE_TEAM_KEY", "")
        if self.base_url is None:
            self.base_url = os.getenv("A1MOBILE_BASE_URL") or DEFAULT_BASE_URL
        if self.dry_run is None:
            self.dry_run = _dry_run_default()
        if self.verified_numbers is None:
            self.verified_numbers = _verified_numbers_from_env()

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + SMS_PATH

    async def send(self, to: str, body: str) -> SendResult:
        number = normalise_phone(to)
        if not number or len(number) < 8:
            return SendResult(
                ok=False,
                status=SendStatus.INVALID_NUMBER,
                to=to,
                body=body,
                detail=f"{to!r} is not a usable phone number.",
            )

        if self.verified_numbers and number not in self.verified_numbers:
            return SendResult(
                ok=False,
                status=SendStatus.UNVERIFIED_NUMBER,
                to=number,
                body=body,
                detail=(
                    f"a1mobile will not deliver to {number}: it is not OTP-verified "
                    f"for this team key. Verify it in their console (or add it to "
                    f"A1MOBILE_VERIFIED_NUMBERS) before sending. Nothing was sent."
                ),
            )

        record = {"to": number, "body": body, "dry_run": bool(self.dry_run)}
        if self.dry_run:
            self.sent.append(record)
            logger.info("DRY RUN sms to %s: %s", number, body[:120])
            return SendResult(
                ok=True,
                status=SendStatus.DRY_RUN,
                to=number,
                body=body,
                detail="Dry run - nothing was sent. Set MESSAGING_DRY_RUN=0 to send.",
            )

        if not self.team_key:
            return SendResult(
                ok=False,
                status=SendStatus.NOT_CONFIGURED,
                to=number,
                body=body,
                detail="A1MOBILE_TEAM_KEY is not set; refusing to attempt a send.",
            )

        import httpx

        payload = {"to": number, BODY_FIELD: body}
        from_number = os.getenv("A1MOBILE_FROM_NUMBER", "")
        if from_number:
            payload["from"] = from_number

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.url,
                    json=payload,
                    headers={
                        "X-Team-Key": self.team_key,
                        "Content-Type": "application/json",
                    },
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sms transport failure to %s: %s", number, exc)
            return SendResult(
                ok=False,
                status=SendStatus.ERROR,
                to=number,
                body=body,
                detail=f"Could not reach {self.url}: {exc}",
            )

        text = (resp.text or "")[:600]
        self.sent.append({**record, "http_status": resp.status_code})

        if resp.status_code < 300:
            logger.info("sms accepted for %s (%s)", number, resp.status_code)
            return SendResult(
                ok=True,
                status=SendStatus.SENT,
                to=number,
                body=body,
                detail=(
                    "a1mobile accepted the message. It sends from their shared "
                    "pool number, so any reply goes somewhere we cannot read - "
                    "this is not evidence of anything."
                ),
                http_status=resp.status_code,
                raw=text,
            )

        if _looks_like_verification_refusal(resp.status_code, text):
            return SendResult(
                ok=False,
                status=SendStatus.UNVERIFIED_NUMBER,
                to=number,
                body=body,
                detail=(
                    f"a1mobile rejected {number} as not OTP-verified "
                    f"(HTTP {resp.status_code}). Verify the number with them, then "
                    f"retry. Provider said: {text.strip()[:200]}"
                ),
                http_status=resp.status_code,
                raw=text,
            )

        return SendResult(
            ok=False,
            status=SendStatus.ERROR,
            to=number,
            body=body,
            detail=f"a1mobile returned HTTP {resp.status_code}: {text.strip()[:200]}",
            http_status=resp.status_code,
            raw=text,
        )


def _looks_like_verification_refusal(status_code: int, text: str) -> bool:
    low = (text or "").lower()
    if any(hint in low for hint in _VERIFICATION_HINTS):
        return True
    # 401 is about our key, not their number; 403 with no explanation is most
    # often the verified-recipient rule, so it is worth naming as the likely
    # cause rather than shrugging at a bare status code.
    return status_code == 403


@dataclass
class NullSender:
    """Records and drops. For tests and for the offline rehearsal path."""

    sent: list[dict[str, Any]] = field(default_factory=list)
    status: SendStatus = SendStatus.DRY_RUN

    async def send(self, to: str, body: str) -> SendResult:
        number = normalise_phone(to)
        self.sent.append({"to": number, "body": body, "dry_run": True})
        return SendResult(
            ok=self.status in (SendStatus.SENT, SendStatus.DRY_RUN),
            status=self.status,
            to=number,
            body=body,
            detail="NullSender - nothing left the process.",
        )
