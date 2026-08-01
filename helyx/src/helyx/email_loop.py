"""The post-call email loop: check for an inbound email, then report back.

Honest statement of what works, because overstating this would be exactly the
failure mode Helyx exists to prevent:

* There is **no email credential in this repo**. ``config/.env`` has
  ``CALLBACK_EMAIL`` and nothing else -- no SMTP host, no app password, no
  IMAP login. a1mobile exposes no email route either (``/api/email`` and
  ``/api/mail`` both 404).
* Therefore ``SMTPTransport`` is real, complete code that is **inert until**
  ``HELYX_SMTP_HOST`` / ``HELYX_SMTP_USER`` / ``HELYX_SMTP_PASSWORD`` are
  supplied. It has not been exercised against a live server.
* ``OutboxTransport`` always runs and writes the exact RFC-822 message to
  ``helyx/var/outbox/``. That part is verified: the message is composed,
  addressed and persisted on every call end.
* Inbound checking works the same way: ``IMAPBackend`` is gated on credentials;
  ``DropBackend`` reads operator-supplied payloads from ``helyx/var/inbox/``
  so the loop is exercisable end to end without a mailbox.

**Recipient lock.** Every send is checked against ``CALLBACK_EMAIL`` and
refused otherwise. Helyx must never mail a third party.
"""

from __future__ import annotations

import json
import logging
import re
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Protocol

from .config import VAR_DIR, secret, settings
from .domain import Channel, Evidence, ProposalStatus

logger = logging.getLogger("helyx.email")

OUTBOX = VAR_DIR / "outbox"
INBOX = VAR_DIR / "inbox"


class EmailRefused(RuntimeError):
    """Raised when a send targets anyone other than the authorised operator."""


@dataclass(frozen=True)
class InboundEmail:
    sender: str
    subject: str
    body: str
    message_id: str = ""
    received_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.sender,
            "subject": self.subject,
            "body": self.body[:2000],
            "message_id": self.message_id,
            "received_at": self.received_at,
        }


@dataclass
class SendOutcome:
    delivered: bool
    transport: str
    detail: str
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivered": self.delivered,
            "transport": self.transport,
            "detail": self.detail,
            "path": self.path,
        }


# --- transports -------------------------------------------------------------


class Transport(Protocol):
    name: str

    def available(self) -> bool: ...

    def send(self, message: EmailMessage) -> SendOutcome: ...


class SMTPTransport:
    """Real SMTP send. Inert unless HELYX_SMTP_* credentials are supplied."""

    name = "smtp"

    def available(self) -> bool:
        return bool(
            secret("HELYX_SMTP_HOST")
            and secret("HELYX_SMTP_USER")
            and secret("HELYX_SMTP_PASSWORD")
        )

    def send(self, message: EmailMessage) -> SendOutcome:
        host = secret("HELYX_SMTP_HOST")
        port = int(secret("HELYX_SMTP_PORT") or "587")
        user = secret("HELYX_SMTP_USER")
        password = secret("HELYX_SMTP_PASSWORD")
        try:
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.starttls()
                server.login(user, password)
                server.send_message(message)
        except Exception as exc:  # noqa: BLE001
            # Never include the exception's full text blindly; it can echo creds.
            return SendOutcome(False, self.name, f"SMTP failed: {type(exc).__name__}")
        return SendOutcome(True, self.name, f"sent via {host}")


class OutboxTransport:
    """Always-available fallback: persist the exact message for inspection."""

    name = "outbox"

    def available(self) -> bool:
        return True

    def send(self, message: EmailMessage) -> SendOutcome:
        OUTBOX.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = OUTBOX / f"{stamp}.eml"
        path.write_text(message.as_string())
        return SendOutcome(
            delivered=False,
            transport=self.name,
            detail="no SMTP credential configured; message composed and stored, not delivered",
            path=str(path),
        )


# --- inbound ----------------------------------------------------------------


def check_inbound(since_subject_hint: str = "") -> InboundEmail | None:
    """Look for an inbound email. Returns the newest match, or None.

    Tries IMAP when credentials exist, then the local drop directory.
    """
    email = _check_imap(since_subject_hint)
    if email is not None:
        return email
    return _check_drop()


def _check_imap(hint: str) -> InboundEmail | None:
    host = secret("HELYX_IMAP_HOST")
    user = secret("HELYX_IMAP_USER")
    password = secret("HELYX_IMAP_PASSWORD")
    if not (host and user and password):
        return None
    import email as email_mod
    import imaplib

    try:
        with imaplib.IMAP4_SSL(host) as imap:
            imap.login(user, password)
            imap.select("INBOX")
            criteria = f'(SUBJECT "{hint}")' if hint else "(UNSEEN)"
            _, data = imap.search(None, criteria)
            ids = (data[0] or b"").split()
            if not ids:
                return None
            _, raw = imap.fetch(ids[-1], "(RFC822)")
            msg = email_mod.message_from_bytes(raw[0][1])
            return InboundEmail(
                sender=str(msg.get("From", "")),
                subject=str(msg.get("Subject", "")),
                body=_body_of(msg),
                message_id=str(msg.get("Message-ID", "")),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("IMAP check failed: %s", type(exc).__name__)
        return None


def _body_of(msg: Any) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode("utf-8", "replace")
        return ""
    payload = msg.get_payload(decode=True)
    return payload.decode("utf-8", "replace") if payload else str(msg.get_payload())


def _check_drop() -> InboundEmail | None:
    """Read operator-supplied inbound payloads from helyx/var/inbox/*.json."""
    if not INBOX.exists():
        return None
    files = sorted(INBOX.glob("*.json"))
    if not files:
        return None
    try:
        raw = json.loads(files[-1].read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("unreadable inbound drop %s: %s", files[-1], exc)
        return None
    return InboundEmail(
        sender=str(raw.get("from", "")),
        subject=str(raw.get("subject", "")),
        body=str(raw.get("body", "")),
        message_id=str(raw.get("message_id", "")),
    )


# --- composition & send -----------------------------------------------------


def compose_summary(snapshot: dict[str, Any], inbound: InboundEmail | None) -> EmailMessage:
    """Build the post-call report. Wording follows the evidence, pessimistically."""
    to_addr = settings().callback_email
    msg = EmailMessage()
    msg["To"] = to_addr
    msg["From"] = secret("HELYX_SMTP_USER") or to_addr
    headline = str(snapshot.get("headline", "No headline"))
    msg["Subject"] = f"[Helyx] {headline[:120]}"
    msg.set_content(_render_body(snapshot, inbound, headline))
    return msg


def _render_body(
    snapshot: dict[str, Any], inbound: InboundEmail | None, headline: str
) -> str:
    lines: list[str] = ["HELYX POST-CALL REPORT", "=" * 60, "", headline, ""]

    neg = snapshot.get("negotiation") or {}
    mandate = neg.get("mandate") or {}
    if mandate:
        lines += [
            "MANDATE",
            f"  Supplier : {mandate.get('counterparty_name')}",
            f"  Order    : {mandate.get('quantity')} x {mandate.get('item')}",
            f"  Target   : ${(mandate.get('target_unit_price_cents') or 0) / 100:.2f}/unit",
            f"  Ceiling  : ${(mandate.get('ceiling_unit_price_cents') or 0) / 100:.2f}/unit",
            f"  Needed by: {mandate.get('needed_by')}",
            f"  Outcome  : {neg.get('outcome')}",
            "",
        ]

    lines.append("PROPOSALS AND THEIR EVIDENCE")
    proposals = snapshot.get("proposals") or []
    if not proposals:
        lines.append("  (none filed)")
    for p in proposals:
        lines += [
            f"  [{str(p['status']).upper()}] {p['quantity']} x {p['item']} at "
            f"${p['unit_price_cents'] / 100:.2f}/unit -> total "
            f"${p['total_cents'] / 100:.2f}",
            f"      basis: {p['why']}",
        ]
        for e in p.get("evidence", []):
            tag = "INDEPENDENT" if e["independent"] else "agent claim"
            lines.append(f"      - {e['channel']} ({tag}): {e['body'][:110]}")
    lines.append("")

    lines.append("INBOUND EMAIL CHECK")
    if inbound is None:
        lines.append("  No inbound email found at the time of this report.")
    else:
        lines += [
            f"  From   : {inbound.sender}",
            f"  Subject: {inbound.subject}",
            f"  Excerpt: {inbound.body[:400]}",
        ]
    lines.append("")

    violations = [
        v
        for turn in (neg.get("turns") or [])
        for v in (turn.get("violations") or [])
    ]
    lines.append("MANDATE ENFORCEMENT")
    if violations:
        lines.append(f"  {len(violations)} unauthorised amount(s) were caught and suppressed:")
        for v in violations:
            lines.append(f"    - {v['text']}: {v['reason']}")
    else:
        lines.append("  No unauthorised amounts were spoken.")
    lines += [
        "",
        "-" * 60,
        "Helyx marks nothing complete on the agent's own word. Any line above",
        "reading CONFIRMED is backed by an independent channel named in its",
        "evidence list; anything else is explicitly UNCONFIRMED.",
    ]
    return "\n".join(lines)


_ADDR = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def send_report(message: EmailMessage) -> SendOutcome:
    """Send to the authorised operator address only."""
    allowed = (settings().callback_email or "").strip().lower()
    if not allowed or not _ADDR.fullmatch(allowed):
        raise EmailRefused("CALLBACK_EMAIL is not a valid address; refusing to send")
    recipients = [a.strip().lower() for a in str(message.get("To", "")).split(",")]
    if recipients != [allowed]:
        raise EmailRefused(
            f"Helyx only mails the operator ({allowed}); refusing {recipients}"
        )

    smtp = SMTPTransport()
    if smtp.available():
        outcome = smtp.send(message)
        if outcome.delivered:
            OutboxTransport().send(message)  # keep a local copy for audit
            return outcome
        logger.warning("SMTP transport failed, falling back to outbox")
    return OutboxTransport().send(message)


def run_email_loop(store: Any) -> dict[str, Any]:
    """Post-call: check inbound mail, attach it as evidence, then report back.

    Inbound email is an ``INDEPENDENT`` channel, so a supplier email that
    restates the terms can confirm a proposal exactly as an SMS would.
    """
    inbound = check_inbound()
    if inbound is not None:
        for proposal in store.proposals:
            proposal.add_evidence(
                Evidence(
                    channel=Channel.INBOUND_EMAIL,
                    body=f"{inbound.subject}\n{inbound.body}",
                    external_ref=inbound.message_id,
                )
            )
        store.emit("email.inbound", **inbound.to_dict())

    snapshot = store.snapshot()
    message = compose_summary(snapshot, inbound)
    outcome = send_report(message)

    record = {
        "inbound_found": inbound is not None,
        "inbound": inbound.to_dict() if inbound else None,
        "subject": message["Subject"],
        "to": message["To"],
        "outcome": outcome.to_dict(),
        "confirmed_after_email": [
            p.id for p in store.proposals if p.status is ProposalStatus.CONFIRMED
        ],
    }
    store.email_log.append(record)
    store.emit("email.report", **record)
    return record
