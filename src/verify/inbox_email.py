"""Email as the independent verification channel.

a1mobile's inbound webhook never fires - not for SMS, not for calls - and there
is no endpoint to read received messages. That removes the channel the design
originally leaned on, so verification moves to email, which we can read
programmatically and which the agent has no ability to write.

Independence is the whole property being preserved. The agent talks on the
phone; the prospect sends an email; a different process reads the mailbox. No
amount of agent insistence puts a message in that inbox.

Two backends, same interface:
  - `check_via_mcp_payload` - for results handed in from the Gmail MCP tools,
    which is what the operator console uses.
  - `check_via_imap`        - standalone fallback needing an app password.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from src.verify.receipts import Channel, Evidence

logger = logging.getLogger("verify.email")


def _norm(text: str) -> str:
    """Lowercase and strip formatting so '$400.00' matches '400'."""
    return re.sub(r"[\s,$]", "", (text or "").lower())


def tokens_present(body: str, expected: list[str]) -> tuple[bool, list[str]]:
    """Are ALL expected tokens in the body?

    Requiring every token is deliberate and matches the SMS matcher. A booking
    for 200 units at 400 dollars must not be satisfied by a mail that merely
    says "confirmed" - that is precisely how a wrong order gets scored right.
    """
    haystack = _norm(body)
    missing = [t for t in expected if _norm(t) not in haystack]
    return (not missing), missing


def verify_claim_via_email(
    claim: Any,
    messages: list[dict[str, Any]],
    expected_tokens: list[str],
    *,
    from_contains: str | None = None,
) -> bool:
    """Promote a claim if a matching email exists.

    Args:
        messages: [{"from":..., "subject":..., "body":..., "date":...}]
        expected_tokens: every one must appear, e.g. ["200", "400", "Friday"].
        from_contains: optionally require the sender to match, so our own
            outbound copy cannot satisfy our own claim.

    Absence of a matching mail attaches nothing. Not finding proof is not proof
    of failure, and recording it as contradiction would overstate what we know.
    """
    for msg in messages:
        sender = str(msg.get("from", ""))
        if from_contains and from_contains.lower() not in sender.lower():
            continue

        body = f"{msg.get('subject', '')}\n{msg.get('body', '')}"
        ok, missing = tokens_present(body, expected_tokens)
        if not ok:
            logger.debug("mail from %s missing %s", sender, missing)
            continue

        claim.attach_evidence(
            Evidence(
                channel=Channel.INBOUND_EMAIL,
                summary=f"Email from {sender}: {str(msg.get('subject',''))[:120]}",
                raw={
                    "from": sender,
                    "subject": msg.get("subject"),
                    "body": str(msg.get("body", ""))[:2000],
                    "date": msg.get("date"),
                },
            )
        )
        logger.info("claim %s VERIFIED via email from %s", claim.id, sender)
        return True

    logger.info("no email matching %s for claim %s", expected_tokens, claim.id)
    return False


def check_via_mcp_payload(
    claim: Any, mcp_messages: list[dict[str, Any]], expected_tokens: list[str]
) -> bool:
    """Adapt Gmail MCP output, whose field names differ from ours."""
    normalised = [
        {
            "from": m.get("from") or m.get("sender") or "",
            "subject": m.get("subject") or "",
            "body": m.get("body") or m.get("snippet") or m.get("text") or "",
            "date": m.get("date") or m.get("internalDate") or "",
        }
        for m in mcp_messages
    ]
    return verify_claim_via_email(claim, normalised, expected_tokens)


def check_via_imap(
    claim: Any,
    expected_tokens: list[str],
    *,
    host: str,
    user: str,
    password: str,
    since_minutes: int = 60,
    folder: str = "INBOX",
) -> bool:
    """Standalone IMAP check. Needs a Gmail app password, not the account one."""
    import email
    import imaplib
    from datetime import timedelta

    since = (datetime.now(timezone.utc) - timedelta(minutes=since_minutes)).strftime("%d-%b-%Y")
    messages: list[dict[str, Any]] = []

    with imaplib.IMAP4_SSL(host) as imap:
        imap.login(user, password)
        imap.select(folder)
        _, data = imap.search(None, f'(SINCE "{since}")')
        for uid in (data[0].split() if data and data[0] else [])[-40:]:
            _, raw = imap.fetch(uid, "(RFC822)")
            if not raw or not raw[0]:
                continue
            msg = email.message_from_bytes(raw[0][1])
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode("utf-8", "replace")
            else:
                payload = msg.get_payload(decode=True)
                body = payload.decode("utf-8", "replace") if payload else ""
            messages.append({
                "from": msg.get("From", ""), "subject": msg.get("Subject", ""),
                "body": body, "date": msg.get("Date", ""),
            })

    return verify_claim_via_email(claim, messages, expected_tokens)
