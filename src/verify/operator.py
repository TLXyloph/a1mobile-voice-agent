"""Operator escalation - how the agent asks a human mid-call.

When the agent wants to exceed the envelope it must stop and ask. That request
has to reach a person whose hands are covered in flour, get an answer, and get
back to a caller who is holding the line - inside about thirty seconds.

Two channels, deliberately redundant:

- **Voice, via VoiceOS.** VoiceOS has no API, so nothing can call into it. What
  it *does* have is hands-free operation of Gmail. So we email the question,
  the operator hears and answers it by voice through VoiceOS, and we watch the
  inbox for the reply. That is an honest integration: VoiceOS does the thing it
  actually does.

- **A click, on the dashboard.** Same request, one button. This exists because
  an email round-trip is 15-45 seconds and a demo cannot be hostage to inbox
  latency. Whichever channel answers first wins; the other is ignored.

The pending store is a plain JSON file rather than a database because the
dashboard is a separate process and needs to see the same queue. At this scale
a file with an flock is simpler than a broker and fails in more obvious ways.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("verify.operator")

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "evidence" / "approvals.json"


class Decision(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    TIMED_OUT = "timed_out"
    """No human answered. Treated as denial - silence must never authorise."""


@dataclass
class ApprovalRequest:
    """One question the agent cannot answer on its own."""

    question: str
    """Plain language, as the operator would hear it read aloud."""

    options: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    """Numbers behind the ask: quote, floor, qty, campaign."""

    id: str = field(default_factory=lambda: f"apr_{uuid.uuid4().hex[:8]}")
    created_at: float = field(default_factory=time.time)
    decision: str | None = None
    note: str = ""
    decided_at: float | None = None
    decided_via: str = ""

    @property
    def pending(self) -> bool:
        return self.decision is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -- the shared queue ----------------------------------------------------


def _load() -> list[dict[str, Any]]:
    if not QUEUE.exists():
        return []
    try:
        return json.loads(QUEUE.read_text())
    except json.JSONDecodeError:
        # A torn write mid-demo must not take the whole system down.
        logger.warning("approvals.json unreadable; starting a fresh queue")
        return []


def _save(rows: list[dict[str, Any]]) -> None:
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2))
    tmp.replace(QUEUE)  # atomic on POSIX; readers never see a partial file


def submit(req: ApprovalRequest) -> str:
    rows = _load()
    rows.append(req.to_dict())
    _save(rows)
    logger.warning("APPROVAL NEEDED [%s]: %s", req.id, req.question)
    return req.id


def get(approval_id: str) -> dict[str, Any] | None:
    return next((r for r in _load() if r["id"] == approval_id), None)


def pending() -> list[dict[str, Any]]:
    return [r for r in _load() if r.get("decision") is None]


def decide(approval_id: str, approved: bool, *, note: str = "", via: str = "dashboard") -> bool:
    """Record a decision. First answer wins; later ones are ignored."""
    rows = _load()
    for row in rows:
        if row["id"] == approval_id:
            if row.get("decision") is not None:
                logger.info("approval %s already decided via %s", approval_id,
                            row.get("decided_via"))
                return False
            row["decision"] = Decision.APPROVED.value if approved else Decision.DENIED.value
            row["note"] = note
            row["decided_at"] = time.time()
            row["decided_via"] = via
            _save(rows)
            logger.info("approval %s -> %s (via %s)", approval_id, row["decision"], via)
            return True
    return False


# -- channel: email out --------------------------------------------------


def _compose(req: ApprovalRequest) -> tuple[str, str]:
    """Subject and body, written to be read aloud by a screen reader or VoiceOS.

    The subject carries the whole question because that is often all the
    operator hears before deciding.
    """
    subject = f"Approve? {req.question}"[:120]
    lines = [req.question, ""]
    if req.options:
        lines += ["Options:", *(f"  - {o}" for o in req.options), ""]
    if req.context:
        lines += ["Details:"]
        lines += [f"  {k}: {v}" for k, v in req.context.items()]
        lines += [""]
    lines += [
        "Reply with YES or NO as the first word. Anything after it is passed",
        "back to the agent as a note.",
        "",
        f"ref: {req.id}",
    ]
    return subject, "\n".join(lines)


async def notify_by_email(req: ApprovalRequest) -> bool:
    """Send the approval request to the operator's inbox.

    Uses SMTP so it works without any provider SDK. Returns False rather than
    raising: a failed notification must not kill the call, because the
    dashboard path is still live.
    """
    host = os.getenv("SMTP_HOST")
    to_addr = os.getenv("OPERATOR_EMAIL")
    if not (host and to_addr):
        logger.info("email channel not configured; dashboard only")
        return False

    import smtplib
    from email.message import EmailMessage

    subject, body = _compose(req)
    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_FROM", to_addr)
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    def _send() -> None:
        with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=10) as s:
            s.starttls()
            if os.getenv("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            s.send_message(msg)

    try:
        await asyncio.to_thread(_send)
        logger.info("approval %s emailed to %s", req.id, to_addr)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("approval email failed (dashboard still live): %s", exc)
        return False


# -- channel: voice reply in --------------------------------------------

_YES = re.compile(r"^\s*(yes|yep|yeah|approve[d]?|ok|okay|do it|go ahead|sure)\b", re.I)
_NO = re.compile(r"^\s*(no|nope|deny|denied|don't|do not|reject|skip)\b", re.I)


def parse_reply(text: str) -> tuple[bool | None, str]:
    """Read a dictated reply. Returns (approved, note).

    Ambiguity returns None, never a guess. A misheard "no" read as approval
    commits the business to an order it cannot fill, so anything that is not
    clearly affirmative or clearly negative goes back to the operator.
    """
    stripped = text.strip()
    if _YES.match(stripped):
        return True, _YES.sub("", stripped, count=1).strip(" .,-")
    if _NO.match(stripped):
        return False, _NO.sub("", stripped, count=1).strip(" .,-")
    return None, stripped


def apply_reply(approval_id: str, text: str, *, via: str = "voiceos") -> bool | None:
    """Apply a dictated/emailed reply to a pending approval."""
    approved, note = parse_reply(text)
    if approved is None:
        logger.warning("ambiguous operator reply for %s: %r", approval_id, text[:80])
        return None
    decide(approval_id, approved, note=note, via=via)
    return approved


# -- the call-side wait --------------------------------------------------


async def ask(
    question: str,
    *,
    options: list[str] | None = None,
    context: dict[str, Any] | None = None,
    timeout: float = 45.0,
    poll: float = 1.0,
) -> tuple[Decision, str]:
    """Ask the operator and wait, while the agent holds the line.

    Timeout resolves to TIMED_OUT, which callers must treat as a denial. An
    unanswered question is not permission - that is the same rule as the
    receipt system, applied to authority instead of evidence.
    """
    req = ApprovalRequest(
        question=question, options=options or [], context=context or {}
    )
    submit(req)
    await notify_by_email(req)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = get(req.id)
        if row and row.get("decision"):
            return Decision(row["decision"]), row.get("note", "")
        await asyncio.sleep(poll)

    decide(req.id, approved=False, note="no answer before timeout", via="timeout")
    logger.warning("approval %s timed out after %.0fs -> treated as denial",
                   req.id, timeout)
    return Decision.TIMED_OUT, ""
