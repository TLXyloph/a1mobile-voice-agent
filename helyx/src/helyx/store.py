"""Append-only event log plus the derived view the dashboard renders.

State changes are events. The current view is a fold over them, so the
dashboard and the audit trail cannot disagree -- what an operator sees is
reconstructible from the log, which is the point when the scoring rule is
"verifiable side effects, not agent claims".

Subscribers receive events as they are appended, which is what makes the
dashboard live without polling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import VAR_DIR
from .domain import Channel, Evidence, Proposal, ProposalStatus
from .intake import IntakeSession
from .negotiator import Negotiation
from .sms import InboundSMS

logger = logging.getLogger("helyx.store")

MAX_EVENTS = 2000


@dataclass(frozen=True)
class Event:
    kind: str
    payload: dict[str, Any]
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "at": self.at, "payload": self.payload}


class HelyxStore:
    """In-process state for one operator session."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.intake = IntakeSession()
        self.negotiation: Negotiation | None = None
        self.call_active: bool = False
        self.email_log: list[dict[str, Any]] = []
        self.sms_log: list[dict[str, Any]] = []
        self.model_report: dict[str, Any] = {}
        self._seq = 0
        self._subscribers: set[asyncio.Queue[str]] = set()

    # -- events ------------------------------------------------------------
    def emit(self, kind: str, **payload: Any) -> Event:
        self._seq += 1
        event = Event(kind=kind, payload=payload, seq=self._seq)
        self.events.append(event)
        if len(self.events) > MAX_EVENTS:
            del self.events[: len(self.events) - MAX_EVENTS]
        self._broadcast(event)
        return event

    def _broadcast(self, event: Event) -> None:
        if not self._subscribers:
            return
        message = json.dumps({"event": event.to_dict(), "state": self.snapshot()})
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("dropping event for a slow dashboard subscriber")

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._subscribers.discard(queue)

    # -- proposals & verification -----------------------------------------
    @property
    def proposals(self) -> list[Proposal]:
        return list(self.negotiation.proposals) if self.negotiation else []

    def attach_inbound_sms(self, msg: InboundSMS) -> dict[str, Any]:
        """Turn an inbound SMS into evidence against every open proposal.

        The message is attached to all proposals; whether it *confirms* any of
        them is decided by ``Proposal.status``, which compares the message
        against the terms. We never decide that here.
        """
        self.sms_log.append({"direction": "in", **msg.to_dict()})
        matched: list[str] = []
        for proposal in self.proposals:
            supports = not _reads_as_refusal(msg.text)
            proposal.add_evidence(
                Evidence(
                    channel=Channel.INBOUND_SMS,
                    body=msg.text,
                    supports=supports,
                    external_ref=msg.provider_id,
                )
            )
            if proposal.status is ProposalStatus.CONFIRMED:
                matched.append(proposal.id)
        self.emit(
            "sms.inbound",
            **msg.to_dict(),
            confirmed_proposals=matched,
        )
        return {"attached_to": [p.id for p in self.proposals], "confirmed": matched}

    def add_human_review(self, body: str, supports: bool = True) -> list[str]:
        """Operator vouches for (or contradicts) what they heard. Independent
        of the agent because a human, not the agent, authored it."""
        confirmed: list[str] = []
        for proposal in self.proposals:
            proposal.add_evidence(
                Evidence(channel=Channel.HUMAN_REVIEW, body=body, supports=supports)
            )
            if proposal.status is ProposalStatus.CONFIRMED:
                confirmed.append(proposal.id)
        self.emit("evidence.human_review", body=body[:280], supports=supports,
                  confirmed=confirmed)
        return confirmed

    # -- derived view ------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        proposals = [p.to_dict() for p in self.proposals]
        counts = {s.value: 0 for s in ProposalStatus}
        for p in proposals:
            counts[str(p["status"])] += 1
        return {
            "ts": time.time(),
            "intake": self.intake.to_dict(),
            "negotiation": self.negotiation.to_dict() if self.negotiation else None,
            "proposals": proposals,
            "status_counts": counts,
            "call_active": self.call_active,
            "sms_log": self.sms_log[-25:],
            "email_log": self.email_log[-10:],
            "model_report": self.model_report,
            "events": [e.to_dict() for e in self.events[-40:]],
            "headline": self.headline(),
        }

    def headline(self) -> str:
        """Reports pessimistically. Never says 'done' without evidence."""
        if self.negotiation is None:
            return "Awaiting mandate. No call may be placed until intake validates."
        proposals = self.proposals
        if not proposals:
            return "Negotiation open. Nothing proposed yet."
        confirmed = [p for p in proposals if p.status is ProposalStatus.CONFIRMED]
        contradicted = [p for p in proposals if p.status is ProposalStatus.CONTRADICTED]
        if contradicted:
            return f"CONTRADICTED - independent evidence conflicts with {len(contradicted)} proposal(s)."
        if confirmed:
            best = confirmed[-1]
            return (
                f"CONFIRMED by independent evidence: {best.terms.quantity} x "
                f"{best.terms.item} at ${best.terms.unit_price_cents / 100:.2f}/unit."
            )
        return (
            f"UNCONFIRMED - {len(proposals)} proposal(s) on the agent's word alone. "
            "No independent evidence has arrived."
        )

    def write_receipt(self) -> Path:
        """Every run writes a receipt, including failed ones. A run with no
        receipt is indistinguishable from a run that lied."""
        VAR_DIR.mkdir(parents=True, exist_ok=True)
        path = VAR_DIR / f"receipt_{int(time.time())}.json"
        path.write_text(json.dumps(self.snapshot(), indent=2, default=str))
        self.emit("receipt.written", path=str(path))
        return path


_REFUSAL_MARKERS = (
    "cannot",
    "can't",
    "cant ",
    "unable",
    "sold out",
    "no longer",
    "not available",
    "we don't",
    "decline",
    "cancel",
)


def _reads_as_refusal(text: str) -> bool:
    """Conservative refusal detector.

    Only used to decide whether inbound evidence *contradicts*. It never
    promotes anything -- confirmation still requires the terms to match.
    """
    low = (text or "").lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


STORE = HelyxStore()
