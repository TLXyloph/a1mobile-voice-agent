"""In-memory board state for the operator dashboard.

The dashboard is a *view*. It owns no invariants: verdicts still come from
`src.verify.receipts.Claim.verdict`, capacity still comes from a real
`CapacityLedger`, and an escalation is still phrased by
`Envelope.permits()`. Nothing here can mark anything done - if this module
could, it would be exactly the fabrication path the whole project exists to
close off.

Two things live here that the rest of the system does not have:

1. **Display metadata.** A `Receipt` knows its claims; it does not know who was
   on the other end of the phone or which campaign paid for the call. The card
   wrappers carry that so the printed receipt reads like a receipt.

2. **An approval queue.** `src.agents.sales_agent.OperatorChannel` reaches the
   human by voice. On a conference floor that path is one bad microphone away
   from dead, so the rail is the fallback: same question, same two answers, but
   clickable. `ApprovalRequest.decision` starts None and None means no - an
   unanswered escalation must never default to yes.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from src.agents.negotiation import Tactic
from src.business.campaign import CAMPAIGNS, Campaign, get_campaign
from src.business.capacity import CapacityLedger
from src.business.discovery import Lead
from src.verify.receipts import Channel, Evidence, Receipt, Verdict

#: Seeded holds must outlive a demo. A real call heartbeats its hold every 45s
#: (`sales_agent.HEARTBEAT_SECONDS`); fixture holds have nobody beating for
#: them, so they get a long TTL instead of silently draining to AVAILABLE
#: halfway through a judging slot.
FIXTURE_TTL = 24 * 3600.0

#: How long after an operator decision the seeded call finishes and prints.
RESOLVE_AFTER = 5.0
#: How long after printing the independent SMS lands and flips the stamp.
EVIDENCE_AFTER = 7.0


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# --------------------------------------------------------------------------
# What a tactic means, in one line a human can read off a projector.
# --------------------------------------------------------------------------

TACTIC_GLOSS: dict[Tactic, str] = {
    Tactic.ASK_CURRENT_SPEND: "learning their number before spending ours",
    Tactic.EMPHASISE_VALUE: "reframing on value, price untouched",
    Tactic.OFFER_CONCESSION: "stepping down the ladder, floor enforced",
    Tactic.RESTRUCTURE: "changing the shape of the deal, not the price",
    Tactic.ESCALATE_TO_OPERATOR: "outside the envelope - waiting on you",
    Tactic.CLOSE: "confirming specifics and asking for it in writing",
    Tactic.WALK_AWAY: "exiting politely to preserve the next call",
}


# --------------------------------------------------------------------------
# Board objects
# --------------------------------------------------------------------------


@dataclass
class LiveCall:
    """One call in flight. `quote` is what is currently on the table."""

    business: str
    phone: str
    campaign_key: str
    status: str = "CONNECTED"
    tactic: Tactic = Tactic.ASK_CURRENT_SPEND
    quote: float | None = None
    qty: int | None = None
    unit: str = ""
    transcript_tail: list[str] = field(default_factory=list)
    started_epoch: float = field(default_factory=time.time)
    ended_epoch: float | None = None
    id: str = field(default_factory=lambda: _uid("call"))

    @property
    def campaign(self) -> Campaign | None:
        return CAMPAIGNS.get(self.campaign_key)

    @property
    def campaign_name(self) -> str:
        c = self.campaign
        return c.name if c else self.campaign_key

    @property
    def elapsed_seconds(self) -> float:
        end = self.ended_epoch if self.ended_epoch is not None else time.time()
        return max(0.0, end - self.started_epoch)

    @property
    def elapsed(self) -> str:
        return _mmss(self.elapsed_seconds)

    @property
    def tactic_gloss(self) -> str:
        return TACTIC_GLOSS.get(self.tactic, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "business": self.business,
            "phone": self.phone,
            "campaign": self.campaign_name,
            "campaign_key": self.campaign_key,
            "status": self.status,
            "tactic": self.tactic.value,
            "tactic_gloss": self.tactic_gloss,
            "quote": self.quote,
            "qty": self.qty,
            "unit": self.unit,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "started_epoch": self.started_epoch,
            "transcript_tail": list(self.transcript_tail),
        }


@dataclass
class ApprovalRequest:
    """An escalation waiting on a human. Two answers, no third.

    `decision` is None until somebody presses a button, and None is a refusal,
    not a pause: `is_pending` is what the rail renders, and `approved` is False
    for anything that is not an explicit approve.
    """

    business: str
    campaign_key: str
    ask: str
    """The one sentence the operator has to decide about."""

    reason: str
    """Why it needs a human - taken from `Envelope.permits()`, verbatim."""

    options: list[str] = field(default_factory=list)
    """Concrete alternatives, best first. The operator picks by approving or
    denying; the options exist so the decision is informed, not blind."""

    call_id: str | None = None
    at_stake: str = ""
    at_stake_total: float | None = None
    """The number the call quotes if this is approved. None leaves it alone."""

    created_epoch: float = field(default_factory=time.time)
    decision: str | None = None
    decided_epoch: float | None = None
    id: str = field(default_factory=lambda: _uid("ask"))

    @property
    def is_pending(self) -> bool:
        return self.decision is None

    @property
    def approved(self) -> bool:
        return self.decision == "approve"

    @property
    def campaign_name(self) -> str:
        c = CAMPAIGNS.get(self.campaign_key)
        return c.name if c else self.campaign_key

    @property
    def waiting_seconds(self) -> float:
        end = self.decided_epoch if self.decided_epoch is not None else time.time()
        return max(0.0, end - self.created_epoch)

    @property
    def waiting(self) -> str:
        return _mmss(self.waiting_seconds)

    def decide(self, decision: str) -> None:
        """Record an answer. Anything not 'approve' is a denial.

        Idempotent by design: a double-click, or a second operator hitting the
        other button a moment later, must not flip a settled decision.
        """
        if self.decision is not None:
            return
        self.decision = "approve" if decision == "approve" else "deny"
        self.decided_epoch = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "business": self.business,
            "campaign": self.campaign_name,
            "ask": self.ask,
            "reason": self.reason,
            "options": list(self.options),
            "at_stake": self.at_stake,
            "call_id": self.call_id,
            "pending": self.is_pending,
            "decision": self.decision,
            "waiting_seconds": round(self.waiting_seconds, 1),
            "created_epoch": self.created_epoch,
        }


@dataclass
class ReceiptCard:
    """A finished call, as printed paper.

    The stamp is derived from the claims, never set: `stamp` reads the same
    `Claim.verdict` property a judge would read, so a receipt cannot show
    VERIFIED because the dashboard felt like it.
    """

    receipt: Receipt
    counterparty: str
    campaign_key: str
    phone: str = ""
    duration_seconds: float = 0.0
    printed_epoch: float = field(default_factory=time.time)
    demo_pending_evidence: bool = False
    """Fixture flag: an inbound SMS is due to land shortly. Only ever set by
    the seeded demo flow, never by a real run."""

    @property
    def campaign_name(self) -> str:
        c = CAMPAIGNS.get(self.campaign_key)
        return c.name if c else self.campaign_key

    @property
    def stamp(self) -> str:
        r = self.receipt
        if r.contradicted:
            return "CONTRADICTED"
        if not r.claims or r.unverified:
            return "UNCONFIRMED"
        return "VERIFIED"

    @property
    def stamp_class(self) -> str:
        return self.stamp.lower()

    @property
    def opened_at(self) -> str:
        return self._clock(self.receipt.started_at)

    @property
    def closed_at(self) -> str:
        return self._clock(self.receipt.ended_at) if self.receipt.ended_at else "--:--:--"

    @staticmethod
    def _clock(iso: str) -> str:
        try:
            return datetime.fromisoformat(iso).strftime("%H:%M:%S")
        except (TypeError, ValueError):
            return "--:--:--"

    @property
    def duration(self) -> str:
        return _mmss(self.duration_seconds)

    def to_dict(self) -> dict[str, Any]:
        d = self.receipt.to_dict()
        d.update(
            {
                "counterparty": self.counterparty,
                "campaign": self.campaign_name,
                "phone": self.phone,
                "stamp": self.stamp,
                "duration_seconds": round(self.duration_seconds, 1),
            }
        )
        return d


@dataclass
class LedgerCard:
    """One campaign's capacity, plus the label the operator thinks in."""

    campaign_key: str
    ledger: CapacityLedger

    @property
    def campaign(self) -> Campaign | None:
        return CAMPAIGNS.get(self.campaign_key)

    @property
    def campaign_name(self) -> str:
        c = self.campaign
        return c.name if c else self.campaign_key

    @property
    def snapshot(self) -> dict[str, Any]:
        return self.ledger.snapshot()

    def bars(self) -> list[dict[str, Any]]:
        """Segment widths as percentages, for the stacked bar. Sums to 100."""
        s = self.snapshot
        total = max(1, s["total"])
        return [
            {"key": "committed", "n": s["committed"], "pct": 100 * s["committed"] / total},
            {"key": "held", "n": s["held"], "pct": 100 * s["held"] / total},
            {"key": "available", "n": s["available"], "pct": 100 * s["available"] / total},
        ]


# --------------------------------------------------------------------------
# The board
# --------------------------------------------------------------------------


class Board:
    """Everything on screen. One process, one board, guarded by a lock.

    The lock matters: a live agent thread filing an approval while the browser
    polls must not tear a render.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.calls: list[LiveCall] = []
        self.approvals: list[ApprovalRequest] = []
        self.ledgers: list[LedgerCard] = []
        self.leads: list[Lead] = []
        self.receipts: list[ReceiptCard] = []
        self.demo: bool = False
        """True when the board holds fixture data whose story advances on a
        timer. Real runs leave this False and nothing moves on its own."""

    # -- writes (the integration surface for a live run) ----------------

    def clear(self) -> None:
        with self._lock:
            self.calls.clear()
            self.approvals.clear()
            self.ledgers.clear()
            self.leads.clear()
            self.receipts.clear()
            self.demo = False

    def register_ledger(self, campaign_key: str, ledger: CapacityLedger) -> LedgerCard:
        with self._lock:
            card = LedgerCard(campaign_key=campaign_key, ledger=ledger)
            self.ledgers.append(card)
            return card

    def open_call(self, call: LiveCall) -> LiveCall:
        with self._lock:
            self.calls.append(call)
            return call

    def close_call(self, call_id: str) -> LiveCall | None:
        with self._lock:
            for c in self.calls:
                if c.id == call_id:
                    c.ended_epoch = time.time()
                    self.calls.remove(c)
                    return c
            return None

    def request_approval(self, req: ApprovalRequest) -> ApprovalRequest:
        with self._lock:
            self.approvals.append(req)
            return req

    def decide(self, approval_id: str, decision: str) -> ApprovalRequest | None:
        """Answer an escalation. Returns the request, or None if unknown."""
        with self._lock:
            req = self.approval(approval_id)
            if req is None:
                return None
            was_pending = req.is_pending
            req.decide(decision)
            if was_pending:
                self._apply_decision(req)
            return req

    def add_receipt(self, card: ReceiptCard) -> ReceiptCard:
        with self._lock:
            self.receipts.insert(0, card)
            return card

    # -- reads ----------------------------------------------------------

    def approval(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            return next((a for a in self.approvals if a.id == approval_id), None)

    def call(self, call_id: str) -> LiveCall | None:
        with self._lock:
            return next((c for c in self.calls if c.id == call_id), None)

    def snapshot(self) -> dict[str, Any]:
        """Everything the page needs, after advancing any timed fixtures."""
        with self._lock:
            self.tick()
            pending = [a for a in self.approvals if a.is_pending]
            settled = [a for a in self.approvals if not a.is_pending]
            settled.sort(key=lambda a: a.decided_epoch or 0, reverse=True)
            return {
                "pending": pending,
                "settled": settled[:3],
                "calls": list(self.calls),
                "ledgers": list(self.ledgers),
                "leads": list(self.leads),
                "receipts": list(self.receipts),
                "counts": {
                    "pending": len(pending),
                    "answered": len(self.approvals) - len(pending),
                    "calls": len(self.calls),
                    "leads": len(self.leads),
                    "receipts": len(self.receipts),
                    "verified": sum(1 for r in self.receipts if r.stamp == "VERIFIED"),
                    "contradicted": sum(
                        1 for r in self.receipts if r.stamp == "CONTRADICTED"
                    ),
                },
            }

    def as_json(self) -> dict[str, Any]:
        snap = self.snapshot()
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counts": snap["counts"],
            "calls": [c.to_dict() for c in snap["calls"]],
            "approvals": {
                "pending": [a.to_dict() for a in snap["pending"]],
                "settled": [a.to_dict() for a in snap["settled"]],
            },
            "capacity": [
                {"campaign": lc.campaign_name, "campaign_key": lc.campaign_key,
                 **lc.snapshot}
                for lc in snap["ledgers"]
            ],
            "leads": [lead.to_dict() for lead in snap["leads"]],
            "receipts": [r.to_dict() for r in snap["receipts"]],
        }

    # -- timed fixtures -------------------------------------------------

    def tick(self) -> None:
        """Advance the seeded story. A no-op on a board holding real data.

        Everything here is bookkeeping the live system would do for itself: a
        decided escalation ends the call, an ended call prints, and an inbound
        SMS - the independent channel - arrives a few seconds later and
        promotes the claim. It is on a timer only because there is no phone
        line in the room.
        """
        if not self.demo:
            return
        now = time.time()
        with self._lock:
            for req in self.approvals:
                if req.is_pending or req.decided_epoch is None:
                    continue
                if now - req.decided_epoch < RESOLVE_AFTER:
                    continue
                if req.call_id and self.call(req.call_id) is not None:
                    self._print_for(req)

            for card in self.receipts:
                if not card.demo_pending_evidence:
                    continue
                if now - card.printed_epoch < EVIDENCE_AFTER:
                    continue
                self._land_sms(card)

    def _apply_decision(self, req: ApprovalRequest) -> None:
        """Reflect the answer on the call immediately, so the press is felt."""
        call = self.call(req.call_id) if req.call_id else None
        if call is None:
            return
        if req.approved:
            call.tactic = Tactic.CLOSE
            call.status = "CLOSING"
            call.transcript_tail = [
                "AGENT: I checked with the kitchen - we can do that.",
                "THEM: Great. Send it over and I'll confirm by text.",
            ]
            if req.at_stake_total is not None:
                call.quote = req.at_stake_total
        else:
            call.tactic = Tactic.RESTRUCTURE
            call.status = "HOLDING FLOOR"
            call.transcript_tail = [
                "AGENT: I can't go under that, but I can change the shape of it.",
                "THEM: What did you have in mind?",
            ]

    def _print_for(self, req: ApprovalRequest) -> None:
        """Close the seeded call and print its receipt, UNVERIFIED at birth."""
        call = self.close_call(req.call_id or "")
        if call is None:
            return
        campaign = call.campaign
        unit = call.unit or (campaign.capacity_units if campaign else "units")
        total = call.quote or 0.0

        receipt = Receipt(task=f"{call.business} - {call.campaign_name}")
        if req.approved:
            claim = receipt.claim(
                description=(
                    f"{call.qty} {unit} at {total:,.2f} USD agreed, "
                    f"operator-approved below floor"
                ),
                expected_side_effect=(
                    f"an SMS from {call.phone} naming {call.qty} and {total:,.2f}"
                ),
                required_channels=(Channel.INBOUND_SMS,),
            )
            claim.attach_evidence(
                Evidence(
                    channel=Channel.AGENT_ASSERTION,
                    summary="buyer said yes on the call and agreed to text confirmation",
                    raw={"call_id": call.id, "approval_id": req.id},
                )
            )
        else:
            claim = receipt.claim(
                description=(
                    f"held the {campaign.envelope.min_price:,.2f} floor and offered a "
                    f"restructure instead of a discount"
                    if campaign
                    else "held the floor and offered a restructure"
                ),
                expected_side_effect=(
                    f"a reply from {call.phone} accepting or declining the restructure"
                ),
                required_channels=(Channel.INBOUND_SMS,),
            )
            claim.attach_evidence(
                Evidence(
                    channel=Channel.AGENT_ASSERTION,
                    summary="operator denied the sub-floor price; agent did not concede",
                    raw={"call_id": call.id, "approval_id": req.id},
                )
            )
        _stamp_times(receipt, time.time(), call.elapsed_seconds)

        self.add_receipt(
            ReceiptCard(
                receipt=receipt,
                counterparty=call.business,
                campaign_key=call.campaign_key,
                phone=call.phone,
                duration_seconds=call.elapsed_seconds,
                demo_pending_evidence=req.approved,
            )
        )

    def _land_sms(self, card: ReceiptCard) -> None:
        """The independent channel arrives. This is the only thing that can
        move a stamp from UNCONFIRMED to VERIFIED, here as everywhere."""
        card.demo_pending_evidence = False
        for claim in card.receipt.claims:
            if claim.verdict is not Verdict.UNVERIFIED:
                continue
            claim.attach_evidence(
                Evidence(
                    channel=Channel.INBOUND_SMS,
                    summary=(
                        f"inbound SMS from {card.phone}: "
                        f"\"confirmed, see you then - thanks\""
                    ),
                    raw={"from": card.phone, "body": "confirmed, see you then - thanks"},
                )
            )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _stamp_times(r: Receipt, ended_epoch: float, duration: float) -> None:
    """Give a fixture receipt a coherent clock.

    `Receipt` timestamps itself at construction, so a seeded one would open and
    close in the same second while claiming a seven-minute call. Nobody would
    be misled by it, but a receipt whose own numbers do not add up is a bad
    advertisement for a project about evidence.
    """
    r.started_at = datetime.fromtimestamp(ended_epoch - duration, timezone.utc).isoformat()
    r.ended_at = datetime.fromtimestamp(ended_epoch, timezone.utc).isoformat()


def _seed_ledger(total: int, unit: str, committed: int, held: int) -> CapacityLedger:
    """A ledger posed mid-week. Holds are taken through the real API so the
    numbers are arithmetic the ledger agrees with, not numbers we typed."""
    ledger = CapacityLedger(total, unit=unit, default_ttl_seconds=FIXTURE_TTL)
    if committed:
        h = ledger.hold(committed, ttl_seconds=FIXTURE_TTL)
        if h:
            ledger.commit(h.id)
    if held:
        ledger.hold(held, ttl_seconds=FIXTURE_TTL)
    return ledger


def _verified_receipt() -> ReceiptCard:
    r = Receipt(task="Bayview Family Clinic - standing weekly breakfast order")
    c = r.claim(
        description="240 muffins/week for 6 weeks at 3.65 USD per unit",
        expected_side_effect="an SMS from +14155550163 naming 240/wk and 3.65",
        required_channels=(Channel.INBOUND_SMS,),
    )
    c.attach_evidence(
        Evidence(
            channel=Channel.AGENT_ASSERTION,
            summary="practice manager agreed to 240/wk at 3.65 starting Aug 18",
        )
    )
    c.attach_evidence(
        Evidence(
            channel=Channel.INBOUND_SMS,
            summary='"Confirming 240/wk at 3.65 from the 18th. - Dana, Bayview"',
            raw={"from": "+14155550163", "body": "Confirming 240/wk at 3.65 from the 18th."},
        )
    )
    printed = time.time() - 3600
    _stamp_times(r, printed, 468)
    return ReceiptCard(
        receipt=r,
        counterparty="Bayview Family Clinic",
        campaign_key="restaurant_catering",
        phone="+1 415 555 0163",
        duration_seconds=468,
        printed_epoch=printed,
    )


def _unverified_receipt() -> ReceiptCard:
    r = Receipt(task="Stockton Judo Academy - five-page site with booking")
    c = r.claim(
        description="site build at 1400 USD, live in two weeks, verbally agreed",
        expected_side_effect="a deposit invoice paid, or a written yes by email",
        required_channels=(Channel.INBOUND_EMAIL,),
    )
    c.attach_evidence(
        Evidence(
            channel=Channel.AGENT_ASSERTION,
            summary="sensei said 'sounds good, send me something in writing'",
        )
    )
    printed = time.time() - 2400
    _stamp_times(r, printed, 312)
    return ReceiptCard(
        receipt=r,
        counterparty="Stockton Judo Academy",
        campaign_key="freelance_webdev",
        phone="+1 209 555 0114",
        duration_seconds=312,
        printed_epoch=printed,
    )


def _contradicted_receipt() -> ReceiptCard:
    r = Receipt(task="Harbor Supply Co - discovery call with Head of Support")
    c = r.claim(
        description="discovery call booked for Thu 10:00 PT with Priya N.",
        expected_side_effect="a calendar event readable from the provider API",
        required_channels=(Channel.PROVIDER_API,),
    )
    c.attach_evidence(
        Evidence(
            channel=Channel.AGENT_ASSERTION,
            summary="agent believed the slot was accepted before the line dropped",
        )
    )
    c.attach_evidence(
        Evidence(
            channel=Channel.PROVIDER_API,
            summary="calendar shows no event Thu 10:00; the invite was never sent",
            raw={"events": []},
            supports=False,
        )
    )
    printed = time.time() - 900
    _stamp_times(r, printed, 203)
    return ReceiptCard(
        receipt=r,
        counterparty="Harbor Supply Co",
        campaign_key="startup_outbound",
        phone="+1 628 555 0177",
        duration_seconds=203,
        printed_epoch=printed,
    )


def _seed_leads() -> list[Lead]:
    return [
        Lead(
            business_name="Stockton Judo Academy",
            phone="+1 209 555 0114",
            website=None,
            qualification_reason="no website found in listing; reachable by phone only",
            score=0.95,
            signals=("no website in source listing",),
        ),
        Lead(
            business_name="Ocean Ave Barbers",
            phone="+1 415 555 0148",
            website="instagram.com/oceanavebarbers",
            qualification_reason=(
                "only web presence is an instagram.com page - no site and no "
                "domain they control"
            ),
            score=0.85,
            signals=("listed website is instagram.com",),
        ),
        Lead(
            business_name="Marin Pediatric Dental",
            phone="+1 415 555 0192",
            website="marinpedsdental.com",
            qualification_reason=(
                "single-page site (94 words), no online booking, no online "
                "ordering, no contact form"
            ),
            score=0.85,
            signals=(
                "1 page(s) linked",
                "94 words of visible text",
                "no booking, ordering, checkout or contact-form signals found",
            ),
        ),
        Lead(
            business_name="Presidio Heights Cleaners",
            phone="+1 415 555 0107",
            website="presidioheightscleaners.com",
            qualification_reason=(
                "brochure site (4 pages, 380 words), no online booking, no "
                "online ordering, no contact form"
            ),
            score=0.65,
            signals=(
                "4 page(s) linked",
                "380 words of visible text",
                "no booking, ordering, checkout or contact-form signals found",
            ),
        ),
        Lead(
            business_name="Ridgeline Athletics",
            phone="+1 415 555 0136",
            website="ridgelineathletics.co",
            qualification_reason=(
                "brochure site (3 pages, 210 words), no online booking, no "
                "online ordering, no contact form"
            ),
            score=0.65,
            signals=(
                "3 page(s) linked",
                "210 words of visible text",
                "no booking, ordering, checkout or contact-form signals found",
            ),
        ),
    ]


def seed(board: Board) -> Board:
    """Fill the board with a plausible mid-shift moment.

    Chosen so the whole thesis is on screen at once: one call is stalled on an
    escalation the operator has to answer, and the receipt rail already shows
    all three verdicts - including a CONTRADICTED one, because a board that
    only ever shows green is not evidence of anything.
    """
    board.clear()
    board.demo = True

    board.register_ledger("restaurant_catering", _seed_ledger(600, "muffins/week", 240, 200))
    board.register_ledger("freelance_webdev", _seed_ledger(4, "site builds/month", 2, 1))
    board.register_ledger("startup_outbound", _seed_ledger(12, "demos/week", 5, 0))

    catering = get_campaign("restaurant_catering")

    call = board.open_call(
        LiveCall(
            business="Ridgeline Athletics",
            phone="+1 415 555 0136",
            campaign_key="restaurant_catering",
            status="ON HOLD - ASKED OPERATOR",
            tactic=Tactic.ESCALATE_TO_OPERATOR,
            quote=744.00,
            qty=200,
            unit="muffins/week",
            started_epoch=time.time() - 214,
            transcript_tail=[
                "THEM: We're paying 3.10 a head at the place on Third.",
                "AGENT: Let me check whether I can match that - can you hold a moment?",
            ],
        )
    )
    board.open_call(
        LiveCall(
            business="Ocean Ave Barbers",
            phone="+1 415 555 0148",
            campaign_key="freelance_webdev",
            status="DIALLING",
            tactic=Tactic.ASK_CURRENT_SPEND,
            quote=None,
            qty=1,
            unit="site builds/month",
            started_epoch=time.time() - 11,
            transcript_tail=[],
        )
    )

    # The reason is not written by hand - it is whatever the envelope says,
    # so the rail and the agent cannot drift apart.
    _, reason = catering.envelope.permits(price=3.10, qty=200, when=date(2026, 8, 24))
    req = ApprovalRequest(
        business="Ridgeline Athletics",
        campaign_key="restaurant_catering",
        ask="Drop to 3.10 per muffin on 200/week for 6 weeks?",
        reason=reason,
        options=[
            "APPROVE 3.10/unit - 620.00/wk, 80.00/wk under the floor, 480.00 over the run",
            "DENY, counter at 3.50 with delivery and setup included (inside envelope)",
            "DENY, offer a 4-week trial at 3.72 instead of a 6-week commitment",
        ],
        at_stake="200 muffins/week x 6 weeks - 4,464.00 at list, 200 units held",
        at_stake_total=620.00,
        call_id=call.id,
    )
    board.request_approval(req)

    board.leads = _seed_leads()
    board.receipts = [
        _contradicted_receipt(),
        _unverified_receipt(),
        _verified_receipt(),
    ]
    return board


#: The one board this process serves.
BOARD = Board()
