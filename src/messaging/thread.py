"""A text conversation that inherits the constraints of the call it follows.

The whole point of this module is continuity. When a call ends, the limits it
ran under are sitting in three places - the campaign `Envelope`, the
`CostModel`, and the `Gate`'s `Facts` - and every one of them is in memory in a
process that is about to exit. If the SMS follow-up starts from a blank slate,
then "we couldn't agree a price on the phone" quietly becomes "we agreed it by
text", and the envelope was never a boundary at all.

So a `Thread` snapshots all of it, and `ThreadStore` writes the snapshot to
sqlite. A restart rebuilds the same `Gate` at the same phase with the same
facts, and the same `CostModel` with the same floor. The reconstruction is
exact, not approximate: per-unit costs are persisted individually as strings
because `CostModel.to_dict()` only exposes their sum, and a floor rebuilt from
a rounded sum is not the same floor.

Two deliberate asymmetries, both fail-toward-refusing-a-deal:

- **`budget_floor` is the *highest* number the prospect has ever named**, not
  the most recent one. A live call countered a stated $385 with $74 and threw
  away $311; taking the max means a later "actually I only have $200" cannot
  reopen that hole. The cost is a deal we decline that we might have made. That
  is the cheaper mistake.
- **Escalation is not a field.** It is unavailable on this channel by
  construction. `CallSession.allow_escalation` is recorded in `meta` for the
  audit trail and then ignored - nobody is watching an SMS thread, so "I'll
  check with the owner" is a lie with a delay on it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields as dc_fields, replace
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from src.agents.flow import Facts, Gate, Phase
from src.business.campaign import (
    Campaign,
    CloseCondition,
    DiscoveryStrategy,
    Envelope,
)
from src.business.pricing import CostModel

#: Where threads live when nobody says otherwise. Under `evidence/` because a
#: thread is part of the audit trail of a run, not scratch state.
DEFAULT_DB = Path(__file__).resolve().parents[2] / "evidence" / "messaging.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalise_phone(number: str) -> str:
    """Loose E.164. Threads are keyed by this, so it has to be stable.

    "(415) 555-0134", "415-555-0134" and "+14155550134" are one prospect, and
    keying them separately would let the same conversation run twice with two
    different sets of inherited limits.
    """
    raw = (number or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return raw
    if raw.startswith("+"):
        return "+" + digits
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return "+" + digits


class Direction(str, Enum):
    INBOUND = "inbound"
    """From the prospect. The only direction that can ever verify anything."""

    OUTBOUND = "outbound"
    """From us. Agent assertion, forever."""


#: Statuses a message row can carry. Free-form by design - `send.SendStatus`
#: values land here verbatim so the dashboard can show *why* nothing was sent.
STATUS_RECORDED = "recorded"


@dataclass
class Message:
    """One text. Persisted verbatim, before any interpretation."""

    direction: Direction
    text: str
    at: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _uid("msg"))
    status: str = STATUS_RECORDED
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "direction": self.direction.value,
            "text": self.text,
            "at": self.at,
            "status": self.status,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(
            direction=Direction(d["direction"]),
            text=d.get("text", ""),
            at=d.get("at") or _now(),
            id=d.get("id") or _uid("msg"),
            status=d.get("status", STATUS_RECORDED),
            meta=d.get("meta") or {},
        )


@dataclass
class ClaimRef:
    """A pointer to a `Claim` living in some `Receipt`, plus what would prove it.

    The `Claim` object itself is not persisted here - it belongs to a receipt,
    and duplicating it would create a second place a verdict could come from,
    which is exactly the thing this codebase refuses to have. What is persisted
    is the id and the tokens an inbound text must contain, so that after a
    restart we can still say *which* message would have verified *what*.
    """

    claim_id: str
    description: str
    tokens: tuple[str, ...] = ()
    verdict: str = "UNVERIFIED"
    """Last verdict observed. A cache for display; never an input to anything."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "description": self.description,
            "tokens": list(self.tokens),
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClaimRef:
        return cls(
            claim_id=d["claim_id"],
            description=d.get("description", ""),
            tokens=tuple(d.get("tokens") or ()),
            verdict=d.get("verdict", "UNVERIFIED"),
        )


# -- (de)serialising the constraint objects ------------------------------


def _cost_model_to_dict(costs: CostModel) -> dict[str, Any]:
    """Every field, as strings. `CostModel.to_dict()` sums the components."""
    return {
        "materials_per_unit": str(costs.materials_per_unit),
        "labor_per_unit": str(costs.labor_per_unit),
        "transport_per_unit": str(costs.transport_per_unit),
        "min_margin_pct": str(costs.min_margin_pct),
        "target_margin_pct": str(costs.target_margin_pct),
        "unit": costs.unit,
        "currency": costs.currency,
    }


def _cost_model_from_dict(d: dict[str, Any]) -> CostModel:
    return CostModel(
        materials_per_unit=Decimal(d["materials_per_unit"]),
        labor_per_unit=Decimal(d["labor_per_unit"]),
        transport_per_unit=Decimal(d["transport_per_unit"]),
        min_margin_pct=Decimal(d["min_margin_pct"]),
        target_margin_pct=Decimal(d["target_margin_pct"]),
        unit=d.get("unit", "unit"),
        currency=d.get("currency", "USD"),
    )


#: Read off the dataclass rather than listed by hand. `Facts` is the contract
#: between the call and this thread, and it grows - a hand-written field list
#: drops the new ones silently on the first restart after someone adds one, and
#: a precondition that vanishes is a precondition that stops blocking.
_FACT_FIELDS: frozenset[str] = frozenset(f.name for f in dc_fields(Facts))


def _campaign_from_dict(d: dict[str, Any]) -> Campaign:
    env = d["envelope"]
    return Campaign(
        name=d["name"],
        vertical=d.get("vertical", ""),
        icp=d.get("icp", ""),
        offer=d.get("offer", ""),
        discovery=DiscoveryStrategy(d["discovery"]),
        close_condition=CloseCondition(d["close_condition"]),
        capacity_units=d.get("capacity_units", "units"),
        envelope=Envelope(
            min_price=float(env["min_price"]),
            max_qty=int(env["max_qty"]),
            earliest_date=date.fromisoformat(env["earliest_date"]),
            latest_date=date.fromisoformat(env["latest_date"]),
            max_discount_pct=float(env["max_discount_pct"]),
            currency=env.get("currency", "USD"),
        ),
        notes=d.get("notes", ""),
        tags=tuple(d.get("tags") or ()),
    )


# -- the thread ----------------------------------------------------------


@dataclass
class Thread:
    """One prospect, one phone number, one inherited set of limits."""

    phone: str
    campaign: Campaign | None = None
    costs: CostModel | None = None
    qty: int = 0
    """Units the call was working with. Not a promise - see `hold_id`."""

    hold_id: str | None = None
    """The capacity hold the call held. It has almost certainly expired by the
    time a text arrives, which is correct: a hold is scoped to a live call. It
    is kept so the thread can say *which* reservation the numbers came from,
    and so a re-hold can be attempted before anything is re-promised."""

    phase: Phase = Phase.OPENING
    facts: Facts = field(default_factory=Facts)
    approved_total: float | None = None
    """Lowest total an operator explicitly authorised *during the call*. It
    carries over; nothing on this channel can add to it."""

    stated_budgets: list[float] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    claims: list[ClaimRef] = field(default_factory=list)

    opted_out: bool = False
    """They texted STOP. Nothing further goes out, ever."""

    call_receipt_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    revision: int = 0

    # -- construction -------------------------------------------------

    @classmethod
    def from_call_session(
        cls, session: Any, phone: str, *, call_receipt_id: str | None = None
    ) -> Thread:
        """Snapshot a finished (or abandoned) `CallSession` into a text thread.

        Typed loosely on purpose: `src.agents.sales_agent` imports livekit, and
        a thread has no business dragging a voice stack into a webhook process.
        Everything read here is a plain attribute.
        """
        state = getattr(session, "state", None)
        gate = getattr(session, "gate", None) or Gate()
        facts = replace(gate.facts)

        budgets: list[float] = []
        if facts.their_budget:
            budgets.append(float(facts.their_budget))
        if state is not None and getattr(state, "their_stated_budget", None):
            budgets.append(float(state.their_stated_budget))

        hold = getattr(session, "hold", None)
        thread = cls(
            phone=normalise_phone(phone),
            campaign=getattr(session, "campaign", None),
            costs=getattr(session, "costs", None),
            qty=int(getattr(state, "qty", 0) or 0),
            hold_id=getattr(hold, "id", None),
            phase=gate.phase,
            facts=facts,
            approved_total=getattr(session, "approved_total", None),
            stated_budgets=budgets,
            call_receipt_id=call_receipt_id
            or getattr(getattr(session, "receipt", None), "id", None),
            meta={
                "call_allowed_escalation": bool(
                    getattr(session, "allow_escalation", False)
                ),
                "escalation_on_sms": False,
                "competitor_named": getattr(state, "competitor_named", None),
                "concessions_made": getattr(state, "concessions_made", 0),
                "current_total": getattr(state, "current_total", None),
            },
        )
        thread._sync_budget()
        return thread

    # -- inherited limits ----------------------------------------------

    @property
    def gate(self) -> Gate:
        """A `Gate` at the phase the call left off, with the call's facts.

        Rebuilt on read rather than held, so that mutating the returned gate
        cannot silently advance the persisted thread. Anything a caller wants
        kept has to go through `note_*` on the thread.
        """
        return Gate(phase=self.phase, facts=replace(self.facts))

    @property
    def budget_floor(self) -> float | None:
        """The highest total the prospect has ever said they would pay.

        Read the module docstring for why this is a max and not a last-write.
        """
        if not self.stated_budgets:
            return self.facts.their_budget
        return max(self.stated_budgets)

    def _sync_budget(self) -> None:
        floor = self.budget_floor
        if floor:
            self.facts.their_budget = floor

    def note_stated_budget(self, amount: float) -> None:
        """Record a number the prospect named. Only ever raises the floor."""
        if amount and amount > 0:
            self.stated_budgets.append(float(amount))
            self.facts.asked_current_spend = True
            self._sync_budget()

    def floor_total(self) -> Decimal | None:
        if self.costs is None or self.qty < 1:
            return None
        return self.costs.floor_price(self.qty)

    def target_total(self) -> Decimal | None:
        if self.costs is None or self.qty < 1:
            return None
        return self.costs.target_price(self.qty)

    @property
    def can_quote_at_all(self) -> bool:
        """False when we have no way to validate a number.

        A thread with no cost model or no quantity is not a thread that may say
        a price. The closer enforces this by refusing every figure, which is
        the right failure: a number nobody can check is the fabrication path.
        """
        return self.costs is not None and self.qty >= 1

    def constraints_summary(self) -> str:
        """The limits, written so they can be dropped into a system prompt."""
        lines = [f"phase at end of call: {self.phase.value}"]
        if self.campaign:
            env = self.campaign.envelope
            lines.append(f"campaign: {self.campaign.name}")
            lines.append(f"offer: {self.campaign.offer}")
            lines.append(
                f"envelope: min {env.currency} {env.min_price:.2f}/unit, "
                f"max qty {env.max_qty}, delivery between {env.earliest_date} "
                f"and {env.latest_date}, max discount {env.max_discount_pct:.0f}%"
            )
        if self.qty:
            lines.append(f"quantity under discussion: {self.qty}")
        floor, target = self.floor_total(), self.target_total()
        if floor is not None and target is not None:
            cur = self.costs.currency if self.costs else "USD"
            lines.append(
                f"hard floor for {self.qty}: {cur} {floor} "
                f"(you may never state a total below this)"
            )
            lines.append(f"target total for {self.qty}: {cur} {target}")
        if self.budget_floor:
            lines.append(
                f"they already said they would pay {self.budget_floor:.2f} - "
                "never state a total below that number"
            )
        if self.approved_total:
            lines.append(
                f"the owner authorised {self.approved_total:.2f} during the call; "
                "anything at or above that is cleared"
            )
        gaps = self.facts.missing_for_quote()
        if gaps:
            lines.append("still unknown: " + "; ".join(gaps))
        return "\n".join(lines)

    # -- messages -------------------------------------------------------

    def add(
        self,
        direction: Direction,
        text: str,
        *,
        status: str = STATUS_RECORDED,
        meta: dict[str, Any] | None = None,
    ) -> Message:
        msg = Message(
            direction=direction, text=text, status=status, meta=meta or {}
        )
        self.messages.append(msg)
        self.updated_at = msg.at
        return msg

    def add_inbound(self, text: str, **meta: Any) -> Message:
        return self.add(Direction.INBOUND, text, meta=meta or {})

    def add_outbound(self, text: str, *, status: str = STATUS_RECORDED, **meta: Any) -> Message:
        return self.add(Direction.OUTBOUND, text, status=status, meta=meta or {})

    @property
    def last_inbound(self) -> Message | None:
        for m in reversed(self.messages):
            if m.direction is Direction.INBOUND:
                return m
        return None

    @property
    def outbound_since_inbound(self) -> int:
        """How many texts we have sent without hearing back.

        Used as a stop condition. "Don't be pushy" is unenforceable in a
        prompt; a counter is not.
        """
        n = 0
        for m in reversed(self.messages):
            if m.direction is Direction.INBOUND:
                break
            n += 1
        return n

    def history_for_model(self, limit: int = 24) -> list[dict[str, str]]:
        """Chat turns, prospect as `user`, us as `assistant`."""
        return [
            {
                "role": "assistant" if m.direction is Direction.OUTBOUND else "user",
                "text": m.text,
            }
            for m in self.messages[-limit:]
        ]

    # -- claims ---------------------------------------------------------

    def track_claim(self, claim_id: str, description: str, tokens: Iterable[str]) -> ClaimRef:
        for ref in self.claims:
            if ref.claim_id == claim_id:
                ref.tokens = tuple(tokens)
                return ref
        ref = ClaimRef(claim_id=claim_id, description=description, tokens=tuple(tokens))
        self.claims.append(ref)
        return ref

    @property
    def open_claims(self) -> list[ClaimRef]:
        return [c for c in self.claims if c.verdict != "VERIFIED"]

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "phone": self.phone,
            "campaign": self.campaign.to_dict() if self.campaign else None,
            "costs": _cost_model_to_dict(self.costs) if self.costs else None,
            "qty": self.qty,
            "hold_id": self.hold_id,
            "phase": self.phase.value,
            "facts": asdict(self.facts),
            "approved_total": self.approved_total,
            "stated_budgets": list(self.stated_budgets),
            "messages": [m.to_dict() for m in self.messages],
            "claims": [c.to_dict() for c in self.claims],
            "opted_out": self.opted_out,
            "call_receipt_id": self.call_receipt_id,
            "meta": self.meta,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Thread:
        # Unknown keys are dropped rather than raising: a thread written by a
        # newer build must still load, minus the facts this build cannot honour.
        f = {k: v for k, v in (d.get("facts") or {}).items() if k in _FACT_FIELDS}
        return cls(
            phone=d["phone"],
            campaign=_campaign_from_dict(d["campaign"]) if d.get("campaign") else None,
            costs=_cost_model_from_dict(d["costs"]) if d.get("costs") else None,
            qty=int(d.get("qty") or 0),
            hold_id=d.get("hold_id"),
            phase=Phase(d.get("phase", Phase.OPENING.value)),
            facts=Facts(**f),
            approved_total=d.get("approved_total"),
            stated_budgets=[float(x) for x in (d.get("stated_budgets") or [])],
            messages=[Message.from_dict(m) for m in (d.get("messages") or [])],
            claims=[ClaimRef.from_dict(c) for c in (d.get("claims") or [])],
            opted_out=bool(d.get("opted_out", False)),
            call_receipt_id=d.get("call_receipt_id"),
            meta=d.get("meta") or {},
            created_at=d.get("created_at") or _now(),
            updated_at=d.get("updated_at") or _now(),
            revision=int(d.get("revision") or 0),
        )

    def summary(self) -> dict[str, Any]:
        """The row a dashboard list view renders."""
        floor, target = self.floor_total(), self.target_total()
        last = self.messages[-1] if self.messages else None
        return {
            "phone": self.phone,
            "campaign": self.campaign.name if self.campaign else None,
            "phase": self.phase.value,
            "qty": self.qty,
            "unit": self.costs.unit if self.costs else None,
            "currency": self.costs.currency if self.costs else "USD",
            "floor_total": str(floor) if floor is not None else None,
            "target_total": str(target) if target is not None else None,
            "their_budget": self.budget_floor,
            "approved_total": self.approved_total,
            "can_quote": self.can_quote_at_all,
            "opted_out": self.opted_out,
            "messages": len(self.messages),
            "awaiting_reply": self.outbound_since_inbound,
            "open_claims": len(self.open_claims),
            "claims": [c.to_dict() for c in self.claims],
            "last_message": last.to_dict() if last else None,
            "call_receipt_id": self.call_receipt_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision": self.revision,
        }


# -- the store -----------------------------------------------------------


class ThreadStore:
    """sqlite-backed thread persistence. stdlib only, one file, no migrations.

    A whole thread is one JSON row. That is a deliberate non-normalisation: the
    thing being persisted is a *snapshot of constraints*, and splitting it
    across tables invites the failure where half of it is restored and the
    limits come back subtly wider than they went in.
    """

    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS threads (
                   phone      TEXT PRIMARY KEY,
                   updated_at TEXT NOT NULL,
                   revision   INTEGER NOT NULL DEFAULT 0,
                   data       TEXT NOT NULL
               )"""
        )
        self._db.commit()

    # -- writes ---------------------------------------------------------

    def save(self, thread: Thread) -> Thread:
        with self._lock:
            thread.revision += 1
            thread.updated_at = _now()
            self._db.execute(
                "INSERT INTO threads (phone, updated_at, revision, data) "
                "VALUES (?,?,?,?) ON CONFLICT(phone) DO UPDATE SET "
                "updated_at=excluded.updated_at, revision=excluded.revision, "
                "data=excluded.data",
                (
                    thread.phone,
                    thread.updated_at,
                    thread.revision,
                    json.dumps(thread.to_dict()),
                ),
            )
            self._db.commit()
        return thread

    def delete(self, phone: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM threads WHERE phone = ?", (normalise_phone(phone),)
            )
            self._db.commit()
            return cur.rowcount > 0

    # -- reads ----------------------------------------------------------

    def load(self, phone: str) -> Thread | None:
        key = normalise_phone(phone)
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM threads WHERE phone = ?", (key,)
            ).fetchone()
        return Thread.from_dict(json.loads(row[0])) if row else None

    def get_or_create(self, phone: str, **defaults: Any) -> Thread:
        """Load, or start a bare thread that can hold messages but not quote.

        A thread created this way has no cost model, so `can_quote_at_all` is
        False and every figure the model tries to emit is refused. That is the
        correct default for a text arriving from a number we have no call
        context for.
        """
        existing = self.load(phone)
        if existing is not None:
            return existing
        thread = Thread(phone=normalise_phone(phone), **defaults)
        return self.save(thread)

    def all(self) -> list[Thread]:
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM threads ORDER BY updated_at DESC"
            ).fetchall()
        return [Thread.from_dict(json.loads(r[0])) for r in rows]

    def state_key(self) -> str:
        """A short hash that changes only when something actually changed.

        Pollers diff this instead of re-rendering. It is computed from
        (phone, revision, updated_at) tuples, so a save that writes identical
        content still bumps it - a redraw nobody needed is cheap, a missed
        update is not.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT phone, revision, updated_at FROM threads ORDER BY phone"
            ).fetchall()
        if not rows:
            return "empty"
        blob = "|".join(f"{p}:{r}:{u}" for p, r, u in rows)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def close(self) -> None:
        with self._lock:
            self._db.close()
