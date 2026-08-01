"""What kind of task is this, really.

The fixed verticals each carry a hand-written intake form. That does not scale,
and worse, it makes every task wear the same clothes: a dentist booking gets
asked about gross margin because the restaurant campaign needed gross margin.
Noise like that is how a product starts feeling generic.

`TaskProfile` is the smallest description of a task that is enough to decide
*which questions are worth asking*. Six things:

    goal            what the user said, verbatim - never paraphrased away
    exchange        what is actually changing hands
    callee          who picks up the phone
    subject         what is being exchanged, in the callee's words
    done_when       the checkable sentence that ends the errand
    limits          what the agent may commit to with nobody watching

The load-bearing branch is `exchange`. **Not every task is a sale.** A booking
has no margin, no cost per unit and no discount authority, so a generator that
invents those fields is not being thorough, it is padding. `unit_economics_apply`
is derived from the exchange unless the caller overrides it, and
`src/generator/questions.py` treats it as a hard filter, not a hint.

Two heuristics fail deliberately in one direction, matching the convention in
`CLAUDE.md`:

* **Physical goods fails toward True.** Asking a units-vs-headcount question
  about a task with no items is a wasted sentence. Skipping it on a task that
  has them cost $311 on a live call, because "thirty" turned into thirty muffins
  before anyone noticed it meant thirty people.
* **Classification fails toward asking more.** An unrecognised goal lands on
  the generic question set rather than a narrow one.

Matching is token-based, never bare substring: `src/tasks/triage.py` already
learned that "Stockton" contains "tock".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any

from src.business.campaign import (
    Campaign,
    CloseCondition,
    DiscoveryStrategy,
    Envelope,
)

#: How far out an unspecified delivery window runs.
DEFAULT_WINDOW_DAYS = 90

#: Stand-in floor for tasks with no unit economics. `Envelope` requires a
#: positive `min_price` to be coherent, and `STARTUP_OUTBOUND` in
#: `src/business/campaign.py` already uses this trick for its free demo: the
#: number is not a price, it is "pricing is not on the table here".
NOMINAL_PRICE = 0.01


class Exchange(str, Enum):
    """What is actually changing hands. The branch everything else hangs off."""

    SALE = "sale"
    """We are selling on the operator's behalf. Margin is real, a floor exists,
    and discount authority has to be bounded. The only kind with unit
    economics by default."""

    PURCHASE = "purchase"
    """We are buying or ordering for the user. There is a spend ceiling but no
    margin - nobody's profit is at stake on our side of the call."""

    BOOKING = "booking"
    """A slot in somebody else's calendar. Dates, names and eligibility
    matter; cost per unit does not exist."""

    INFORMATION = "information"
    """We need an answer and nothing else changes. The deliverable is the
    answer, written down somewhere we can check."""

    ADMIN = "admin"
    """Changing a record that already exists - cancel, reschedule, dispute,
    correct. Needs a reference number more than it needs a price."""


#: Exchanges where a per-unit cost, a floor and a discount ladder are real.
ECONOMIC_EXCHANGES: frozenset[Exchange] = frozenset({Exchange.SALE})

#: What ends each kind of task, as an observable side effect. Every value here
#: is a `CloseCondition`, so it inherits the rule that a campaign cannot close
#: on the agent's own say-so.
DEFAULT_CLOSE: dict[Exchange, CloseCondition] = {
    Exchange.SALE: CloseCondition.WRITTEN_CONFIRMATION,
    Exchange.PURCHASE: CloseCondition.WRITTEN_CONFIRMATION,
    Exchange.BOOKING: CloseCondition.BOOKED_MEETING,
    Exchange.INFORMATION: CloseCondition.WRITTEN_CONFIRMATION,
    Exchange.ADMIN: CloseCondition.WRITTEN_CONFIRMATION,
}

DEFAULT_UNITS: dict[Exchange, str] = {
    Exchange.SALE: "units",
    Exchange.PURCHASE: "items",
    Exchange.BOOKING: "appointments",
    Exchange.INFORMATION: "answers",
    Exchange.ADMIN: "changes",
}


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    """Lowercased word tokens. Tokenising is the word-boundary guarantee."""
    return set(_WORD.findall(text.lower()))


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text.lower()) is not None


#: Checked in this order. Sale markers win over booking markers because "book
#: twelve discovery calls with prospects" is a sales campaign whose close
#: condition happens to be a meeting - the economics are still real.
_MARKERS: list[tuple[Exchange, frozenset[str], tuple[str, ...]]] = [
    (
        Exchange.SALE,
        frozenset(
            {
                "sell", "selling", "sale", "sales", "pitch", "outbound",
                "prospect", "prospects", "prospecting", "lead", "leads",
                "upsell", "quote", "quoting", "client", "clients", "customer",
                "customers", "campaign",
            }
        ),
        ("close deals", "win business", "drum up", "new business", "cold call"),
    ),
    (
        Exchange.ADMIN,
        frozenset(
            {
                "cancel", "reschedule", "refund", "dispute", "chargeback",
                "complain", "complaint", "escalate", "correct", "amend",
            }
        ),
        ("change my", "close my account", "move my appointment", "get my money back"),
    ),
    (
        Exchange.BOOKING,
        frozenset(
            {
                "book", "booking", "appointment", "appointments", "reserve",
                "reservation", "schedule", "slot", "table", "lesson", "class",
                "dentist", "doctor", "haircut", "trim", "checkup", "cleaning",
                "consult", "consultation", "viewing", "session",
            }
        ),
        ("tee time", "sign up for", "get on the calendar", "set up a time"),
    ),
    (
        Exchange.PURCHASE,
        frozenset(
            {
                "order", "buy", "purchase", "cater", "catering", "deliver",
                "delivery", "pickup", "restock", "rent", "hire", "source",
            }
        ),
        ("pick up", "place an order", "have delivered"),
    ),
    (
        Exchange.INFORMATION,
        frozenset(
            {
                "ask", "confirm", "check", "verify", "enquire", "inquire",
                "hours", "availability", "whether", "price", "prices",
            }
        ),
        ("find out", "do they", "are they", "how much does", "call around"),
    ),
]

#: Nouns that mean somebody has to count physical things. Deliberately broad -
#: over-asking the units question is cheap.
_GOODS_TOKENS: frozenset[str] = frozenset(
    {
        "muffin", "muffins", "cake", "cakes", "cupcake", "cupcakes", "bagel",
        "bagels", "pizza", "pizzas", "sandwich", "sandwiches", "platter",
        "platters", "tray", "trays", "box", "boxes", "dozen", "dozens",
        "lunch", "lunches", "breakfast", "meal", "meals", "coffee", "pastry",
        "pastries", "bouquet", "bouquets", "flower", "flowers", "print",
        "prints", "copies", "shirt", "shirts", "unit", "units", "item",
        "items", "part", "parts", "pallet", "pallets", "case", "cases",
        "bottle", "bottles", "kit", "kits", "sign", "signs", "badge",
        "badges", "chair", "chairs", "widget", "widgets", "portion",
        "portions", "serving", "servings", "head", "heads", "burrito",
        "burritos", "taco", "tacos", "salad", "salads", "donut", "donuts",
        "cater", "catering", "food", "drinks", "snacks", "supplies", "stock",
    }
)

#: Words that mean a number in the goal is a headcount, which is precisely the
#: ambiguity that has to be asked about rather than resolved silently.
_PEOPLE_TOKENS: frozenset[str] = frozenset(
    {"people", "person", "guests", "attendees", "staff", "team", "heads", "pax"}
)

_QUANTITY = re.compile(r"\b\d+\b")


def classify(goal: str) -> Exchange:
    """Best guess at what kind of task this is. Generic when unsure."""
    toks = _tokens(goal)
    for exchange, words, phrases in _MARKERS:
        if toks & words:
            return exchange
        if any(_has_phrase(goal, p) for p in phrases):
            return exchange
    return Exchange.INFORMATION


def mentions_physical_goods(goal: str, exchange: Exchange | None = None) -> bool:
    """Does completing this task require counting physical things?

    Biased toward True. The cost of a false positive is one extra question;
    the cost of a false negative is a mispriced order nobody can see is wrong.
    """
    toks = _tokens(goal)
    if toks & _GOODS_TOKENS:
        return True
    if toks & _PEOPLE_TOKENS and _QUANTITY.search(goal):
        # "lunch for 30 people" - the count is people, the order is items, and
        # that gap is the entire failure mode.
        return True
    if exchange is Exchange.PURCHASE and _QUANTITY.search(goal):
        return True
    return False


# ---------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardLimits:
    """What the agent may commit to alone. Everything optional, because an
    unstated limit must stay unstated rather than become a plausible default -
    the same rule `src/webapp/intake.py` enforces on the interview."""

    earliest_date: date | None = None
    latest_date: date | None = None
    max_qty: int | None = None
    spend_ceiling: float | None = None
    """Total the user will not exceed. A purchase limit, not a margin."""

    min_price: float | None = None
    """Per-unit floor. Only meaningful when unit economics apply."""

    max_discount_pct: float | None = None
    currency: str = "USD"
    never_do: tuple[str, ...] = ()
    """Plain-English prohibitions, read to the agent as absolute."""

    def to_envelope(self, *, economics: bool) -> Envelope:
        """A coherent `Envelope`, filling only what `Envelope` requires.

        Missing authority resolves to *no* authority: an unstated discount cap
        becomes 0%, not "probably fine". An unstated floor on a task with no
        economics becomes `NOMINAL_PRICE`, which reads as "price is not on the
        table" rather than as a real number somebody chose.
        """
        today = date.today()
        earliest = self.earliest_date or today
        latest = self.latest_date or (earliest + timedelta(days=DEFAULT_WINDOW_DAYS))
        if latest < earliest:
            latest = earliest

        floor = self.min_price if (self.min_price or 0) > 0 else NOMINAL_PRICE

        discount = self.max_discount_pct if economics else 0.0
        if discount is None:
            discount = 0.0
        discount = max(0.0, min(100.0, float(discount)))

        return Envelope(
            min_price=float(floor),
            max_qty=int(self.max_qty) if (self.max_qty or 0) > 0 else 1,
            earliest_date=earliest,
            latest_date=latest,
            max_discount_pct=discount,
            currency=self.currency or "USD",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "earliest_date": self.earliest_date.isoformat() if self.earliest_date else None,
            "latest_date": self.latest_date.isoformat() if self.latest_date else None,
            "max_qty": self.max_qty,
            "spend_ceiling": self.spend_ceiling,
            "min_price": self.min_price,
            "max_discount_pct": self.max_discount_pct,
            "currency": self.currency,
            "never_do": list(self.never_do),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> HardLimits:
        data = data or {}

        def _date(key: str) -> date | None:
            raw = data.get(key)
            if not raw:
                return None
            try:
                return date.fromisoformat(str(raw)[:10])
            except ValueError:
                return None

        def _num(key: str) -> float | None:
            raw = data.get(key)
            try:
                return float(raw) if raw not in (None, "") else None
            except (TypeError, ValueError):
                return None

        qty = _num("max_qty")
        never = data.get("never_do") or ()
        if isinstance(never, str):
            never = [never]
        return cls(
            earliest_date=_date("earliest_date"),
            latest_date=_date("latest_date"),
            max_qty=int(qty) if qty else None,
            spend_ceiling=_num("spend_ceiling"),
            min_price=_num("min_price"),
            max_discount_pct=_num("max_discount_pct"),
            currency=str(data.get("currency") or "USD"),
            never_do=tuple(str(n) for n in never if str(n).strip()),
        )


def slugify(text: str, *, fallback: str = "task") -> str:
    """A filesystem-safe directory name. Also the traversal guard: the output
    of this function cannot contain a separator or a dot segment."""
    out = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    out = re.sub(r"-{2,}", "-", out)[:48].strip("-")
    return out or fallback


@dataclass(frozen=True)
class TaskProfile:
    """Enough about a task to decide what to ask and what to show."""

    goal: str
    exchange: Exchange
    callee: str
    """Who picks up. Specific enough that a human could look up the number."""

    subject: str = ""
    """What is being exchanged, phrased as the person on the phone would hear
    it. Empty is allowed at draft time and flagged by `problems()`."""

    done_when: str = ""
    """The checkable sentence. If you cannot write it, the task has no receipt
    - which is the same standard `Claim.expected_side_effect` holds itself to."""

    physical_goods: bool = False
    unit_label: str = ""
    unit_economics_override: bool | None = None
    """`None` means derive from `exchange`. Set it only when the caller knows
    better than the classifier - a booking with a real per-seat margin, say."""

    close_condition: CloseCondition | None = None
    discovery: DiscoveryStrategy = DiscoveryStrategy.SEEDED_LIST
    limits: HardLimits = field(default_factory=HardLimits)
    vertical: str = ""
    notes: str = ""

    # -- derived ------------------------------------------------------

    @property
    def unit_economics_apply(self) -> bool:
        """The filter `questions.py` applies to every generated question."""
        if self.unit_economics_override is not None:
            return self.unit_economics_override
        return self.exchange in ECONOMIC_EXCHANGES

    @property
    def slug(self) -> str:
        return slugify(self.goal, fallback=self.exchange.value)

    @property
    def units(self) -> str:
        return self.unit_label or DEFAULT_UNITS[self.exchange]

    @property
    def closes_on(self) -> CloseCondition:
        return self.close_condition or DEFAULT_CLOSE[self.exchange]

    def required_fields(self) -> tuple[str, ...]:
        """Field ids this task cannot be run without.

        Every id here must have an entry in `questions.CANONICAL`;
        `tests/test_generator.py` pins that, so adding a requirement without a
        way to ask for it fails loudly instead of producing an unfillable form.
        """
        out: list[str] = ["callee", "done_definition", "confirm_to", "deadline"]

        if self.physical_goods:
            # The $311 question. Not conditional on anything else.
            out += ["quantity", "units_basis"]

        if self.unit_economics_apply:
            out += ["unit_price_floor", "max_discount_pct"]

        if self.exchange is Exchange.PURCHASE:
            out.append("spend_ceiling")
        elif self.exchange is Exchange.BOOKING:
            out += ["preferred_windows", "on_whose_behalf", "urgency"]
        elif self.exchange is Exchange.INFORMATION:
            out.append("questions_to_ask")
        elif self.exchange is Exchange.ADMIN:
            out += ["account_reference", "fallback_if_refused"]

        seen: dict[str, None] = {}
        for f in out:
            seen.setdefault(f, None)
        return tuple(seen)

    # -- validity -----------------------------------------------------

    def problems(self) -> list[str]:
        out: list[str] = []
        if not self.goal.strip():
            out.append("goal must not be empty")
        if not self.callee.strip():
            out.append("callee must not be empty - somebody has to pick up")
        if self.unit_economics_apply and not self.physical_goods and not self.unit_label:
            out.append("unit economics apply but no unit_label was set")
        out += [f"limits: {p}" for p in self.to_envelope().problems()]
        return out

    @property
    def is_valid(self) -> bool:
        return not self.problems()

    # -- conversions --------------------------------------------------

    def to_envelope(self) -> Envelope:
        return self.limits.to_envelope(economics=self.unit_economics_apply)

    def to_campaign(self, name: str | None = None) -> Campaign:
        """Hand the generated profile to the engine that already exists.

        This is the whole point of the meta-layer: it emits configuration for
        `src/business/campaign.py`, it does not become a second engine.
        """
        return Campaign(
            name=name or (self.goal.strip()[:80] or "generated task"),
            vertical=self.vertical or self.exchange.value,
            icp=self.callee.strip() or "unspecified",
            offer=self.subject.strip() or self.goal.strip(),
            discovery=self.discovery,
            close_condition=self.closes_on,
            capacity_units=self.units,
            envelope=self.to_envelope(),
            notes=self.notes,
            tags=(self.exchange.value,) + (("goods",) if self.physical_goods else ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "slug": self.slug,
            "exchange": self.exchange.value,
            "callee": self.callee,
            "subject": self.subject,
            "done_when": self.done_when,
            "physical_goods": self.physical_goods,
            "unit_label": self.unit_label,
            "units": self.units,
            "unit_economics_apply": self.unit_economics_apply,
            "close_condition": self.closes_on.value,
            "discovery": self.discovery.value,
            "limits": self.limits.to_dict(),
            "vertical": self.vertical,
            "notes": self.notes,
            "required_fields": list(self.required_fields()),
            "problems": self.problems(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskProfile:
        """Tolerant of anything a language model might hand back."""
        goal = str(data.get("goal") or "").strip()

        raw_exchange = str(data.get("exchange") or "").strip().lower()
        try:
            exchange = Exchange(raw_exchange)
        except ValueError:
            exchange = classify(goal)

        goods = data.get("physical_goods")
        if goods is None:
            goods = mentions_physical_goods(goal, exchange)

        raw_close = str(data.get("close_condition") or "").strip().lower()
        try:
            close = CloseCondition(raw_close)
        except ValueError:
            close = None

        raw_disc = str(data.get("discovery") or "").strip().lower()
        try:
            discovery = DiscoveryStrategy(raw_disc)
        except ValueError:
            discovery = DiscoveryStrategy.SEEDED_LIST

        override = data.get("unit_economics_apply")
        if isinstance(override, str):
            override = override.strip().lower() in ("true", "yes", "1")
        # Only record an override when it disagrees with the derived value, so
        # a model echoing the default does not freeze it in place.
        derived = exchange in ECONOMIC_EXCHANGES
        override = None if override is None or bool(override) == derived else bool(override)

        return cls(
            goal=goal,
            exchange=exchange,
            callee=str(data.get("callee") or "").strip(),
            subject=str(data.get("subject") or "").strip(),
            done_when=str(data.get("done_when") or "").strip(),
            physical_goods=bool(goods),
            unit_label=str(data.get("unit_label") or "").strip(),
            unit_economics_override=override,
            close_condition=close,
            discovery=discovery,
            limits=HardLimits.from_dict(data.get("limits")),
            vertical=str(data.get("vertical") or "").strip(),
            notes=str(data.get("notes") or "").strip(),
        )


def heuristic_profile(goal: str) -> TaskProfile:
    """A usable profile with no model involved.

    This is the offline spine. Conference wifi is a real threat model, and a
    meta-layer that stops working when the LLM 502s takes the demo with it.
    """
    goal = (goal or "").strip()
    exchange = classify(goal)
    return TaskProfile(
        goal=goal,
        exchange=exchange,
        callee="",
        subject=goal,
        physical_goods=mentions_physical_goods(goal, exchange),
    )
