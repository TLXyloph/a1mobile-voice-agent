"""What the owner is asked, in what order, and what the answers become.

Today the business profile is a row of env vars - COST_MATERIALS, CAPACITY_TOTAL,
MIN_MARGIN_PCT. The person who knows those numbers is the one person who will
never open a shell to set them. So the numbers stay at whatever the demo was
seeded with, and the agent negotiates on behalf of a business that does not exist.

This module is the translation layer: plain English in, `CostModel` /
`CapacityLedger` / `Envelope` out. Two things follow from that.

First, **intake is where unit semantics get pinned down.** The $311 call went
wrong because "thirty people" became thirty muffins, and by the time it was an
integer the difference was invisible. `src/agents/flow.py` blocks a quote until
the agent confirms items-vs-people on the call; `items_per_person` is the same
question asked once, up front, by someone who actually knows the answer.

Second, **nothing here defaults a missing number.** A zero materials cost is not
a conservative guess - it is the one value that makes every price look
profitable, so an unanswered cost stays unanswered and `to_config()` refuses to
build.

The rules each answer is judged against live in `intake_fields.py`; this file is
the script and the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import Any, Mapping

from src.business.campaign import Envelope
from src.business.capacity import CapacityLedger
from src.business.pricing import CENT, CostModel
from src.mcp import intake_fields as f
from src.mcp.intake_fields import Question, Refusal, plural, unit_of

__all__ = [
    "BY_FIELD",
    "FIELDS",
    "QUESTIONS",
    "BusinessConfig",
    "Question",
    "Refusal",
    "missing_fields",
    "next_question",
    "required_fields",
    "to_config",
    "transport_per_unit",
]

#: Asked in this order. Costs come after the unit so every prompt can name it.
QUESTIONS: tuple[Question, ...] = (
    Question(
        field="unit",
        prompt=(
            "When you sell one single thing, what do you call it? Give me the "
            "singular - 'muffin', 'crew-hour', 'website'."
        ),
        why="Every price, capacity number and receipt is denominated in this.",
        coerce=f.unit_name,
    ),
    Question(
        field="items_per_person",
        prompt=lambda a: (
            f"When someone orders for a group, how many {plural(unit_of(a))} does each "
            f"person get? If one person means one {unit_of(a)}, say 1."
        ),
        why=(
            "A caller saying 'thirty people' is not ordering thirty units. Getting "
            "this wrong once quoted $74 against a $385 budget."
        ),
        coerce=lambda v: f.positive_int(v, what="items per person"),
    ),
    Question(
        field="capacity_period",
        prompt="Do you think about how much you can make by the week, or by the month?",
        why="Fixes what the capacity number below is counted over.",
        coerce=lambda v: f.choice(
            v, {"week": ("weekly", "wk"), "month": ("monthly", "mo")}, what="that"
        ),
    ),
    Question(
        field="capacity_total",
        prompt=lambda a: (
            f"In one {a.get('capacity_period', 'week')}, how many {plural(unit_of(a))} "
            "can you actually make and get out the door? Not your best ever - the "
            "number you could hit reliably."
        ),
        why="The hard ceiling the agent may sell against. Oversell is unrecoverable.",
        coerce=lambda v: f.positive_int(v, what="capacity"),
    ),
    Question(
        field="materials_per_unit",
        prompt=lambda a: (
            f"What do the raw materials for one {unit_of(a)} cost you? Ingredients, "
            "parts, packaging - your cost, not what you charge."
        ),
        why="First of the three cost lines the margin floor is computed from.",
        coerce=lambda v: f.money(v, what="materials cost"),
    ),
    Question(
        field="labor_per_unit",
        prompt=lambda a: (
            f"What does the labour for one {unit_of(a)} cost you? Count your own time "
            "at what you would pay someone else to do it."
        ),
        why="Unpaid owner time is the most common reason a 'profitable' price is not.",
        coerce=lambda v: f.money(v, what="labour cost"),
    ),
    Question(
        field="transport_basis",
        prompt=lambda a: (
            f"Is your delivery cost per order, or per {unit_of(a)}? Say 'per delivery' "
            "if one run costs the same whether it is ten or a hundred."
        ),
        why=(
            "CostModel wants a per-unit number. Which one they mean decides whether "
            "we divide - guessing here silently misprices every large order."
        ),
        coerce=lambda v: f.choice(
            v,
            {
                "per_delivery": ("delivery", "order", "run", "trip", "drop", "flat"),
                "per_unit": ("unit", "each", "item", "piece"),
            },
            what="how delivery is costed",
        ),
    ),
    Question(
        field="transport_cost",
        prompt=lambda a: (
            "What does getting one order out the door cost you - fuel, driver, "
            "courier fee?"
            if a.get("transport_basis") == "per_delivery"
            else f"What does delivery add to the cost of a single {unit_of(a)}?"
        ),
        why="Third cost line. Left out, the floor sits below break-even on delivery.",
        coerce=lambda v: f.money(v, what="delivery cost"),
    ),
    Question(
        field="units_per_delivery",
        prompt=lambda a: (
            f"Roughly how many {plural(unit_of(a))} go out in a typical delivery? I "
            "need it to spread that delivery cost across the order."
        ),
        why="The divisor that turns a per-order cost into the per-unit one CostModel takes.",
        coerce=lambda v: f.positive_int(v, what="units per delivery"),
        asked_when=lambda a: a.get("transport_basis") == "per_delivery",
    ),
    Question(
        field="min_margin_pct",
        prompt=(
            "What is the thinnest margin you would still take the job at? As a "
            "percentage of the price - say 30 for thirty percent."
        ),
        why="The hard floor. No agent, under any pressure, may offer below it.",
        coerce=lambda v: f.percent(v, what="your minimum margin"),
    ),
    Question(
        field="target_margin_pct",
        prompt="And what margin do you actually want on a good order?",
        why=(
            "Anything between this and the floor is held for you rather than "
            "refused - the agent cannot close it alone."
        ),
        coerce=lambda v: f.percent(v, what="your target margin"),
        check=f.target_not_below_floor,
    ),
    Question(
        field="max_discount_pct",
        prompt=(
            "If a customer pushes for a discount, how much can the agent give away "
            "before it has to stop and ask you? Say 0 if never."
        ),
        why="Discount headroom the agent spends without a human. Above it, it escalates.",
        coerce=lambda v: f.percent(v, what="the discount the agent may give"),
    ),
    Question(
        field="earliest_date",
        prompt=(
            "What is the soonest you could start a brand new order? Say a number of "
            "days, like 3, or an actual date."
        ),
        why="Stops the agent promising a same-day start on stock bought on Thursdays.",
        coerce=lambda v: f.date_or_days(v, what="your soonest date"),
    ),
    Question(
        field="latest_date",
        prompt="And how far ahead will you take bookings? Again, days or a date.",
        why="The far edge of the window the agent may commit to unsupervised.",
        coerce=lambda v: f.date_or_days(v, what="how far ahead you book"),
        check=f.latest_after_earliest,
    ),
    Question(
        field="blackout_days",
        prompt=(
            "Any days of the week you never deliver? Name them, or say 'none' if you "
            "are open all week."
        ),
        why="Carried on the profile so a blacked-out date is caught before it is agreed.",
        coerce=f.weekdays,
    ),
    Question(
        field="approval_mode",
        prompt=(
            "Last one. Do you want to sign off on every single order, or only the "
            "ones that fall outside the limits you just gave me?"
        ),
        why="Whether the envelope is a standing authorisation or advisory only.",
        coerce=lambda v: f.choice(
            v,
            {
                "every_order": ("every", "all", "each", "everything", "always"),
                "out_of_envelope": (
                    "outside", "out of", "only the", "unusual", "exception", "envelope",
                ),
            },
            what="when you want to be asked",
        ),
    ),
)

FIELDS: tuple[str, ...] = tuple(q.field for q in QUESTIONS)
BY_FIELD: dict[str, Question] = {q.field: q for q in QUESTIONS}


def required_fields(answers: Mapping[str, Any]) -> tuple[str, ...]:
    """Fields this particular profile needs, conditionals resolved."""
    return tuple(q.field for q in QUESTIONS if q.applies(answers))


def missing_fields(answers: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(name for name in required_fields(answers) if answers.get(name) is None)


def next_question(answers: Mapping[str, Any]) -> Question | None:
    """The first unanswered applicable question, or None when the interview ends."""
    for q in QUESTIONS:
        if q.applies(answers) and answers.get(q.field) is None:
            return q
    return None


# ---------------------------------------------------------------------------
# Answers -> the objects the rest of the codebase already consumes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BusinessConfig:
    """What intake produces. If this builds, the profile is callable-with."""

    costs: CostModel
    ledger: CapacityLedger
    envelope: Envelope
    ledger_args: dict[str, Any]
    items_per_person: int
    capacity_period: str
    blackout_days: tuple[str, ...]
    approval_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "costs": self.costs.to_dict(),
            "ledger": self.ledger_args,
            "envelope": self.envelope.to_dict(),
            "items_per_person": self.items_per_person,
            "capacity_period": self.capacity_period,
            "blackout_days": list(self.blackout_days),
            "approval_mode": self.approval_mode,
        }


def transport_per_unit(answers: Mapping[str, Any]) -> Decimal:
    """Per-unit delivery cost, amortised if they quoted it per order.

    Rounded *up* to the cent, for the same reason `pricing._ceil_cents` rounds
    floors up: a transport cost rounded down is a cost the floor does not cover.
    """
    cost = answers["transport_cost"]
    if answers.get("transport_basis") != "per_delivery":
        return Decimal(cost)
    per_delivery = Decimal(answers["units_per_delivery"])
    return (Decimal(cost) / per_delivery).quantize(CENT, rounding=ROUND_CEILING)


def to_config(answers: Mapping[str, Any]) -> BusinessConfig:
    """Build the real objects. A broken profile fails here, not mid-call.

    Every constraint is enforced by the class that owns it - `CostModel` rejects
    an inverted margin pair, `Envelope.problems()` rejects an inverted date
    window - so intake cannot drift from what the agent will actually run.
    """
    if gaps := missing_fields(answers):
        raise ValueError("profile is incomplete; still missing: " + ", ".join(gaps))

    unit = str(answers["unit"])
    costs = CostModel(
        materials_per_unit=answers["materials_per_unit"],
        labor_per_unit=answers["labor_per_unit"],
        transport_per_unit=transport_per_unit(answers),
        min_margin_pct=answers["min_margin_pct"],
        target_margin_pct=answers["target_margin_pct"],
        unit=unit,
    )
    if costs.unit_cost <= 0:
        # Not a technicality: at zero cost every price clears every margin, so
        # the floor stops being a floor and the agent can give the job away.
        raise ValueError(
            f"the three costs for one {unit} add up to zero, which makes every "
            "price look profitable. At least one of materials, labour or delivery "
            "must be a real number."
        )

    ledger_args: dict[str, Any] = {
        "total": int(answers["capacity_total"]),
        "unit": plural(unit),
    }
    ledger = CapacityLedger(**ledger_args)

    envelope = Envelope(
        min_price=float(costs.floor_price(1)),
        max_qty=int(answers["capacity_total"]),
        earliest_date=answers["earliest_date"],
        latest_date=answers["latest_date"],
        max_discount_pct=float(answers["max_discount_pct"]),
    )
    if problems := envelope.problems():
        raise ValueError("envelope is not usable: " + "; ".join(problems))

    return BusinessConfig(
        costs=costs,
        ledger=ledger,
        envelope=envelope,
        ledger_args=ledger_args,
        items_per_person=int(answers["items_per_person"]),
        capacity_period=str(answers["capacity_period"]),
        blackout_days=tuple(answers["blackout_days"]),
        approval_mode=str(answers["approval_mode"]),
    )
