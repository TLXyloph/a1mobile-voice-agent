"""The negotiating mandate: the user-supplied parameters that bound the agent.

Helyx is a *buyer-side* agent for restaurants and bakehouses. The operator is
placing a wholesale or catering order (loaves, pastry trays, par-baked stock)
and Helyx phones the supplier to negotiate it.

Because the operator is buying, the walk-away limit is a **ceiling**: the
highest unit price Helyx is permitted to utter or accept. ``target`` is what a
good outcome looks like; ``ceiling`` is the point past which walking away is
correct.

Validation lives here because this is the system boundary -- everything
downstream may assume a Mandate is internally consistent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

MAX_QUANTITY = 100_000
MAX_UNIT_PRICE_CENTS = 5_000_00  # $5,000/unit -- a sanity bound, not a policy
MAX_ROUNDS_LIMIT = 12


class MandateError(ValueError):
    """Raised when supplied parameters cannot form a coherent mandate."""


#: The fields the operator must supply before any call may be placed. This list
#: is the single source of truth for intake completeness -- the intake LLM does
#: not get to declare itself finished (see intake.py).
REQUIRED_FIELDS: tuple[str, ...] = (
    "item",
    "quantity",
    "target_unit_price_cents",
    "ceiling_unit_price_cents",
    "needed_by",
    "counterparty_name",
)

FIELD_PROMPTS: dict[str, str] = {
    "item": "What are you ordering? (e.g. 'sourdough loaves', 'croissant trays')",
    "quantity": "How many units?",
    "target_unit_price_cents": "What unit price are you aiming for?",
    "ceiling_unit_price_cents": "What is the most per unit you would ever pay -- the walk-away point?",
    "needed_by": "What date do you need it by? (YYYY-MM-DD)",
    "counterparty_name": "Which bakehouse or restaurant should Helyx call?",
    "counterparty_phone": "What number should Helyx call? (optional for a dry run)",
    "opening_unit_price_cents": "Optional: what should the opening offer be? Defaults below target.",
    "constraints": "Any constraints? (allergens, delivery vs pickup, deposit terms)",
    "max_rounds": "Optional: how many concession rounds before walking away? (default 4)",
}


@dataclass(frozen=True)
class Mandate:
    """A validated, internally consistent negotiating brief.

    Invariant enforced at construction:
        0 < opening <= target <= ceiling <= MAX_UNIT_PRICE_CENTS
    """

    item: str
    quantity: int
    target_unit_price_cents: int
    ceiling_unit_price_cents: int
    needed_by: date
    counterparty_name: str
    counterparty_phone: str = ""
    opening_unit_price_cents: int = 0
    max_rounds: int = 4
    constraints: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.item.strip():
            raise MandateError("item must be non-empty")
        if not self.counterparty_name.strip():
            raise MandateError("counterparty_name must be non-empty")
        if not 0 < self.quantity <= MAX_QUANTITY:
            raise MandateError(f"quantity must be in 1..{MAX_QUANTITY}")
        if not 0 < self.target_unit_price_cents <= MAX_UNIT_PRICE_CENTS:
            raise MandateError("target_unit_price_cents out of range")
        if not 0 < self.ceiling_unit_price_cents <= MAX_UNIT_PRICE_CENTS:
            raise MandateError("ceiling_unit_price_cents out of range")
        if self.target_unit_price_cents > self.ceiling_unit_price_cents:
            raise MandateError(
                "target_unit_price_cents must not exceed ceiling_unit_price_cents "
                "(you cannot aim above your own walk-away point)"
            )
        if not 0 < self.max_rounds <= MAX_ROUNDS_LIMIT:
            raise MandateError(f"max_rounds must be in 1..{MAX_ROUNDS_LIMIT}")

        if self.opening_unit_price_cents == 0:
            # Default opening: 12% under target, so there is room to concede.
            object.__setattr__(
                self,
                "opening_unit_price_cents",
                max(1, int(round(self.target_unit_price_cents * 0.88))),
            )
        if self.opening_unit_price_cents > self.target_unit_price_cents:
            raise MandateError(
                "opening_unit_price_cents must not exceed target_unit_price_cents"
            )

    @property
    def ceiling_total_cents(self) -> int:
        return self.quantity * self.ceiling_unit_price_cents

    @property
    def target_total_cents(self) -> int:
        return self.quantity * self.target_unit_price_cents

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "quantity": self.quantity,
            "target_unit_price_cents": self.target_unit_price_cents,
            "ceiling_unit_price_cents": self.ceiling_unit_price_cents,
            "opening_unit_price_cents": self.opening_unit_price_cents,
            "needed_by": self.needed_by.isoformat(),
            "counterparty_name": self.counterparty_name,
            "counterparty_phone": self.counterparty_phone,
            "max_rounds": self.max_rounds,
            "constraints": list(self.constraints),
            "ceiling_total_cents": self.ceiling_total_cents,
            "target_total_cents": self.target_total_cents,
        }


# --- boundary parsing -------------------------------------------------------

_MONEY = re.compile(r"^\$?\s*(\d+(?:\.\d{1,2})?)$")


def parse_money_to_cents(value: Any) -> int:
    """Accept 4.25, '4.25', '$4.25', 425 (already cents is NOT assumed).

    Strings and floats are read as dollars. Integers are ambiguous, so they are
    also read as dollars -- the caller passes cents explicitly by name.
    """
    if isinstance(value, bool):
        raise MandateError("money value must be a number, not a boolean")
    if isinstance(value, (int, float)):
        dollars = float(value)
    elif isinstance(value, str):
        m = _MONEY.match(value.strip())
        if not m:
            raise MandateError(f"could not read {value!r} as a money amount")
        dollars = float(m.group(1))
    else:
        raise MandateError(f"could not read {type(value).__name__} as a money amount")
    if dollars <= 0:
        raise MandateError("money amount must be positive")
    return int(round(dollars * 100))


def parse_quantity(value: Any) -> int:
    if isinstance(value, bool):
        raise MandateError("quantity must be a number")
    if isinstance(value, str):
        digits = re.sub(r"[^\d]", "", value)
        if not digits:
            raise MandateError(f"could not read {value!r} as a quantity")
        value = int(digits)
    if not isinstance(value, int):
        value = int(value)
    if not 0 < value <= MAX_QUANTITY:
        raise MandateError(f"quantity must be in 1..{MAX_QUANTITY}")
    return value


def parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise MandateError("needed_by must be a date or YYYY-MM-DD string")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise MandateError(f"could not read {value!r} as a YYYY-MM-DD date") from exc


def missing_fields(raw: dict[str, Any]) -> list[str]:
    """Which required fields are still absent. Computed, never asserted."""
    return [f for f in REQUIRED_FIELDS if raw.get(f) in (None, "", [], {})]


def build_mandate(raw: dict[str, Any]) -> Mandate:
    """Validate a loose dict from intake into a Mandate. Raises MandateError."""
    absent = missing_fields(raw)
    if absent:
        raise MandateError(f"missing required fields: {', '.join(absent)}")

    constraints = raw.get("constraints") or ()
    if isinstance(constraints, str):
        constraints = tuple(c.strip() for c in constraints.split(",") if c.strip())
    else:
        constraints = tuple(str(c) for c in constraints)

    opening_raw = raw.get("opening_unit_price_cents")
    return Mandate(
        item=str(raw["item"]).strip(),
        quantity=parse_quantity(raw["quantity"]),
        target_unit_price_cents=_as_cents(raw["target_unit_price_cents"]),
        ceiling_unit_price_cents=_as_cents(raw["ceiling_unit_price_cents"]),
        needed_by=parse_date(raw["needed_by"]),
        counterparty_name=str(raw["counterparty_name"]).strip(),
        counterparty_phone=str(raw.get("counterparty_phone") or "").strip(),
        opening_unit_price_cents=_as_cents(opening_raw) if opening_raw else 0,
        max_rounds=int(raw.get("max_rounds") or 4),
        constraints=constraints,
    )


def _as_cents(value: Any) -> int:
    """Fields named ``*_cents`` may arrive as literal cents from the UI or as
    dollar strings from a human/LLM. Integers >= 1000 with no decimal point are
    treated as cents; everything else is read as dollars."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1000:
        return value
    return parse_money_to_cents(value)
