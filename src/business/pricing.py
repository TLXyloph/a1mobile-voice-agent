"""A margin floor that lives in code, not in a prompt.

The failure mode: an LLM told "you may discount up to 10%" will, on the third
push from a buyer who says "that's still too high, my last supplier did $2.10",
concede fifteen and apologise about it. A system prompt is advice; negotiation
pressure is the exact input that turns advice into a rounding error.

So the ladder is arithmetic, not instruction. `suggest_concession` is the only
way a price moves down and it is clamped at `floor_price`, which means the worst
case an agent can reach is offering the floor - never a cent under it.

Same shape as `src/verify/receipts.py`: the agent may only ever *propose*.
`QuoteCheck.verdict` is derived from the numbers and has no setter, and anything
between the hard floor and the operator's stated target comes back
REQUIRES_APPROVAL - a proposal a human, not the agent, has to confirm.

Money is `Decimal` throughout. Floats accumulate error, and a floor that is
wrong by a fraction of a cent in the wrong direction is not a floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from enum import Enum
from typing import Any

#: Anything `Decimal(str(x))` accepts. Coerced to Decimal at construction.
Money = Decimal | float | int | str

CENT = Decimal("0.01")
HUNDRED = Decimal("100")


def _money(value: Money) -> Decimal:
    """Coerce via `str` so 1.15 lands on Decimal('1.15'), not its float shadow."""
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _ceil_cents(value: Decimal) -> Decimal:
    """Round a floor *up* to the cent.

    Rounding to nearest would let the published floor sit a half-cent below the
    real one, which is the one direction that is not allowed to be wrong.
    """
    return value.quantize(CENT, rounding=ROUND_CEILING)


class QuoteVerdict(str, Enum):
    """The only three answers about a proposed price."""

    OK = "OK"
    """At or above the operator's target. The agent may offer this itself."""

    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    """Above the floor, under target. Needs an operator, not more talking."""

    BELOW_FLOOR = "BELOW_FLOOR"
    """Loses money against the stated minimum margin. Not offerable, ever."""


@dataclass(frozen=True)
class CostModel:
    """Per-unit cost of delivering the thing, plus the two margin lines.

    Vertical-agnostic: for a bakery a unit is a muffin, for a contractor it is a
    crew-hour, for an agency it is a site build.
    """

    materials_per_unit: Money
    labor_per_unit: Money
    transport_per_unit: Money
    min_margin_pct: Money
    """The hard floor. Gross margin on *price*: price = cost / (1 - m).

    A 25% floor on $3.00 of cost is $4.00, not $3.75. Reading this as a markup
    on cost quietly sells the whole book under the operator's minimum, so the
    convention is stated here and pinned by a test.
    """

    target_margin_pct: Money | None = None
    """What the operator actually wants. Defaults to the floor.

    Quotes landing between the two are not refused - they are held for a human.
    """

    unit: str = "unit"
    currency: str = "USD"

    def __post_init__(self) -> None:
        for name in (
            "materials_per_unit",
            "labor_per_unit",
            "transport_per_unit",
            "min_margin_pct",
        ):
            object.__setattr__(self, name, _money(getattr(self, name)))
        target = self.min_margin_pct if self.target_margin_pct is None else self.target_margin_pct
        object.__setattr__(self, "target_margin_pct", _money(target))

        if min(self.materials_per_unit, self.labor_per_unit, self.transport_per_unit) < 0:
            raise ValueError("per-unit costs cannot be negative")
        if not (0 <= self.min_margin_pct < HUNDRED):
            # At 100% the price would have to be infinite; above it, negative.
            raise ValueError("min_margin_pct must be in [0, 100)")
        if not (0 <= self.target_margin_pct < HUNDRED):
            raise ValueError("target_margin_pct must be in [0, 100)")
        if self.target_margin_pct < self.min_margin_pct:
            raise ValueError("target_margin_pct cannot be below min_margin_pct")

    # -- costs and floors -----------------------------------------------

    @property
    def unit_cost(self) -> Decimal:
        return self.materials_per_unit + self.labor_per_unit + self.transport_per_unit

    def total_cost(self, qty: int) -> Decimal:
        return self.unit_cost * _check_qty(qty)

    def floor_price(self, qty: int) -> Decimal:
        """The absolute minimum total this business can accept for `qty`.

        Nothing in this module returns a number below this one.
        """
        return self._price_at(self.total_cost(qty), self.min_margin_pct)

    def target_price(self, qty: int) -> Decimal:
        """What the operator wants for `qty`. The agent opens at or above this."""
        return self._price_at(self.total_cost(qty), self.target_margin_pct)

    def _price_at(self, cost: Decimal, margin_pct: Decimal) -> Decimal:
        return _ceil_cents(cost / (Decimal(1) - margin_pct / HUNDRED))

    # -- the two operations an agent is allowed to perform ---------------

    def validate_quote(self, qty: int, proposed_total: Money) -> QuoteCheck:
        """Judge a proposed total. The agent proposes; this decides."""
        return QuoteCheck(
            qty=_check_qty(qty),
            proposed_total=_money(proposed_total),
            floor=self.floor_price(qty),
            target=self.target_price(qty),
            total_cost=self.total_cost(qty),
            unit=self.unit,
            currency=self.currency,
        )

    def suggest_concession(self, current_total: Money, qty: int, step_pct: Money) -> Decimal:
        """The next rung down the ladder, clamped at the floor.

        Cannot return below `floor_price(qty)` for any step size, any number of
        applications, or even when handed a `current_total` that is already
        under the floor - in that case it walks the price back *up* to it.
        """
        step = _money(step_pct)
        if not (0 < step < HUNDRED):
            raise ValueError("step_pct must be in (0, 100)")
        floor = self.floor_price(qty)
        # Ceiling again: a concession rounded down is a discount nobody approved.
        nxt = _ceil_cents(_money(current_total) * (Decimal(1) - step / HUNDRED))
        return max(nxt, floor)

    def concession_ladder(
        self, qty: int, start_total: Money, step_pct: Money, max_steps: int = 10
    ) -> list[Decimal]:
        """Every rung from `start_total` down to the floor, floor included.

        Handy for showing an operator what the agent is permitted to say before
        it says any of it.
        """
        rungs: list[Decimal] = []
        current = _money(start_total)
        floor = self.floor_price(qty)
        for _ in range(max_steps):
            current = self.suggest_concession(current, qty, step_pct)
            rungs.append(current)
            if current <= floor:
                break
        return rungs

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "currency": self.currency,
            "unit_cost": str(self.unit_cost),
            "min_margin_pct": str(self.min_margin_pct),
            "target_margin_pct": str(self.target_margin_pct),
        }


@dataclass(frozen=True)
class QuoteCheck:
    """The verdict on one proposed total, and the numbers behind it."""

    qty: int
    proposed_total: Decimal
    floor: Decimal
    target: Decimal
    total_cost: Decimal
    unit: str = "unit"
    currency: str = "USD"

    @property
    def verdict(self) -> QuoteVerdict:
        """Derived, never stored. There is no setter, by design."""
        if self.proposed_total < self.floor:
            return QuoteVerdict.BELOW_FLOOR
        if self.proposed_total < self.target:
            return QuoteVerdict.REQUIRES_APPROVAL
        return QuoteVerdict.OK

    @property
    def approved(self) -> bool:
        """True only for OK.

        REQUIRES_APPROVAL is deliberately falsy here: an agent reading this
        field can only ever decline to close on its own authority.
        """
        return self.verdict is QuoteVerdict.OK

    @property
    def shortfall(self) -> Decimal:
        """How far under the floor, or 0 when it clears."""
        return max(Decimal("0.00"), self.floor - self.proposed_total)

    @property
    def margin_pct(self) -> Decimal | None:
        """Gross margin of the proposal, or None if the total is not positive."""
        if self.proposed_total <= 0:
            return None
        pct = (self.proposed_total - self.total_cost) / self.proposed_total * HUNDRED
        return pct.quantize(CENT)

    @property
    def reason(self) -> str:
        """One sentence, written to be readable to an operator or a log."""
        c = self.currency
        head = f"{c} {self.proposed_total} for {self.qty} {self.unit}"
        if self.verdict is QuoteVerdict.BELOW_FLOOR:
            return (
                f"{head} is {c} {self.shortfall} below the hard floor of "
                f"{c} {self.floor}; the lowest acceptable total is {c} {self.floor}."
            )
        if self.verdict is QuoteVerdict.REQUIRES_APPROVAL:
            return (
                f"{head} clears the {c} {self.floor} floor but is under the "
                f"{c} {self.target} target; needs operator approval before it is offered."
            )
        return f"{head} is at or above the {c} {self.target} target."

    def to_dict(self) -> dict[str, Any]:
        return {
            "qty": self.qty,
            "verdict": self.verdict.value,
            "approved": self.approved,
            "proposed_total": str(self.proposed_total),
            "floor": str(self.floor),
            "target": str(self.target),
            "total_cost": str(self.total_cost),
            "margin_pct": None if self.margin_pct is None else str(self.margin_pct),
            "reason": self.reason,
        }


def _check_qty(qty: int) -> int:
    """Validate at the boundary: a non-positive qty makes every floor meaningless."""
    if not isinstance(qty, int) or isinstance(qty, bool):
        raise TypeError("qty must be an int")
    if qty < 1:
        raise ValueError("qty must be at least 1")
    return qty
