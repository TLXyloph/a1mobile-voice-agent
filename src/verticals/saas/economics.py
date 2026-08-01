"""Subscription economics: the floor is payback and contract value, not unit cost.

`src/business/pricing.py` prices a *thing*. Materials, labour, transport, a
margin on top, one transaction. That model is right for muffins and wrong for
software in a way that is dangerous rather than merely imprecise: the marginal
cost of a seat is a few dollars of infrastructure, so a per-unit margin floor
sits so far below any plausible price that it never fires. An agent handed that
floor has, in practice, no floor at all - it can give away 60% and still be
told the quote "clears".

What actually bounds a subscription deal is two numbers:

  **total contract value** - what the account is worth across the whole term,
  after every lever has been pulled; and
  **CAC payback** - how many months of gross profit it takes to earn back what
  it cost to win. A 90%-margin deal that takes twenty months to repay its own
  acquisition cost is a cash-flow hole with a good-looking margin line.

Both floors are checked, and *either* one failing rejects the deal. The test
suite pins the interesting half of that: a deal can clear on margin and on
contract value and still be bad, because payback is a separate dimension that
margin cannot see.

## The stacking problem

The real hazard is not one big concession. It is four small ones. Price per
seat, seat count, term length, free months and the onboarding fee are five
independent levers, and a buyer pushes them one at a time:

    "can you do 10% off?"           - within the 10% cap, allowed
    "throw in the first two months" - within the 2-month cap, allowed
    "and waive the setup fee"       - waivers are allowed

Every answer is inside its own limit. The deal that comes out the other end is
25% off, not 10%. An agent checking levers one at a time will approve it, which
is why `lever_within_cap()` is documented as *not* permission and
`DealFloor.evaluate()` - which only ever looks at the combined, fully-applied
model - is the only thing that can clear a deal.

`effective_discount_pct` is the number to say out loud in that argument: it
compares the effective per-seat-month rate across the whole term against list,
so free months and a shortened term and a headline discount all land in one
figure.

Money is `Decimal`, same reasoning as `pricing.py`. Rounding is deliberately
asymmetric: revenue rounds *down*, payback rounds *up*. Every rounding error
therefore makes the deal look slightly worse than it is, which is the only
direction a floor is allowed to be wrong in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import Enum
from typing import Any

#: Anything `Decimal(str(x))` accepts.
Money = Decimal | float | int | str

CENT = Decimal("0.01")
HUNDRED = Decimal(100)
ZERO = Decimal(0)

#: How far past the end of the term payback is allowed to be projected before
#: we call it "never". Beyond the term, payback assumes the account renews at
#: the same terms - the standard convention, and an assumption, so it is capped.
MAX_PAYBACK_HORIZON_MONTHS = Decimal(120)


def _money(value: Money) -> Decimal:
    """Coerce via `str` so 0.1 lands on Decimal('0.1'), not its float shadow."""
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _floor2(value: Decimal) -> Decimal:
    """Round revenue and margin *down*. Never overstate what we are getting."""
    return value.quantize(CENT, rounding=ROUND_FLOOR)


def _ceil2(value: Decimal) -> Decimal:
    """Round payback *up*. Never understate how long the money is out."""
    return value.quantize(CENT, rounding=ROUND_CEILING)


#: Public aliases. Other modules in this vertical need the same coercion and
#: the same asymmetric rounding; re-implementing either would be a bug, and a
#: quiet one.
as_money = _money
round_down = _floor2
round_up = _ceil2


class Lever(str, Enum):
    """The five ways a subscription deal gets cheaper.

    Named so a transcript can be replayed against the model: every concession
    a buyer extracts is one of these, and the combined model is what gets
    judged.
    """

    DISCOUNT_PCT = "discount_pct"
    """Percentage points off the list rate per seat."""

    FREE_MONTHS = "free_months"
    """Months at the front of the term that are served but not billed."""

    ONBOARDING_WAIVER = "onboarding_waiver"
    """Dollars knocked off the one-time onboarding fee."""

    SEAT_REDUCTION = "seat_reduction"
    """Seats removed from the order. Cheaper for them, smaller for us."""

    TERM_REDUCTION = "term_reduction"
    """Months removed from the commitment. The quietest of the five: it does
    not change any headline number, and it can halve the contract."""


@dataclass(frozen=True)
class Concession:
    """One thing given away, in the units of its lever."""

    lever: Lever
    amount: Money
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", _money(self.amount))
        if self.amount < 0:
            raise ValueError("a concession cannot be negative; that is a price rise")

    @property
    def described(self) -> str:
        """Said the way it would be said on the call, not the way it is stored.

        `Decimal("10.0")` renders as "10", because "10.0% off list" reads like
        a machine wrote it and this string ends up in transcripts and in the UI.
        """
        unit = {
            Lever.DISCOUNT_PCT: "% off list",
            Lever.FREE_MONTHS: " free month(s)",
            Lever.ONBOARDING_WAIVER: " off onboarding",
            Lever.SEAT_REDUCTION: " fewer seats",
            Lever.TERM_REDUCTION: " months shorter term",
        }[self.lever]
        amount = _money(self.amount).normalize()
        # normalize() turns 2500 into 2.5E+3; undo that for anything integral.
        if amount == amount.to_integral_value():
            amount = amount.quantize(Decimal(1))
        return f"{amount}{unit}"


@dataclass(frozen=True)
class SubscriptionModel:
    """One proposed subscription deal, fully specified.

    Everything derived from it is a property with no setter, for the same
    reason `Claim.verdict` is: numbers an agent can assign are numbers an agent
    can talk itself into.

    Degenerate inputs (zero seats, zero term, zero price) are *allowed* and
    return safe answers rather than raising. A half-built deal is a real state
    during a live call - the buyer has said "about twenty seats" and nothing
    else - and a pricing model that throws mid-call is worse than one that says
    "no revenue, no margin, no payback".
    """

    price_per_seat_month: Money
    seats: int
    term_months: int
    discount_pct: Money = 0
    free_months: int = 0
    onboarding_fee: Money = 0
    monthly_cost_to_serve_per_seat: Money = 0
    cac: Money = 0
    currency: str = "USD"
    plan: str = "standard"

    def __post_init__(self) -> None:
        for name in (
            "price_per_seat_month",
            "discount_pct",
            "onboarding_fee",
            "monthly_cost_to_serve_per_seat",
            "cac",
        ):
            object.__setattr__(self, name, _money(getattr(self, name)))
        for name in ("seats", "term_months", "free_months"):
            object.__setattr__(self, name, int(getattr(self, name)))

        if self.price_per_seat_month < 0:
            raise ValueError("price_per_seat_month cannot be negative")
        if self.onboarding_fee < 0:
            raise ValueError("onboarding_fee cannot be negative")
        if self.monthly_cost_to_serve_per_seat < 0:
            raise ValueError("monthly_cost_to_serve_per_seat cannot be negative")
        if self.cac < 0:
            raise ValueError("cac cannot be negative")
        if self.seats < 0 or self.term_months < 0 or self.free_months < 0:
            raise ValueError("seats, term_months and free_months cannot be negative")
        if not (ZERO <= self.discount_pct <= HUNDRED):
            raise ValueError("discount_pct must be within 0-100")

    # -- rates ------------------------------------------------------------

    @property
    def net_price_per_seat_month(self) -> Decimal:
        """List rate after the headline discount. Still not what they pay:
        free months and the term are not in this number."""
        return self.price_per_seat_month * (HUNDRED - self.discount_pct) / HUNDRED

    @property
    def list_monthly(self) -> Decimal:
        return self.price_per_seat_month * self.seats

    @property
    def billed_monthly(self) -> Decimal:
        """What lands on an invoice in a month that is actually billed."""
        return self.net_price_per_seat_month * self.seats

    @property
    def billable_months(self) -> int:
        """Term minus free months, never below zero.

        Free months beyond the term are nonsense but not an exception: they
        simply mean nothing is ever billed, which is the honest reading.
        """
        return max(0, self.term_months - self.free_months)

    # -- contract value ----------------------------------------------------

    @property
    def subscription_revenue(self) -> Decimal:
        return self.billed_monthly * self.billable_months

    @property
    def total_contract_value(self) -> Decimal:
        """Everything they will pay across the term. The first real floor."""
        return _floor2(self.subscription_revenue + self.onboarding_fee)

    @property
    def effective_monthly_rate(self) -> Decimal:
        """Contract value spread over every month of the term, free ones included.

        This is the number that catches stacking. A 10% discount with two free
        months on a twelve-month term is not a 10% deal, and this says so.
        """
        if self.term_months <= 0:
            return ZERO
        return _floor2(self.total_contract_value / self.term_months)

    @property
    def effective_rate_per_seat_month(self) -> Decimal:
        """Recurring revenue per seat per month, averaged over the whole term.

        The one-time onboarding fee is deliberately **not** in here. A rate is
        a rate: folding a setup fee into it makes a list-price deal look like
        it is running at a premium to list, which is nonsense and, worse, makes
        the discount figure below go negative. The fee is not ignored - it is
        in `total_contract_value` and in payback, which is where a one-time
        payment actually belongs.
        """
        if self.term_months <= 0 or self.seats <= 0:
            return ZERO
        return _floor2(
            self.subscription_revenue / (Decimal(self.term_months) * self.seats)
        )

    @property
    def effective_discount_pct(self) -> Decimal | None:
        """Every rate lever, expressed as one discount off list. None if list is 0.

        Discount, free months and a shortened term all land here, because all
        three change what a seat costs per month across the term. Say this
        number, not the headline one: 10% off with two free months on a
        twelve-month term is a 25% deal, and the gap between "ten" and
        "twenty-five" is the value that got away while every individual answer
        sounded reasonable.

        A seat reduction does not show up here, and should not - selling fewer
        seats is a smaller deal, not a cheaper one. An onboarding waiver does
        not either; it shows up in contract value and payback.
        """
        if self.price_per_seat_month <= 0:
            return None
        gap = self.price_per_seat_month - self.effective_rate_per_seat_month
        return _ceil2(gap / self.price_per_seat_month * HUNDRED)

    @property
    def all_in_rate_per_seat_month(self) -> Decimal:
        """Cash view: everything they pay, spread over seats and months."""
        if self.term_months <= 0 or self.seats <= 0:
            return ZERO
        return _floor2(
            self.total_contract_value / (Decimal(self.term_months) * self.seats)
        )

    # -- cost and margin ---------------------------------------------------

    @property
    def monthly_cost_to_serve(self) -> Decimal:
        """Support, infra and success, per month. Free months cost this too -
        that is the entire reason free months are not free."""
        if self.term_months <= 0:
            return ZERO
        return self.monthly_cost_to_serve_per_seat * self.seats

    @property
    def cost_to_serve_total(self) -> Decimal:
        return _ceil2(self.monthly_cost_to_serve * self.term_months)

    @property
    def gross_profit(self) -> Decimal:
        """Contract value less cost to serve. CAC is deliberately excluded:
        it is an acquisition cost, and it is judged by payback, not margin."""
        return _floor2(self.total_contract_value - self.cost_to_serve_total)

    @property
    def gross_margin_pct(self) -> Decimal | None:
        """None when there is no revenue - not zero, and never a division."""
        tcv = self.total_contract_value
        if tcv <= 0:
            return None
        return _floor2(self.gross_profit / tcv * HUNDRED)

    @property
    def steady_monthly_gross_profit(self) -> Decimal:
        """Gross profit in an ordinary billed month, once the front-loading
        (onboarding fee in, free months out) is behind us."""
        if self.term_months <= 0:
            return ZERO
        return self.billed_monthly - self.monthly_cost_to_serve

    # -- payback -----------------------------------------------------------

    def month_gross_profit(self, month: int) -> Decimal:
        """Gross profit in `month` (1-indexed). CAC excluded; it is the target.

        Months past the end of the term assume a renewal on the same terms.
        That is an assumption, so `months_to_cac_payback` flags any payback
        that lands out there via `pays_back_within_term`.
        """
        if month < 1:
            raise ValueError("month is 1-indexed")
        if self.term_months <= 0:
            # No term, no service, no cost - just whatever was paid up front.
            return self.onboarding_fee if month == 1 else ZERO

        revenue = self.onboarding_fee if month == 1 else ZERO
        billed = month > self.term_months or month > self.free_months
        if billed:
            revenue += self.billed_monthly
        return revenue - self.monthly_cost_to_serve

    @property
    def months_to_cac_payback(self) -> Decimal | None:
        """Months until cumulative gross profit repays CAC. None means never.

        Simulated month by month rather than `cac / monthly_profit`, because
        the shortcut is wrong exactly where it matters: an onboarding fee pays
        back instantly and free months push payback out, and the closed form
        sees neither.
        """
        cac = self.cac
        if cac <= 0:
            return Decimal("0.00")

        cumulative = ZERO
        for month in range(1, self.term_months + 1):
            previous = cumulative
            cumulative += self.month_gross_profit(month)
            if cumulative >= cac and cumulative > previous:
                # Interpolate inside the month it crossed.
                fraction = (cac - previous) / (cumulative - previous)
                return _ceil2(Decimal(month - 1) + fraction)

        steady = self.steady_monthly_gross_profit
        if steady <= 0:
            return None
        projected = Decimal(self.term_months) + (cac - cumulative) / steady
        if projected > MAX_PAYBACK_HORIZON_MONTHS:
            return None
        return _ceil2(projected)

    @property
    def pays_back_within_term(self) -> bool:
        payback = self.months_to_cac_payback
        return payback is not None and payback <= self.term_months

    # -- concessions -------------------------------------------------------

    def with_concessions(self, *concessions: Concession) -> SubscriptionModel:
        """Apply concessions and return the deal that actually results.

        Discounts and free months accumulate additively - two 5% concessions
        are 10% off, not 9.75%. That is how they get said on a call, and the
        arithmetic should match the conversation rather than be quietly
        kinder to us than the buyer thinks it is.
        """
        discount = self.discount_pct
        free = self.free_months
        fee = self.onboarding_fee
        seats = self.seats
        term = self.term_months

        for c in concessions:
            if c.lever is Lever.DISCOUNT_PCT:
                discount = min(HUNDRED, discount + _money(c.amount))
            elif c.lever is Lever.FREE_MONTHS:
                free += int(c.amount)
            elif c.lever is Lever.ONBOARDING_WAIVER:
                fee = max(ZERO, fee - _money(c.amount))
            elif c.lever is Lever.SEAT_REDUCTION:
                seats = max(0, seats - int(c.amount))
            elif c.lever is Lever.TERM_REDUCTION:
                term = max(0, term - int(c.amount))

        return replace(
            self,
            discount_pct=discount,
            free_months=free,
            onboarding_fee=fee,
            seats=seats,
            term_months=term,
        )

    def to_dict(self) -> dict[str, Any]:
        payback = self.months_to_cac_payback
        margin = self.gross_margin_pct
        effective = self.effective_discount_pct
        return {
            "plan": self.plan,
            "currency": self.currency,
            "price_per_seat_month": str(self.price_per_seat_month),
            "seats": self.seats,
            "term_months": self.term_months,
            "discount_pct": str(self.discount_pct),
            "free_months": self.free_months,
            "onboarding_fee": str(self.onboarding_fee),
            "monthly_cost_to_serve_per_seat": str(self.monthly_cost_to_serve_per_seat),
            "cac": str(self.cac),
            "billable_months": self.billable_months,
            "total_contract_value": str(self.total_contract_value),
            "effective_monthly_rate": str(self.effective_monthly_rate),
            "effective_rate_per_seat_month": str(self.effective_rate_per_seat_month),
            "all_in_rate_per_seat_month": str(self.all_in_rate_per_seat_month),
            "effective_discount_pct": None if effective is None else str(effective),
            "cost_to_serve_total": str(self.cost_to_serve_total),
            "gross_profit": str(self.gross_profit),
            "gross_margin_pct": None if margin is None else str(margin),
            "months_to_cac_payback": None if payback is None else str(payback),
            "pays_back_within_term": self.pays_back_within_term,
        }


class DealVerdict(str, Enum):
    """Two answers, not three.

    `pricing.QuoteVerdict` has a REQUIRES_APPROVAL band because a bakery owner
    is in the next room. This campaign runs with escalation off - see
    `src/verticals/saas/campaign.py` - so a middle state would be a state with
    nowhere to go. The agent closes inside the floor or walks.
    """

    CLEARS = "CLEARS"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class DealFloor:
    """What a deal has to be worth, and how fast it has to pay for itself.

    The two mandatory fields are the real floors. The `max_*` fields below them
    are per-lever caps: guardrails on a single ask, useful for answering a
    buyer quickly, and *not* a substitute for `evaluate()`. See
    `lever_within_cap`.
    """

    min_contract_value: Money
    """Below this the account is not worth the onboarding slot it occupies."""

    max_payback_months: Money
    """Longest we will wait to earn back CAC. Keep this at or under the term:
    a payback that only arrives on renewal is a payback we have not been
    promised."""

    min_gross_margin_pct: Money = 0
    """A backstop, not the binding constraint. Software margin is high almost
    regardless of what you do to price, which is exactly why it must not be
    the only check."""

    max_discount_pct: Money = 0
    max_free_months: int = 0
    max_onboarding_waiver: Money = 0
    max_seat_reduction: int = 0
    max_term_reduction: int = 0
    """Per-lever caps on a single ask."""

    min_seats: int = 1
    min_term_months: int = 1
    """Shape floors. A two-seat, one-month deal can clear every money test and
    still not be a customer."""

    currency: str = "USD"

    def __post_init__(self) -> None:
        for name in (
            "min_contract_value",
            "max_payback_months",
            "min_gross_margin_pct",
            "max_discount_pct",
            "max_onboarding_waiver",
        ):
            object.__setattr__(self, name, _money(getattr(self, name)))
        if self.min_contract_value < 0:
            raise ValueError("min_contract_value cannot be negative")
        if self.max_payback_months <= 0:
            raise ValueError("max_payback_months must be > 0")
        if not (ZERO <= self.min_gross_margin_pct < HUNDRED):
            raise ValueError("min_gross_margin_pct must be in [0, 100)")
        if not (ZERO <= self.max_discount_pct <= HUNDRED):
            raise ValueError("max_discount_pct must be within 0-100")

    def cap_for(self, lever: Lever) -> Decimal:
        return {
            Lever.DISCOUNT_PCT: self.max_discount_pct,
            Lever.FREE_MONTHS: Decimal(self.max_free_months),
            Lever.ONBOARDING_WAIVER: self.max_onboarding_waiver,
            Lever.SEAT_REDUCTION: Decimal(self.max_seat_reduction),
            Lever.TERM_REDUCTION: Decimal(self.max_term_reduction),
        }[lever]

    def lever_within_cap(self, concession: Concession) -> tuple[bool, str]:
        """Is this one ask inside its own cap?

        **True here is not permission to close.** It answers "may I say yes to
        this specific question", which is the question a buyer asks and the
        wrong question to end a negotiation on. Three concessions can each
        return True and produce a deal `evaluate()` rejects outright - that is
        the whole failure mode this module exists to catch, and there is a test
        named after it. Always finish with `evaluate()` on the combined model.
        """
        cap = self.cap_for(concession.lever)
        amount = _money(concession.amount)
        if amount > cap:
            return False, (
                f"{concession.described} exceeds the {cap} cap on "
                f"{concession.lever.value}"
            )
        return True, (
            f"{concession.described} is inside the {cap} cap on "
            f"{concession.lever.value} - but the deal is only cleared by "
            f"evaluating all concessions together"
        )

    def evaluate(self, model: SubscriptionModel) -> DealCheck:
        """The only gate. Judges the whole deal, however it got that way."""
        return DealCheck(model=model, floor=self)

    def evaluate_with(
        self, model: SubscriptionModel, concessions: Iterable[Concession]
    ) -> DealCheck:
        """Apply concessions to `model`, then judge the result."""
        return self.evaluate(model.with_concessions(*concessions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_contract_value": str(self.min_contract_value),
            "max_payback_months": str(self.max_payback_months),
            "min_gross_margin_pct": str(self.min_gross_margin_pct),
            "max_discount_pct": str(self.max_discount_pct),
            "max_free_months": self.max_free_months,
            "max_onboarding_waiver": str(self.max_onboarding_waiver),
            "max_seat_reduction": self.max_seat_reduction,
            "max_term_reduction": self.max_term_reduction,
            "min_seats": self.min_seats,
            "min_term_months": self.min_term_months,
            "currency": self.currency,
        }


@dataclass(frozen=True)
class DealCheck:
    """The verdict on one fully-specified deal, and every reason behind it.

    `breaches` and `verdict` are derived on read. Nothing here is stored, so
    there is no field an agent can set to make a rejected deal acceptable.
    """

    model: SubscriptionModel
    floor: DealFloor

    @property
    def breaches(self) -> tuple[str, ...]:
        """Every floor this deal fails, in the order they matter. All of them,
        not the first - an operator fixing one wants to see the other two."""
        m, f = self.model, self.floor
        out: list[str] = []
        cur = f.currency

        tcv = m.total_contract_value
        if tcv < f.min_contract_value:
            out.append(
                f"contract value {cur} {tcv} is below the {cur} "
                f"{f.min_contract_value} minimum (short by {cur} "
                f"{_ceil2(f.min_contract_value - tcv)})"
            )

        payback = m.months_to_cac_payback
        if payback is None:
            out.append(
                f"CAC of {cur} {m.cac} never pays back on these terms - "
                f"gross profit per billed month is {cur} "
                f"{_floor2(m.steady_monthly_gross_profit)}"
            )
        elif payback > f.max_payback_months:
            out.append(
                f"CAC payback of {payback} months exceeds the "
                f"{f.max_payback_months}-month limit"
            )

        margin = m.gross_margin_pct
        if margin is None:
            out.append("no contract value, so there is no margin to measure")
        elif margin < f.min_gross_margin_pct:
            out.append(
                f"gross margin {margin}% is under the {f.min_gross_margin_pct}% floor"
            )

        if m.seats < f.min_seats:
            out.append(f"{m.seats} seats is below the {f.min_seats}-seat minimum")
        if m.term_months < f.min_term_months:
            out.append(
                f"a {m.term_months}-month term is shorter than the "
                f"{f.min_term_months}-month minimum"
            )
        return tuple(out)

    @property
    def verdict(self) -> DealVerdict:
        """Derived, never stored. There is no setter, by design."""
        return DealVerdict.REJECTED if self.breaches else DealVerdict.CLEARS

    @property
    def approved(self) -> bool:
        return self.verdict is DealVerdict.CLEARS

    @property
    def headline(self) -> str:
        m = self.model
        payback = m.months_to_cac_payback
        pb = "never" if payback is None else f"{payback} months"
        return (
            f"{m.currency} {m.total_contract_value} over {m.term_months} months "
            f"({m.seats} seats), margin "
            f"{'n/a' if m.gross_margin_pct is None else str(m.gross_margin_pct) + '%'}, "
            f"payback {pb}"
        )

    @property
    def reason(self) -> str:
        """One paragraph, written to be read to an operator or logged verbatim."""
        if self.approved:
            effective = self.model.effective_discount_pct
            tail = (
                ""
                if effective is None
                else f" Effective discount across every lever is {effective}%."
            )
            return f"CLEARS. {self.headline}.{tail}"
        return "REJECTED. " + " ".join(f"{b}." for b in self.breaches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "approved": self.approved,
            "headline": self.headline,
            "reason": self.reason,
            "breaches": list(self.breaches),
            "deal": self.model.to_dict(),
            "floor": self.floor.to_dict(),
        }
