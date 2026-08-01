"""The SaaS campaign: a generic Envelope built out of subscription economics.

`src/business/campaign.py` is the engine and it stays untouched - if this
vertical needed a new field on `Envelope`, the abstraction would be wrong. It
does not. What it needs is a *translation*, and the translation is the point:

    Envelope.min_price   is a per-seat-month rate, and it is **derived** from
                         the contract-value floor rather than chosen. The unit
                         price is the shadow the real floor casts, not the
                         thing itself.
    Envelope.max_qty     is seats we can onboard this month. Capacity here is
                         implementation bandwidth, not inventory: overselling
                         it produces a churned account rather than a backlog.
    Envelope.max_discount_pct
                         is the cap on a *single headline ask*. It is the most
                         dangerous field in this file, because an agent that
                         treats it as sufficient will approve a 25% deal one
                         10% answer at a time. `SaasPlaybook.decide()` exists
                         so that never has to be a judgement call.

## No escalation

`STARTUP_OUTBOUND` in the shared campaign module sets `max_discount_pct=0`, so
every pricing question escalates by construction. That is right for a first
discovery call and wrong for this one: here the agent is closing a
subscription, and it has real pricing authority.

What it does not have is anyone to ask. The limits are absolute. There are
exactly three outcomes - close inside the floor, counter with the best thing
the floor still allows, or thank them and end the call - and `Outcome` has no
fourth member. Escalation is not disabled here, it is absent, which is a
different and stronger property: there is no code path that produces
"let me check with someone" and therefore no way for the model to discover one
under pressure.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.business.campaign import (
    Campaign,
    CloseCondition,
    DiscoveryStrategy,
    Envelope,
)
from src.verticals.saas.economics import (
    Concession,
    DealCheck,
    DealFloor,
    Lever,
    SubscriptionModel,
    as_money,
    round_up,
)

#: Where the numbers live. Nothing in this module hardcodes a price.
CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "saas.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or CONFIG_PATH).read_text())


def model_from_config(cfg: dict[str, Any]) -> SubscriptionModel:
    return SubscriptionModel(**cfg["list_terms"])


def floor_from_config(cfg: dict[str, Any]) -> DealFloor:
    return DealFloor(**cfg["floor"])


def implied_seat_floor(model: SubscriptionModel, floor: DealFloor) -> Decimal:
    """The per-seat-month rate the contract-value floor implies for this shape.

    Returns 0 for a degenerate deal (no seats, no term). That makes the
    resulting `Envelope` invalid, which is correct - there is no coherent unit
    floor for a contract with no units - and it is why `problems()` is checked
    rather than assumed.
    """
    seat_months = Decimal(model.seats) * model.term_months
    if seat_months <= 0:
        return Decimal(0)
    # Round up: a floor rounded down is a floor slightly below the real one.
    return round_up(floor.min_contract_value / seat_months)


def envelope_for(
    model: SubscriptionModel,
    floor: DealFloor,
    *,
    seats_onboardable_per_month: int,
    earliest_date: date,
    latest_date: date,
) -> Envelope:
    """Translate subscription economics into the generic envelope."""
    return Envelope(
        min_price=float(implied_seat_floor(model, floor)),
        max_qty=int(seats_onboardable_per_month),
        earliest_date=earliest_date,
        latest_date=latest_date,
        max_discount_pct=float(floor.max_discount_pct),
        currency=floor.currency,
    )


def campaign_for(
    model: SubscriptionModel,
    floor: DealFloor,
    cfg: dict[str, Any],
) -> Campaign:
    """Build the outbound campaign for a subscription product."""
    c = cfg["campaign"]
    window = c["window"]
    return Campaign(
        name=c["name"],
        vertical="b2b_saas",
        icp=c["icp"],
        offer=c["offer"],
        discovery=DiscoveryStrategy.SEEDED_LIST,
        # A subscription is closed by a countersigned order form or a written
        # yes, never by an audible one. WRITTEN_CONFIRMATION draws only from
        # INDEPENDENT_CHANNELS, which `Campaign.problems()` re-checks.
        close_condition=CloseCondition.WRITTEN_CONFIRMATION,
        capacity_units="seats onboarded/month",
        envelope=envelope_for(
            model,
            floor,
            seats_onboardable_per_month=cfg["capacity"]["seats_onboardable_per_month"],
            earliest_date=date.fromisoformat(window["earliest_date"]),
            latest_date=date.fromisoformat(window["latest_date"]),
        ),
        notes=c["notes"],
        tags=("saas", "subscription", "b2b", "no-escalation"),
    )


# -- what the agent is allowed to do on a call ----------------------------


@dataclass(frozen=True)
class Outcome:
    """The three endings. There is deliberately no fourth."""

    CLOSE = "close"
    COUNTER = "counter"
    WALK_AWAY = "walk_away"


@dataclass(frozen=True)
class Decision:
    """What to do about a set of concessions, and the line to say."""

    outcome: str
    line: str
    check: DealCheck

    @property
    def may_close(self) -> bool:
        return self.outcome == Outcome.CLOSE

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "line": self.line,
            "check": self.check.to_dict(),
        }


@dataclass(frozen=True)
class SaasPlaybook:
    """Campaign, list terms and floor, kept together because they are one thing.

    The agent talks to `decide()` and nothing else. Handing it the `Envelope`
    directly would re-expose the per-lever trap this vertical exists to close.
    """

    campaign: Campaign
    model: SubscriptionModel
    floor: DealFloor
    seats_onboardable_per_month: int

    # -- validation --------------------------------------------------

    def problems(self) -> list[str]:
        out = list(self.campaign.problems())
        list_check = self.floor.evaluate(self.model)
        if not list_check.approved:
            # A list price that does not clear our own floor means the product
            # is mispriced, not that a deal is bad. Surface it here rather than
            # discovering it on the third call.
            out.extend(f"list terms: {b}" for b in list_check.breaches)
        if self.seats_onboardable_per_month <= 0:
            out.append("no onboarding capacity: every deal would be unservable")
        return out

    @property
    def is_valid(self) -> bool:
        return not self.problems()

    # -- the gate ----------------------------------------------------

    def evaluate(self, *concessions: Concession) -> DealCheck:
        """Judge the deal that results from these concessions, all at once."""
        return self.floor.evaluate_with(self.model, concessions)

    def decide(self, *concessions: Concession) -> Decision:
        """Close, counter, or walk. Never escalate - there is nobody to ask.

        A rejected ask does not end the call: `counter_offer()` finds the best
        thing still inside the floor, so the agent always has something true to
        say next. Only when nothing clears does it walk.
        """
        check = self.evaluate(*concessions)
        if check.approved:
            return Decision(
                outcome=Outcome.CLOSE,
                line=(
                    f"That works. {check.model.seats} seats at "
                    f"{check.model.currency} {check.model.net_price_per_seat_month} "
                    f"per seat per month for {check.model.term_months} months. "
                    f"I'll send the order form now - reply to it and we're set."
                ),
                check=check,
            )

        counter = self.counter_offer()
        if counter is None:
            return Decision(
                outcome=Outcome.WALK_AWAY,
                line=(
                    "I can't get there on price, and I'd rather tell you that "
                    "than waste your time. If the numbers change, call me back."
                ),
                check=check,
            )

        best = counter.model
        return Decision(
            outcome=Outcome.COUNTER,
            line=(
                f"I can't do all of that. What I can do is "
                f"{best.discount_pct}% off, "
                f"{best.free_months} free month(s), and "
                f"{best.currency} {best.onboarding_fee} onboarding - "
                f"{best.currency} {best.effective_rate_per_seat_month} per seat "
                f"per month averaged across the term. That's my best."
            ),
            check=counter,
        )

    # -- headroom ----------------------------------------------------

    def max_permitted(self, lever: Lever, step: Decimal | str = "0.5") -> Decimal:
        """The largest single ask on `lever` that still clears the *whole* floor.

        Walked upward in steps rather than solved analytically, because the
        floor is three interacting constraints and the point of this vertical
        is that they do not decompose. Slow, exact, and correct when the model
        changes shape.
        """
        step_d = as_money(step)
        if step_d <= 0:
            raise ValueError("step must be positive")
        cap = self.floor.cap_for(lever)
        best = Decimal(0)
        amount = Decimal(0)
        while amount <= cap:
            trial = Concession(lever=lever, amount=amount)
            if self.floor.evaluate_with(self.model, [trial]).approved:
                best = amount
            else:
                break
            amount += step_d
        return best

    def counter_offer(self) -> DealCheck | None:
        """The deepest single concession we can still stand behind, or None.

        Deliberately single-lever. A counter built by stacking two levers is
        the exact move this vertical is trying to prevent an agent from making
        by instinct, so the safe path does not model it either.
        """
        candidates: list[DealCheck] = []
        for lever in (Lever.DISCOUNT_PCT, Lever.FREE_MONTHS, Lever.ONBOARDING_WAIVER):
            amount = self.max_permitted(lever, step="1" if lever is not Lever.ONBOARDING_WAIVER else "100")
            if amount <= 0:
                continue
            check = self.floor.evaluate_with(
                self.model, [Concession(lever=lever, amount=amount)]
            )
            if check.approved:
                candidates.append(check)
        if not candidates:
            base = self.floor.evaluate(self.model)
            return base if base.approved else None
        # Give away the least: the cheapest concession that is still a real one.
        return max(candidates, key=lambda c: c.model.total_contract_value)

    def concession_report(self, *concessions: Concession) -> dict[str, Any]:
        """Per-lever caps next to the combined verdict, in one object.

        This is the artifact for the UI and the demo. Every row can say "yes,
        within cap" while the footer says REJECTED, and seeing those two facts
        adjacent is the whole argument for evaluating deals as a whole.
        """
        rows = []
        for c in concessions:
            ok, reason = self.floor.lever_within_cap(c)
            rows.append(
                {
                    "lever": c.lever.value,
                    "amount": str(as_money(c.amount)),
                    "described": c.described,
                    "within_cap": ok,
                    "reason": reason,
                    "note": c.note,
                }
            )
        check = self.evaluate(*concessions)
        return {
            "levers": rows,
            "all_levers_within_cap": all(r["within_cap"] for r in rows),
            "combined": check.to_dict(),
            "stacking_trap": (
                all(r["within_cap"] for r in rows) and not check.approved
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign": self.campaign.to_dict(),
            "list_terms": self.model.to_dict(),
            "floor": self.floor.to_dict(),
            "seats_onboardable_per_month": self.seats_onboardable_per_month,
            "valid": self.is_valid,
            "problems": self.problems(),
        }


def build_playbook(cfg: dict[str, Any] | None = None) -> SaasPlaybook:
    """The configured playbook. One call, everything wired."""
    cfg = cfg or load_config()
    model = model_from_config(cfg)
    floor = floor_from_config(cfg)
    return SaasPlaybook(
        campaign=campaign_for(model, floor, cfg),
        model=model,
        floor=floor,
        seats_onboardable_per_month=cfg["capacity"]["seats_onboardable_per_month"],
    )


def model_from_terms(terms: dict[str, Any], cfg: dict[str, Any]) -> SubscriptionModel:
    """A model from partial terms, backfilled from the configured list terms.

    Used by the app: a proposed deal on a form supplies four fields, and the
    cost-to-serve and CAC have to come from somewhere that is not the URL.
    """
    base = dict(cfg["list_terms"])
    base.update({k: v for k, v in terms.items() if v is not None})
    return SubscriptionModel(**base)


def concessions_from(terms: dict[str, Any], cfg: dict[str, Any]) -> list[Concession]:
    """Read a proposed deal as the list of concessions that produced it."""
    base = cfg["list_terms"]
    out: list[Concession] = []
    discount = as_money(terms.get("discount_pct", 0)) - as_money(base["discount_pct"])
    if discount > 0:
        out.append(Concession(Lever.DISCOUNT_PCT, discount))
    free = int(terms.get("free_months", 0)) - int(base["free_months"])
    if free > 0:
        out.append(Concession(Lever.FREE_MONTHS, free))
    waived = as_money(base["onboarding_fee"]) - as_money(
        terms.get("onboarding_fee", base["onboarding_fee"])
    )
    if waived > 0:
        out.append(Concession(Lever.ONBOARDING_WAIVER, waived))
    seats_cut = int(base["seats"]) - int(terms.get("seats", base["seats"]))
    if seats_cut > 0:
        out.append(Concession(Lever.SEAT_REDUCTION, seats_cut))
    term_cut = int(base["term_months"]) - int(
        terms.get("term_months", base["term_months"])
    )
    if term_cut > 0:
        out.append(Concession(Lever.TERM_REDUCTION, term_cut))
    return out


def summarise(playbook: SaasPlaybook, concessions: Iterable[Concession]) -> str:
    """One line for a log or a transcript."""
    return playbook.decide(*concessions).check.reason
