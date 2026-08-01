"""Live negotiation policy.

"Too expensive" is the most predictable friction a judge can plant, and it is
where a generic voice agent fails in one of two ways: it folds immediately, or
it bluffs a number it cannot honour. This module exists so the agent does
neither.

Three ideas carry the design:

1. **The competitor book is pre-computed.** Looking up a rival's catering prices
   mid-call costs five to ten seconds of dead air, and dead air is the single
   most bot-revealing thing on a phone line. Prices are scraped before the
   campaign runs, so mid-call it is a dictionary lookup.

2. **Concessions are a ladder with a floor, not a judgement call.** An LLM told
   "you may discount up to ten percent" will concede past that under pressure,
   because sounding agreeable is what conversational training rewards. The agent
   proposes; `pricing.validate_quote` disposes. This module only decides *what
   move to make next*, never *what price is acceptable*.

3. **Stop conditions are counted, not vibed.** "Don't be pushy" is unenforceable
   in a prompt. Two declines ends it. One discount escalation, not three.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("agents.negotiation")


class Tactic(str, Enum):
    """The next move. Ordered roughly by how much they cost us."""

    ASK_CURRENT_SPEND = "ask_current_spend"
    """Cheapest and best opening. What they volunteer beats what we guess."""

    EMPHASISE_VALUE = "emphasise_value"
    """Non-price move: freshness, delivery included, dietary options."""

    OFFER_CONCESSION = "offer_concession"
    """Step down the ladder. Costs margin - use after value framing failed."""

    RESTRUCTURE = "restructure"
    """Change the deal shape, not the price: smaller order, different date,
    simpler item. Often lands where a discount would not."""

    ESCALATE_TO_OPERATOR = "escalate_to_operator"
    """Out of envelope. Ask the human, by voice, while holding the line."""

    CLOSE = "close"
    """They are agreeable. Stop selling and confirm specifics."""

    WALK_AWAY = "walk_away"
    """Thank them and exit. A polite exit preserves a future call."""


@dataclass
class CompetitorPrice:
    """One rival's price for a comparable order."""

    vendor: str
    item: str
    unit_price: float
    min_qty: int = 1
    delivery_included: bool = False
    source_url: str | None = None

    def estimate_total(self, qty: int) -> float:
        return round(self.unit_price * max(qty, self.min_qty), 2)


class CompetitorBook:
    """Pre-scraped rival pricing, keyed by loose vendor name.

    Built before the campaign runs. Lookup is a dict hit so the agent never
    stalls mid-sentence.
    """

    def __init__(self, prices: list[CompetitorPrice] | None = None) -> None:
        self._by_vendor: dict[str, list[CompetitorPrice]] = {}
        for p in prices or []:
            self._by_vendor.setdefault(self._key(p.vendor), []).append(p)

    @staticmethod
    def _key(vendor: str) -> str:
        return re.sub(r"[^a-z0-9]", "", vendor.lower())

    def lookup(self, vendor: str) -> list[CompetitorPrice]:
        """Find by loose name match. Callers say 'Round Table', not the slug."""
        key = self._key(vendor)
        if key in self._by_vendor:
            return self._by_vendor[key]
        # Substring both directions: "roundtablepizza" vs "roundtable".
        for known, prices in self._by_vendor.items():
            if known and (known in key or key in known):
                return prices
        return []

    def undercut_target(self, vendor: str, qty: int, by_pct: float = 10.0) -> float | None:
        """What total would beat this vendor by `by_pct`?

        Returns None when we have no data - and the caller must then NOT invent
        a number. Guessing a rival's price and being wrong is a credibility hit
        we cannot recover from mid-call.
        """
        prices = self.lookup(vendor)
        if not prices:
            return None
        best = min(p.estimate_total(qty) for p in prices)
        return round(best * (1 - by_pct / 100), 2)

    @classmethod
    def load(cls, path: str | Path) -> CompetitorBook:
        data = json.loads(Path(path).read_text())
        return cls([CompetitorPrice(**row) for row in data])

    def save(self, path: str | Path) -> None:
        rows = [
            p.__dict__ for prices in self._by_vendor.values() for p in prices
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(rows, indent=2))


@dataclass
class NegotiationState:
    """What has happened so far in this call. Drives the stop conditions."""

    opening_total: float
    current_total: float
    qty: int

    concessions_made: int = 0
    declines_received: int = 0
    value_framed: bool = False
    restructure_offered: bool = False
    asked_current_spend: bool = False
    competitor_named: str | None = None
    their_stated_budget: float | None = None
    """What they said they would pay US. A floor for our quote."""

    competitor_price: float | None = None
    """What they pay their current supplier. A number to beat, never a floor."""
    escalation_pending: bool = False
    history: list[str] = field(default_factory=list)

    def record(self, note: str) -> None:
        self.history.append(note)
        logger.info("negotiation: %s", note)

    @property
    def total_discount_pct(self) -> float:
        if self.opening_total <= 0:
            return 0.0
        return round((1 - self.current_total / self.opening_total) * 100, 2)


@dataclass
class Limits:
    """Operator-set bounds. Outside these, the agent must ask, not decide."""

    max_concessions: int = 1
    """One discount escalation. A second reads as desperation and trains the
    buyer to keep pushing."""

    max_declines: int = 2
    """Two 'no's and we exit politely. This is what 'not pushy' means in code."""

    max_discount_pct: float = 10.0
    floor_total: float = 0.0
    """Hard money floor from the cost model. Never quote below this."""


def next_tactic(state: NegotiationState, limits: Limits) -> Tactic:
    """Decide the next move.

    Deliberately ordered so the cheap moves are exhausted before we spend
    margin, and so both exits (close, walk away) are always reachable.
    """
    if state.declines_received >= limits.max_declines:
        return Tactic.WALK_AWAY

    # Learn before spending. What they say they pay beats any estimate.
    if not state.asked_current_spend:
        return Tactic.ASK_CURRENT_SPEND

    # Free move: reframe on value before touching price.
    if not state.value_framed:
        return Tactic.EMPHASISE_VALUE

    # Their budget is knowable and below our floor -> only the operator can say.
    if state.their_stated_budget is not None:
        if state.their_stated_budget < limits.floor_total:
            return Tactic.ESCALATE_TO_OPERATOR
        if state.their_stated_budget >= state.current_total:
            return Tactic.CLOSE

    if state.concessions_made < limits.max_concessions:
        return Tactic.OFFER_CONCESSION

    # Out of discounts. Changing the deal shape is cheaper than more money.
    if not state.restructure_offered:
        return Tactic.RESTRUCTURE

    # Everything inside the envelope is spent. The human decides now.
    return Tactic.ESCALATE_TO_OPERATOR


def concession_step(
    state: NegotiationState, limits: Limits, step_pct: float = 5.0
) -> float:
    """Next price down the ladder, clamped at the floor and the discount cap.

    Clamping happens here as well as in the pricing validator. Two independent
    clamps is not redundancy worth removing - this one keeps the *spoken* number
    sane, and the validator is the last line before a commitment is made.
    """
    proposed = state.current_total * (1 - step_pct / 100)

    cap_by_pct = state.opening_total * (1 - limits.max_discount_pct / 100)
    floor = max(limits.floor_total, cap_by_pct)

    return round(max(proposed, floor), 2)


TACTIC_SCRIPTS: dict[Tactic, str] = {
    Tactic.ASK_CURRENT_SPEND: (
        "Ask, warmly and in one sentence, who they currently order from and "
        "roughly what they pay. Do not guess a number at them - if you are wrong "
        "you lose credibility you cannot get back. Let them tell you."
    ),
    Tactic.EMPHASISE_VALUE: (
        "Name one or two concrete differences that are true and cost us nothing: "
        "baked that morning, delivery and setup included, dietary options handled. "
        "Do not mention price. Do not disparage whoever they use now."
    ),
    Tactic.OFFER_CONCESSION: (
        "Offer the specific lower total you have been given, framed as something "
        "you can do for them rather than a discount you were holding back. One "
        "sentence, then stop talking and let them respond."
    ),
    Tactic.RESTRUCTURE: (
        "Change the shape, not the price: a smaller order, a simpler item, a "
        "different day. Ask which of those would work rather than proposing one."
    ),
    Tactic.ESCALATE_TO_OPERATOR: (
        "Tell them you want to check whether you can do that, and ask if they can "
        "hold for a moment. Then call ask_operator. Do not promise anything "
        "before the answer comes back."
    ),
    Tactic.CLOSE: (
        "Stop selling. Confirm quantity, date, delivery time and total out loud, "
        "then ask them to text or email confirmation so you both have it in "
        "writing."
    ),
    Tactic.WALK_AWAY: (
        "Thank them genuinely, say you will follow up another time, and end the "
        "call. Do not make a final pitch on the way out."
    ),
}


def guidance(state: NegotiationState, limits: Limits) -> dict[str, Any]:
    """The instruction packet handed to the agent for its next turn."""
    tactic = next_tactic(state, limits)
    packet: dict[str, Any] = {
        "tactic": tactic.value,
        "instruction": TACTIC_SCRIPTS[tactic],
        "discount_so_far_pct": state.total_discount_pct,
    }
    if tactic is Tactic.OFFER_CONCESSION:
        packet["quote_this_total"] = concession_step(state, limits)
    if tactic is Tactic.ESCALATE_TO_OPERATOR:
        packet["why"] = (
            f"buyer budget {state.their_stated_budget} below floor {limits.floor_total}"
            if state.their_stated_budget is not None
            else "envelope exhausted"
        )
    return packet
