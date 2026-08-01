"""Negotiation limits enforced as arithmetic, not as prompt instructions.

An LLM told "never pay more than $12" will pay more than $12 under pressure.
So in Helyx the model never chooses a number. A pure function chooses the
price; the model only chooses the *words* around it. Two mechanisms:

``ConcessionLadder``
    Deterministic schedule of offers from the opening price to the ceiling,
    conceding in shrinking increments. ``offer_for_round(r)`` is a pure
    function of the mandate -- same input, same number, every time.

``MandateGuard``
    (a) ``evaluate()`` decides accept / counter / walk-away by comparison, and
        the walk-away is a arithmetic consequence of round count and ceiling,
        not a judgement call the model can be argued out of.
    (b) ``scan_utterance()`` re-reads whatever the model actually said and
        flags any money amount outside the authorised envelope. This is the
        backstop for the case where the model ignores its instructions: the
        utterance is caught *after* generation and before it is spoken.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .mandate import Mandate

#: Ratio controlling how fast concessions shrink. 0.5 => each successive
#: concession is about half the previous one, which reads as "approaching a
#: real limit" rather than "will keep moving if pushed".
DECAY = 0.5


class Move(str, Enum):
    ACCEPT = "accept"
    COUNTER = "counter"
    WALK_AWAY = "walk_away"


@dataclass(frozen=True)
class Decision:
    move: Move
    unit_price_cents: int  # the price to accept or to counter with; 0 if walking
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "move": self.move.value,
            "unit_price_cents": self.unit_price_cents,
            "reason": self.reason,
        }


class ConcessionLadder:
    """Pure arithmetic schedule of authorised offers."""

    def __init__(self, mandate: Mandate) -> None:
        self.mandate = mandate

    def offer_for_round(self, round_index: int) -> int:
        """Authorised unit price for a given round. Never exceeds the ceiling."""
        if round_index < 0:
            raise ValueError("round_index must be >= 0")
        m = self.mandate
        last = m.max_rounds - 1
        if round_index >= last or last == 0:
            # Final round is the ceiling: our best and last price.
            return m.ceiling_unit_price_cents if last > 0 else m.opening_unit_price_cents
        span = m.ceiling_unit_price_cents - m.opening_unit_price_cents
        if span <= 0:
            return m.opening_unit_price_cents
        denom = 1.0 - DECAY**last
        frac = (1.0 - DECAY**round_index) / denom
        offer = m.opening_unit_price_cents + int(round(span * frac))
        return min(offer, m.ceiling_unit_price_cents)

    def schedule(self) -> list[int]:
        """The full ladder, for display on the dashboard and for audit."""
        return [self.offer_for_round(r) for r in range(self.mandate.max_rounds)]


_MONEY_IN_TEXT = re.compile(
    r"\$\s*(\d[\d,]*(?:\.\d{1,2})?)"  # $4.25 / $1,440
    r"|(\d[\d,]*(?:\.\d{1,2})?)\s*(?:dollars|bucks)\b",
    re.IGNORECASE,
)


@dataclass
class Violation:
    amount_cents: int
    text: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "amount_cents": self.amount_cents,
            "text": self.text,
            "reason": self.reason,
        }


@dataclass
class MandateGuard:
    """Enforces the mandate against both decisions and generated speech."""

    mandate: Mandate
    ladder: ConcessionLadder = field(init=False)

    def __post_init__(self) -> None:
        self.ladder = ConcessionLadder(self.mandate)

    # -- decisions ---------------------------------------------------------
    def authorized_offer(self, round_index: int) -> int:
        return self.ladder.offer_for_round(round_index)

    def is_final_round(self, round_index: int) -> bool:
        return round_index >= self.mandate.max_rounds - 1

    def evaluate(self, counter_unit_cents: int, round_index: int) -> Decision:
        """Decide the next move purely by comparison. No model involved."""
        if counter_unit_cents <= 0:
            raise ValueError("counter_unit_cents must be positive")
        m = self.mandate

        if counter_unit_cents <= m.target_unit_price_cents:
            return Decision(
                Move.ACCEPT,
                counter_unit_cents,
                f"counter {_usd(counter_unit_cents)} is at or below target "
                f"{_usd(m.target_unit_price_cents)}",
            )

        if counter_unit_cents > m.ceiling_unit_price_cents:
            if self.is_final_round(round_index):
                return Decision(
                    Move.WALK_AWAY,
                    0,
                    f"counter {_usd(counter_unit_cents)} exceeds ceiling "
                    f"{_usd(m.ceiling_unit_price_cents)} and rounds are exhausted",
                )
            nxt = self.authorized_offer(round_index + 1)
            return Decision(
                Move.COUNTER,
                nxt,
                f"counter {_usd(counter_unit_cents)} exceeds ceiling "
                f"{_usd(m.ceiling_unit_price_cents)}; authorised counter {_usd(nxt)}",
            )

        # target < counter <= ceiling
        if self.is_final_round(round_index):
            return Decision(
                Move.ACCEPT,
                counter_unit_cents,
                f"counter {_usd(counter_unit_cents)} is within ceiling "
                f"{_usd(m.ceiling_unit_price_cents)} and rounds are exhausted",
            )

        nxt = self.authorized_offer(round_index + 1)
        if nxt >= counter_unit_cents:
            # Never bid against ourselves. Our own ladder has climbed to or past
            # what they are already asking, so their price is the better deal
            # and countering would offer them *more* money than they requested.
            return Decision(
                Move.ACCEPT,
                counter_unit_cents,
                f"counter {_usd(counter_unit_cents)} is at or below our next "
                f"authorised offer {_usd(nxt)}; accepting theirs rather than bidding up",
            )
        return Decision(
            Move.COUNTER,
            nxt,
            f"counter {_usd(counter_unit_cents)} above target; authorised counter {_usd(nxt)}",
        )

    def may_accept(self, unit_cents: int) -> bool:
        return 0 < unit_cents <= self.mandate.ceiling_unit_price_cents

    # -- speech backstop ---------------------------------------------------
    def scan_utterance(self, text: str) -> list[Violation]:
        """Flag any money amount the agent uttered that breaches the mandate.

        An amount is permitted when it reads either as a unit price at or below
        the ceiling, or as a bulk total inside the authorised total band. Any
        other amount is a violation and the utterance must not be spoken.
        """
        m = self.mandate
        low_total = m.quantity * m.opening_unit_price_cents
        high_total = m.ceiling_total_cents
        out: list[Violation] = []

        for match in _MONEY_IN_TEXT.finditer(text or ""):
            raw = match.group(1) or match.group(2)
            cents = int(round(float(raw.replace(",", "")) * 100))
            if cents <= m.ceiling_unit_price_cents:
                continue  # fine as a unit price
            if low_total <= cents <= high_total:
                continue  # fine as an order total
            if cents > high_total:
                reason = (
                    f"{_usd(cents)} exceeds the authorised total "
                    f"{_usd(high_total)} for {m.quantity} units"
                )
            else:
                reason = (
                    f"{_usd(cents)} is above the unit ceiling "
                    f"{_usd(m.ceiling_unit_price_cents)} and is not a valid order total"
                )
            out.append(Violation(cents, match.group(0).strip(), reason))
        return out

    def safe_line(self, round_index: int, pushing_over_ceiling: bool = False) -> str:
        """Deterministic fallback used when generated speech is rejected.

        Varied by situation so a call that leans on the fallback does not read
        as the same sentence repeated. Every variant names only the authorised
        price, so ``scan_utterance`` passes it by construction.
        """
        price = self.authorized_offer(round_index)
        m = self.mandate
        when = m.needed_by.isoformat()

        if pushing_over_ceiling:
            variants = [
                f"That is further than I can go on this order. I can do {_usd(price)} "
                f"per unit for the {m.quantity}, delivered by {when}.",
                f"I hear you on costs, but that is outside what I am authorised to "
                f"agree. {_usd(price)} per unit is where I can be.",
                f"I cannot meet that. My position is {_usd(price)} per unit for "
                f"{m.quantity} {m.item} by {when}.",
            ]
        else:
            variants = [
                f"For {m.quantity} {m.item}, I can do {_usd(price)} per unit, "
                f"delivered by {when}. Does that work on your side?",
                f"I can move to {_usd(price)} per unit on the {m.quantity}, "
                f"still needing them by {when}.",
                f"Let me put {_usd(price)} per unit on the table for the full "
                f"{m.quantity}. Can you work with that for {when}?",
            ]
        return variants[round_index % len(variants)]


def _usd(cents: int) -> str:
    return f"${cents / 100:,.2f}"
