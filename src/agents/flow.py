"""A state machine over the sales call.

The $74 incident is the argument for this module. On a live call the buyer said
"thirty people"; the agent reserved 30 units, priced them at the configured
margin, and offered $74 against a stated budget of $385. Every individual guard
passed - the floor was $57.86 for 30 units, so $74 cleared it - because each
tool only validated *itself*. Nothing owned the question of whether the call had
reached a state where quoting was legitimate at all.

Prompts cannot fix that. "Ask what a unit means before you quote" is a request,
and a model under conversational pressure will skip it. So ordering becomes
structure: every tool declares the phase it may run in and the facts that must
already be known, and the gate refuses anything else.

    OPENING -> DISCOVERY -> QUALIFIED -> QUOTED -> {NEGOTIATING, CLOSING} -> CLOSED
                                 ^            |
                                 +-- ESCALATED +

The refusal is not an error. It is returned to the model as the next instruction,
so being blocked from quoting reads as "go find out how many items per person"
rather than as a tool failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    OPENING = "opening"
    """Greeting. Nothing learned yet."""

    DISCOVERY = "discovery"
    """Learning what they buy, from whom, at what price, how much."""

    QUALIFIED = "qualified"
    """Quantity is known *in units* and capacity is reserved."""

    QUOTED = "quoted"
    """A validated price has been put to them."""

    NEGOTIATING = "negotiating"
    """They pushed back. Concessions and restructuring live here."""

    ESCALATED = "escalated"
    """Waiting on the operator. No commitment may be made in this phase."""

    CLOSING = "closing"
    """Terms agreed; gathering written confirmation."""

    CLOSED = "closed"
    """Order filed as a claim, or the call ended."""


#: Directed edges. Anything absent is unreachable.
TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.OPENING: frozenset({Phase.DISCOVERY, Phase.CLOSED}),
    Phase.DISCOVERY: frozenset({Phase.QUALIFIED, Phase.DISCOVERY, Phase.CLOSED}),
    Phase.QUALIFIED: frozenset({Phase.QUOTED, Phase.ESCALATED, Phase.DISCOVERY, Phase.CLOSED}),
    Phase.QUOTED: frozenset({Phase.NEGOTIATING, Phase.CLOSING, Phase.ESCALATED, Phase.CLOSED}),
    Phase.NEGOTIATING: frozenset({Phase.QUOTED, Phase.ESCALATED, Phase.CLOSING, Phase.CLOSED}),
    Phase.ESCALATED: frozenset({Phase.QUOTED, Phase.NEGOTIATING, Phase.CLOSING, Phase.CLOSED}),
    Phase.CLOSING: frozenset({Phase.CLOSED, Phase.NEGOTIATING}),
    Phase.CLOSED: frozenset(),
}


@dataclass
class Facts:
    """What the call has actually established.

    These are preconditions, not preferences. `units_confirmed` exists solely
    because of the $74 call: a headcount is not a unit count, and the difference
    is invisible once it has become an integer.
    """

    units_confirmed: bool = False
    """The agent has established how many ITEMS, not how many people."""

    headcount: int | None = None
    units: int | None = None

    items_per_person: float = 1.0
    """How many items one person consumes, from the business profile.

    Defaults to 1.0, which means "no opinion" - without a profile, units equal
    to headcount is a perfectly legitimate order and warning about it would be
    noise. Once a profile says two pastries each, thirty items for thirty people
    is the $311 mistake and worth interrupting for.
    """
    capacity_held: bool = False
    asked_current_spend: bool = False
    their_budget: float | None = None
    price_validated: bool = False
    operator_approved_total: float | None = None

    def expected_units(self, headcount: int) -> float:
        """What the profile says this headcount should need."""
        return headcount * self.items_per_person

    def conversion_looks_wrong(
        self, units: int | None, headcount: int | None
    ) -> str | None:
        """Warn when a headcount looks like it was used as an item count.

        Returns None when there is nothing to say. Deliberately quiet in three
        cases: no headcount to compare against, a default ratio of 1.0 (no
        profile, so no expectation), and small shortfalls. Ordering a little
        under the ratio is a real choice a buyer makes; ordering half is the
        unit-confusion bug, and only that deserves an interruption. A guard that
        fires on 58-vs-60 gets ignored by the time it matters.
        """
        if units is None or headcount is None:
            return None
        if self.items_per_person <= 1.0:
            return None

        expected = self.expected_units(headcount)
        if units >= expected * 0.9:
            return None

        return (
            f"{units} items for {headcount} people looks low - the profile says "
            f"{self.items_per_person:g} each, so about {expected:g}. Say that "
            "assumption out loud and let them correct it."
        )

    def missing_for_quote(self) -> list[str]:
        gaps = []
        if not self.units_confirmed:
            gaps.append(
                "how many ITEMS they need (a headcount is not an item count - "
                "ask how many per person, or state your assumption out loud)"
            )
        if not self.capacity_held:
            gaps.append("a capacity reservation (call check_capacity)")
        if not self.asked_current_spend:
            gaps.append("what they pay now (ask before you name a price)")
        return gaps


@dataclass
class Gate:
    """Enforces phase and preconditions. One per call."""

    phase: Phase = Phase.OPENING
    facts: Facts = field(default_factory=Facts)
    history: list[str] = field(default_factory=list)

    # -- transitions --------------------------------------------------

    def can_move(self, to: Phase) -> bool:
        return to in TRANSITIONS[self.phase]

    def move(self, to: Phase) -> bool:
        """Advance if the edge exists. Never raises - a refused transition is
        information for the model, not a crash on a live call."""
        if not self.can_move(to):
            self.history.append(f"BLOCKED {self.phase.value} -> {to.value}")
            return False
        self.history.append(f"{self.phase.value} -> {to.value}")
        self.phase = to
        return True

    # -- tool admission -----------------------------------------------

    def allow_capacity(self, qty: int) -> str | None:
        """None means proceed; a string is the instruction to return instead."""
        if self.phase is Phase.CLOSED:
            return "BLOCKED. This call is over. Do not reserve anything."
        if not self.facts.units_confirmed:
            return (
                f"BLOCKED. Before reserving {qty}, confirm whether {qty} is the "
                "number of ITEMS or the number of PEOPLE. If they gave a "
                "headcount, ask how many items each - or say your assumption "
                "aloud so they can correct you. Then call confirm_units."
            )
        return None

    def allow_quote(self, total: float) -> str | None:
        if self.phase in (Phase.OPENING, Phase.DISCOVERY):
            gaps = self.facts.missing_for_quote()
            return (
                "BLOCKED - too early to quote. Still needed: "
                + "; ".join(gaps)
                + ". Do not name a price yet."
            )
        if self.phase is Phase.CLOSED:
            return "BLOCKED. This call is over. Do not quote."
        if gaps := self.facts.missing_for_quote():
            return "BLOCKED - still needed: " + "; ".join(gaps) + "."

        budget = self.facts.their_budget
        if budget and total < budget:
            return (
                f"BLOCKED. They already offered {budget:.2f}; quoting {total:.2f} "
                f"discards {budget - total:.2f}. Offer {budget:.2f} or more."
            )
        return None

    def allow_close(self) -> str | None:
        if not self.facts.price_validated:
            return (
                "BLOCKED. No validated price has been agreed. Run propose_price "
                "and get their agreement before filing an order."
            )
        if self.phase is Phase.ESCALATED:
            return (
                "BLOCKED. You are waiting on the owner. Do not file an order "
                "before the answer comes back."
            )
        if self.phase is Phase.CLOSED:
            return "BLOCKED. This order was already filed."
        return None

    # -- observations -------------------------------------------------

    def note_units(self, units: int, headcount: int | None = None) -> None:
        self.facts.units_confirmed = True
        self.facts.units = units
        self.facts.headcount = headcount
        if self.phase is Phase.OPENING:
            self.move(Phase.DISCOVERY)

    def note_capacity_held(self) -> None:
        self.facts.capacity_held = True
        if self.phase in (Phase.OPENING, Phase.DISCOVERY):
            self.move(Phase.DISCOVERY) if self.phase is Phase.OPENING else None
            self.move(Phase.QUALIFIED)

    def note_position(self, budget: float | None) -> None:
        self.facts.asked_current_spend = True
        if budget:
            self.facts.their_budget = budget
        if self.phase is Phase.OPENING:
            self.move(Phase.DISCOVERY)

    def note_quoted(self) -> None:
        self.facts.price_validated = True
        if self.phase in (Phase.QUALIFIED, Phase.NEGOTIATING, Phase.ESCALATED):
            self.move(Phase.QUOTED)

    def note_pushback(self) -> None:
        if self.phase is Phase.QUOTED:
            self.move(Phase.NEGOTIATING)

    def note_escalating(self) -> None:
        self.move(Phase.ESCALATED)

    def note_operator(self, approved_total: float | None) -> None:
        if approved_total:
            self.facts.operator_approved_total = approved_total
        if self.phase is Phase.ESCALATED:
            self.move(Phase.QUOTED if self.facts.price_validated else Phase.QUALIFIED)

    def note_closed(self) -> None:
        self.move(Phase.CLOSING)
        self.move(Phase.CLOSED)

    def describe(self) -> str:
        gaps = self.facts.missing_for_quote()
        return (
            f"phase={self.phase.value} "
            f"units={self.facts.units} budget={self.facts.their_budget} "
            f"blocking={gaps or 'nothing'}"
        )
