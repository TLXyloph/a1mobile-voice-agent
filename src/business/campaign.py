"""One campaign engine, configured per vertical.

We are not building a catering app, a freelance app and a sales app. We are
building one loop - find who to call, offer them something, stay inside the
operator's limits, stop when a side effect proves it closed - and configuring
it three ways. Everything vertical-specific in this file is *data*. If a
fourth vertical needs new code here, the abstraction is wrong.

A campaign answers exactly four questions:

  who      `icp` + `discovery` - who to find, and by what strategy
  what     `offer` + `capacity_units` - what we sell and what we can supply
  bounds   `envelope` - what the agent may agree to with nobody watching
  done     `close_condition` - which independent side effect ends the loop

The envelope is the load-bearing part. A voice agent negotiating live will be
pushed ("can you do 200 instead of 100?", "what about Tuesday?", "any chance
of 30% off?"). Every one of those is either inside the envelope - decide now,
no human needed - or outside it, in which case the only correct behaviour is
to escalate rather than improvise. `Envelope.permits()` is what makes that a
mechanical check instead of a judgement call made by a language model under
pressure.

`close_condition` deliberately points at `src.verify.receipts.Channel`. A
campaign cannot declare itself closed by the agent saying so: every acceptable
close channel is drawn from `INDEPENDENT_CHANNELS`, which is asserted in
tests. The anti-fabrication invariant follows the campaign into every vertical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

from src.verify.receipts import INDEPENDENT_CHANNELS, Channel


class DiscoveryStrategy(str, Enum):
    """How this campaign sources the list of people to contact."""

    NO_WEBSITE = "no_website"
    """Absence as a signal: businesses with no site, or a brochure with no
    working functionality. See `src.business.discovery`."""

    EVENT_SIGNAL = "event_signal"
    """A dated public trigger implies a need - a hiring post, an office move,
    an announced offsite. Timing is the whole edge; a stale signal is noise."""

    SEEDED_LIST = "seeded_list"
    """The operator supplies the list (their own CRM, an ICP export). We do not
    invent targets - inventing a prospect is the outbound version of
    fabricating a result."""


class CloseCondition(str, Enum):
    """What observable side effect means this campaign closed a deal.

    Not "the prospect said yes on the call." Agent-audible agreement is a
    reason to keep going, never a reason to stop.
    """

    WRITTEN_CONFIRMATION = "written_confirmation"
    """They put it in writing - a reply SMS/email, or the order in the POS."""

    DELIVERED_ARTIFACT = "delivered_artifact"
    """Something exists that did not before, and can be fetched and checked."""

    BOOKED_MEETING = "booked_meeting"
    """A calendar event exists on both sides, readable from the provider."""


#: Which independent channels can satisfy each close condition. "Acceptable"
#: is any-of, not all-of: a signed email and a POS record are equally good
#: proof of a written confirmation. Every entry must be an independent channel
#: - `tests/test_campaign.py` pins that, so a well-meaning edit that adds
#: AGENT_ASSERTION here goes red instead of quietly re-opening the DQ path.
ACCEPTABLE_EVIDENCE: dict[CloseCondition, frozenset[Channel]] = {
    CloseCondition.WRITTEN_CONFIRMATION: frozenset(
        {Channel.INBOUND_SMS, Channel.INBOUND_EMAIL, Channel.PROVIDER_API}
    ),
    CloseCondition.DELIVERED_ARTIFACT: frozenset(
        {Channel.WEB_CHECK, Channel.INBOUND_EMAIL}
    ),
    CloseCondition.BOOKED_MEETING: frozenset(
        {Channel.PROVIDER_API, Channel.INBOUND_EMAIL, Channel.WEB_CHECK}
    ),
}


@dataclass(frozen=True)
class Envelope:
    """The bounds the agent may act inside without asking a human.

    Read it as a standing authorisation from the operator: "agree to anything
    in here on my behalf; for anything else, call me." Frozen because an agent
    that can widen its own mandate mid-call has no mandate.
    """

    min_price: float
    """Floor per unit. Below this the job is not worth doing."""

    max_qty: int
    """Ceiling in `capacity_units`. This is a supply limit, not a wish."""

    earliest_date: date
    latest_date: date
    """Inclusive delivery window the agent may commit to."""

    max_discount_pct: float
    """Headroom off list, 0-100. Anything deeper needs the operator."""

    currency: str = "USD"

    def problems(self) -> list[str]:
        """Coherence check. Empty list means the envelope is usable."""
        out: list[str] = []
        if self.min_price <= 0:
            out.append(f"min_price must be > 0, got {self.min_price}")
        if self.max_qty <= 0:
            out.append(f"max_qty must be > 0, got {self.max_qty}")
        if not 0 <= self.max_discount_pct <= 100:
            out.append(
                f"max_discount_pct must be within 0-100, got {self.max_discount_pct}"
            )
        if self.latest_date < self.earliest_date:
            out.append(
                f"date range inverted: {self.earliest_date} .. {self.latest_date}"
            )
        if not self.currency:
            out.append("currency must be set")
        return out

    @property
    def is_valid(self) -> bool:
        return not self.problems()

    def permits(
        self,
        *,
        price: float | None = None,
        qty: int | None = None,
        when: date | None = None,
        discount_pct: float | None = None,
    ) -> tuple[bool, str]:
        """May the agent agree to this unilaterally?

        Returns (permitted, reason). The reason is written to be said out loud
        on the call or handed to the operator as an escalation, so it names the
        term and the bound rather than saying "outside policy".

        Only the terms passed are checked; a `None` is "not on the table",
        which is not the same as "anything goes".
        """
        if price is not None and price < self.min_price:
            return False, (
                f"price {price:.2f} {self.currency} is below the floor of "
                f"{self.min_price:.2f} - needs operator approval"
            )
        if qty is not None and qty > self.max_qty:
            return False, (
                f"quantity {qty} exceeds committed capacity of {self.max_qty} "
                f"- needs operator approval"
            )
        if when is not None and not (self.earliest_date <= when <= self.latest_date):
            return False, (
                f"date {when} falls outside the agreed window "
                f"{self.earliest_date} .. {self.latest_date} - needs operator approval"
            )
        if discount_pct is not None and discount_pct > self.max_discount_pct:
            return False, (
                f"discount {discount_pct:.1f}% exceeds the {self.max_discount_pct:.1f}% "
                f"the agent may give - needs operator approval"
            )
        return True, "within envelope"

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_price": self.min_price,
            "max_qty": self.max_qty,
            "earliest_date": self.earliest_date.isoformat(),
            "latest_date": self.latest_date.isoformat(),
            "max_discount_pct": self.max_discount_pct,
            "currency": self.currency,
        }


@dataclass(frozen=True)
class Campaign:
    """A configured instance of the outbound loop. Vertical-agnostic."""

    name: str
    vertical: str
    """Free-text label only - no behaviour keys off it. If code ever branches
    on `vertical`, that branch belongs in the campaign data instead."""

    icp: str
    """Who to target, specific enough that a human could screen a list by it."""

    offer: str
    """What we are selling, phrased as the prospect would hear it."""

    discovery: DiscoveryStrategy
    close_condition: CloseCondition
    capacity_units: str
    """What the operator actually sells, in their units: 'muffins/week'."""

    envelope: Envelope
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def close_evidence(self) -> frozenset[Channel]:
        """Channels that can prove this campaign closed. Any one suffices."""
        return ACCEPTABLE_EVIDENCE[self.close_condition]

    def problems(self) -> list[str]:
        """Everything wrong with this campaign. Empty list means shippable."""
        out = [f"envelope: {p}" for p in self.envelope.problems()]
        for label, value in (
            ("name", self.name),
            ("vertical", self.vertical),
            ("icp", self.icp),
            ("offer", self.offer),
            ("capacity_units", self.capacity_units),
        ):
            if not value.strip():
                out.append(f"{label} must not be empty")
        if not ACCEPTABLE_EVIDENCE.get(self.close_condition):
            out.append(f"no evidence channel can close {self.close_condition.value}")
        # An agent-satisfiable close condition would let the campaign mark
        # itself done by talking. Refuse to run rather than risk the DQ.
        rogue = self.close_evidence - INDEPENDENT_CHANNELS
        if rogue:
            out.append(
                "close condition accepts non-independent evidence: "
                + ", ".join(sorted(c.value for c in rogue))
            )
        return out

    @property
    def is_valid(self) -> bool:
        return not self.problems()

    def escalation_for(self, **terms: Any) -> str | None:
        """The escalation message for these terms, or None if the agent may
        just agree. Thin wrapper so call sites read as intent, not policy."""
        ok, reason = self.envelope.permits(**terms)
        return None if ok else reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vertical": self.vertical,
            "icp": self.icp,
            "offer": self.offer,
            "discovery": self.discovery.value,
            "close_condition": self.close_condition.value,
            "close_evidence": sorted(c.value for c in self.close_evidence),
            "capacity_units": self.capacity_units,
            "envelope": self.envelope.to_dict(),
            "notes": self.notes,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# The three demo campaigns.
#
# Everything below is configuration. Note that no two of them share a
# discovery strategy or a close condition, and none of them needed a new field
# - that is the claim the engine is making, stated as data.
# ---------------------------------------------------------------------------

RESTAURANT_CATERING = Campaign(
    name="Weekday corporate catering - fall block",
    vertical="restaurant",
    icp=(
        "Offices, clinics and schools within 3 miles that already feed 15-60 "
        "people on a recurring weekday schedule, and have shown a dated signal "
        "of an upcoming need (hiring spree, announced offsite, new office)."
    ),
    offer=(
        "A standing weekly breakfast catering order at a fixed per-head price, "
        "delivered before 8:30am, cancellable with 48 hours notice."
    ),
    discovery=DiscoveryStrategy.EVENT_SIGNAL,
    close_condition=CloseCondition.WRITTEN_CONFIRMATION,
    capacity_units="muffins/week",
    envelope=Envelope(
        min_price=3.50,          # per muffin; below this the kitchen loses money
        max_qty=600,             # what the ovens can actually do in a week
        earliest_date=date(2026, 8, 10),
        latest_date=date(2026, 11, 20),
        max_discount_pct=15.0,
        currency="USD",
    ),
    notes=(
        "Do not agree to same-week starts: prep is bought on a Thursday. A "
        "date inside the window is still subject to the 48-hour rule, which "
        "the agent states rather than negotiates."
    ),
    tags=("catering", "b2b", "recurring"),
)

FREELANCE_WEBDEV = Campaign(
    name="Local businesses with no working website",
    vertical="freelance_services",
    icp=(
        "Owner-operated local businesses with no website at all, a social page "
        "standing in for one, or a single-page brochure with no booking, "
        "ordering or contact functionality - and a phone number we can call."
    ),
    offer=(
        "A five-page site with online booking wired up, live in two weeks, "
        "fixed price, and they own the domain and the code."
    ),
    discovery=DiscoveryStrategy.NO_WEBSITE,
    close_condition=CloseCondition.DELIVERED_ARTIFACT,
    capacity_units="site builds/month",
    envelope=Envelope(
        min_price=1200.00,       # per build; below this it is not worth a week
        max_qty=4,               # one person, four builds a month, honestly
        earliest_date=date(2026, 8, 3),
        latest_date=date(2026, 12, 18),
        max_discount_pct=20.0,
        currency="USD",
    ),
    notes=(
        "The pitch only survives contact with the owner if the qualification "
        "reason is true and specific. 'You have no website' said to someone "
        "who has a good one ends the call and the reputation. Discovery is "
        "therefore biased to exclude when uncertain."
    ),
    tags=("freelance", "smb", "web"),
)

STARTUP_OUTBOUND = Campaign(
    name="Seed-stage outbound - discovery calls",
    vertical="b2b_saas",
    icp=(
        "Taken verbatim from the startup's own customer definition: heads of "
        "support at 50-500 person e-commerce companies running Zendesk, who "
        "have publicly complained about ticket backlog. The operator owns this "
        "definition; we do not widen it to make the list longer."
    ),
    offer=(
        "A 20-minute call showing their own backlog triaged automatically, "
        "using a sample of their public help-centre content."
    ),
    discovery=DiscoveryStrategy.SEEDED_LIST,
    close_condition=CloseCondition.BOOKED_MEETING,
    capacity_units="demos/week",
    envelope=Envelope(
        min_price=0.01,          # the demo is free; the term is the pilot fee
        max_qty=12,              # founder-led demos, one calendar
        earliest_date=date(2026, 8, 4),
        latest_date=date(2026, 9, 30),
        max_discount_pct=0.0,    # no pricing authority on a first call at all
        currency="USD",
    ),
    notes=(
        "The agent may book a meeting and nothing else. Any pricing question "
        "escalates by construction (max_discount_pct=0), which is the correct "
        "behaviour for a first conversation, not a limitation."
    ),
    tags=("outbound", "b2b", "meetings"),
)

CAMPAIGNS: dict[str, Campaign] = {
    "restaurant_catering": RESTAURANT_CATERING,
    "freelance_webdev": FREELANCE_WEBDEV,
    "startup_outbound": STARTUP_OUTBOUND,
}


def get_campaign(key: str) -> Campaign:
    """Look up a preconfigured campaign, failing loudly on a typo."""
    try:
        return CAMPAIGNS[key]
    except KeyError:
        raise KeyError(
            f"unknown campaign {key!r}; known: {', '.join(sorted(CAMPAIGNS))}"
        ) from None
