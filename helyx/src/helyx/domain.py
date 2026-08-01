"""The anti-fabrication core of Helyx.

One rule shapes this module: **the agent may only ever propose.** Nothing the
agent says can move a deal into a confirmed state. Confirmation is a *derived*
property computed from evidence that arrived through a channel the agent cannot
write to.

Mechanically that is enforced three ways:

1. ``Proposal.status`` is a read-only property. There is no setter, so no code
   path -- including a future careless one -- can assign ``CONFIRMED``.
2. ``Channel.AGENT_ASSERTION`` is recorded for provenance but is excluded from
   ``INDEPENDENT_CHANNELS``. Volume does not help: a thousand agent assertions
   still derive ``UNCONFIRMED``.
3. Agreement is a stricter bar than delivery. ``Channel.PROVIDER_API`` proves
   *we sent something* (a1mobile returns a message id) but cannot prove the
   counterparty *agreed*. Only ``AGREEMENT_CHANNELS`` can confirm terms.

``tests/test_domain.py`` pins all three. If those tests go red, the
disqualification condition is reachable again.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Channel(str, Enum):
    """Where a piece of evidence came from."""

    AGENT_ASSERTION = "agent_assertion"
    INBOUND_SMS = "inbound_sms"
    INBOUND_EMAIL = "inbound_email"
    HUMAN_REVIEW = "human_review"
    PROVIDER_API = "provider_api"


#: Channels the agent cannot author. AGENT_ASSERTION is deliberately absent.
INDEPENDENT_CHANNELS: frozenset[Channel] = frozenset(
    {
        Channel.INBOUND_SMS,
        Channel.INBOUND_EMAIL,
        Channel.HUMAN_REVIEW,
        Channel.PROVIDER_API,
    }
)

#: Channels that can establish that a counterparty *agreed to terms*.
#: PROVIDER_API is excluded: a delivery receipt proves transmission, not assent.
AGREEMENT_CHANNELS: frozenset[Channel] = frozenset(
    {
        Channel.INBOUND_SMS,
        Channel.INBOUND_EMAIL,
        Channel.HUMAN_REVIEW,
    }
)


class ProposalStatus(str, Enum):
    UNCONFIRMED = "unconfirmed"
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class Terms:
    """A concrete, checkable set of order terms. All money in integer cents."""

    item: str
    quantity: int
    unit_price_cents: int
    fulfilment_date: str  # ISO-8601 date, e.g. "2026-08-14"

    def __post_init__(self) -> None:
        if not self.item.strip():
            raise ValueError("item must be non-empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.unit_price_cents <= 0:
            raise ValueError("unit_price_cents must be positive")

    @property
    def total_cents(self) -> int:
        return self.quantity * self.unit_price_cents

    def expected_tokens(self) -> list[str]:
        """The numeric facts an independent confirmation must restate.

        Requiring *every* token is deliberate. A bare "confirmed" must not
        satisfy an order for 120 loaves at $4.25 -- that is exactly how a wrong
        order gets scored as right.
        """
        return [str(self.quantity), f"{self.unit_price_cents / 100:.2f}"]

    def item_words(self) -> list[str]:
        """Words that identify the product, for a soft item check.

        Soft on purpose: a bakehouse replying to an order for "sourdough
        loaves" may well write only "loaves". Requiring the *first* word would
        reject a genuine confirmation, and a missed confirmation is as bad as a
        fabricated one. The numeric tokens carry the strictness.
        """
        return [w for w in re.findall(r"[a-z]+", self.item.lower()) if len(w) >= 4]

    def matches(self, body: str) -> tuple[bool, list[str]]:
        """Does this text independently restate these terms?

        Both conditions must hold:
          * every numeric token appears, bounded so "1200" does not satisfy "120"
          * at least one identifying product word appears
        """
        missing = [t for t in self.expected_tokens() if not _token_in(body, t)]
        words = self.item_words()
        if words and not any(_normalise(w) in _normalise(body) for w in words):
            missing.append(f"item({'|'.join(words)})")
        return (not missing), missing


@dataclass(frozen=True)
class Evidence:
    """A single observation about a proposal, tagged with its channel."""

    channel: Channel
    body: str
    observed_at: datetime = field(default_factory=_now)
    supports: bool = True  # False => this evidence contradicts the proposal
    external_ref: str = ""  # provider message id, email id, etc.

    @property
    def is_independent(self) -> bool:
        return self.channel in INDEPENDENT_CHANNELS

    @property
    def can_establish_agreement(self) -> bool:
        return self.channel in AGREEMENT_CHANNELS


def _normalise(text: str) -> str:
    """Strip whitespace, commas and currency marks so '$4.25' matches '4.25'."""
    return re.sub(r"[\s,$]", "", (text or "").lower())


def _token_in(body: str, token: str) -> bool:
    """Boundary-aware numeric match against the normalised text.

    Bare substring matching is wrong here: it lets a total of "1200" satisfy a
    quantity of "120", and "$14.25" satisfy a price of "4.25". Both would turn
    an unrelated number into a confirmation.
    """
    hay = _normalise(body)
    needle = _normalise(token)
    if not needle:
        return False
    if re.fullmatch(r"[\d.]+", needle):
        return re.search(rf"(?<![\d.]){re.escape(needle)}(?![\d.])", hay) is not None
    return needle in hay


def tokens_present(body: str, expected: list[str]) -> tuple[bool, list[str]]:
    """True only when *all* expected tokens appear in body."""
    missing = [t for t in expected if not _token_in(body, t)]
    return (not missing), missing


@dataclass
class Proposal:
    """Something the agent claims was agreed. Born UNCONFIRMED, always."""

    terms: Terms
    counterparty: str
    id: str = field(default_factory=lambda: _new_id("prop"))
    filed_at: datetime = field(default_factory=_now)
    filed_by: Channel = Channel.AGENT_ASSERTION
    _evidence: list[Evidence] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # A proposal is by definition the agent's word. Anything else is
        # evidence, not a proposal.
        if self.filed_by is not Channel.AGENT_ASSERTION:
            raise ValueError("proposals are filed by the agent; use add_evidence otherwise")

    # -- evidence ----------------------------------------------------------
    def add_evidence(self, evidence: Evidence) -> None:
        self._evidence.append(evidence)

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence)

    @property
    def independent_evidence(self) -> tuple[Evidence, ...]:
        return tuple(e for e in self._evidence if e.is_independent)

    # -- derived state (NO SETTER, deliberately) ---------------------------
    @property
    def status(self) -> ProposalStatus:
        """Derived from evidence only. There is no way to assign this."""
        contradicting = [
            e for e in self._evidence if e.is_independent and not e.supports
        ]
        if contradicting:
            return ProposalStatus.CONTRADICTED

        if self.confirming_evidence is not None:
            return ProposalStatus.CONFIRMED
        return ProposalStatus.UNCONFIRMED

    @property
    def confirming_evidence(self) -> Evidence | None:
        for e in self._evidence:
            if not e.can_establish_agreement or not e.supports:
                continue
            if self.terms.matches(e.body)[0]:
                return e
        return None

    def why(self) -> str:
        """Human-readable justification for the current status."""
        st = self.status
        if st is ProposalStatus.CONFIRMED:
            e = self.confirming_evidence
            assert e is not None
            return f"confirmed by {e.channel.value} ({e.external_ref or 'no ref'})"
        if st is ProposalStatus.CONTRADICTED:
            e = next(x for x in self._evidence if x.is_independent and not x.supports)
            return f"contradicted by {e.channel.value}"
        n_agent = sum(1 for e in self._evidence if e.channel is Channel.AGENT_ASSERTION)
        n_ind = len(self.independent_evidence)
        return (
            f"no independent agreement evidence yet "
            f"({n_agent} agent assertion(s), {n_ind} independent, none matching terms)"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "counterparty": self.counterparty,
            "item": self.terms.item,
            "quantity": self.terms.quantity,
            "unit_price_cents": self.terms.unit_price_cents,
            "total_cents": self.terms.total_cents,
            "fulfilment_date": self.terms.fulfilment_date,
            "filed_at": self.filed_at.isoformat(),
            "status": self.status.value,
            "why": self.why(),
            "expected_tokens": self.terms.expected_tokens(),
            "evidence": [
                {
                    "channel": e.channel.value,
                    "independent": e.is_independent,
                    "supports": e.supports,
                    "body": e.body[:280],
                    "ref": e.external_ref,
                    "at": e.observed_at.isoformat(),
                }
                for e in self._evidence
            ],
        }
