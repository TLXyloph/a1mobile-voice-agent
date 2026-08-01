"""Which direction a text points, and therefore what it can prove.

`src/verify/receipts.py` already settles the principle: `Channel.INBOUND_SMS`
is in `INDEPENDENT_CHANNELS`, `Channel.AGENT_ASSERTION` is not, and `Verdict`
is derived with no setter. This module is the SMS-shaped application of it, and
it exists mostly so that the asymmetry is written down in one obvious place:

    outbound text  -> Channel.AGENT_ASSERTION -> can never verify anything
    inbound  text  -> Channel.INBOUND_SMS     -> can promote a Claim

The subtle failure this closes is self-verification. Our outbound number and
the prospect's number are both just strings; a bug (or an agent handed a "send
to yourself" tool) that files our own text as inbound would hand the agent a
way to verify its own claim by texting itself. `record_outbound` is therefore
the *only* function here that touches an outbound message, and it is hardcoded
to AGENT_ASSERTION with no channel parameter to override.

Matching is all-tokens, copied in spirit from `webhooks.find_confirmation`: a
message saying only "confirmed" must not satisfy a claim about 120 muffins at
$420. Numeric tokens compare by value, not substring, so "120" is not matched
by "1200" and "$420" satisfies "420.00".
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from src.verify.receipts import Channel, Claim, Evidence

logger = logging.getLogger("messaging.evidence")

_NUMBER = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?")


def _as_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", "").lstrip("$").strip())
    except (InvalidOperation, AttributeError):
        return None


def numbers_in(text: str) -> list[Decimal]:
    """Every numeric value in a message body, commas stripped."""
    out: list[Decimal] = []
    for raw in _NUMBER.findall(text or ""):
        value = _as_decimal(raw)
        if value is not None:
            out.append(value)
    return out


def matches(body: str, tokens: Iterable[str]) -> bool:
    """True only if EVERY token is present.

    Requiring all of them is the whole point. A booking claim for "4 people at
    7pm" satisfied by a bare "confirmed" is how a wrong booking gets scored as
    a right one - and on this project that is the disqualifying kind of wrong.
    """
    tokens = [t for t in (tokens or []) if str(t).strip()]
    if not tokens:
        # No tokens means nothing was asserted, so nothing can be confirmed.
        # Returning True here would let any inbound text verify any claim.
        return False

    low = (body or "").lower()
    values = numbers_in(body or "")
    for token in tokens:
        token = str(token).strip()
        wanted = _as_decimal(token)
        if wanted is not None and _NUMBER.fullmatch(token.replace("$", "").strip()):
            if not any(v == wanted for v in values):
                return False
        elif not re.search(re.escape(token.lower()), low):
            return False
    return True


def tokens_for(qty: int | None, total: float | Decimal | None, *, when: str = "") -> tuple[str, ...]:
    """The default proof tokens for an order claim.

    Quantity and total, because those are the two numbers that make an order a
    different order if they are wrong. `when` is included as free text when
    given; it is the term most often paraphrased, so requiring it is a judgement
    call left to the caller.
    """
    out: list[str] = []
    if qty:
        out.append(str(int(qty)))
    if total is not None:
        dec = total if isinstance(total, Decimal) else Decimal(str(total))
        out.append(str(dec.quantize(Decimal("0.01"))))
    if when:
        out.append(when)
    return tuple(out)


# -- the outbound side: powerless by construction ------------------------


def record_outbound(claim: Claim, text: str, *, to: str = "") -> Evidence:
    """File our own text against a claim as AGENT_ASSERTION.

    Recorded because honesty rate is measurable and worth measuring; powerless
    because `AGENT_ASSERTION` is excluded from `INDEPENDENT_CHANNELS`. There is
    no channel argument. There is no supports=False path either - a text we
    wrote is not evidence against ourselves any more than it is evidence for
    us.
    """
    ev = Evidence(
        channel=Channel.AGENT_ASSERTION,
        summary=f"outbound SMS to {to or 'prospect'}: {text[:280]}",
        raw={"direction": "outbound", "to": to, "body": text},
    )
    claim.attach_evidence(ev)
    return ev


# -- the inbound side: the only thing that can promote ------------------


def record_inbound(claim: Claim, body: str, *, sender: str = "", raw: Any = None) -> Evidence:
    """File a prospect's text as INBOUND_SMS evidence.

    Call this only when `matches()` has already said the body confirms the
    claim. Attaching a non-matching message would verify the wrong thing.
    """
    ev = Evidence(
        channel=Channel.INBOUND_SMS,
        summary=f"inbound SMS from {sender or 'prospect'}: {body[:280]}",
        raw=raw if raw is not None else {"from": sender, "body": body},
    )
    claim.attach_evidence(ev)
    logger.info("claim %s promoted via inbound SMS from %s", claim.id, sender)
    return ev


def try_verify(claim: Claim, tokens: Iterable[str], body: str, *, sender: str = "", raw: Any = None) -> bool:
    """Attach inbound evidence iff the body contains every token.

    Never attaches contradicting evidence on a miss. A text that does not
    mention the numbers is absence of proof, not proof of absence, and
    recording it as a contradiction would turn a vague reply into a failure.
    """
    if not matches(body, tokens):
        logger.info("inbound from %s does not match %s", sender, list(tokens))
        return False
    record_inbound(claim, body, sender=sender, raw=raw)
    return True


# -- process-local claim registry ---------------------------------------


@dataclass
class RegisteredClaim:
    """A live `Claim` object plus the phone and tokens that would prove it."""

    claim: Claim
    phone: str
    tokens: tuple[str, ...]


class ClaimRegistry:
    """Where the webhook finds the `Claim` object an inbound text should hit.

    Process-local and intentionally not persisted. A `Claim` belongs to a
    `Receipt`; storing a copy here would create a second place a verdict could
    be read from, and the one thing this codebase will not have is two sources
    of truth about whether something happened.

    The consequence is honest and worth stating: after a restart, an inbound
    text that would have verified a claim is still captured and still stored on
    the thread, but there is no live claim to promote. The route reports that as
    `evidence_pending` rather than pretending either way.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, RegisteredClaim] = {}

    def register(self, phone: str, claim: Claim, tokens: Iterable[str]) -> RegisteredClaim:
        entry = RegisteredClaim(claim=claim, phone=phone, tokens=tuple(tokens))
        with self._lock:
            self._by_id[claim.id] = entry
        return entry

    def get(self, claim_id: str) -> RegisteredClaim | None:
        with self._lock:
            return self._by_id.get(claim_id)

    def for_phone(self, phone: str) -> list[RegisteredClaim]:
        with self._lock:
            return [e for e in self._by_id.values() if e.phone == phone]

    def clear(self) -> None:
        with self._lock:
            self._by_id.clear()


#: The default registry the router uses. Swap in tests via `routes.set_registry`.
REGISTRY = ClaimRegistry()
