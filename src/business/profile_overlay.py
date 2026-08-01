"""Let the owner's answers override the campaign's hardcoded envelope.

`src/business/campaign.py` ships three campaigns with envelopes written by us:
RESTAURANT_CATERING allows 600 muffins and a 15% discount between two dates in
2026. Those numbers were a reasonable guess when nobody had asked the bakery.

Intake asks the bakery. Once it has, the guess is not just redundant, it is
actively wrong in the one direction that costs money: the agent would still be
authorised to sell 600 against a stated capacity of 400, and to give away 15%
when the owner said 10. Two sources of truth for the same limit, and the looser
one wins by being the one the code reads.

This closes that. `apply_business_profile()` rebuilds the envelope from the
profile the MCP intake server wrote, and everything it cannot find it leaves
exactly as the campaign had it. A half-finished profile therefore degrades to
the old behaviour rather than to an empty envelope - missing is "no opinion",
never "no limit".

`min_price` is deliberately not read from the environment at all. It is derived
from the CostModel's own floor, so the price the agent may not go under is
always the one the cost model computes, and the two cannot drift apart.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from datetime import date

from src.business.campaign import Campaign, Envelope
from src.business.pricing import CostModel

logger = logging.getLogger("business.profile_overlay")


def _float(name: str) -> float | None:
    """A malformed env value is ignored with a warning, never guessed at."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; keeping the campaign's value", name, raw)
        return None


def _int(name: str) -> int | None:
    value = _float(name)
    return None if value is None else int(value)


def _date(name: str) -> date | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        logger.warning("%s=%r is not an ISO date; keeping the campaign's value", name, raw)
        return None


def envelope_from_profile(base: Envelope, costs: CostModel) -> Envelope:
    """The campaign's envelope, with every limit the owner stated applied.

    Returns `base` unchanged when the profile says nothing, and refuses to
    return an incoherent envelope: if the overlay would produce something
    `Envelope.problems()` rejects - an inverted date window, a discount over
    100 - the campaign's original is kept and the reason is logged, because a
    broken envelope mid-call is worse than a stale one.
    """
    max_qty = _int("CAPACITY_TOTAL")
    discount = _float("MAX_DISCOUNT_PCT")
    earliest = _date("EARLIEST_DATE")
    latest = _date("LATEST_DATE")

    candidate = replace(
        base,
        # Always from the cost model, so the floor and the envelope agree.
        min_price=float(costs.floor_price(1)),
        max_qty=base.max_qty if max_qty is None else max_qty,
        max_discount_pct=base.max_discount_pct if discount is None else discount,
        earliest_date=base.earliest_date if earliest is None else earliest,
        latest_date=base.latest_date if latest is None else latest,
    )

    if problems := candidate.problems():
        logger.warning(
            "business profile would make the envelope unusable (%s); keeping the "
            "campaign's envelope", "; ".join(problems),
        )
        return base
    return candidate


def apply_business_profile(campaign: Campaign, costs: CostModel) -> Campaign:
    """The campaign the owner actually authorised, not the one we shipped."""
    updated = envelope_from_profile(campaign.envelope, costs)
    if updated == campaign.envelope:
        return campaign
    logger.info(
        "business profile applied: max_qty %s->%s, discount %.1f%%->%.1f%%, "
        "window %s..%s -> %s..%s, floor %.2f",
        campaign.envelope.max_qty, updated.max_qty,
        campaign.envelope.max_discount_pct, updated.max_discount_pct,
        campaign.envelope.earliest_date, campaign.envelope.latest_date,
        updated.earliest_date, updated.latest_date, updated.min_price,
    )
    return replace(campaign, envelope=updated)
