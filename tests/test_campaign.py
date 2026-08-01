"""The campaign engine must be generic, and its envelopes must be coherent.

Two things are pinned here. First, that the three demo campaigns are valid
configuration rather than three special cases - if the engine ever needs a
vertical-specific field, one of these stops validating. Second, that a
campaign cannot declare itself closed on the agent's own say-so: every
acceptable close channel has to come from `INDEPENDENT_CHANNELS`, which is the
same invariant `tests/test_receipts.py` guards, followed into the business layer.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.business.campaign import (  # noqa: E402
    ACCEPTABLE_EVIDENCE,
    CAMPAIGNS,
    FREELANCE_WEBDEV,
    RESTAURANT_CATERING,
    STARTUP_OUTBOUND,
    Campaign,
    CloseCondition,
    DiscoveryStrategy,
    Envelope,
    get_campaign,
)
from src.verify.receipts import INDEPENDENT_CHANNELS  # noqa: E402

ALL = (RESTAURANT_CATERING, FREELANCE_WEBDEV, STARTUP_OUTBOUND)


@pytest.mark.parametrize("campaign", ALL, ids=lambda c: c.vertical)
def test_preconfigured_campaigns_validate(campaign: Campaign):
    assert campaign.problems() == [], campaign.problems()
    assert campaign.is_valid


@pytest.mark.parametrize("campaign", ALL, ids=lambda c: c.vertical)
def test_envelope_bounds_are_coherent(campaign: Campaign):
    env = campaign.envelope
    assert env.min_price > 0
    assert env.max_qty > 0
    assert 0 <= env.max_discount_pct <= 100
    assert env.earliest_date <= env.latest_date
    assert env.currency


def test_incoherent_envelopes_are_rejected():
    bad = Envelope(
        min_price=0.0,
        max_qty=-1,
        earliest_date=date(2026, 9, 1),
        latest_date=date(2026, 8, 1),
        max_discount_pct=140.0,
    )
    problems = bad.problems()
    assert not bad.is_valid
    assert len(problems) == 4, problems
    joined = " ".join(problems)
    for term in ("min_price", "max_qty", "max_discount_pct", "inverted"):
        assert term in joined


def test_a_campaign_with_a_bad_envelope_does_not_validate():
    campaign = Campaign(
        name="broken", vertical="test", icp="x", offer="y",
        discovery=DiscoveryStrategy.SEEDED_LIST,
        close_condition=CloseCondition.BOOKED_MEETING,
        capacity_units="things/week",
        envelope=Envelope(
            min_price=-5.0, max_qty=10,
            earliest_date=date(2026, 8, 1), latest_date=date(2026, 8, 30),
            max_discount_pct=10.0,
        ),
    )
    assert not campaign.is_valid
    assert any("envelope" in p for p in campaign.problems())


def test_empty_required_text_is_rejected():
    campaign = Campaign(
        name="  ", vertical="test", icp="who", offer="",
        discovery=DiscoveryStrategy.NO_WEBSITE,
        close_condition=CloseCondition.DELIVERED_ARTIFACT,
        capacity_units="builds/month",
        envelope=FREELANCE_WEBDEV.envelope,
    )
    problems = campaign.problems()
    assert any("name" in p for p in problems)
    assert any("offer" in p for p in problems)


# -- the envelope is what stops an agent improvising under pressure ---------


def test_terms_inside_the_envelope_need_no_operator():
    env = RESTAURANT_CATERING.envelope
    ok, reason = env.permits(price=4.00, qty=400, when=date(2026, 9, 1),
                             discount_pct=10.0)
    assert ok is True
    assert reason == "within envelope"


@pytest.mark.parametrize(
    "terms, expected",
    [
        ({"price": 2.00}, "below the floor"),
        ({"qty": 5000}, "exceeds committed capacity"),
        ({"when": date(2027, 1, 5)}, "outside the agreed window"),
        ({"discount_pct": 60.0}, "exceeds the"),
    ],
)
def test_terms_outside_the_envelope_escalate_with_a_specific_reason(terms, expected):
    ok, reason = RESTAURANT_CATERING.envelope.permits(**terms)
    assert ok is False
    assert expected in reason
    assert "operator approval" in reason
    # The escalation helper must agree with the raw check.
    assert RESTAURANT_CATERING.escalation_for(**terms) == reason


def test_unspecified_terms_are_not_silently_approved():
    """Passing nothing means nothing was checked, not that anything goes."""
    ok, _ = RESTAURANT_CATERING.envelope.permits()
    assert ok is True
    # ... but the moment a term is named it is bounded.
    assert RESTAURANT_CATERING.escalation_for(price=0.01) is not None


def test_startup_outbound_has_no_pricing_authority_at_all():
    """max_discount_pct=0 must mean every discount ask escalates."""
    assert STARTUP_OUTBOUND.escalation_for(discount_pct=0.5) is not None
    assert STARTUP_OUTBOUND.escalation_for(discount_pct=0.0) is None


def test_envelope_is_frozen_so_an_agent_cannot_widen_its_own_mandate():
    with pytest.raises(Exception):
        FREELANCE_WEBDEV.envelope.max_discount_pct = 90.0  # type: ignore[misc]


# -- the anti-fabrication invariant, carried into the business layer --------


@pytest.mark.parametrize("condition", list(CloseCondition), ids=lambda c: c.value)
def test_every_close_condition_is_closable_only_by_independent_evidence(condition):
    channels = ACCEPTABLE_EVIDENCE[condition]
    assert channels, f"{condition} has no way to close"
    assert channels <= INDEPENDENT_CHANNELS, channels - INDEPENDENT_CHANNELS


def test_a_campaign_closable_by_agent_assertion_fails_validation():
    """Guard the guard: if someone widens ACCEPTABLE_EVIDENCE, this catches it."""
    from src.verify.receipts import Channel

    original = ACCEPTABLE_EVIDENCE[CloseCondition.BOOKED_MEETING]
    ACCEPTABLE_EVIDENCE[CloseCondition.BOOKED_MEETING] = frozenset(
        {Channel.AGENT_ASSERTION}
    )
    try:
        assert not STARTUP_OUTBOUND.is_valid
        assert any("non-independent" in p for p in STARTUP_OUTBOUND.problems())
    finally:
        ACCEPTABLE_EVIDENCE[CloseCondition.BOOKED_MEETING] = original
    assert STARTUP_OUTBOUND.is_valid


# -- the actual architectural claim: the differences are only data ----------


def test_the_three_campaigns_differ_only_in_configuration():
    assert {type(c) for c in ALL} == {Campaign}
    assert len({c.discovery for c in ALL}) == 3
    assert len({c.close_condition for c in ALL}) == 3
    assert len({c.vertical for c in ALL}) == 3
    assert len({c.capacity_units for c in ALL}) == 3


def test_campaigns_serialise_for_the_demo_screen():
    payload = FREELANCE_WEBDEV.to_dict()
    assert payload["discovery"] == DiscoveryStrategy.NO_WEBSITE.value
    assert payload["close_condition"] == CloseCondition.DELIVERED_ARTIFACT.value
    assert payload["envelope"]["earliest_date"] == "2026-08-03"
    assert payload["capacity_units"] == "site builds/month"
    assert payload["close_evidence"]


def test_registry_lookup_fails_loudly_on_a_typo():
    assert get_campaign("freelance_webdev") is FREELANCE_WEBDEV
    assert set(CAMPAIGNS.values()) == set(ALL)
    with pytest.raises(KeyError, match="unknown campaign"):
        get_campaign("freelance_web_dev")
