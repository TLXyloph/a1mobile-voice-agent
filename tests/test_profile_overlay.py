"""The owner's stated limits must beat the ones we guessed for them.

RESTAURANT_CATERING ships an envelope allowing 600 units and 15% off. A bakery
that told intake "400 a week, 10% maximum" has said something narrower, and the
only interesting question about this module is whether the narrower number is
the one the agent ends up holding.

The other half is what happens when the profile is partial or wrong. Missing has
to mean "no opinion" and leave the campaign's value standing - if a blank env var
read as zero, `max_qty=0` would silently make every order too big to accept and
the agent would escalate on all of them.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.business.campaign import RESTAURANT_CATERING  # noqa: E402
from src.business.pricing import CostModel  # noqa: E402
from src.business.profile_overlay import (  # noqa: E402
    apply_business_profile,
    envelope_from_profile,
)

BASE = RESTAURANT_CATERING
COSTS = CostModel(
    materials_per_unit="0.80",
    labor_per_unit="0.40",
    transport_per_unit="0.15",
    min_margin_pct="30",
    target_margin_pct="45",
    unit="muffin",
)

PROFILE_KEYS = ("CAPACITY_TOTAL", "MAX_DISCOUNT_PCT", "EARLIEST_DATE", "LATEST_DATE")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No test inherits a profile from the developer's shell or config/.env."""
    for key in PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)


def set_profile(monkeypatch, **values: str) -> None:
    for key, value in values.items():
        monkeypatch.setenv(key, value)


# -- the point of the module ------------------------------------------------


def test_the_owners_capacity_beats_the_campaigns_guess(monkeypatch):
    set_profile(monkeypatch, CAPACITY_TOTAL="400")

    envelope = envelope_from_profile(BASE.envelope, COSTS)

    assert BASE.envelope.max_qty == 600, "fixture drift: the guess should be looser"
    assert envelope.max_qty == 400
    permitted, reason = envelope.permits(qty=500)
    assert permitted is False and "400" in reason


def test_the_owners_discount_ceiling_beats_the_campaigns_guess(monkeypatch):
    set_profile(monkeypatch, MAX_DISCOUNT_PCT="10")

    envelope = envelope_from_profile(BASE.envelope, COSTS)

    assert BASE.envelope.max_discount_pct == 15.0
    assert envelope.max_discount_pct == 10.0
    assert envelope.permits(discount_pct=12)[0] is False


def test_the_owners_delivery_window_replaces_the_shipped_dates(monkeypatch):
    set_profile(monkeypatch, EARLIEST_DATE="2026-09-01", LATEST_DATE="2026-09-30")

    envelope = envelope_from_profile(BASE.envelope, COSTS)

    assert envelope.earliest_date == date(2026, 9, 1)
    assert envelope.latest_date == date(2026, 9, 30)
    assert envelope.permits(when=date(2026, 8, 15))[0] is False


def test_min_price_always_tracks_the_cost_model_floor(monkeypatch):
    """Never read from env, so the floor and the envelope cannot drift apart."""
    set_profile(monkeypatch, CAPACITY_TOTAL="400")

    envelope = envelope_from_profile(BASE.envelope, COSTS)

    assert envelope.min_price == float(COSTS.floor_price(1))
    assert envelope.permits(price=envelope.min_price - 0.01)[0] is False


def test_a_dearer_cost_model_raises_the_floor_the_agent_may_offer(monkeypatch):
    dearer = CostModel(
        materials_per_unit="2.00", labor_per_unit="1.00", transport_per_unit="0.25",
        min_margin_pct="30", target_margin_pct="45", unit="muffin",
    )

    envelope = envelope_from_profile(BASE.envelope, dearer)

    assert envelope.min_price > BASE.envelope.min_price


# -- partial and broken profiles --------------------------------------------


def test_an_empty_profile_leaves_every_stated_limit_alone():
    envelope = envelope_from_profile(BASE.envelope, COSTS)

    assert envelope.max_qty == BASE.envelope.max_qty
    assert envelope.max_discount_pct == BASE.envelope.max_discount_pct
    assert envelope.earliest_date == BASE.envelope.earliest_date
    assert envelope.latest_date == BASE.envelope.latest_date


def test_a_blank_value_is_no_opinion_not_zero(monkeypatch):
    """The dangerous reading: max_qty=0 would escalate on every order."""
    set_profile(monkeypatch, CAPACITY_TOTAL="", MAX_DISCOUNT_PCT="   ")

    envelope = envelope_from_profile(BASE.envelope, COSTS)

    assert envelope.max_qty == BASE.envelope.max_qty
    assert envelope.max_discount_pct == BASE.envelope.max_discount_pct


@pytest.mark.parametrize(
    ("key", "junk"),
    [
        ("CAPACITY_TOTAL", "four hundred"),
        ("MAX_DISCOUNT_PCT", "ten percent"),
        ("EARLIEST_DATE", "next tuesday"),
        ("LATEST_DATE", "2026-13-45"),
    ],
)
def test_an_unparseable_value_keeps_the_campaigns_value(monkeypatch, key, junk):
    set_profile(monkeypatch, **{key: junk})

    envelope = envelope_from_profile(BASE.envelope, COSTS)

    assert envelope.problems() == []
    assert getattr(envelope, key.lower().replace("capacity_total", "max_qty")) is not None


def test_an_incoherent_overlay_is_refused_whole(monkeypatch):
    """An inverted window would be worse mid-call than a stale one."""
    set_profile(monkeypatch, EARLIEST_DATE="2026-12-01", LATEST_DATE="2026-08-01")

    envelope = envelope_from_profile(BASE.envelope, COSTS)

    assert envelope == BASE.envelope
    assert envelope.problems() == []


def test_a_zero_capacity_is_refused_rather_than_silently_blocking_everything(monkeypatch):
    set_profile(monkeypatch, CAPACITY_TOTAL="0")

    envelope = envelope_from_profile(BASE.envelope, COSTS)

    assert envelope == BASE.envelope


# -- the campaign wrapper ---------------------------------------------------


def test_apply_business_profile_returns_a_campaign_that_is_still_valid(monkeypatch):
    set_profile(monkeypatch, CAPACITY_TOTAL="400", MAX_DISCOUNT_PCT="10")

    campaign = apply_business_profile(BASE, COSTS)

    assert campaign.is_valid
    assert campaign.envelope.max_qty == 400
    # Everything that is not the envelope is untouched.
    assert campaign.name == BASE.name
    assert campaign.close_condition is BASE.close_condition
    assert BASE.envelope.max_qty == 600, "the shipped campaign must not be mutated"


def test_apply_business_profile_is_identity_when_there_is_no_profile():
    assert apply_business_profile(BASE, COSTS).envelope.max_qty == BASE.envelope.max_qty


def test_the_overlay_cannot_introduce_a_non_independent_close_channel(monkeypatch):
    """The anti-fabrication invariant survives the overlay."""
    set_profile(monkeypatch, CAPACITY_TOTAL="400")

    campaign = apply_business_profile(BASE, COSTS)

    assert campaign.problems() == []
    assert campaign.close_evidence == BASE.close_evidence


# -- both transports must enforce the same limits -------------------------

def test_vapi_and_livekit_build_the_same_envelope(monkeypatch):
    """The owner caps discounts once; both paths must honour it.

    Otherwise they get 10% on one transport and 15% on the other, and which
    one applies depends on plumbing they never see.
    """
    monkeypatch.setenv("MAX_DISCOUNT_PCT", "10")
    monkeypatch.setenv("CAPACITY_TOTAL", "400")

    from src.agents import vapi_bridge
    from src.agents.run_call import build_call_session

    vapi_agent = vapi_bridge._new_agent("test-parity")
    live_session = build_call_session()

    v, l = vapi_agent.s.campaign.envelope, live_session.campaign.envelope
    assert v.max_discount_pct == l.max_discount_pct == 10.0
    assert v.max_qty == l.max_qty
    assert v.earliest_date == l.earliest_date
    assert v.latest_date == l.latest_date
