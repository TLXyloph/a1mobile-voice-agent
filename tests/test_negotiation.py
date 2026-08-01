"""Negotiation must never concede past the floor and must always be able to exit.

The two failures that matter: quoting below cost (loses money on every order),
and never stopping (the "pushy" failure, which is what loses ratings).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.negotiation import (  # noqa: E402
    CompetitorBook,
    CompetitorPrice,
    Limits,
    NegotiationState,
    Tactic,
    concession_step,
    guidance,
    next_tactic,
)


def _state(**kw) -> NegotiationState:
    base = dict(opening_total=500.0, current_total=500.0, qty=200)
    base.update(kw)
    return NegotiationState(**base)


# -- stop conditions ------------------------------------------------------


def test_two_declines_ends_the_call():
    s = _state(declines_received=2)
    assert next_tactic(s, Limits()) is Tactic.WALK_AWAY


def test_walk_away_wins_over_every_other_move():
    """Exhausted-and-declined must exit, not loop on escalation."""
    s = _state(
        declines_received=5,
        asked_current_spend=True,
        value_framed=True,
        concessions_made=9,
        restructure_offered=True,
    )
    assert next_tactic(s, Limits()) is Tactic.WALK_AWAY


def test_opens_by_asking_not_discounting():
    """The first move must never spend margin."""
    assert next_tactic(_state(), Limits()) is Tactic.ASK_CURRENT_SPEND


def test_value_framing_precedes_any_discount():
    s = _state(asked_current_spend=True)
    assert next_tactic(s, Limits()) is Tactic.EMPHASISE_VALUE


def test_only_one_concession_then_restructure():
    s = _state(asked_current_spend=True, value_framed=True, concessions_made=1)
    assert next_tactic(s, Limits(max_concessions=1)) is Tactic.RESTRUCTURE


def test_envelope_exhausted_escalates_rather_than_conceding():
    s = _state(
        asked_current_spend=True,
        value_framed=True,
        concessions_made=1,
        restructure_offered=True,
    )
    assert next_tactic(s, Limits(max_concessions=1)) is Tactic.ESCALATE_TO_OPERATOR


def test_budget_below_floor_escalates_never_self_authorises():
    s = _state(asked_current_spend=True, value_framed=True, their_stated_budget=200.0)
    assert next_tactic(s, Limits(floor_total=380.0)) is Tactic.ESCALATE_TO_OPERATOR


def test_budget_above_current_price_closes():
    s = _state(asked_current_spend=True, value_framed=True, their_stated_budget=600.0)
    assert next_tactic(s, Limits(floor_total=380.0)) is Tactic.CLOSE


# -- the money floor ------------------------------------------------------


def test_concession_stops_at_whichever_limit_binds_first():
    """Two independent clamps: the cost floor and the discount cap.

    Whichever is *higher* must win, because both are hard limits and the safe
    answer is the more restrictive one. Here a 10% cap on a 500 opening (=450)
    binds before the 390 cost floor does.
    """
    s = _state(current_total=400.0)
    assert concession_step(s, Limits(floor_total=390.0, max_discount_pct=10.0),
                           step_pct=50.0) == 450.0


def test_cost_floor_binds_when_it_is_the_tighter_limit():
    s = _state(current_total=400.0)
    # A generous 50% discount cap (=250) leaves the 390 cost floor binding.
    assert concession_step(s, Limits(floor_total=390.0, max_discount_pct=50.0),
                           step_pct=50.0) == 390.0


def test_repeated_concessions_converge_to_floor_not_through_it():
    s = _state(current_total=500.0)
    limits = Limits(floor_total=400.0, max_discount_pct=100.0)
    for _ in range(40):
        s.current_total = concession_step(s, limits, step_pct=20.0)
    assert s.current_total >= 400.0


def test_discount_cap_binds_even_when_floor_is_low():
    """max_discount_pct must hold independently of the cost floor."""
    s = _state(current_total=500.0)
    limits = Limits(floor_total=1.0, max_discount_pct=10.0)
    for _ in range(20):
        s.current_total = concession_step(s, limits, step_pct=25.0)
    assert s.current_total >= 450.0, "10% cap on a 500 opening means 450 minimum"


# -- competitor book ------------------------------------------------------


def _book() -> CompetitorBook:
    return CompetitorBook(
        [CompetitorPrice(vendor="Round Table Pizza", item="party pack",
                         unit_price=2.50, min_qty=20)]
    )


def test_loose_vendor_name_matching():
    assert _book().lookup("round table")
    assert _book().lookup("Round Table Pizza")
    assert _book().lookup("RoundTable")


def test_unknown_vendor_returns_none_rather_than_a_guess():
    """Inventing a rival's price and being wrong is unrecoverable mid-call."""
    assert _book().undercut_target("Totally Unknown Caterer", 100) is None


def test_undercut_beats_competitor_by_requested_margin():
    target = _book().undercut_target("Round Table", 100, by_pct=10.0)
    assert target == 225.0  # 100 * 2.50 = 250, less 10%


def test_min_qty_is_respected_in_estimate():
    p = CompetitorPrice(vendor="X", item="y", unit_price=2.0, min_qty=50)
    assert p.estimate_total(10) == 100.0  # charged for 50, not 10


# -- guidance packet ------------------------------------------------------


def test_concession_guidance_carries_a_concrete_number():
    s = _state(asked_current_spend=True, value_framed=True)
    g = guidance(s, Limits(floor_total=400.0))
    assert g["tactic"] == Tactic.OFFER_CONCESSION.value
    assert g["quote_this_total"] >= 400.0


def test_escalation_guidance_explains_itself():
    s = _state(asked_current_spend=True, value_framed=True, their_stated_budget=100.0)
    g = guidance(s, Limits(floor_total=400.0))
    assert g["tactic"] == Tactic.ESCALATE_TO_OPERATOR.value
    assert "floor" in g["why"]


def test_every_tactic_has_a_script():
    from src.agents.negotiation import TACTIC_SCRIPTS

    for tactic in Tactic:
        assert tactic in TACTIC_SCRIPTS, f"{tactic} has no spoken guidance"
        assert len(TACTIC_SCRIPTS[tactic]) > 40
