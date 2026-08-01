"""The state machine must make the $74 sequence unreachable.

Per-tool validation passed on that call - the floor was correct for 30 units.
What was missing was any owner of the question "has this call reached a state
where quoting is legitimate?" These tests pin that owner.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.flow import Gate, Phase, TRANSITIONS  # noqa: E402


def test_cannot_reserve_before_units_are_confirmed():
    g = Gate()
    assert "BLOCKED" in (g.allow_capacity(30) or "")
    assert "ITEMS or the number of PEOPLE" in g.allow_capacity(30)


def test_confirming_units_unblocks_reservation():
    g = Gate(); g.note_units(60, headcount=30)
    assert g.allow_capacity(60) is None


def test_cannot_quote_before_discovery():
    g = Gate()
    out = g.allow_quote(74.0)
    assert "too early to quote" in out


def test_quote_blocked_until_every_precondition_met():
    g = Gate(); g.note_units(60, 30)
    assert "BLOCKED" in g.allow_quote(150.0)      # no capacity, no spend asked
    g.note_capacity_held()
    assert "BLOCKED" in g.allow_quote(150.0)      # still never asked their spend
    g.note_position(385.0)
    assert g.allow_quote(400.0) is None           # all preconditions satisfied


def test_the_74_dollar_sequence_is_unreachable():
    """The exact live failure, replayed."""
    g = Gate()
    g.note_units(30, headcount=30)   # the mistake: people counted as items
    g.note_capacity_held()
    g.note_position(385.0)           # they said they pay 385
    blocked = g.allow_quote(74.0)
    assert blocked and "discards 311.00" in blocked


def test_quoting_at_or_above_their_number_is_allowed():
    g = Gate(); g.note_units(60, 30); g.note_capacity_held(); g.note_position(385.0)
    assert g.allow_quote(385.0) is None
    assert g.allow_quote(500.0) is None


def test_cannot_close_without_a_validated_price():
    g = Gate(); g.note_units(60); g.note_capacity_held(); g.note_position(400.0)
    assert "BLOCKED" in g.allow_close()
    g.note_quoted()
    assert g.allow_close() is None


def test_cannot_close_while_waiting_on_the_operator():
    """A commitment made before approval arrives is the whole failure mode."""
    g = Gate(); g.note_units(60); g.note_capacity_held()
    g.note_position(400.0); g.note_quoted()
    g.note_escalating()
    assert "waiting on the owner" in g.allow_close()
    g.note_operator(400.0)
    assert g.allow_close() is None


def test_closed_is_terminal():
    g = Gate(); g.note_units(60); g.note_capacity_held()
    g.note_position(400.0); g.note_quoted(); g.note_closed()
    assert g.phase is Phase.CLOSED
    assert "BLOCKED" in g.allow_capacity(10)
    assert "over" in g.allow_quote(500.0)


def test_illegal_transitions_are_refused_not_raised():
    g = Gate()
    assert g.move(Phase.CLOSING) is False
    assert g.phase is Phase.OPENING
    assert any("BLOCKED" in h for h in g.history)


def test_every_phase_except_closed_can_reach_closed():
    """A call must always be able to end. No state may trap the agent."""
    for phase, edges in TRANSITIONS.items():
        if phase is Phase.CLOSED:
            continue
        assert Phase.CLOSED in edges, f"{phase} cannot terminate"


def test_units_equal_to_headcount_is_flagged_not_blocked():
    g = Gate(); g.note_units(30, headcount=30)
    assert g.facts.units_confirmed  # allowed, but the tool warns the model


# -- items_per_person: the field behind the $311 loss ----------------------

def test_headcount_converts_using_the_profile_ratio():
    from src.agents.flow import Facts
    f = Facts(items_per_person=2.0)
    assert f.expected_units(30) == 60


def test_thirty_items_for_thirty_people_is_flagged():
    """The exact live mistake, when the profile says two each."""
    from src.agents.flow import Facts
    f = Facts(items_per_person=2.0)
    w = f.conversion_looks_wrong(30, 30)
    assert w and "looks low" in w and "60" in w


def test_a_correct_conversion_is_not_flagged():
    from src.agents.flow import Facts
    assert Facts(items_per_person=2.0).conversion_looks_wrong(60, 30) is None


def test_small_shortfalls_are_not_nagged():
    """Warn on the case that matters, not on 58 vs 60."""
    from src.agents.flow import Facts
    assert Facts(items_per_person=2.0).conversion_looks_wrong(58, 30) is None


def test_no_headcount_means_no_warning():
    from src.agents.flow import Facts
    assert Facts(items_per_person=2.0).conversion_looks_wrong(30, None) is None


def test_default_ratio_of_one_never_warns_on_equality():
    """Without a profile the ratio is 1, so units == headcount is legitimate."""
    from src.agents.flow import Facts
    assert Facts().conversion_looks_wrong(30, 30) is None
