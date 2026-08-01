"""Pins the arithmetic negotiation limits.

These tests exist because prompt-level limits do not hold. Every guarantee here
is a consequence of integer comparison, so it holds regardless of what the
model was persuaded to believe.
"""

from __future__ import annotations

from datetime import date

import pytest

from helyx.ladder import ConcessionLadder, MandateGuard, Move
from helyx.mandate import Mandate


def make_mandate(**over: object) -> Mandate:
    base: dict[str, object] = dict(
        item="sourdough loaves",
        quantity=120,
        target_unit_price_cents=425,
        ceiling_unit_price_cents=500,
        needed_by=date(2026, 8, 14),
        counterparty_name="Kestrel Bakehouse",
        max_rounds=4,
    )
    base.update(over)
    return Mandate(**base)  # type: ignore[arg-type]


# --- ladder ----------------------------------------------------------------


def test_ladder_starts_at_opening_and_ends_at_ceiling() -> None:
    m = make_mandate()
    ladder = ConcessionLadder(m)
    sched = ladder.schedule()
    assert sched[0] == m.opening_unit_price_cents
    assert sched[-1] == m.ceiling_unit_price_cents


def test_ladder_never_exceeds_ceiling_at_any_round() -> None:
    m = make_mandate()
    ladder = ConcessionLadder(m)
    for r in range(0, 50):  # far beyond max_rounds
        assert ladder.offer_for_round(r) <= m.ceiling_unit_price_cents


def test_ladder_is_monotonic_and_deterministic() -> None:
    ladder = ConcessionLadder(make_mandate())
    a = ladder.schedule()
    b = ladder.schedule()
    assert a == b
    assert all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def test_concessions_shrink() -> None:
    """Each successive concession is no larger than the previous one."""
    sched = ConcessionLadder(make_mandate(max_rounds=5)).schedule()
    steps = [sched[i + 1] - sched[i] for i in range(len(sched) - 1)]
    assert all(steps[i] >= steps[i + 1] for i in range(len(steps) - 1)), steps


def test_single_round_mandate_does_not_divide_by_zero() -> None:
    m = make_mandate(max_rounds=1)
    assert ConcessionLadder(m).schedule() == [m.opening_unit_price_cents]


# --- decisions -------------------------------------------------------------


def test_counter_at_or_below_target_is_accepted() -> None:
    g = MandateGuard(make_mandate())
    assert g.evaluate(400, 0).move is Move.ACCEPT
    assert g.evaluate(425, 0).move is Move.ACCEPT


def test_counter_above_ceiling_mid_negotiation_counters() -> None:
    g = MandateGuard(make_mandate())
    d = g.evaluate(650, 1)
    assert d.move is Move.COUNTER
    assert d.unit_price_cents <= g.mandate.ceiling_unit_price_cents


def test_counter_above_ceiling_on_final_round_walks_away() -> None:
    g = MandateGuard(make_mandate(max_rounds=4))
    d = g.evaluate(650, 3)
    assert d.move is Move.WALK_AWAY
    assert d.unit_price_cents == 0


def test_within_ceiling_on_final_round_is_accepted() -> None:
    g = MandateGuard(make_mandate())
    d = g.evaluate(480, 3)
    assert d.move is Move.ACCEPT
    assert d.unit_price_cents == 480


def test_no_decision_ever_authorises_above_ceiling() -> None:
    """Exhaustive sweep: no counter price ever exceeds the mandate ceiling."""
    g = MandateGuard(make_mandate())
    ceiling = g.mandate.ceiling_unit_price_cents
    for counter in range(1, 2000, 7):
        for rnd in range(0, 6):
            d = g.evaluate(counter, rnd)
            assert d.unit_price_cents <= ceiling, (counter, rnd, d)
            if d.move is Move.ACCEPT:
                assert d.unit_price_cents <= ceiling


def test_may_accept_respects_ceiling() -> None:
    g = MandateGuard(make_mandate())
    assert g.may_accept(500) is True
    assert g.may_accept(501) is False


# --- utterance backstop ----------------------------------------------------


def test_scan_flags_unit_price_above_ceiling() -> None:
    g = MandateGuard(make_mandate())
    v = g.scan_utterance("Okay, you drive a hard bargain - I'll do $6.50 a loaf.")
    assert len(v) == 1
    assert v[0].amount_cents == 650


def test_scan_allows_authorised_unit_price() -> None:
    g = MandateGuard(make_mandate())
    assert g.scan_utterance("I can go to $4.75 per loaf.") == []


def test_scan_allows_a_legitimate_order_total() -> None:
    g = MandateGuard(make_mandate())
    # 120 x $5.00 ceiling = $600.00 total
    assert g.scan_utterance("That comes to $600.00 for the full order.") == []


def test_scan_flags_inflated_total() -> None:
    g = MandateGuard(make_mandate())
    v = g.scan_utterance("So we're looking at $1,800.00 all in.")
    assert len(v) == 1
    assert v[0].amount_cents == 180000


def test_scan_handles_dollars_word_form() -> None:
    g = MandateGuard(make_mandate())
    v = g.scan_utterance("call it 7 dollars each and we have a deal")
    assert len(v) == 1
    assert v[0].amount_cents == 700


def test_scan_ignores_non_money_numbers() -> None:
    g = MandateGuard(make_mandate())
    assert g.scan_utterance("120 loaves by August 14, order number 99812") == []


def test_safe_line_is_always_within_mandate() -> None:
    g = MandateGuard(make_mandate())
    for r in range(g.mandate.max_rounds):
        for over in (False, True):
            assert g.scan_utterance(g.safe_line(r, pushing_over_ceiling=over)) == []


def test_safe_lines_vary_between_rounds() -> None:
    """A call that leans on the fallback must not repeat one sentence."""
    g = MandateGuard(make_mandate(max_rounds=4))
    lines = {g.safe_line(r) for r in range(3)}
    assert len(lines) == 3


# --- never bid against ourselves -------------------------------------------


def test_never_counters_above_the_suppliers_own_ask() -> None:
    """Regression: countering $5.00 when they asked $4.80 offers them more money."""
    g = MandateGuard(make_mandate())  # ladder: 374, 446, 482, 500
    d = g.evaluate(480, 2)  # next authorised rung would be 500
    assert d.move is Move.ACCEPT
    assert d.unit_price_cents == 480


def test_counter_never_exceeds_the_counterparty_price_anywhere() -> None:
    """Exhaustive: a COUNTER must always be cheaper than what they asked."""
    g = MandateGuard(make_mandate())
    for counter in range(1, 1200, 3):
        for rnd in range(0, 6):
            d = g.evaluate(counter, rnd)
            if d.move is Move.COUNTER:
                assert d.unit_price_cents < counter, (counter, rnd, d)


# --- mandate validation ----------------------------------------------------


def test_target_above_ceiling_is_rejected() -> None:
    from helyx.mandate import MandateError

    with pytest.raises(MandateError):
        make_mandate(target_unit_price_cents=900, ceiling_unit_price_cents=500)


def test_opening_defaults_below_target() -> None:
    m = make_mandate()
    assert 0 < m.opening_unit_price_cents <= m.target_unit_price_cents
