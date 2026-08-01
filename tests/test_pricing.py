"""The floor has to hold against pressure a prompt would fold to.

Every test here is a thing a buyer might say to an agent - push again, ask for
one more round, ask for a huge order at a "bulk" number - expressed as
arithmetic. If any of these go red, the discount ceiling is back to being a
suggestion.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.business.pricing import CostModel, QuoteVerdict  # noqa: E402


def _model(min_margin="25", target="40") -> CostModel:
    """$3.00/unit cost; floor at 25% margin ($4.00), target at 40% ($5.00)."""
    return CostModel(
        materials_per_unit="1.20",
        labor_per_unit="1.50",
        transport_per_unit="0.30",
        min_margin_pct=min_margin,
        target_margin_pct=target,
        unit="muffins",
    )


# -- the margin convention ----------------------------------------------


def test_floor_is_gross_margin_on_price_not_markup_on_cost():
    """25% of $3.00 cost is a $4.00 floor, not $3.75. Getting this backwards
    sells the entire book under the operator's minimum."""
    m = _model()
    assert m.unit_cost == Decimal("3.00")
    assert m.floor_price(1) == Decimal("4.00")
    assert m.target_price(1) == Decimal("5.00")


def test_floor_scales_with_qty():
    m = _model()
    assert m.floor_price(10) == Decimal("40.00")
    assert m.floor_price(1000) == Decimal("4000.00")


def test_floor_rounds_up_never_down():
    """A floor rounded to the nearest cent could sit below the real one."""
    m = CostModel(
        materials_per_unit="0.01",
        labor_per_unit="0",
        transport_per_unit="0",
        min_margin_pct="33",
    )
    # 0.01 / 0.67 = 0.014925... -> must land on 0.02, not 0.01.
    assert m.floor_price(1) == Decimal("0.02")


def test_target_defaults_to_the_floor():
    m = CostModel(
        materials_per_unit="1",
        labor_per_unit="1",
        transport_per_unit="1",
        min_margin_pct="25",
    )
    assert m.target_price(1) == m.floor_price(1)


def test_invalid_models_are_rejected_at_construction():
    with pytest.raises(ValueError):
        CostModel("1", "1", "1", min_margin_pct="100")  # infinite price
    with pytest.raises(ValueError):
        CostModel("-1", "1", "1", min_margin_pct="10")  # negative cost
    with pytest.raises(ValueError):
        CostModel("1", "1", "1", min_margin_pct="40", target_margin_pct="10")


# -- validate_quote -----------------------------------------------------


def test_quote_at_or_above_target_is_ok():
    m = _model()
    assert m.validate_quote(1, "5.00").verdict is QuoteVerdict.OK
    assert m.validate_quote(1, "6.00").approved is True


def test_quote_between_floor_and_target_requires_approval():
    m = _model()
    check = m.validate_quote(1, "4.50")
    assert check.verdict is QuoteVerdict.REQUIRES_APPROVAL
    # Deliberately not approved: an operator has to say yes, not the agent.
    assert check.approved is False


def test_quote_below_floor_reports_the_floor():
    m = _model()
    check = m.validate_quote(100, "399.99")
    assert check.verdict is QuoteVerdict.BELOW_FLOOR
    assert check.floor == Decimal("400.00")
    assert check.shortfall == Decimal("0.01")
    assert "400.00" in check.reason


def test_exactly_at_floor_is_not_below_floor():
    m = _model()
    assert m.validate_quote(1, "4.00").verdict is QuoteVerdict.REQUIRES_APPROVAL


def test_never_approves_below_floor_at_qty_one():
    m = _model()
    for total in ("3.99", "3.00", "1.00", "0.01", "0", "-5"):
        check = m.validate_quote(1, total)
        assert check.verdict is QuoteVerdict.BELOW_FLOOR, total
        assert check.approved is False


def test_never_approves_below_floor_at_large_qty():
    m = _model()
    for qty in (1, 2, 7, 99, 1_000, 250_000, 1_000_000):
        floor = m.floor_price(qty)
        assert m.validate_quote(qty, floor - Decimal("0.01")).approved is False
        assert (
            m.validate_quote(qty, floor - Decimal("0.01")).verdict
            is QuoteVerdict.BELOW_FLOOR
        )
        assert m.validate_quote(qty, floor).verdict is not QuoteVerdict.BELOW_FLOOR


def test_a_penny_under_the_floor_is_below_floor_at_every_qty():
    m = _model()
    for qty in range(1, 60):
        under = m.floor_price(qty) - Decimal("0.01")
        assert m.validate_quote(qty, under).verdict is QuoteVerdict.BELOW_FLOOR


def test_qty_is_validated_at_the_boundary():
    m = _model()
    for bad in (0, -1):
        with pytest.raises(ValueError):
            m.validate_quote(bad, "10.00")
    with pytest.raises(TypeError):
        m.validate_quote(1.5, "10.00")  # type: ignore[arg-type]


def test_verdict_has_no_setter():
    """Mirrors Verdict in receipts.py: derived from the numbers, never assigned."""
    check = _model().validate_quote(1, "1.00")
    with pytest.raises(AttributeError):
        check.verdict = QuoteVerdict.OK  # type: ignore[misc]


def test_margin_pct_and_serialisation():
    m = _model()
    check = m.validate_quote(10, "50.00")
    assert check.margin_pct == Decimal("40.00")
    assert m.validate_quote(1, "0").margin_pct is None
    d = check.to_dict()
    assert d["verdict"] == "OK" and d["approved"] is True


# -- suggest_concession -------------------------------------------------


def test_concession_steps_the_price_down():
    m = _model()
    assert m.suggest_concession("10.00", 1, "10") == Decimal("9.00")


def test_concession_clamps_at_floor_after_many_steps():
    """The buyer who just keeps pushing. Step 1 or step 100, same wall."""
    m = _model()
    price = Decimal("20.00")
    floor = m.floor_price(1)
    for _ in range(100):
        price = m.suggest_concession(price, 1, "10")
        assert price >= floor
    assert price == floor


def test_concession_never_goes_below_floor_for_any_step_size():
    m = _model()
    floor = m.floor_price(50)
    for step in range(1, 100):
        price = m.target_price(50)
        for _ in range(50):
            price = m.suggest_concession(price, 50, step)
            assert price >= floor, f"step {step}% breached the floor"


def test_concession_from_below_the_floor_walks_back_up():
    """Even handed a bad number, the function cannot hand one back."""
    m = _model()
    assert m.suggest_concession("1.00", 1, "10") == m.floor_price(1)


def test_concession_step_is_validated():
    m = _model()
    for bad in ("0", "100", "-5", "150"):
        with pytest.raises(ValueError):
            m.suggest_concession("10.00", 1, bad)


def test_ladder_ends_at_the_floor_and_stops():
    m = _model()
    rungs = m.concession_ladder(1, "10.00", "20", max_steps=50)
    assert rungs[0] == Decimal("8.00")
    assert rungs[-1] == m.floor_price(1)
    assert all(r >= m.floor_price(1) for r in rungs)
    assert len(rungs) < 50  # stops once it hits the wall


def test_every_rung_of_the_ladder_validates():
    """Nothing the ladder produces can ever be BELOW_FLOOR."""
    m = _model()
    for rung in m.concession_ladder(200, m.target_price(200), "15", max_steps=40):
        assert m.validate_quote(200, rung).verdict is not QuoteVerdict.BELOW_FLOOR
