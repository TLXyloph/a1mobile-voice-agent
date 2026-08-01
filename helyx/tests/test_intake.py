"""Intake: the model fills slots, but only arithmetic decides readiness."""

from __future__ import annotations

from helyx.intake import IntakeAgent, IntakeSession
from helyx.mandate import REQUIRED_FIELDS

from .conftest import FakeLLM

FULL = {
    "item": "sourdough loaves",
    "quantity": 120,
    "target_unit_price": 4.25,
    "ceiling_unit_price": 5.00,
    "needed_by": "2026-08-14",
    "counterparty_name": "Kestrel Bakehouse",
    "counterparty_phone": "+15551230000",
}


def test_new_session_is_not_ready() -> None:
    s = IntakeSession()
    assert s.ready is False
    assert set(s.missing) == set(REQUIRED_FIELDS)


def test_partial_extraction_leaves_session_blocked(fake_llm: FakeLLM) -> None:
    fake_llm.replies = [("Got it.", {"item": "sourdough loaves", "quantity": 120})]
    s = IntakeAgent(fake_llm).turn(IntakeSession(), "120 sourdough loaves please")
    assert s.ready is False
    assert "target_unit_price_cents" in s.missing
    assert s.next_question


def test_model_claiming_completion_does_not_unblock(fake_llm: FakeLLM) -> None:
    """The failure this whole design exists to prevent, at the intake layer."""
    fake_llm.replies = [
        (
            "All set! I have everything I need and the mandate is complete and locked.",
            {"item": "sourdough loaves"},
        )
    ]
    s = IntakeAgent(fake_llm).turn(IntakeSession(), "just do it")
    assert s.ready is False
    assert len(s.missing) > 0


def test_full_extraction_becomes_ready(fake_llm: FakeLLM) -> None:
    fake_llm.replies = [("Captured.", dict(FULL))]
    s = IntakeAgent(fake_llm).turn(IntakeSession(), "the whole order")
    assert s.missing == []
    assert s.ready is True
    m = s.mandate()
    assert m.quantity == 120
    assert m.target_unit_price_cents == 425
    assert m.ceiling_unit_price_cents == 500


def test_incoherent_limits_block_even_when_all_slots_filled(fake_llm: FakeLLM) -> None:
    """Target above the walk-away point is not a valid mandate."""
    bad = dict(FULL, target_unit_price=9.00, ceiling_unit_price=5.00)
    fake_llm.replies = [("Captured.", bad)]
    s = IntakeAgent(fake_llm).turn(IntakeSession(), "aim high")
    assert s.missing == []
    assert s.ready is False
    assert "ceiling" in s.validation_error()


def test_slots_accumulate_across_turns(fake_llm: FakeLLM) -> None:
    agent = IntakeAgent(fake_llm)
    s = IntakeSession()
    fake_llm.replies = [
        ("ok", {"item": "croissant trays", "quantity": 40}),
        ("ok", {"target_unit_price": 18.0, "ceiling_unit_price": 22.0}),
        ("ok", {"needed_by": "2026-09-01", "counterparty_name": "Aurel Bakehouse"}),
    ]
    for text in ("40 croissant trays", "aim 18, cap 22", "by Sept 1 from Aurel"):
        s = agent.turn(s, text)
    assert s.ready is True
    assert s.mandate().item == "croissant trays"


def test_llm_outage_does_not_corrupt_slots() -> None:
    class Broken:
        model = fallback_model = "x"

        def complete(self, *a: object, **k: object) -> object:
            from helyx.llm import LLMError

            raise LLMError("gateway down")

    s = IntakeSession(slots={"item": "loaves"})
    out = IntakeAgent(Broken()).turn(s, "hello")  # type: ignore[arg-type]
    assert out.slots == {"item": "loaves"}
    assert out.ready is False
    assert "could not reach" in out.last_reply
