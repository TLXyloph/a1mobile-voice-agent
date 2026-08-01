"""Negotiator: the model's words are checked against the mandate after the fact."""

from __future__ import annotations

from datetime import date

from helyx.domain import Channel, ProposalStatus
from helyx.ladder import Move
from helyx.mandate import Mandate
from helyx.negotiator import Negotiation, Negotiator, Outcome

from .conftest import FakeLLM


def mandate(**over: object) -> Mandate:
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


def test_agent_line_breaching_ceiling_is_replaced(fake_llm: FakeLLM) -> None:
    """The headline failure mode: model caves under pressure and names $6.50."""
    fake_llm.speech = ["Fine - $6.50 a loaf, you win."]
    neg = Negotiation(mandate=mandate())
    turn = Negotiator(fake_llm).turn(neg, "We can't go below $6.50.")

    assert turn.replaced is True
    assert turn.violations and turn.violations[0].amount_cents == 650
    assert "6.50" not in turn.agent_said
    # whatever replaced it is itself inside the mandate
    assert neg.guard.scan_utterance(turn.agent_said) == []


def test_authorised_line_passes_through_untouched(fake_llm: FakeLLM) -> None:
    neg = Negotiation(mandate=mandate())
    authorised = neg.guard.authorized_offer(0)
    line = f"I can do ${authorised / 100:.2f} per loaf."
    fake_llm.speech = [line]
    turn = Negotiator(fake_llm).turn(neg, "What's your offer?")
    assert turn.replaced is False
    assert turn.agent_said == line


def test_filed_proposal_is_unconfirmed_and_agent_sourced(fake_llm: FakeLLM) -> None:
    fake_llm.replies = [
        ("Noted.", {"unit_price": 4.10, "quantity": 120, "fulfilment_date": "2026-08-14"})
    ]
    neg = Negotiation(mandate=mandate())
    Negotiator(fake_llm).turn(neg, "We can do $4.10 each.")

    assert len(neg.proposals) == 1
    p = neg.proposals[0]
    assert p.status is ProposalStatus.UNCONFIRMED
    assert p.evidence[0].channel is Channel.AGENT_ASSERTION
    assert p.terms.unit_price_cents == 410


def test_below_target_counter_is_accepted_pending_confirmation(fake_llm: FakeLLM) -> None:
    fake_llm.replies = [("Great.", {"unit_price": 4.10, "quantity": 120})]
    neg = Negotiation(mandate=mandate())
    turn = Negotiator(fake_llm).turn(neg, "$4.10 each works for us.")
    assert turn.decision is not None and turn.decision.move is Move.ACCEPT
    # "accepted" still means nothing is confirmed
    assert neg.outcome is Outcome.ACCEPTED_PENDING_CONFIRMATION
    assert neg.proposals[0].status is ProposalStatus.UNCONFIRMED


def test_price_above_ceiling_on_final_round_walks_away(fake_llm: FakeLLM) -> None:
    fake_llm.replies = [("Understood.", {"unit_price": 7.50, "quantity": 120})]
    neg = Negotiation(mandate=mandate(), round_index=3)
    Negotiator(fake_llm).turn(neg, "Final answer, $7.50.")
    assert neg.outcome is Outcome.WALKED_AWAY


def test_agent_still_speaks_when_it_files_a_proposal(fake_llm: FakeLLM) -> None:
    """Regression: models return empty content alongside a tool call, which
    silently forced the deterministic fallback on every single turn."""
    fake_llm.replies = [("", {"unit_price": 4.60, "quantity": 120})]
    fake_llm.speech = ["I hear you, but I can stretch to $4.46 a loaf and no further."]
    neg = Negotiation(mandate=mandate())
    turn = Negotiator(fake_llm).turn(neg, "We need $4.60.")

    assert len(neg.proposals) == 1  # it heard them
    assert turn.replaced is False  # and it spoke in its own words
    assert "4.46" in turn.agent_said


def test_speech_pass_is_told_the_arithmetic_decision(fake_llm: FakeLLM) -> None:
    """The model writes the words, but the price in its brief is the guard's."""
    fake_llm.replies = [("", {"unit_price": 9.00, "quantity": 120})]
    fake_llm.speech = ["No can do."]
    neg = Negotiation(mandate=mandate())
    Negotiator(fake_llm).turn(neg, "It's $9.00 a loaf.")

    # The last call is the speech pass; its instruction is the final message.
    instruction = fake_llm.calls[-1][-1]["content"]
    assert "$4.46" in instruction  # the authorised rung, chosen by arithmetic
    assert "ONLY figure" in instruction
    assert "9.00" not in instruction  # their price is never handed to the speaker


def test_agent_has_no_tool_that_confirms() -> None:
    """There is deliberately no 'mark_confirmed' capability."""
    from helyx.negotiator import FILE_PROPOSAL_TOOL

    names = {FILE_PROPOSAL_TOOL["function"]["name"]}
    assert names == {"file_proposal"}
    for banned in ("confirm", "complete", "book", "verify", "done"):
        assert banned not in str(names)


def test_running_out_of_rounds_walks_away(fake_llm: FakeLLM) -> None:
    neg = Negotiation(mandate=mandate(max_rounds=2))
    fake_llm.replies = [("ok", None), ("ok", None)]
    n = Negotiator(fake_llm)
    n.turn(neg, "hmm")
    n.turn(neg, "still thinking")
    assert neg.finished is True
    assert neg.outcome is Outcome.WALKED_AWAY


def test_llm_outage_still_produces_a_safe_line() -> None:
    class Broken:
        model = fallback_model = "x"

        def complete(self, *a: object, **k: object) -> object:
            from helyx.llm import LLMError

            raise LLMError("down")

    neg = Negotiation(mandate=mandate())
    turn = Negotiator(Broken()).turn(neg, "what's your price?")  # type: ignore[arg-type]
    assert turn.agent_said
    assert neg.guard.scan_utterance(turn.agent_said) == []
