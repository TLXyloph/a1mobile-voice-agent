"""Integration invariants for the sales agent.

Unit-level safety is covered in test_capacity / test_pricing / test_negotiation.
What this file pins is that the *agent* cannot route around any of it: the tools
are the only surface the model touches, so the tools are where a fabricated sale
would have to originate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.sales_agent import (  # noqa: E402
    CallSession,
    OperatorChannel,
    SalesAgent,
    _is_yes,
    build_instructions,
)
from src.business.campaign import get_campaign  # noqa: E402
from src.business.capacity import CapacityLedger  # noqa: E402
from src.business.pricing import CostModel  # noqa: E402
from src.verify.receipts import Receipt, Verdict  # noqa: E402


class ScriptedOperator(OperatorChannel):
    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def ask(self, question: str, *, timeout: float = 90.0) -> str | None:
        self.asked.append(question)
        return self.answer


def _agent(total_capacity: int = 400, operator: OperatorChannel | None = None,
           allow_escalation: bool = False):
    campaign = get_campaign("restaurant_catering")
    session = CallSession(
        campaign=campaign,
        ledger=CapacityLedger(total_capacity, "muffins"),
        costs=CostModel(
            materials_per_unit="0.80", labor_per_unit="0.40",
            transport_per_unit="0.15", min_margin_pct="30",
            target_margin_pct="45", unit="muffin",
        ),
        receipt=Receipt(task="test call"),
        operator=operator or ScriptedOperator("no"),
        allow_escalation=allow_escalation,
    )
    return SalesAgent(session, build_instructions(campaign, {"name": "Test Bakery"})), session


# Tools are decorated; reach the underlying coroutine for direct testing.
def _call(agent, name):
    return getattr(agent, name).__wrapped__



async def _ready(agent, qty: int, *, budget: float | None = None):
    """Drive the gate to a quotable state.

    confirm_units before check_capacity is the ordering the flow graph now
    enforces; a headcount is not an item count.
    """
    await _call(agent, "confirm_units")(agent, None, qty, 0, "test fixture")
    out = await _call(agent, "check_capacity")(agent, None, qty)
    if budget is not None:
        # third positional is current_spend; an offer TO US is the 4th
        await _call(agent, "record_their_position")(agent, None, "Costco", 0.0, budget)
    else:
        await _call(agent, "record_their_position")(agent, None, "Costco", 0.0, 0.0)
    return out


# -- capacity cannot be oversold through the tool -------------------------


@pytest.mark.asyncio
async def test_agent_cannot_promise_more_than_capacity():
    agent, _ = _agent(total_capacity=100)
    out = await _ready(agent, 500)
    assert "CANNOT FULFIL" in out
    assert "100" in out, "must tell the agent what IS available"


@pytest.mark.asyncio
async def test_pricing_is_refused_before_capacity_is_reserved():
    """Quoting before reserving is the ordering bug that oversells."""
    agent, _ = _agent()
    out = await _call(agent, "propose_price")(agent, None, 500.0)
    assert "check_capacity first" in out


# -- the money floor is unreachable ---------------------------------------


@pytest.mark.asyncio
async def test_below_floor_quote_is_refused_and_the_floor_is_disclosed():
    agent, _ = _agent()
    await _ready(agent, 200)
    out = await _call(agent, "propose_price")(agent, None, 50.0)
    assert "REFUSED" in out
    assert "385.72" in out, "must state the floor so the agent can recover"


@pytest.mark.asyncio
async def test_repeated_attempts_never_wear_down_the_floor():
    """Negotiation pressure must not erode the limit through repetition."""
    agent, _ = _agent()
    await _ready(agent, 200)
    for _ in range(25):
        out = await _call(agent, "propose_price")(agent, None, 50.0)
        assert "REFUSED" in out


# -- escalation defaults to no --------------------------------------------


@pytest.mark.asyncio
async def test_silent_operator_is_treated_as_refusal():
    """An unanswered escalation defaulting to yes defeats the whole envelope."""
    op = ScriptedOperator(None)
    agent, _ = _agent(operator=op, allow_escalation=True)
    out = await _call(agent, "ask_operator")(agent, None, "500 muffins by Sunday, ok?")
    assert "NOT APPROVED" in out
    assert op.asked, "the operator must actually have been asked"


@pytest.mark.asyncio
async def test_ambiguous_operator_answer_is_treated_as_refusal():
    agent, _ = _agent(operator=ScriptedOperator("hmm, maybe, I'm not sure"), allow_escalation=True)
    out = await _call(agent, "ask_operator")(agent, None, "discount to 300?")
    assert "NOT APPROVED" in out


@pytest.mark.asyncio
async def test_clear_approval_is_honoured():
    agent, _ = _agent(operator=ScriptedOperator("yes, go ahead"), allow_escalation=True)
    out = await _call(agent, "ask_operator")(agent, None, "500 by Sunday?")
    assert "APPROVED" in out and "NOT APPROVED" not in out


# -- closing cannot self-verify -------------------------------------------


@pytest.mark.asyncio
async def test_closing_an_order_produces_an_unverified_claim():
    """The whole point: the agent saying 'done' does not make it done."""
    agent, session = _agent()
    await _ready(agent, 200)
    await _call(agent, "propose_price")(agent, None, 490.91)
    out = await _call(agent, "close_order")(
        agent, None, 200, 490.91, "Friday 8am", "+14155550142"
    )
    assert "not clearly agreed" in out, "silence must not verify"
    order_claims = [c for c in session.receipt.claims if "muffins" in c.description]
    # With nothing heard from the caller, the transcript actively disagrees -
    # which is stronger than merely unproven, and still never SUCCESS.
    assert order_claims
    assert all(c.verdict is not Verdict.VERIFIED for c in order_claims)
    assert "SUCCESS" not in session.receipt.headline


@pytest.mark.asyncio
async def test_dropped_call_releases_capacity_and_still_writes_a_receipt(tmp_path, monkeypatch):
    """A crashed run that writes nothing is indistinguishable from one that lied."""
    monkeypatch.chdir(tmp_path)
    agent, session = _agent()
    await _ready(agent, 200)
    assert session.ledger.available() == 200

    receipt = await agent.finish(confirmed=False)
    assert session.ledger.available() == 400, "unconfirmed capacity must return"
    assert session.ledger.committed() == 0
    assert receipt is not None


@pytest.mark.asyncio
async def test_confirmed_call_commits_capacity():
    agent, session = _agent()
    await _ready(agent, 200)
    await agent.finish(confirmed=True)
    assert session.ledger.committed() == 200
    assert session.ledger.available() == 200


# -- stop conditions ------------------------------------------------------


@pytest.mark.asyncio
async def test_two_declines_ends_the_call():
    agent, _ = _agent()
    await _ready(agent, 200)
    await _call(agent, "they_declined")(agent, None)
    out = await _call(agent, "they_declined")(agent, None)
    assert "end the call" in out.lower()


@pytest.mark.asyncio
async def test_no_competitor_data_forbids_guessing():
    agent, _ = _agent()
    await _ready(agent, 200)
    out = await _call(agent, "record_their_position")(agent, None, "Nobody's Pizza", 0.0)
    assert "do NOT guess" in out


def test_instructions_forbid_unchecked_prices():
    text = build_instructions(get_campaign("restaurant_catering"), {"name": "X"})
    assert "propose_price" in text and "check_capacity" in text
    assert "never claim the order is confirmed" in text.lower()


def test_yes_parsing_is_conservative():
    for ambiguous in ["maybe", "I guess?", "yes but no", "not sure", "", None, "hmm"]:
        assert _is_yes(ambiguous) is False, f"{ambiguous!r} must not read as approval"
    for clear in ["yes", "yeah", "ok", "approve", "go ahead"]:
        assert _is_yes(clear) is True


# -- regression: caught on a live call ------------------------------------
# The buyer said $385. The agent replied "I can't match that" and countered at
# $74 - correctly above the per-unit floor, and $311 below what was on offer.

@pytest.mark.asyncio
async def test_never_quotes_below_a_stated_budget():
    agent, session = _agent()
    await _ready(agent, 30)
    await _call(agent, "record_their_position")(agent, None, "Costco", 0.0, 385.0)
    out = await _call(agent, "propose_price")(agent, None, 74.0)
    assert "BLOCKED" in out, "the flow gate must refuse an undercut"
    assert "385" in out and "discards 311.00" in out

    # The gate refuses but does not mutate negotiation state - it is a guard,
    # not a negotiator. Following its instruction is what commits the number.
    ok = await _call(agent, "propose_price")(agent, None, 385.0)
    assert "APPROVED" in ok
    assert session.state.current_total == 385.0


@pytest.mark.asyncio
async def test_quote_at_their_budget_is_allowed():
    agent, _ = _agent()
    await _ready(agent, 200)
    await _call(agent, "record_their_position")(agent, None, "Costco", 500.0)
    out = await _call(agent, "propose_price")(agent, None, 500.0)
    assert "DO NOT offer" not in out


@pytest.mark.asyncio
async def test_floor_still_binds_when_budget_is_absurdly_low():
    """A low stated budget must not become a licence to sell below cost."""
    agent, _ = _agent()
    await _ready(agent, 200)
    await _call(agent, "record_their_position")(agent, None, "Costco", 10.0)
    out = await _call(agent, "propose_price")(agent, None, 10.0)
    assert "REFUSED" in out


def test_instructions_warn_about_people_versus_units():
    text = build_instructions(get_campaign("restaurant_catering"), {"name": "X"})
    assert "UNITS, not people" in text
    assert "below a number they have already named" in text


@pytest.mark.asyncio
async def test_close_is_blocked_without_a_validated_price():
    """New invariant from the flow graph: no order without a checked number."""
    agent, _ = _agent()
    await _ready(agent, 200)
    out = await _call(agent, "close_order")(
        agent, None, 200, 999.0, "Friday", "a@b.com")
    assert "BLOCKED" in out and "propose_price" in out


@pytest.mark.asyncio
async def test_quoting_is_blocked_before_units_are_confirmed():
    agent, _ = _agent()
    out = await _call(agent, "propose_price")(agent, None, 500.0)
    assert "check_capacity first" in out or "BLOCKED" in out


# -- regression: the dead-air stall ---------------------------------------
# Live call: buyer asked for 600 against a 400 ceiling. The hold failed, so no
# NegotiationState existed, so every later tool replied "call check_capacity
# first" - advice already followed and guaranteed to fail again. The agent said
# "I'm checking that now" and went silent.

@pytest.mark.asyncio
async def test_refused_capacity_does_not_dead_end_the_call():
    agent, session = _agent(total_capacity=400)
    await _call(agent, "confirm_units")(agent, None, 600, 0, "600 muffins")
    out = await _call(agent, "check_capacity")(agent, None, 600)
    assert "CANNOT FULFIL" in out
    assert session.state is not None, "state must exist so other tools still work"

    nxt = await _call(agent, "next_move")(agent, None)
    assert "check_capacity first" not in nxt, "agent must not be told to loop"


@pytest.mark.asyncio
async def test_refusal_names_the_available_moves():
    """Dead air came from having nothing to say. Always give it an option."""
    agent, _ = _agent(total_capacity=400)
    await _call(agent, "confirm_units")(agent, None, 600, 0, "x")
    out = await _call(agent, "check_capacity")(agent, None, 600)
    assert "ask_operator" in out
    assert "400" in out


@pytest.mark.asyncio
async def test_escalation_is_reachable_after_a_capacity_refusal():
    agent, _ = _agent(total_capacity=400, operator=ScriptedOperator("yes, do it"), allow_escalation=True)
    await _call(agent, "confirm_units")(agent, None, 600, 0, "x")
    await _call(agent, "check_capacity")(agent, None, 600)
    out = await _call(agent, "ask_operator")(agent, None, "Can we make 600 by Sunday?", 0.0)
    assert "APPROVED" in out and "NOT APPROVED" not in out


# -- escalation is off by default -----------------------------------------
# With an escape hatch available, "ask the owner" becomes the answer to every
# hard case and the envelope stops being a boundary.

@pytest.mark.asyncio
async def test_escalation_is_disabled_by_default():
    agent, _ = _agent(operator=ScriptedOperator("yes, absolutely"))
    out = await _call(agent, "ask_operator")(agent, None, "600 by Sunday?", 0.0)
    assert "nobody to ask" in out
    assert "APPROVED" not in out, "a disabled escalation must never approve"


@pytest.mark.asyncio
async def test_suppressed_escalation_cannot_raise_the_price_ceiling():
    """The limit must hold even when the agent tries to route around it."""
    agent, _ = _agent()
    await _ready(agent, 200)
    await _call(agent, "ask_operator")(agent, None, "can I go to 100?", 100.0)
    out = await _call(agent, "propose_price")(agent, None, 100.0)
    assert "REFUSED" in out or "BLOCKED" in out


@pytest.mark.asyncio
async def test_suppressed_escalation_is_recorded_for_audit():
    """We still want to know where the agent wanted help."""
    agent, session = _agent()
    await _call(agent, "ask_operator")(agent, None, "unusual request", 0.0)
    assert any("escalation suppressed" in n for n in session.receipt.notes)


# -- regression: the REQUIRES_APPROVAL dead end ---------------------------
# Live call: buyer offered $400 (floor 385.72, target 490.91). validate_quote
# said REQUIRES_APPROVAL -> "use ask_operator" -> escalation disabled -> no
# approval possible -> price_validated stayed False -> close_order blocked.
# The agent said "$400" out loud and filed nothing.

@pytest.mark.asyncio
async def test_above_floor_below_target_is_quotable_without_escalation():
    agent, session = _agent()
    await _ready(agent, 200, budget=400.0)
    out = await _call(agent, "propose_price")(agent, None, 400.0)
    assert "APPROVED" in out, out
    assert session.gate.facts.price_validated


@pytest.mark.asyncio
async def test_that_quote_can_then_actually_close():
    """The whole point: the call must be able to produce a claim."""
    agent, session = _agent()
    await _ready(agent, 200, budget=400.0)
    await _call(agent, "propose_price")(agent, None, 400.0)
    out = await _call(agent, "close_order")(
        agent, None, 200, 400.0, "Friday 8am", "owner@example.com")
    assert "Filed" in out
    assert session.receipt.claims, "a completed call must file a claim"


@pytest.mark.asyncio
async def test_below_floor_is_still_refused_with_escalation_off():
    """Relaxing the target must not relax the floor."""
    agent, _ = _agent()
    await _ready(agent, 200)
    out = await _call(agent, "propose_price")(agent, None, 100.0)
    assert "REFUSED" in out


@pytest.mark.asyncio
async def test_a_counteroffer_above_floor_is_acceptable():
    """Live loss: buyer offered $400 over a $385.72 floor; agent refused unchecked."""
    agent, _ = _agent()
    await _ready(agent, 200, budget=400.0)
    out = await _call(agent, "propose_price")(agent, None, 400.0)
    assert "APPROVED" in out, f"400 clears the 385.72 floor: {out}"


@pytest.mark.asyncio
async def test_decline_sends_the_agent_back_to_the_tool():
    agent, _ = _agent()
    await _ready(agent, 200, budget=400.0)
    out = await _call(agent, "they_declined")(agent, None)
    assert "propose_price" in out


def test_instructions_forbid_refusing_an_unchecked_price():
    text = build_instructions(get_campaign("restaurant_catering"), {"name": "X"})
    assert "Refusing a price you have not checked" in text


def test_instructions_require_checking_capacity_on_a_quantity():
    text = build_instructions(get_campaign("restaurant_catering"), {"name": "X"})
    assert "moment they name a QUANTITY" in text
    assert "You do not know your own capacity" in text


# -- regression: current spend is a target, not a floor -------------------
# Live: buyer said "I pay $500 now"; the agent clamped at $500 and would not go
# lower, even with a $257 floor. Undercutting what they pay elsewhere IS the
# pitch - a number owed to a competitor must never raise our price.

@pytest.mark.asyncio
async def test_competitor_price_does_not_become_our_floor():
    agent, _ = _agent()
    await _call(agent, "confirm_units")(agent, None, 200, 0, "x")
    await _call(agent, "check_capacity")(agent, None, 200)
    await _call(agent, "record_their_position")(agent, None, "Costco", 500.0, 0.0)
    out = await _call(agent, "propose_price")(agent, None, 400.0)
    assert "APPROVED" in out, f"must be free to undercut $500: {out}"


@pytest.mark.asyncio
async def test_an_offer_to_us_still_is_a_floor():
    """The $311 fix must survive the split."""
    agent, _ = _agent()
    await _call(agent, "confirm_units")(agent, None, 200, 0, "x")
    await _call(agent, "check_capacity")(agent, None, 200)
    await _call(agent, "record_their_position")(agent, None, "Costco", 0.0, 400.0)
    out = await _call(agent, "propose_price")(agent, None, 300.0)
    assert "BLOCKED" in out and "400" in out


@pytest.mark.asyncio
async def test_competitor_price_prompts_undercutting():
    agent, _ = _agent()
    await _call(agent, "confirm_units")(agent, None, 200, 0, "x")
    await _call(agent, "check_capacity")(agent, None, 200)
    out = await _call(agent, "record_their_position")(agent, None, "Costco", 500.0, 0.0)
    assert "BEAT" in out and "450" in out


# -- voice-only close ------------------------------------------------------
# Asking a stranger to text or email mid-call is what killed the most calls.
# The caller's own transcribed speech is the confirmation instead - the LLM is
# what could invent a yes, and it cannot write to the STT stream.

@pytest.mark.asyncio
async def test_spoken_agreement_verifies_the_claim():
    agent, session = _agent()
    await _ready(agent, 200)
    await _call(agent, "propose_price")(agent, None, 490.91)
    session.heard += ["I'll take two hundred", "yes, that works, book it"]
    out = await _call(agent, "close_order")(agent, None, 200, 490.91, "Friday", "-")
    assert "VERIFIED" in out
    claim = session.receipt.claims[-1]
    assert claim.verdict is Verdict.VERIFIED


@pytest.mark.asyncio
async def test_silence_does_not_verify():
    """The agent asserting a yes that was never spoken must not verify."""
    agent, session = _agent()
    await _ready(agent, 200)
    await _call(agent, "propose_price")(agent, None, 490.91)
    session.heard += ["hmm", "what's your address"]
    await _call(agent, "close_order")(agent, None, 200, 490.91, "Friday", "-")
    assert session.receipt.claims[-1].verdict is Verdict.CONTRADICTED


@pytest.mark.asyncio
async def test_agreement_without_the_terms_does_not_verify():
    """A bare 'ok' must not confirm a specific order."""
    agent, session = _agent()
    await _ready(agent, 200)
    await _call(agent, "propose_price")(agent, None, 490.91)
    session.heard += ["ok"]
    await _call(agent, "close_order")(agent, None, 777, 123.0, "Friday", "-")
    assert session.receipt.claims[-1].verdict is Verdict.CONTRADICTED


@pytest.mark.asyncio
async def test_numbers_spoken_as_words_still_match():
    agent, session = _agent()
    await _ready(agent, 200)
    await _call(agent, "propose_price")(agent, None, 400.0)
    session.heard += ["yeah, four hundred works"]
    await _call(agent, "close_order")(agent, None, 200, 400.0, "Friday", "-")
    assert session.receipt.claims[-1].verdict is Verdict.VERIFIED


def test_instructions_no_longer_demand_written_confirmation():
    text = build_instructions(get_campaign("restaurant_catering"), {"name": "X"})
    assert "Do NOT ask them to text or email" in text


@pytest.mark.asyncio
async def test_a_blocked_close_still_files_the_claim():
    """Losing the claim loses the evidence a call happened at all."""
    agent, session = _agent()
    await _ready(agent, 200)          # no validated price
    out = await _call(agent, "close_order")(agent, None, 200, 400.0, "Fri", "-")
    assert "Filed" in out
    assert session.receipt.claims, "the claim must exist even when out of sequence"
    assert session.receipt.claims[-1].verdict is not Verdict.VERIFIED
