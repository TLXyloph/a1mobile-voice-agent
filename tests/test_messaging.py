"""The SMS channel must not be a way around the limits the call ran under.

Two families of test live here, and they are the two ways this feature could
lose the hackathon rather than win it:

1. **A price the voice agent was forbidden to say must not be sayable by text.**
   Below the floor, under a budget the prospect already named, per-person
   instead of per-unit, or waiting on an approval nobody can give.
2. **The agent must not be able to verify itself.** An outbound text is
   `Channel.AGENT_ASSERTION`. If a test in here goes green while the agent
   texts itself into a VERIFIED claim, the automatic-disqualification path is
   open again.
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.agents.flow import Phase  # noqa: E402
from src.business.campaign import (  # noqa: E402
    Campaign,
    CloseCondition,
    DiscoveryStrategy,
    Envelope,
)
from src.business.pricing import CostModel  # noqa: E402
from src.messaging import evidence as ev  # noqa: E402
from src.messaging import routes  # noqa: E402
from src.messaging.closer import (  # noqa: E402
    Basis,
    ReplyStatus,
    ScriptedResponder,
    check_draft,
    extract_figures,
    generate_reply,
    read_inbound,
)
from src.messaging.send import A1MobileSMS, NullSender, SendStatus  # noqa: E402
from src.messaging.thread import Direction, Thread, ThreadStore  # noqa: E402
from src.verify.receipts import Channel, Receipt, Verdict  # noqa: E402

PHONE = "+14155550134"


# -- fixtures --------------------------------------------------------------


def _campaign() -> Campaign:
    return Campaign(
        name="Weekday catering",
        vertical="restaurant",
        icp="offices nearby",
        offer="a standing weekly breakfast order",
        discovery=DiscoveryStrategy.EVENT_SIGNAL,
        close_condition=CloseCondition.WRITTEN_CONFIRMATION,
        capacity_units="muffins/week",
        envelope=Envelope(
            min_price=3.50,
            max_qty=600,
            earliest_date=date(2026, 8, 10),
            latest_date=date(2026, 11, 20),
            max_discount_pct=15.0,
        ),
    )


def _costs() -> CostModel:
    # 120 units: cost 1.93*120 = 231.60 -> floor 231.60/0.75 = 308.80,
    # target 231.60/0.60 = 386.00.
    return CostModel(
        materials_per_unit="1.10",
        labor_per_unit="0.55",
        transport_per_unit="0.28",
        min_margin_pct="25",
        target_margin_pct="40",
        unit="muffin",
    )


def _thread(**over) -> Thread:
    t = Thread(
        phone=PHONE,
        campaign=_campaign(),
        costs=_costs(),
        qty=120,
        phase=Phase.QUOTED,
    )
    t.facts.units_confirmed = True
    t.facts.units = 120
    t.facts.capacity_held = True
    t.facts.asked_current_spend = True
    t.facts.price_validated = True
    for k, v in over.items():
        setattr(t, k, v)
    return t


def test_fixture_numbers_are_what_the_tests_assume():
    """If the arithmetic moves, the assertions below stop meaning anything."""
    t = _thread()
    assert t.floor_total() == Decimal("308.80")
    assert t.target_total() == Decimal("386.00")


# -- the price guard -------------------------------------------------------


def test_extracts_currency_figures_in_several_shapes():
    figures = extract_figures("It's $420, or 385 dollars, or USD 400.00 flat.")
    assert [str(f.amount) for f in figures] == ["420", "385", "400.00"]


def test_times_are_not_mistaken_for_prices():
    assert extract_figures("I'll drop them at 8:30 tomorrow, 7.30am at a push") == []


def test_per_unit_figure_is_multiplied_out():
    figures = extract_figures("$3.50 each")
    assert figures[0].basis is Basis.PER_UNIT
    assert figures[0].implied_total(120) == Decimal("420.00")


def test_below_floor_price_is_refused():
    """The $308.80 floor is the floor by text as much as by voice."""
    check = check_draft(_thread(), "I can do the whole order for $250.")
    assert not check.ok
    assert any(i.kind == "below_floor" for i in check.issues)
    assert "308.80" in check.instruction


@pytest.mark.asyncio
async def test_below_floor_draft_is_regenerated_and_never_sent():
    thread = _thread()
    responder = ScriptedResponder(
        replies=[
            "Happy to do the lot for $250 - shall I put it in?",
            "I can do the full 120 for $420. Reply with the quantity and total "
            "if that works and I'll get it prepped.",
        ]
    )
    reply = await generate_reply(thread, responder)

    assert reply.status is ReplyStatus.REGENERATED
    assert "250" not in reply.text
    assert "$420" in reply.text
    assert reply.rejected and reply.rejected[0]["issues"][0]["kind"] == "below_floor"


@pytest.mark.asyncio
async def test_model_that_never_complies_gets_a_number_free_fallback():
    thread = _thread()
    responder = ScriptedResponder(replies=["$250 final."] * 5, default="$250 final.")
    reply = await generate_reply(thread, responder, attempts=3)

    assert reply.status is ReplyStatus.FALLBACK
    assert extract_figures(reply.text) == []
    assert reply.attempts == 3
    assert check_draft(thread, reply.text).ok


def test_reply_never_undercuts_a_stated_budget():
    """The $311 lesson: their number is our conversational floor, not a ceiling."""
    thread = _thread()
    thread.note_stated_budget(500.0)

    # 420 clears the margin floor and the target, and is still an underquote.
    check = check_draft(thread, "I can do all 120 for $420.")
    assert not check.ok
    assert [i.kind for i in check.issues] == ["undercuts_budget"]
    assert "80.00" in check.instruction  # the money it would have discarded

    assert check_draft(thread, "All 120 comes to $500 even.").ok


def test_a_later_lower_number_cannot_reopen_the_gap():
    thread = _thread()
    thread.note_stated_budget(500.0)
    thread.note_stated_budget(400.0)
    assert thread.budget_floor == 500.0
    assert not check_draft(thread, "Let's call it $420.").ok


def test_per_person_pricing_is_refused():
    check = check_draft(_thread(), "It works out at $6.50 per person.")
    assert not check.ok
    assert any(i.kind == "per_person" for i in check.issues)


def test_price_between_floor_and_target_needs_an_approval_nobody_can_give():
    check = check_draft(_thread(), "I can do 120 for $350.")
    assert not check.ok
    assert any(i.kind == "requires_approval" for i in check.issues)


def test_an_approval_from_the_call_carries_over():
    thread = _thread(approved_total=340.0)
    assert check_draft(thread, "I can do 120 for $350.").ok


def test_no_cost_model_means_no_number_at_all():
    """A figure nobody can validate is the fabrication path, so it is refused."""
    bare = Thread(phone=PHONE)
    check = check_draft(bare, "We can do that for $200.")
    assert not check.ok
    assert any(i.kind == "unvalidatable" for i in check.issues)


def test_escalation_language_is_refused():
    check = check_draft(_thread(), "Let me check with the owner and come back to you.")
    assert not check.ok
    assert any(i.kind == "escalation" for i in check.issues)


def test_claiming_the_deal_is_closed_is_refused():
    check = check_draft(_thread(), "You're all set for Friday, I've booked you in.")
    assert not check.ok
    assert any(i.kind == "claims_closed" for i in check.issues)


def test_asking_for_written_confirmation_is_allowed():
    """The agent's job is to elicit evidence, so this must not trip the guard."""
    assert check_draft(
        _thread(), "Reply with the quantity and total and I'll get it prepped."
    ).ok


def test_gate_blocks_a_quote_the_call_never_earned():
    """Continuity: a call still in discovery cannot start quoting by text."""
    thread = _thread(phase=Phase.DISCOVERY)
    thread.facts.capacity_held = False
    check = check_draft(thread, "I can do 120 for $420.")
    assert not check.ok
    assert any(i.kind == "gate_blocked" for i in check.issues)


# -- evidence: the agent cannot verify itself ------------------------------


def _claim():
    receipt = Receipt(task="120 muffins for Friday")
    return receipt, receipt.claim(
        "120 muffins for 420.00, delivered Friday",
        "an SMS from the office states 120 and 420.00",
    )


@pytest.mark.asyncio
async def test_agent_cannot_verify_its_own_claim_by_texting_itself():
    """The core invariant, in SMS form.

    The agent sends a text containing every token the claim needs, to its own
    number, and then that same text is offered back through the outbound path
    as many times as it likes. The claim stays UNVERIFIED.
    """
    receipt, claim = _claim()
    tokens = ev.tokens_for(120, Decimal("420.00"))
    body = "Confirming 120 muffins at 420.00 for Friday."

    for _ in range(5):
        ev.record_outbound(claim, body, to=PHONE)

    assert claim.verdict is Verdict.UNVERIFIED
    assert all(e.channel is Channel.AGENT_ASSERTION for e in claim.evidence)
    assert not any(e.is_independent for e in claim.evidence)
    # The same words arriving from the prospect DO verify - the direction is
    # the whole difference.
    assert ev.try_verify(claim, tokens, body, sender=PHONE)
    assert claim.verdict is Verdict.VERIFIED
    assert receipt.headline.startswith("SUCCESS")


@pytest.mark.asyncio
async def test_outbound_through_the_router_cannot_verify():
    """End to end: send a perfect confirmation out, claim stays UNVERIFIED."""
    store = ThreadStore(":memory:")
    routes.set_store(store)
    registry = ev.ClaimRegistry()
    routes.set_registry(registry)
    routes.set_sender(NullSender())

    thread = _thread()
    store.save(thread)
    _, claim = _claim()
    routes.link_claim(thread, claim, ev.tokens_for(120, Decimal("420.00")))
    store.save(thread)

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        r = client.post(
            "/messages/send",
            json={
                "phone": PHONE,
                "text": "Just to confirm the numbers: 120 muffins, 420.00 total.",
            },
        )
    assert r.status_code == 200
    assert claim.verdict is Verdict.UNVERIFIED
    assert [e.channel for e in claim.evidence] == [Channel.AGENT_ASSERTION]


def test_matching_inbound_promotes_the_claim_to_verified():
    _, claim = _claim()
    tokens = ev.tokens_for(120, Decimal("420.00"))
    assert ev.try_verify(
        claim,
        tokens,
        "Confirmed - 120 muffins, $420 total, Friday 8am works.",
        sender=PHONE,
    )
    assert claim.verdict is Verdict.VERIFIED


def test_partial_match_does_not_verify():
    """'Confirmed' with no numbers confirms nothing checkable."""
    _, claim = _claim()
    tokens = ev.tokens_for(120, Decimal("420.00"))
    assert not ev.try_verify(claim, tokens, "Confirmed!", sender=PHONE)
    assert claim.verdict is Verdict.UNVERIFIED
    assert claim.evidence == []


def test_wrong_numbers_do_not_verify():
    _, claim = _claim()
    tokens = ev.tokens_for(120, Decimal("420.00"))
    assert not ev.try_verify(claim, tokens, "Confirmed 1200 muffins for $420", sender=PHONE)
    assert claim.verdict is Verdict.UNVERIFIED


def test_a_claim_with_no_tokens_cannot_be_verified_by_anything():
    _, claim = _claim()
    assert not ev.try_verify(claim, (), "confirmed whatever you like", sender=PHONE)


@pytest.mark.asyncio
async def test_inbound_through_the_router_verifies():
    store = ThreadStore(":memory:")
    routes.set_store(store)
    registry = ev.ClaimRegistry()
    routes.set_registry(registry)
    sender = NullSender()
    routes.set_sender(sender)
    routes.set_responder(None)

    thread = _thread()
    store.save(thread)
    _, claim = _claim()
    routes.link_claim(thread, claim, ev.tokens_for(120, Decimal("420.00")))
    store.save(thread)

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        r = client.post(
            "/messages/inbound/generic",
            json={"from": PHONE, "body": "Yes - 120 muffins at 420.00, Friday."},
        )
        body = r.json()
        assert body["verified_claims"] == [claim.id]
        assert claim.verdict is Verdict.VERIFIED

        vague = client.post(
            "/messages/inbound/generic", json={"from": PHONE, "body": "sounds good"}
        )
        assert vague.json()["verified_claims"] == []

    assert sender.sent == []  # no responder configured, so nothing drafted


# -- persistence -----------------------------------------------------------


def test_threads_persist_across_a_reconstructed_store(tmp_path):
    """A restart must restore the limits exactly, not approximately."""
    db = tmp_path / "threads.db"
    store = ThreadStore(db)
    thread = _thread()
    thread.note_stated_budget(500.0)
    thread.add_inbound("can you do 500?")
    thread.add_outbound("500 works - reply with the quantity and total.")
    thread.track_claim("claim_abc", "120 muffins for 500.00", ("120", "500.00"))
    store.save(thread)
    store.close()

    reopened = ThreadStore(db)
    back = reopened.load(PHONE)
    assert back is not None
    assert back.qty == 120
    assert back.phase is Phase.QUOTED
    assert back.budget_floor == 500.0
    assert back.floor_total() == Decimal("308.80")
    assert back.target_total() == Decimal("386.00")
    assert back.campaign is not None and back.campaign.envelope.max_qty == 600
    assert back.facts.units_confirmed and back.facts.capacity_held
    assert [m.direction for m in back.messages] == [Direction.INBOUND, Direction.OUTBOUND]
    assert back.claims[0].tokens == ("120", "500.00")

    # And the guard behaves identically on the restored thread.
    assert not check_draft(back, "I can do $420.").ok
    assert check_draft(back, "That's $500 for the 120.").ok
    reopened.close()


def test_state_key_is_stable_until_something_changes(tmp_path):
    store = ThreadStore(tmp_path / "s.db")
    assert store.state_key() == "empty"
    thread = _thread()
    store.save(thread)
    first = store.state_key()
    assert store.state_key() == first
    thread.add_inbound("hello?")
    store.save(thread)
    assert store.state_key() != first
    store.close()


def test_opt_out_is_recorded_and_blocks_replies():
    thread = _thread()
    read = read_inbound(thread, "STOP")
    assert read["opt_out"] and thread.opted_out


@pytest.mark.asyncio
async def test_opted_out_thread_generates_nothing():
    thread = _thread(opted_out=True)
    reply = await generate_reply(thread, ScriptedResponder(replies=["hello again"]))
    assert reply.status is ReplyStatus.BLOCKED
    assert not reply.sendable


def test_inbound_budget_statement_raises_the_floor():
    thread = _thread()
    read = read_inbound(thread, "Our budget is $500 for the whole thing.")
    assert read["stated_budget"] == 500.0
    assert thread.budget_floor == 500.0


# -- sending ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_sends_nothing(monkeypatch):
    """Nothing leaves the process, and the transport is never even imported."""
    sender = A1MobileSMS(team_key="not-a-real-key", dry_run=True)

    def explode(*a, **k):  # pragma: no cover - only runs if dry run leaks
        raise AssertionError("dry run must not open a connection")

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", explode)

    result = await sender.send(PHONE, "hello")
    assert result.ok and result.status is SendStatus.DRY_RUN
    assert sender.sent == [{"to": PHONE, "body": "hello", "dry_run": True}]


@pytest.mark.asyncio
async def test_dry_run_is_the_default_under_pytest():
    assert A1MobileSMS(team_key="x").dry_run is True
    assert A1MobileSMS(team_key="x", base_url="https://hack.a1mobile.com").url == (
        "https://hack.a1mobile.com/api/sms"
    )


@pytest.mark.asyncio
async def test_unverified_number_surfaces_a_clear_error():
    """a1mobile only delivers to OTP-verified numbers. Say so, loudly."""
    sender = A1MobileSMS(
        team_key="k", dry_run=False, verified_numbers={"+14155550100"}
    )
    result = await sender.send(PHONE, "hello")

    assert not result.ok
    assert result.status is SendStatus.UNVERIFIED_NUMBER
    assert result.needs_operator
    assert "OTP-verified" in result.detail and PHONE in result.detail
    assert sender.sent == []  # refused before any request


@pytest.mark.asyncio
async def test_provider_rejection_is_classified_as_unverified(monkeypatch):
    import httpx

    class _Resp:
        status_code = 403
        text = '{"error":"recipient not verified"}'

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    sender = A1MobileSMS(team_key="k", dry_run=False)
    result = await sender.send(PHONE, "hello")
    assert result.status is SendStatus.UNVERIFIED_NUMBER
    assert "403" in result.detail


@pytest.mark.asyncio
async def test_missing_team_key_is_not_a_silent_no_op():
    sender = A1MobileSMS(team_key="", dry_run=False)
    result = await sender.send(PHONE, "hello")
    assert result.status is SendStatus.NOT_CONFIGURED
    assert not result.ok and result.needs_operator


# -- the router's read surface ---------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_json_shape():
    store = ThreadStore(":memory:")
    routes.set_store(store)
    routes.set_registry(ev.ClaimRegistry())
    routes.set_sender(NullSender())
    routes.set_responder(None)
    store.save(_thread())

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        state = client.get("/messages/state").json()
        assert state["threads"] == 1 and state["state"] != "empty"
        assert client.get("/messages/state").json()["state"] == state["state"]

        listing = client.get("/messages/threads").json()
        row = listing["threads"][0]
        assert row["phone"] == PHONE
        assert row["floor_total"] == "308.80" and row["target_total"] == "386.00"
        assert row["can_quote"] is True

        detail = client.get(f"/messages/threads/{PHONE}").json()
        assert detail["constraints"]["escalation_available"] is False
        assert detail["constraints"]["floor_total"] == "308.80"
        assert detail["messages"] == []

        missing = client.get("/messages/threads/+15550000000").json()
        assert missing["ok"] is False


@pytest.mark.asyncio
async def test_router_refuses_an_operator_typed_underquote():
    store = ThreadStore(":memory:")
    routes.set_store(store)
    routes.set_registry(ev.ClaimRegistry())
    sender = NullSender()
    routes.set_sender(sender)
    store.save(_thread())

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        r = client.post("/messages/send", json={"phone": PHONE, "text": "Fine, $200."})
    body = r.json()
    assert body["ok"] is False
    assert body["check"]["issues"][0]["kind"] == "below_floor"
    assert sender.sent == []


@pytest.mark.asyncio
async def test_inbound_drafts_a_reply_and_records_it_as_agent_assertion():
    store = ThreadStore(":memory:")
    routes.set_store(store)
    registry = ev.ClaimRegistry()
    routes.set_registry(registry)
    sender = NullSender()
    routes.set_sender(sender)
    routes.set_responder(
        ScriptedResponder(
            replies=[
                "$250 and it's yours.",
                "The best I can do on 120 is $420. Reply with the quantity and "
                "total and I'll lock the slot.",
            ]
        )
    )

    thread = _thread()
    store.save(thread)
    _, claim = _claim()
    routes.link_claim(thread, claim, ev.tokens_for(120, Decimal("420.00")))
    store.save(thread)

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        body = client.post(
            "/messages/inbound/generic",
            json={"from": PHONE, "body": "Can you do better than 500?"},
        ).json()

    assert body["reply"]["status"] == ReplyStatus.REGENERATED.value
    assert "250" not in body["reply"]["text"]
    assert body["send"]["status"] == SendStatus.DRY_RUN.value
    assert len(sender.sent) == 1
    # The reply went out; the claim is untouched by it.
    assert claim.verdict is Verdict.UNVERIFIED
    assert all(e.channel is Channel.AGENT_ASSERTION for e in claim.evidence)


@pytest.mark.asyncio
async def test_inbound_after_a_restart_reports_evidence_pending():
    """No live Claim to promote, so say that rather than guess either way."""
    store = ThreadStore(":memory:")
    routes.set_store(store)
    routes.set_registry(ev.ClaimRegistry())  # empty: the process restarted
    routes.set_sender(NullSender())
    routes.set_responder(None)

    thread = _thread()
    thread.track_claim("claim_ghost", "120 muffins for 420.00", ("120", "420.00"))
    store.save(thread)

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        body = client.post(
            "/messages/inbound/generic",
            json={"from": PHONE, "body": "yes, 120 at 420.00"},
        ).json()
    assert body["evidence_pending"] == ["claim_ghost"]
    assert body["verified_claims"] == []


def test_the_suite_never_writes_to_the_real_evidence_inbox():
    """evidence/ is a judged artifact. Tests must not seed it with fiction."""
    assert routes._inbox_path() != routes.INBOX
    assert "evidence" not in str(routes._inbox_path())


def test_twilio_and_a1mobile_payload_shapes_normalise():
    assert routes._normalise("twilio", {"From": "+1555", "Body": "hi"})["body"] == "hi"
    assert routes._normalise("a1mobile", {"sender": "+1555", "text": "hi"})["from"] == "+1555"
    telnyx = {
        "data": {
            "payload": {
                "from": {"phone_number": "+1555"},
                "to": [{"phone_number": "+1666"}],
                "text": "hi",
            }
        }
    }
    assert routes._normalise("telnyx", telnyx) == {
        "from": "+1555",
        "to": "+1666",
        "body": "hi",
    }
