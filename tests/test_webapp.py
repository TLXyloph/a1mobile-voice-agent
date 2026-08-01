"""What the intake app must never do.

Four of these are load-bearing and the rest are scaffolding:

* `test_launch_is_refused_on_an_incomplete_spec` - the disabled button in the
  browser is a courtesy. If this goes red, a half-filled brief can dial a real
  stranger, and whatever the agent then agrees to was never authorised.

* `test_no_pricing_task_does_not_fabricate_costs` - an errand has no materials
  cost. A model asked for JSON will happily supply one, and that number becomes
  a real floor on a real call. `to_cost_model()` must answer None rather than
  zeros dressed up as data.

* `test_units_versus_headcount_is_asked_explicitly` - the $74 call. A headcount
  and an item count are both integers by the time they reach a tool, so the
  distinction has to be asked out loud during intake, not inferred later.

* `test_interview_terminates_even_when_the_model_is_useless` - an interview is
  a loop with a language model in it. Termination has to be a property of the
  code, not a hope about the prompt.

`test_launched_session_forbids_escalation` is the fifth: nothing in this app may
build a session that can hand a decision to a human mid-call.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from src.business.campaign import Campaign, CloseCondition, Envelope
from src.business.pricing import CostModel
from src.webapp import intake
from src.webapp.app import app, reset, use_dialer, use_handoff, use_responder
from src.webapp.dialer import FakeDialer
from src.webapp.intake import MAX_TURNS, ScriptedResponder, advance
from src.webapp.spec import ERRAND, SALE, Economics, TaskSpec

TODAY = date.today()
SOON = (TODAY + timedelta(days=7)).isoformat()
LATER = (TODAY + timedelta(days=90)).isoformat()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def bakery_patch() -> dict:
    """A complete catering sale, as the intake agent would have patched it in."""
    return {
        "kind": SALE,
        "objective": "Sell a standing weekly breakfast catering order",
        "business_name": "Rosewater Bakehouse",
        "vertical": "restaurant",
        "targets": [{"name": "Marlow Dental", "phone": "+14155550142"}],
        "offer": "A weekly breakfast delivery at a fixed per-head price",
        "unit_label": "pastries",
        "units_basis": "2 pastries per person",
        "capacity_total": 600,
        "economics": {
            "materials_per_unit": 0.80,
            "labor_per_unit": 0.40,
            "transport_per_unit": 0.15,
            "min_margin_pct": 30,
            "target_margin_pct": 45,
        },
        "max_discount_pct": 15,
        "earliest_date": SOON,
        "latest_date": LATER,
        "max_qty": 400,
        "close_condition": "written_confirmation",
        "done_definition": "They text the order back with quantity and total",
        "confirm_to": "+14155550111",
    }


def dentist_patch() -> dict:
    """A booking errand. No unit economics exist for this job, and that is fine."""
    return {
        "kind": ERRAND,
        "objective": "Book the earliest cleaning across three dentists",
        "business_name": "Sam Bandi",
        "targets": [
            {"name": "Marlow Dental", "phone": "+14155550142"},
            {"name": "Bay Family Dental", "phone": "415 555 0188"},
        ],
        "offer": "A new-patient cleaning appointment, earliest available",
        "unit_label": "appointments",
        "units_basis": "one appointment for one person, not one per practice",
        "capacity_total": 1,
        "pricing_note": "Booking errand - nothing is being sold, so no costs apply.",
        "earliest_date": SOON,
        "latest_date": LATER,
        "max_qty": 1,
        "close_condition": "booked_meeting",
        "done_definition": "A confirmed appointment time emailed back",
        "confirm_to": "sam@example.com",
    }


def spec_from(patch: dict) -> TaskSpec:
    s = TaskSpec()
    s.apply(patch)
    return s


@pytest.fixture
def client(tmp_path) -> TestClient:
    reset()
    use_dialer(FakeDialer())
    use_responder(ScriptedResponder(default='{"say":"noted","spec":{}}'))
    # Never let a test run overwrite the operator's live handoff file.
    use_handoff(tmp_path / "webapp.json")
    return TestClient(app)


# ---------------------------------------------------------------------------
# the spec starts empty and stays honest
# ---------------------------------------------------------------------------


def test_a_fresh_spec_is_incomplete_and_cannot_launch() -> None:
    s = TaskSpec()
    assert not s.is_complete
    assert not s.can_launch
    assert s.missing_fields()
    assert s.blockers()


def test_a_complete_sale_can_launch() -> None:
    s = spec_from(bakery_patch())
    assert s.missing_fields() == []
    assert s.is_complete
    assert s.can_launch


def test_a_complete_errand_can_launch_without_any_pricing() -> None:
    s = spec_from(dentist_patch())
    assert s.missing_fields() == []
    assert s.can_launch
    assert "economics" not in {f.id for f in s.required_fields()}


def test_a_described_target_with_no_number_blocks_launch() -> None:
    patch = bakery_patch() | {"targets": [{"name": "The dojo on Clement",
                                           "find_hint": "search maps"}]}
    s = spec_from(patch)
    assert s.is_complete          # the brief is filled in ...
    assert not s.can_launch       # ... but a description cannot be dialled
    assert any("dial" in b for b in s.blockers())


def test_garbage_values_are_dropped_not_coerced_to_zero() -> None:
    s = TaskSpec()
    s.apply({"capacity_total": "quite a lot", "max_discount_pct": "some"})
    assert s.capacity_total is None
    assert s.max_discount_pct is None
    assert "capacity_total" in s.missing_fields()


def test_inverted_date_window_is_a_problem_not_a_launchable_spec() -> None:
    s = spec_from(bakery_patch() | {"earliest_date": LATER, "latest_date": SOON})
    assert not s.is_complete
    assert any("inverted" in p for p in s.problems())


# ---------------------------------------------------------------------------
# no fabricated economics
# ---------------------------------------------------------------------------


def test_no_pricing_task_does_not_fabricate_costs() -> None:
    s = spec_from(dentist_patch())

    assert s.economics is None
    assert s.to_cost_model() is None, "an errand must not be handed a cost model"
    assert s.to_dict()["has_pricing"] is False
    assert s.to_dict()["economics"] is None

    # And the envelope it does build must carry no pricing authority at all.
    env = s.to_envelope()
    assert env.is_valid
    assert env.max_discount_pct == 0.0
    permitted, reason = env.permits(discount_pct=5.0)
    assert not permitted and "discount" in reason


def test_partial_economics_are_not_completed_by_the_system() -> None:
    s = spec_from(bakery_patch())
    s.economics = Economics(materials_per_unit=0.80)  # only one number known
    assert not s.economics.is_complete
    assert s.to_cost_model() is None
    assert "economics" in s.missing_fields()
    assert not s.can_launch


def test_zero_cost_model_is_only_reachable_through_the_session_builder() -> None:
    """`CallSession` needs a cost model; asking whether one exists must not lie."""
    s = spec_from(dentist_patch())
    session = s.to_call_session()
    assert session.costs.unit_cost == 0
    assert float(session.costs.floor_price(1)) == 0.0
    assert s.to_cost_model() is None  # the question still answers honestly


# ---------------------------------------------------------------------------
# units versus headcount
# ---------------------------------------------------------------------------


def test_units_versus_headcount_is_asked_explicitly() -> None:
    s = spec_from({k: v for k, v in bakery_patch().items() if k != "units_basis"})
    assert "units_basis" in s.missing_fields()

    fid, question = intake.next_question(s)
    assert fid == "units_basis"
    low = question.lower()
    assert "people" in low and "items" in low
    assert "per person" in low


def test_the_headcount_rule_is_in_the_system_prompt() -> None:
    prompt = intake.build_system_prompt(TaskSpec()).lower()
    assert "units are not headcount" in prompt
    assert "thirty people" in prompt
    assert "three dentists" in prompt


def test_the_errand_phrasing_keeps_the_same_distinction() -> None:
    s = spec_from({k: v for k, v in dentist_patch().items() if k != "units_basis"})
    fid, question = intake.next_question(s)
    assert fid == "units_basis"
    assert "people" in question.lower() and "bookings" in question.lower()


def test_units_basis_survives_into_the_campaign_the_agent_reads() -> None:
    campaign = spec_from(bakery_patch()).to_campaign()
    assert "2 pastries per person" in campaign.notes


# ---------------------------------------------------------------------------
# the interview terminates
# ---------------------------------------------------------------------------


def _filling_responder(patches: list[dict]) -> ScriptedResponder:
    return ScriptedResponder(
        replies=[json.dumps({"say": f"ok {i}", "spec": p})
                 for i, p in enumerate(patches)],
        default='{"say":"anything else?","spec":{}}',
    )


@pytest.mark.asyncio
async def test_the_interview_terminates_with_a_complete_spec() -> None:
    patch = bakery_patch()
    # One field per turn, which is the slowest an honest interview can go.
    steps = [{k: v} for k, v in patch.items()]
    convo = intake.start()
    responder = _filling_responder(steps)

    turns = 0
    while not convo.finished and turns < MAX_TURNS + 5:
        await advance(convo, "here you go", responder)
        turns += 1

    assert convo.finished
    assert not convo.stalled
    assert convo.spec.is_complete
    assert turns <= len(steps) + 2, "the interview should not wander"


@pytest.mark.asyncio
async def test_interview_terminates_even_when_the_model_is_useless() -> None:
    """A model that never extracts anything must still stop, and say so."""
    convo = intake.start()
    responder = ScriptedResponder(default="I'm afraid I can't help with that.")

    turns = 0
    while not convo.finished and turns < MAX_TURNS * 2:
        await advance(convo, "the bakery thing", responder)
        turns += 1

    assert convo.finished
    assert convo.stalled
    assert not convo.spec.is_complete
    assert turns <= MAX_TURNS + 1
    assert "still open" in convo.messages[-1].text.lower()


@pytest.mark.asyncio
async def test_a_model_failure_falls_back_to_the_field_list() -> None:
    class Broken:
        async def respond(self, system, messages):
            raise RuntimeError("502 from the gateway")

    convo = intake.start()
    result = await advance(convo, "I run a bakery", Broken())
    assert not result.used_model
    assert result.reply == intake.next_question(convo.spec)[1]


@pytest.mark.asyncio
async def test_prose_replies_never_move_the_spec() -> None:
    convo = intake.start()
    await advance(convo, "hi", ScriptedResponder(default="Sure, what's the job?"))
    assert convo.spec.missing_fields() == TaskSpec().missing_fields()


@pytest.mark.asyncio
async def test_fenced_json_is_parsed() -> None:
    reply = '```json\n{"say":"got it","spec":{"business_name":"Rosewater"}}\n```'
    convo = intake.start()
    await advance(convo, "Rosewater Bakehouse", ScriptedResponder(replies=[reply]))
    assert convo.spec.business_name == "Rosewater"


# ---------------------------------------------------------------------------
# conversion produces objects the existing engine accepts
# ---------------------------------------------------------------------------


def test_sale_converts_to_valid_campaign_envelope_and_cost_model() -> None:
    s = spec_from(bakery_patch())

    envelope = s.to_envelope()
    assert isinstance(envelope, Envelope)
    assert envelope.problems() == []
    assert envelope.max_discount_pct == 15.0
    assert envelope.max_qty == 400

    costs = s.to_cost_model()
    assert isinstance(costs, CostModel)
    assert costs.unit == "pastry", "the cost model prices one unit, not a plural"
    assert float(costs.unit_cost) == pytest.approx(1.35)
    # 30% floor on $1.35 of cost is 1.35 / 0.70, not 1.35 * 1.30.
    assert float(costs.floor_price(1)) == pytest.approx(1.93, abs=0.01)

    campaign = s.to_campaign()
    assert isinstance(campaign, Campaign)
    assert campaign.problems() == []
    assert campaign.is_valid
    assert campaign.capacity_units == "pastries"
    assert campaign.close_condition is CloseCondition.WRITTEN_CONFIRMATION


def test_envelope_floor_comes_from_the_cost_model_not_a_guess() -> None:
    s = spec_from(bakery_patch())
    assert s.to_envelope().min_price == pytest.approx(
        float(s.to_cost_model().floor_price(1))
    )


def test_errand_converts_to_a_valid_campaign_with_no_cost_model() -> None:
    s = spec_from(dentist_patch())
    campaign = s.to_campaign()
    assert campaign.problems() == []
    assert campaign.close_condition is CloseCondition.BOOKED_MEETING
    assert s.to_cost_model() is None
    # Every close channel must still be one the agent cannot talk its way into.
    assert campaign.close_evidence


def test_the_campaign_close_condition_cannot_be_satisfied_by_the_agent() -> None:
    from src.verify.receipts import INDEPENDENT_CHANNELS

    for patch in (bakery_patch(), dentist_patch()):
        campaign = spec_from(patch).to_campaign()
        assert campaign.close_evidence <= INDEPENDENT_CHANNELS


def test_call_session_forbids_escalation() -> None:
    for patch in (bakery_patch(), dentist_patch()):
        session = spec_from(patch).to_call_session()
        assert session.allow_escalation is False
        assert session.ledger.available() > 0
        assert session.receipt.claims == []
        assert session.receipt.headline.startswith("NO CLAIMS")


@pytest.mark.asyncio
async def test_the_no_operator_channel_answers_no() -> None:
    session = spec_from(bakery_patch()).to_call_session()
    assert await session.operator.ask("can I go 40% off?") is None


# ---------------------------------------------------------------------------
# the HTTP surface
# ---------------------------------------------------------------------------


def test_launch_is_refused_on_an_incomplete_spec(client: TestClient) -> None:
    sid = client.post("/api/session").json()["id"]
    dialer = FakeDialer()
    use_dialer(dialer)

    res = client.post("/launch", json={"session_id": sid})

    assert res.status_code == 409
    assert res.json()["blockers"]
    assert dialer.calls == [], "an incomplete brief must not ring anybody"


def test_launch_is_refused_when_no_target_is_dialable(client: TestClient) -> None:
    sid = client.post("/api/session").json()["id"]
    patch = bakery_patch() | {"targets": [{"name": "the dojo", "find_hint": "maps"}]}
    client.post("/api/spec", json={"session_id": sid, "patch": patch})
    dialer = FakeDialer()
    use_dialer(dialer)

    res = client.post("/launch", json={"session_id": sid})
    assert res.status_code == 409
    assert dialer.calls == []


def test_launch_places_the_call_and_returns_a_room(client: TestClient) -> None:
    sid = client.post("/api/session").json()["id"]
    state = client.post(
        "/api/spec", json={"session_id": sid, "patch": bakery_patch()}
    ).json()
    assert state["spec"]["can_launch"] is True

    dialer = FakeDialer()
    use_dialer(dialer)
    res = client.post("/launch", json={"session_id": sid})

    assert res.status_code == 200
    body = res.json()
    assert body["room"].startswith("call-")
    assert body["dial"]["ok"] is True
    assert len(dialer.calls) == 1
    assert dialer.calls[0]["to"] == "+14155550142"
    assert dialer.calls[0]["room"] == body["room"]

    # The whole brief travels with the dispatch, not just a task string.
    meta = json.loads(dialer.calls[0]["metadata"])
    assert meta["units_basis"] == "2 pastries per person"
    assert meta["objective"] == bakery_patch()["objective"]


def test_launch_result_is_pollable(client: TestClient) -> None:
    sid = client.post("/api/session").json()["id"]
    client.post("/api/spec", json={"session_id": sid, "patch": bakery_patch()})
    use_dialer(FakeDialer())
    room = client.post("/launch", json={"session_id": sid}).json()["room"]

    poll = client.get(f"/api/call/{room}")
    assert poll.status_code == 200
    body = poll.json()
    assert body["allow_escalation"] is False
    assert body["receipt"] is None, "no receipt is honest; a synthesised one is not"


def test_a_brief_can_only_be_launched_once(client: TestClient) -> None:
    sid = client.post("/api/session").json()["id"]
    client.post("/api/spec", json={"session_id": sid, "patch": bakery_patch()})
    dialer = FakeDialer()
    use_dialer(dialer)

    assert client.post("/launch", json={"session_id": sid}).status_code == 200
    second = client.post("/launch", json={"session_id": sid})

    assert second.status_code == 409
    assert len(dialer.calls) == 1, "a retried launch must not ring a second time"


def test_launch_writes_the_brief_where_the_call_worker_can_read_it(
    client: TestClient, tmp_path
) -> None:
    handoff = tmp_path / "handoff.json"
    use_handoff(handoff)
    sid = client.post("/api/session").json()["id"]
    client.post("/api/spec", json={"session_id": sid, "patch": bakery_patch()})
    use_dialer(FakeDialer())

    client.post("/launch", json={"session_id": sid})

    written = json.loads(handoff.read_text())["active"]
    assert written["allow_escalation"] is False
    assert written["campaign"]["capacity_units"] == "pastries"
    assert written["worker_env"]["MIN_MARGIN_PCT"] == "30.0"


def test_launch_on_an_unknown_session_is_404(client: TestClient) -> None:
    assert client.post("/launch", json={"session_id": "nope"}).status_code == 404


def test_a_failed_dial_is_reported_as_a_failure(client: TestClient) -> None:
    sid = client.post("/api/session").json()["id"]
    client.post("/api/spec", json={"session_id": sid, "patch": bakery_patch()})
    use_dialer(FakeDialer(answered=False, detail="no answer"))

    res = client.post("/launch", json={"session_id": sid})
    assert res.status_code == 502
    assert res.json()["dial"]["ok"] is False


def test_message_round_trip_updates_the_panel(client: TestClient) -> None:
    sid = client.post("/api/session").json()["id"]
    use_responder(ScriptedResponder(
        replies=[json.dumps({"say": "Got it.",
                             "spec": {"business_name": "Rosewater Bakehouse"}})],
        default='{"say":"and then?","spec":{}}',
    ))
    body = client.post(
        "/api/message", json={"session_id": sid, "text": "I run a bakery"}
    ).json()

    assert body["spec"]["business_name"] == "Rosewater Bakehouse"
    assert "business_name" in body["changed"]
    assert body["messages"][-1]["text"] == "Got it."


def test_empty_messages_are_rejected(client: TestClient) -> None:
    sid = client.post("/api/session").json()["id"]
    assert client.post(
        "/api/message", json={"session_id": sid, "text": "   "}
    ).status_code == 400


def test_the_page_is_self_contained(client: TestClient) -> None:
    """Conference wifi. Nothing may be fetched from another host.

    Checked on the attributes that actually cause a request, not on raw
    substrings: an SVG namespace looks like a URL and fetches nothing, while a
    bare `<script src>` fetches something and would blank the page.
    """
    html = client.get("/").text
    assert "<title>" in html
    refs = re.findall(r'(?:src|href|action)\s*=\s*"([^"]*)"', html)
    assert refs, "expected at least the favicon reference"
    for ref in refs:
        assert not ref.startswith(("http://", "https://", "//")), (
            f"{ref} would need the network to render"
        )
    assert "cdn." not in html
    assert "@import" not in html


def test_the_page_never_promises_a_human_mid_call(client: TestClient) -> None:
    html = client.get("/").text.lower()
    assert "no escalation" in html
    for forbidden in ("ask the owner", "we'll check with", "escalate to a human",
                      "a human will"):
        assert forbidden not in html


def test_the_field_registry_is_served_for_the_panel(client: TestClient) -> None:
    body = client.get("/api/fields").json()
    ids = {f["id"] for f in body["fields"]}
    assert "units_basis" in ids
    assert {g["id"] for g in body["groups"]} >= {"economics", "limits", "done"}


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json()["ok"] is True
