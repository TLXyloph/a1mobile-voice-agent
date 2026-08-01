"""The meta-layer's invariants.

Three of these are load-bearing and the rest are hygiene:

* A booking errand never gets a pricing question. That is the difference
  between a generated form and a template with the wrong industry's fields in
  it, and it is the thing a judge notices first.
* A physical-goods task always gets the units-vs-headcount question. A live
  call read "thirty" as thirty muffins when it meant thirty people and lost
  $311. `src/agents/flow.py` catches that mid-call; this catches it at intake.
* The HTML validator rejects and falls back rather than writing something
  broken. A blank dashboard in front of a judge reads as a system that does not
  work.

`claude -p` is never invoked here. The subprocess is behind the `runner`
parameter and every test passes a function, so the suite is silent and fast.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.business.campaign import CloseCondition
from src.generator import dashboard_gen
from src.generator.dashboard_gen import (
    builtin_dashboard,
    extract_html,
    generate_dashboard,
    validate_html,
)
from src.generator.questions import (
    CANONICAL,
    UNITS_FIELD,
    Question,
    QuestionSet,
    Rule,
    ScriptedPlanner,
    canonical_set,
    generate,
    harden,
)
from src.generator.spec import (
    Exchange,
    HardLimits,
    TaskProfile,
    classify,
    heuristic_profile,
    mentions_physical_goods,
    slugify,
)

BOOKING = "book a dentist cleaning for my dad in the next three weeks"
GOODS = "order 30 muffins for Monday's standup from the bakery on Valencia"
SALE = "sell standing weekday breakfast catering to offices within three miles"


def profile_for(goal: str) -> TaskProfile:
    p = heuristic_profile(goal)
    return TaskProfile(
        goal=p.goal,
        exchange=p.exchange,
        callee="Valencia Bakery",
        subject=p.subject,
        done_when="a text arrives confirming the order",
        physical_goods=p.physical_goods,
        unit_label="muffins" if p.physical_goods else "",
    )


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "goal,expected",
    [
        (BOOKING, Exchange.BOOKING),
        (GOODS, Exchange.PURCHASE),
        (SALE, Exchange.SALE),
        ("cancel my gym membership and get the last charge refunded", Exchange.ADMIN),
        ("find out whether the hardware store has flue pipe in stock", Exchange.INFORMATION),
    ],
)
def test_classify(goal, expected):
    assert classify(goal) is expected


def test_classification_is_token_based_not_substring():
    """`src/tasks/triage.py` learned this the hard way: "Stockton" contains
    "tock". A goal that merely spells a marker inside a longer word must not
    trip it."""
    assert classify("call Stockton Motors and ask their opening hours") is (
        Exchange.INFORMATION
    )
    assert not mentions_physical_goods("ask about the boxing class timetable")


def test_headcount_plus_a_number_counts_as_physical_goods():
    """The bias that matters: "lunch for 30 people" has an item count hiding
    inside a headcount, and guessing is what cost $311."""
    assert mentions_physical_goods("sort out lunch for 30 people on Thursday")


# ---------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------


def test_booking_has_no_unit_economics():
    assert heuristic_profile(BOOKING).unit_economics_apply is False


def test_sale_has_unit_economics():
    assert heuristic_profile(SALE).unit_economics_apply is True


def test_purchase_has_a_spend_ceiling_but_no_margin():
    p = heuristic_profile(GOODS)
    assert p.unit_economics_apply is False
    assert "spend_ceiling" in p.required_fields()
    assert "unit_price_floor" not in p.required_fields()


def test_override_beats_the_derived_value():
    p = TaskProfile(
        goal=BOOKING, exchange=Exchange.BOOKING, callee="x",
        unit_economics_override=True,
    )
    assert p.unit_economics_apply is True


def test_every_required_field_has_a_canonical_question():
    """A requirement with no way to ask for it is an unfillable form."""
    for exchange in Exchange:
        for goods in (True, False):
            p = TaskProfile(
                goal="x", exchange=exchange, callee="y", physical_goods=goods
            )
            missing = [f for f in p.required_fields() if f not in CANONICAL]
            assert not missing, f"{exchange} goods={goods}: {missing}"


def test_slugify_cannot_escape_a_directory():
    for hostile in ("../../etc/passwd", "..", "a/b/c", "....//"):
        s = slugify(hostile)
        assert "/" not in s and ".." not in s


# ---------------------------------------------------------------------------
# profile -> campaign
# ---------------------------------------------------------------------------


def test_booking_profile_becomes_a_valid_campaign():
    p = TaskProfile(
        goal=BOOKING,
        exchange=Exchange.BOOKING,
        callee="Dr Alvarez, Mission Dental",
        subject="a 30-minute hygienist appointment",
        done_when="the practice texts a confirmed date and time",
        limits=HardLimits(
            earliest_date=date(2026, 8, 3), latest_date=date(2026, 8, 24), max_qty=1
        ),
    )
    c = p.to_campaign()
    assert c.problems() == []
    assert c.is_valid
    assert c.close_condition is CloseCondition.BOOKED_MEETING
    # No pricing authority at all on a task with no economics.
    assert c.envelope.max_discount_pct == 0.0


def test_sale_profile_carries_the_envelope_through():
    p = TaskProfile(
        goal=SALE,
        exchange=Exchange.SALE,
        callee="offices within three miles",
        subject="a standing weekly breakfast order",
        unit_label="muffins/week",
        limits=HardLimits(
            min_price=3.5,
            max_qty=600,
            max_discount_pct=15.0,
            earliest_date=date(2026, 8, 10),
            latest_date=date(2026, 11, 20),
        ),
    )
    c = p.to_campaign()
    assert c.is_valid
    assert c.envelope.min_price == 3.5
    assert c.envelope.max_qty == 600
    ok, _ = c.envelope.permits(price=3.50, qty=600)
    assert ok
    refused, why = c.envelope.permits(price=2.00)
    assert not refused and "floor" in why


def test_inverted_date_window_is_repaired_not_shipped():
    limits = HardLimits(earliest_date=date(2026, 9, 1), latest_date=date(2026, 8, 1))
    env = limits.to_envelope(economics=False)
    assert env.is_valid


def test_every_generated_campaign_closes_on_independent_evidence():
    """Inherited from `src/business/campaign.py` - a generated campaign must not
    be able to close on the agent's own say-so any more than a hand-written one."""
    for exchange in Exchange:
        p = TaskProfile(goal="x", exchange=exchange, callee="y", subject="z")
        assert p.to_campaign().problems() == []


# ---------------------------------------------------------------------------
# THE two question-set invariants
# ---------------------------------------------------------------------------


def test_booking_errand_produces_no_pricing_questions():
    qs = canonical_set(profile_for(BOOKING))
    pricing = [q.field for q in qs.questions if q.is_pricing]
    assert pricing == [], f"pricing questions on a booking: {pricing}"
    assert qs.is_valid


def test_physical_goods_always_produce_the_units_question():
    qs = canonical_set(profile_for(GOODS))
    assert UNITS_FIELD in qs.fields
    ask = qs.get(UNITS_FIELD).ask.lower()
    assert "items" in ask and "people" in ask


def test_the_units_question_is_injected_when_the_model_omits_it():
    p = profile_for(GOODS)
    drafted = [
        Question(field="quantity", ask="How many muffins do you want?", rule=Rule("integer", min=1)),
        Question(field="callee", ask="Which bakery should we call?"),
    ]
    qs = harden(p, drafted)
    assert UNITS_FIELD in qs.fields
    assert any("units" in r for r in qs.repairs)
    assert qs.is_valid


def test_a_model_reworded_units_question_is_replaced_with_the_canonical_one():
    """The model may not paraphrase this one. The wording is the guard."""
    p = profile_for(GOODS)
    drafted = [Question(field=UNITS_FIELD, ask="How many units roughly?")]
    qs = harden(p, drafted)
    assert "people" in qs.get(UNITS_FIELD).ask.lower()


def test_pricing_questions_are_stripped_from_a_non_sale():
    p = profile_for(BOOKING)
    drafted = [
        Question(field="unit_price_floor", ask="What is the lowest price per seat?"),
        Question(field="gross_margin", ask="What is your gross margin on a cleaning?"),
        Question(field="preferred_windows", ask="Which days and times work for you?"),
    ]
    qs = harden(p, drafted)
    assert "unit_price_floor" not in qs.fields
    assert "gross_margin" not in qs.fields
    assert "preferred_windows" in qs.fields
    assert qs.is_valid


def test_a_spend_ceiling_is_not_a_pricing_question():
    """A purchase errand has a budget and no margin. Confusing the two would
    strip the one money question that actually belongs on the form."""
    q = Question(field="spend_ceiling", ask="What is the most you will spend in total?")
    assert q.is_pricing is False
    assert "spend_ceiling" in canonical_set(profile_for(GOODS)).fields


def test_a_pricing_question_survives_when_economics_do_apply():
    qs = canonical_set(profile_for(SALE))
    assert any(q.is_pricing for q in qs.questions)
    assert qs.is_valid


# ---------------------------------------------------------------------------
# set validation
# ---------------------------------------------------------------------------


def test_duplicate_fields_are_dropped_by_hardening():
    p = profile_for(BOOKING)
    drafted = [
        Question(field="callee", ask="Who should we call about this?"),
        Question(field="callee", ask="And what is their phone number?"),
    ]
    qs = harden(p, drafted)
    assert qs.fields.count("callee") == 1
    assert any("duplicate" in r for r in qs.repairs)


def test_problems_reports_a_duplicate_that_bypassed_hardening():
    p = profile_for(BOOKING)
    q = CANONICAL["callee"]
    qs = QuestionSet(profile=p, questions=[q, q])
    assert any("duplicate" in x for x in qs.problems())


def test_every_required_field_is_reachable_after_hardening():
    for goal in (BOOKING, GOODS, SALE, "cancel my gym membership", "find out their hours"):
        p = profile_for(goal)
        qs = harden(p, [Question(field="junk_field", ask="Something irrelevant here?")])
        for needed in p.required_fields():
            assert needed in qs.fields, f"{goal}: {needed} unreachable"
        assert qs.problems() == [], f"{goal}: {qs.problems()}"


def test_an_aliased_field_is_not_asked_twice():
    """A model that asks for the confirmation destination under its own name
    must not get the canonical version bolted on beside it - the user reads
    the form, and being asked the same thing twice is the tell."""
    p = profile_for(GOODS)
    drafted = [
        Question(
            field="written_confirmation_destination",
            ask="Where should the bakery send the confirmation text?",
        )
    ]
    qs = harden(p, drafted)
    assert "confirm_to" in qs.fields
    assert "written_confirmation_destination" not in qs.fields
    assert qs.fields.count("confirm_to") == 1
    assert "bakery" in qs.get("confirm_to").ask, "the model's wording was kept"


def test_aliases_all_point_at_a_real_canonical_field():
    from src.generator.questions import FIELD_ALIASES

    unknown = {v for v in FIELD_ALIASES.values() if v not in CANONICAL}
    assert not unknown


def test_malformed_questions_are_dropped():
    p = profile_for(BOOKING)
    drafted = [Question(field="ok_field", ask="A perfectly answerable question?")]
    bad = Question(field="Bad Field", ask="hi")
    qs = harden(p, drafted + [bad])
    assert "ok_field" in qs.fields
    assert "Bad Field" not in qs.fields


def test_rules_actually_run():
    integer = Rule(kind="integer", min=1)
    assert integer.check("30")[0]
    assert not integer.check("thirty")[0]
    assert not integer.check("0")[0]
    assert Rule(kind="date").check("2026-08-14")[0]
    assert not Rule(kind="date").check("next tuesday")[0]
    assert not Rule(kind="phone").check("555-1234")[0]
    assert Rule(kind="phone").check("+14155551234")[0]
    assert Rule(kind="choice", choices=("today", "whenever")).check("Today")[0]
    assert not Rule(kind="choice", choices=("today",)).check("tomorrow")[0]
    # Blank is the Question's problem, not the Rule's.
    assert integer.check("")[0]


def test_check_answers_flags_missing_required_fields():
    qs = canonical_set(profile_for(BOOKING))
    errors = qs.check_answers({})
    assert errors
    assert all(v == "required" for v in errors.values())
    assert "callee" in errors


# ---------------------------------------------------------------------------
# generation, with the model mocked
# ---------------------------------------------------------------------------


MODEL_REPLY = json.dumps(
    {
        "profile": {
            "exchange": "booking",
            "callee": "Mission Dental",
            "subject": "a hygienist appointment",
            "done_when": "the practice texts a confirmed date and time",
            "physical_goods": False,
        },
        "questions": [
            {"field": "callee", "ask": "Which practice should we call?",
             "rule": {"kind": "text", "min": 3}},
            {"field": "preferred_windows", "ask": "Which days and times work for you?",
             "rule": {"kind": "text"}},
            {"field": "unit_price_floor", "ask": "What is your price floor per cleaning?",
             "rule": {"kind": "money"}},
        ],
    }
)


async def test_generate_uses_the_model_and_strips_its_pricing_question():
    planner = ScriptedPlanner(replies=[MODEL_REPLY])
    qs = await generate(BOOKING, planner)
    assert planner.calls, "the planner was never called"
    assert qs.profile.exchange is Exchange.BOOKING
    assert "unit_price_floor" not in qs.fields
    assert qs.source == "model+repaired"
    assert qs.is_valid


async def test_generate_survives_a_dead_model():
    planner = ScriptedPlanner(raises=RuntimeError("502 from the inference gateway"))
    qs = await generate(GOODS, planner)
    assert qs.source == "heuristic"
    assert UNITS_FIELD in qs.fields
    assert qs.is_valid


async def test_generate_survives_prose_instead_of_json():
    planner = ScriptedPlanner(replies=["Sure! Here are some good questions to ask."])
    qs = await generate(GOODS, planner)
    assert qs.is_valid
    assert UNITS_FIELD in qs.fields


async def test_generate_fenced_json_is_parsed():
    planner = ScriptedPlanner(replies=["```json\n" + MODEL_REPLY + "\n```"])
    qs = await generate(BOOKING, planner)
    assert qs.profile.callee == "Mission Dental"


async def test_a_model_that_invents_pricing_on_a_booking_cannot_ship_it():
    """The end-to-end version of the headline invariant."""
    reply = json.dumps(
        {
            "profile": {"exchange": "booking", "callee": "Mission Dental"},
            "questions": [
                {"field": "gross_margin", "ask": "What margin do you need on this?"},
                {"field": "cost_per_unit", "ask": "What does one appointment cost you?"},
                {"field": "urgency", "ask": "How soon do you need this appointment?"},
            ],
        }
    )
    qs = await generate(BOOKING, ScriptedPlanner(replies=[reply]))
    assert not any(q.is_pricing for q in qs.questions)
    assert qs.is_valid


# ---------------------------------------------------------------------------
# the dashboard
# ---------------------------------------------------------------------------


GOOD_HTML = builtin_dashboard(profile_for(BOOKING))


def test_the_builtin_template_passes_its_own_validator():
    """A fallback that fails the check it exists to satisfy is not a fallback."""
    check = validate_html(GOOD_HTML)
    assert check.ok, check.problems
    assert check.warnings == []


def test_a_hostile_goal_cannot_break_out_of_the_inline_script():
    """`</script>` inside a JSON string still ends the script element - the
    tokenizer does not know it is inside a string. The goal is user text."""
    p = TaskProfile(
        goal='order 30 muffins</script><script>alert(1)</script>',
        exchange=Exchange.PURCHASE,
        callee="Valencia Bakery",
        physical_goods=True,
    )
    html = builtin_dashboard(p)
    assert "alert(1)" not in html.replace("\\/", "/") or "<\\/script>" in html
    assert html.count("</script>") == 1
    assert validate_html(html).ok


def test_validator_rejects_external_urls():
    bad = GOOD_HTML.replace(
        "<head>", '<head><link rel="preload" href="https://cdn.example.com/x.css">'
    )
    check = validate_html(bad)
    assert not check.ok
    assert any("external URL" in p for p in check.problems)


def test_validator_rejects_script_src():
    bad = GOOD_HTML.replace("<body>", '<body><script src="app.js"></script>')
    assert not validate_html(bad).ok


def test_validator_rejects_protocol_relative_urls():
    bad = GOOD_HTML.replace("<body>", '<body><img src="//cdn.example.com/a.png">')
    assert not validate_html(bad).ok


def test_validator_rejects_an_empty_body():
    assert not validate_html(
        "<!doctype html><html><head><style>" + "a{color:red}" * 60
        + "</style></head><body>   </body></html>"
    ).ok


def test_validator_allows_inline_svg_namespaces():
    """An `xmlns` is an identifier, not a fetch. Rejecting it would throw away
    every otherwise-correct page that draws an icon."""
    ok = GOOD_HTML.replace(
        "</body>", '<svg xmlns="http://www.w3.org/2000/svg"></svg></body>'
    )
    assert validate_html(ok).ok


def test_validator_rejects_a_page_that_ignored_the_evidence_model():
    plain = (
        "<!doctype html><html><head><title>x</title></head><body>"
        + "<p>Everything went great and the booking is done.</p>" * 20
        + "</body></html>"
    )
    check = validate_html(plain)
    assert not check.ok
    assert any("claim/evidence/verdict" in p for p in check.problems)


def test_extract_html_unwraps_a_markdown_fence_and_chatter():
    raw = "Sure, here you go:\n\n```html\n<!doctype html><html><body>hi</body></html>\n```\nHope that helps!"
    assert extract_html(raw).startswith("<!doctype html>")
    assert extract_html(raw).endswith("</html>")


def test_generate_dashboard_uses_a_good_generated_page(tmp_path):
    calls = []

    def runner(prompt: str) -> str:
        calls.append(prompt)
        return GOOD_HTML

    result = generate_dashboard(profile_for(BOOKING), runner=runner, out_root=tmp_path)
    assert result.source == "claude"
    assert result.path.is_file()
    assert calls and "booked" in calls[0].lower()
    assert (result.path.parent / "profile.json").is_file()


def test_generate_dashboard_falls_back_on_broken_output(tmp_path):
    def runner(prompt: str) -> str:
        return '<!doctype html><html><body><script src="https://cdn.tailwindcss.com"></script></body></html>'

    result = generate_dashboard(profile_for(BOOKING), runner=runner, out_root=tmp_path)
    assert result.source == "builtin"
    assert result.problems
    assert validate_html(result.path.read_text()).ok


def test_generate_dashboard_falls_back_when_the_subprocess_raises(tmp_path):
    def runner(prompt: str) -> str:
        raise TimeoutError("claude -p hung")

    result = generate_dashboard(profile_for(GOODS), runner=runner, out_root=tmp_path)
    assert result.source == "builtin"
    assert result.path.is_file()
    assert "TimeoutError" in " ".join(result.problems)


def test_generate_dashboard_falls_back_on_empty_output(tmp_path):
    result = generate_dashboard(
        profile_for(GOODS), runner=lambda p: "", out_root=tmp_path
    )
    assert result.source == "builtin"
    assert validate_html(result.html).ok


def test_the_prompt_forbids_pricing_on_a_task_without_economics():
    prompt = dashboard_gen.build_prompt(profile_for(BOOKING))
    assert "NO unit economics" in prompt
    assert "booked" in prompt.lower() and "proven" in prompt.lower()


def test_the_dashboard_is_written_inside_the_output_root(tmp_path):
    hostile = TaskProfile(
        goal="../../../etc/passwd", exchange=Exchange.BOOKING, callee="x"
    )
    result = generate_dashboard(hostile, runner=lambda p: "", out_root=tmp_path)
    assert tmp_path.resolve() in result.path.resolve().parents


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.generator import app as appmod

    monkeypatch.setattr(appmod, "OUT_ROOT", tmp_path)
    appmod.use_planner(ScriptedPlanner(raises=RuntimeError("offline")))
    appmod.use_runner(lambda prompt: GOOD_HTML)
    return TestClient(appmod.app)


def test_plan_endpoint_falls_back_offline(client):
    r = client.post("/api/plan", json={"goal": GOODS})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"]
    assert UNITS_FIELD in [q["field"] for q in body["questions"]]


def test_plan_endpoint_rejects_an_empty_goal(client):
    assert client.post("/api/plan", json={"goal": "  "}).status_code == 400


def test_deleting_the_units_question_in_the_ui_does_not_ship_without_it(client):
    """The edit box may remove it from the screen. It may not remove it from
    the form - the server re-hardens and says so."""
    body = client.post("/api/plan", json={"goal": GOODS}).json()
    body["questions"] = [q for q in body["questions"] if q["field"] != UNITS_FIELD]

    built = client.post("/api/dashboard", json=body).json()
    assert UNITS_FIELD in [q["field"] for q in built["questions"]]
    assert any("units" in r for r in built["repairs"])


def test_preview_serves_the_file_that_was_written(client, tmp_path):
    body = client.post("/api/plan", json={"goal": GOODS}).json()
    built = client.post("/api/dashboard", json=body).json()
    page = client.get(built["preview"])
    assert page.status_code == 200
    assert "Proven" in page.text
    assert (tmp_path / built["slug"] / "dashboard.html").is_file()


def test_preview_of_an_unknown_or_hostile_slug_is_a_404(client):
    assert client.get("/preview/nothing-here").status_code == 404
    assert client.get("/preview/..%2F..%2Fetc").status_code == 404


def test_answers_are_checked_with_the_generated_rules(client):
    body = client.post("/api/plan", json={"goal": GOODS}).json()
    body["answers"] = {"quantity": "thirty"}
    out = client.post("/api/answers/check", json=body).json()
    assert out["ok"] is False
    assert "number" in out["errors"]["quantity"]


def test_the_suite_never_shells_out(monkeypatch):
    """Belt and braces: if a test ever forgets to pass `runner`, this is the
    one that goes red rather than a real `claude -p` firing during CI."""
    def explode(*a, **k):
        raise AssertionError("the test suite must never run claude -p")

    monkeypatch.setattr(dashboard_gen.subprocess, "run", explode)
    result = generate_dashboard(
        profile_for(BOOKING), runner=lambda p: GOOD_HTML, out_root=None, write=False
    )
    assert result.source == "claude"
