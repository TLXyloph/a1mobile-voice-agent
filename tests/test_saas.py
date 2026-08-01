"""The B2B SaaS vertical: subscription economics, the campaign, and the board.

Three properties are pinned here, and they are the three that would make this
vertical dangerous if they broke.

**Payback is a floor in its own right.** A subscription deal can clear a margin
test comfortably and still be a bad deal, because software margin is high
almost regardless of what you do to price. `test_clears_margin_but_breaches_payback`
is the case: 87% gross margin, contract value well over the minimum, and
rejected anyway because the CAC takes twenty months to come back.

**Concessions stack.** `test_stacked_concessions_breach_the_floor` gives away
three things, each one inside its own cap and each one individually acceptable
as a whole deal, and the combination is a 25% discount that fails two floors at
once. An agent checking levers one at a time approves it. That test is the
reason `lever_within_cap()` is documented as not being permission.

**A deal is closed by evidence.** `test_cannot_close_on_the_agents_word` is the
same invariant `tests/test_receipts.py` guards, followed into the CRM - which is
where fabricated success actually hides, because nobody re-reads a transcript
but everybody believes a board.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The app binds a pipeline at import. Point it at memory before that happens,
# or importing the test suite writes to the demo database.
os.environ.setdefault("SAAS_DB", ":memory:")

from src.verify.receipts import Channel, Verdict  # noqa: E402
from src.verticals.saas.campaign import (  # noqa: E402
    Decision,
    Outcome,
    build_playbook,
    campaign_for,
    concessions_from,
    envelope_for,
    floor_from_config,
    implied_seat_floor,
    load_config,
    model_from_config,
    model_from_terms,
)
from src.verticals.saas.economics import (  # noqa: E402
    Concession,
    DealFloor,
    DealVerdict,
    Lever,
    SubscriptionModel,
)
from src.verticals.saas.pipeline import (  # noqa: E402
    ALLOWED_TRANSITIONS,
    EvidenceTopic,
    InvalidTransition,
    Pipeline,
    Prospect,
    Stage,
    UnknownProspect,
    UnverifiedClose,
    seed_samples,
)

D = Decimal


# -- fixtures -------------------------------------------------------------


@pytest.fixture
def cfg() -> dict:
    return load_config()


@pytest.fixture
def deal() -> SubscriptionModel:
    """The reference deal. Every number below is hand-checkable from these.

    25 seats x $40 x 12 months = $12,000 of subscription, plus a $2,500
    onboarding fee = $14,500 of contract value. Cost to serve is $6/seat/month,
    so $1,800 over the term. CAC is $9,000.
    """
    return SubscriptionModel(
        price_per_seat_month=40,
        seats=25,
        term_months=12,
        onboarding_fee=2500,
        monthly_cost_to_serve_per_seat=6,
        cac=9000,
    )


@pytest.fixture
def floor() -> DealFloor:
    return DealFloor(
        min_contract_value=11500,
        max_payback_months=12,
        min_gross_margin_pct=70,
        max_discount_pct=10,
        max_free_months=2,
        max_onboarding_waiver=2500,
        min_seats=10,
        min_term_months=12,
    )


@pytest.fixture
def pipe() -> Pipeline:
    p = Pipeline(":memory:")
    yield p
    p.close()


def _prospect(**kw) -> Prospect:
    base = dict(company="Acme Reconciliation", contact="Jo Diaz", seats=20)
    base.update(kw)
    return Prospect(**base)


# -- the arithmetic -------------------------------------------------------


def test_contract_value_is_subscription_plus_onboarding(deal: SubscriptionModel):
    assert deal.subscription_revenue == D("12000")
    assert deal.total_contract_value == D("14500.00")
    assert deal.billable_months == 12


def test_cost_to_serve_and_margin(deal: SubscriptionModel):
    assert deal.cost_to_serve_total == D("1800.00")
    assert deal.gross_profit == D("12700.00")
    # 12700 / 14500 = 87.586..., floored so we never overstate margin.
    assert deal.gross_margin_pct == D("87.58")


def test_payback_accounts_for_the_onboarding_fee(deal: SubscriptionModel):
    """Month one collects the fee, so payback is faster than CAC/monthly profit.

    The closed form would say 9000 / 850 = 10.6 months. The truth is 7.65,
    because $2,500 lands up front. A model that misses this rejects deals that
    are actually fine.
    """
    assert deal.month_gross_profit(1) == D("3350")
    assert deal.month_gross_profit(2) == D("850")
    assert deal.months_to_cac_payback == D("7.65")
    assert deal.pays_back_within_term


def test_free_months_are_served_and_therefore_cost_money(deal: SubscriptionModel):
    two_free = replace(deal, free_months=2)
    assert two_free.billable_months == 10
    assert two_free.subscription_revenue == D("10000")
    # Cost to serve is unchanged: we support them for all twelve months.
    assert two_free.cost_to_serve_total == deal.cost_to_serve_total
    assert two_free.month_gross_profit(2) == D("-150")
    assert two_free.months_to_cac_payback == D("10.00")


def test_onboarding_fee_is_not_folded_into_the_rate(deal: SubscriptionModel):
    """A list-price deal is a 0% discount, not a negative one."""
    assert deal.effective_rate_per_seat_month == D("40.00")
    assert deal.effective_discount_pct == D("0.00")
    # The fee is not lost - it is in the cash view and in contract value.
    assert deal.all_in_rate_per_seat_month == D("48.33")


def test_effective_discount_sees_every_rate_lever(deal: SubscriptionModel):
    stacked = replace(deal, discount_pct=10, free_months=2)
    # 36/seat for 10 of 12 months = 30 effective, against a 40 list rate.
    assert stacked.effective_rate_per_seat_month == D("30.00")
    assert stacked.effective_discount_pct == D("25.00")


def test_shortening_the_term_is_a_concession_too(deal: SubscriptionModel):
    """The quietest lever: no headline number changes and the deal halves."""
    short = deal.with_concessions(Concession(Lever.TERM_REDUCTION, 6))
    assert short.term_months == 6
    assert short.total_contract_value == D("8500.00")


# -- the two floors -------------------------------------------------------


def test_reference_deal_clears(deal: SubscriptionModel, floor: DealFloor):
    check = floor.evaluate(deal)
    assert check.verdict is DealVerdict.CLEARS
    assert check.approved
    assert check.breaches == ()


def test_clears_margin_but_breaches_payback(deal: SubscriptionModel, floor: DealFloor):
    """The case a unit-cost model cannot see.

    Identical deal, more expensive to win. Margin is untouched at 87%, contract
    value is untouched at $14,500, and the deal is still wrong: the money is
    out for twenty months. Margin and payback are different questions and only
    one of them is about time.
    """
    expensive = replace(deal, cac=20000)

    assert expensive.gross_margin_pct == D("87.58")
    assert expensive.gross_margin_pct > floor.min_gross_margin_pct
    assert expensive.total_contract_value > floor.min_contract_value

    check = floor.evaluate(expensive)
    assert check.verdict is DealVerdict.REJECTED
    assert len(check.breaches) == 1
    assert "payback" in check.breaches[0]
    assert "20.59" in check.breaches[0]


def test_a_deal_that_never_pays_back_is_rejected_not_infinite(floor: DealFloor):
    """Cost to serve above the billed rate means payback never arrives."""
    upside_down = SubscriptionModel(
        price_per_seat_month=10,
        seats=25,
        term_months=12,
        monthly_cost_to_serve_per_seat=15,
        cac=5000,
    )
    assert upside_down.months_to_cac_payback is None
    check = floor.evaluate(upside_down)
    assert not check.approved
    assert any("never pays back" in b for b in check.breaches)


# -- the stacking trap ----------------------------------------------------


@pytest.fixture
def three_asks() -> tuple[Concession, Concession, Concession]:
    """Three things a buyer asks for, one at a time, in this order."""
    return (
        Concession(Lever.DISCOUNT_PCT, 10, note="can you do ten percent?"),
        Concession(Lever.FREE_MONTHS, 2, note="throw in the first two months"),
        Concession(Lever.ONBOARDING_WAIVER, 2500, note="and drop the setup fee"),
    )


def test_each_ask_is_individually_permitted(
    deal: SubscriptionModel, floor: DealFloor, three_asks
):
    """Not merely inside its cap - each one is a fine deal *on its own*.

    This is the stronger version of the claim, and it is what makes the next
    test damning. There is no lever here that a careful human would refuse in
    isolation.
    """
    for ask in three_asks:
        within_cap, _ = floor.lever_within_cap(ask)
        assert within_cap, f"{ask.described} should be inside its own cap"

        alone = floor.evaluate_with(deal, [ask])
        assert alone.approved, f"{ask.described} alone should clear: {alone.breaches}"


def test_stacked_concessions_breach_the_floor(
    deal: SubscriptionModel, floor: DealFloor, three_asks
):
    """Three individually-permitted concessions that together must be rejected.

    Ten percent off, two free months, no setup fee. Every answer was inside
    policy. The result is a 25% effective discount, $5,500 of contract value
    gone, and a payback that lands two and a half months past the end of the
    term.

    Note which check does *not* fire: gross margin is still 80%, comfortably
    over the 70% floor. An agent watching margin sees nothing wrong at all.
    """
    combined = floor.evaluate_with(deal, three_asks)

    assert combined.verdict is DealVerdict.REJECTED
    assert combined.model.total_contract_value == D("9000.00")
    assert combined.model.effective_discount_pct == D("25.00")
    assert combined.model.months_to_cac_payback == D("14.40")

    # The margin backstop is silent. Both real floors are not.
    assert combined.model.gross_margin_pct == D("80.00")
    assert combined.model.gross_margin_pct > floor.min_gross_margin_pct
    assert len(combined.breaches) == 2
    assert any("contract value" in b for b in combined.breaches)
    assert any("payback" in b for b in combined.breaches)


def test_the_report_names_the_trap(deal: SubscriptionModel, three_asks):
    """The UI needs 'all levers within cap' and 'rejected' side by side."""
    playbook = build_playbook()
    report = playbook.concession_report(*three_asks)
    assert report["all_levers_within_cap"] is True
    assert report["combined"]["approved"] is False
    assert report["stacking_trap"] is True


def test_order_of_concessions_does_not_change_the_verdict(
    deal: SubscriptionModel, floor: DealFloor, three_asks
):
    """A buyer who asks in a different order must not get a different deal."""
    forwards = floor.evaluate_with(deal, three_asks)
    backwards = floor.evaluate_with(deal, tuple(reversed(three_asks)))
    assert (
        forwards.model.total_contract_value == backwards.model.total_contract_value
    )
    assert forwards.verdict is backwards.verdict


# -- degenerate input -----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"price_per_seat_month": 0, "seats": 0, "term_months": 0},
        {"price_per_seat_month": 40, "seats": 0, "term_months": 12},
        {"price_per_seat_month": 40, "seats": 25, "term_months": 0},
        {"price_per_seat_month": 0, "seats": 25, "term_months": 12},
        {"price_per_seat_month": 40, "seats": 25, "term_months": 3, "free_months": 99},
    ],
    ids=["all-zero", "no-seats", "no-term", "free-product", "free-past-term"],
)
def test_degenerate_deals_do_not_divide_by_zero(kwargs, floor: DealFloor):
    """Half-built deals are a real mid-call state, not a programming error.

    "About twenty seats, I'll get back to you on the term" has to price to
    something rather than raise, so every derived number is defined for a deal
    with a zero in it. They all report badly, which is correct.
    """
    model = SubscriptionModel(cac=9000, monthly_cost_to_serve_per_seat=6, **kwargs)

    assert model.total_contract_value >= 0
    assert model.effective_monthly_rate >= 0
    assert model.effective_rate_per_seat_month >= 0
    assert model.all_in_rate_per_seat_month >= 0
    assert model.cost_to_serve_total >= 0
    assert model.billable_months >= 0
    # None rather than a fabricated number, in both places it can happen.
    assert model.gross_margin_pct is None or model.gross_margin_pct <= 100
    assert model.months_to_cac_payback is None or model.months_to_cac_payback >= 0

    check = floor.evaluate(model)
    assert not check.approved
    assert check.reason  # renders without raising


def test_free_months_beyond_the_term_bill_nothing(floor: DealFloor):
    model = SubscriptionModel(
        price_per_seat_month=40, seats=25, term_months=3, free_months=99
    )
    assert model.billable_months == 0
    assert model.subscription_revenue == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"price_per_seat_month": -1, "seats": 5, "term_months": 12},
        {"price_per_seat_month": 40, "seats": -5, "term_months": 12},
        {"price_per_seat_month": 40, "seats": 5, "term_months": 12, "cac": -1},
        {"price_per_seat_month": 40, "seats": 5, "term_months": 12, "discount_pct": 101},
    ],
)
def test_negative_and_impossible_inputs_are_refused(bad):
    with pytest.raises(ValueError):
        SubscriptionModel(**bad)


def test_a_concession_cannot_be_a_price_rise():
    with pytest.raises(ValueError):
        Concession(Lever.DISCOUNT_PCT, -5)


def test_verdict_is_derived_and_has_no_setter(deal: SubscriptionModel, floor: DealFloor):
    """Same shape as Claim.verdict: nothing an agent can assign."""
    check = floor.evaluate_with(deal, [Concession(Lever.DISCOUNT_PCT, 90)])
    assert check.verdict is DealVerdict.REJECTED
    with pytest.raises((AttributeError, TypeError)):
        check.verdict = DealVerdict.CLEARS  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        check.model = deal  # type: ignore[misc]


# -- the campaign ---------------------------------------------------------


def test_campaign_built_from_a_subscription_model_validates(cfg):
    playbook = build_playbook(cfg)
    assert playbook.problems() == [], playbook.problems()
    assert playbook.is_valid
    assert playbook.campaign.problems() == []
    assert playbook.campaign.envelope.is_valid


def test_envelope_unit_floor_is_derived_from_the_contract_floor(cfg):
    """The per-seat floor is a shadow the contract-value floor casts.

    $11,500 over 25 seats x 12 months is $38.34 a seat-month, rounded up. It is
    not a number anybody chose, which is the point: a unit floor chosen
    independently of contract value is a unit floor that does not bind.
    """
    model, fl = model_from_config(cfg), floor_from_config(cfg)
    assert implied_seat_floor(model, fl) == D("38.34")
    assert playbook_envelope(cfg).min_price == pytest.approx(38.34)


def playbook_envelope(cfg):
    return campaign_for(model_from_config(cfg), floor_from_config(cfg), cfg).envelope


def test_capacity_is_onboarding_bandwidth_not_inventory(cfg):
    campaign = build_playbook(cfg).campaign
    assert campaign.capacity_units == "seats onboarded/month"
    assert campaign.envelope.max_qty == cfg["capacity"]["seats_onboardable_per_month"]


def test_close_condition_only_accepts_independent_evidence(cfg):
    from src.verify.receipts import INDEPENDENT_CHANNELS

    campaign = build_playbook(cfg).campaign
    assert campaign.close_evidence
    assert campaign.close_evidence <= INDEPENDENT_CHANNELS


def test_there_is_no_escalation_outcome(cfg, three_asks):
    """Limits are absolute on this campaign. Three endings, and no fourth.

    Checked structurally rather than by inspection: if somebody adds an
    ESCALATE member to `Outcome` to unblock a demo, this goes red.
    """
    outcomes = {
        v for k, v in vars(Outcome).items() if not k.startswith("_") and isinstance(v, str)
    }
    assert outcomes == {"close", "counter", "walk_away"}

    playbook = build_playbook(cfg)
    decision = playbook.decide(*three_asks)
    assert isinstance(decision, Decision)
    assert decision.outcome in outcomes
    assert not decision.may_close
    assert "check with" not in decision.line.lower()
    assert "ask" not in decision.line.lower()


def test_a_rejected_ask_produces_a_counter_not_a_dead_end(cfg, three_asks):
    """Walking away is the last resort, not the first refusal."""
    playbook = build_playbook(cfg)
    decision = playbook.decide(*three_asks)
    assert decision.outcome == Outcome.COUNTER
    assert decision.check.approved  # the counter itself clears the floor
    assert decision.line


def test_the_agent_closes_when_the_deal_clears(cfg):
    playbook = build_playbook(cfg)
    decision = playbook.decide(Concession(Lever.DISCOUNT_PCT, 5))
    assert decision.outcome == Outcome.CLOSE
    assert decision.may_close


def test_max_permitted_stops_at_the_first_breach(cfg):
    playbook = build_playbook(cfg)
    best = playbook.max_permitted(Lever.DISCOUNT_PCT, step="1")
    assert best == D("10")  # the cap binds before the floor does, here
    assert playbook.floor.evaluate_with(
        playbook.model, [Concession(Lever.DISCOUNT_PCT, best)]
    ).approved


def test_terms_are_read_back_as_the_concessions_that_made_them(cfg):
    proposed = {"discount_pct": 10, "free_months": 2, "onboarding_fee": 0}
    levers = {c.lever for c in concessions_from(proposed, cfg)}
    assert levers == {
        Lever.DISCOUNT_PCT,
        Lever.FREE_MONTHS,
        Lever.ONBOARDING_WAIVER,
    }


def test_partial_terms_fall_back_to_list_not_to_zero(cfg):
    """A blank field must not silently produce a free deal that clears."""
    model = model_from_terms({"seats": 30}, cfg)
    assert model.seats == 30
    assert model.price_per_seat_month == D("40")
    assert model.cac == D("9000")


def test_a_degenerate_model_makes_an_invalid_envelope(cfg):
    empty = SubscriptionModel(price_per_seat_month=0, seats=0, term_months=0)
    envelope = envelope_for(
        empty,
        floor_from_config(cfg),
        seats_onboardable_per_month=10,
        earliest_date=__import__("datetime").date(2026, 8, 1),
        latest_date=__import__("datetime").date(2026, 9, 1),
    )
    assert not envelope.is_valid
    assert any("min_price" in p for p in envelope.problems())


# -- the pipeline ---------------------------------------------------------


def test_stage_transitions_are_a_whitelist(pipe: Pipeline):
    p = pipe.add(_prospect())
    assert p.stage is Stage.TARGETED

    # Skipping a stage is refused even when it plausibly happened on one call.
    with pytest.raises(InvalidTransition):
        pipe.advance(p.id, Stage.DEMO_BOOKED)
    with pytest.raises(InvalidTransition):
        pipe.advance(p.id, Stage.QUALIFIED)

    pipe.advance(p.id, Stage.CONTACTED)
    pipe.advance(p.id, Stage.QUALIFIED)
    assert pipe.get(p.id).stage is Stage.QUALIFIED

    # No going back, and no re-entering the stage you are already in.
    with pytest.raises(InvalidTransition):
        pipe.advance(p.id, Stage.CONTACTED)
    with pytest.raises(InvalidTransition):
        pipe.advance(p.id, Stage.QUALIFIED)


def test_terminal_stages_are_terminal(pipe: Pipeline):
    p = pipe.add(_prospect())
    pipe.advance(p.id, Stage.CLOSED_LOST)
    assert ALLOWED_TRANSITIONS[Stage.CLOSED_LOST] == frozenset()
    for stage in Stage:
        with pytest.raises(InvalidTransition):
            pipe.advance(p.id, stage)


def test_walking_away_never_needs_evidence(pipe: Pipeline):
    """The asymmetry: refusing to claim a win is always allowed."""
    for stage in (Stage.TARGETED, Stage.CONTACTED, Stage.QUALIFIED, Stage.DEMO_BOOKED):
        assert Stage.CLOSED_LOST in ALLOWED_TRANSITIONS[stage]
    p = pipe.add(_prospect(company="Walkaway Inc"))
    pipe.advance(p.id, Stage.CONTACTED)
    pipe.advance(p.id, Stage.CLOSED_LOST, detail="wanted 40% off")
    assert pipe.get(p.id).stage is Stage.CLOSED_LOST


def _to_demo_booked(pipe: Pipeline, **kw) -> Prospect:
    p = pipe.add(_prospect(**kw))
    pipe.advance(p.id, Stage.CONTACTED)
    pipe.advance(p.id, Stage.QUALIFIED)
    pipe.advance(p.id, Stage.DEMO_BOOKED)
    return p


def test_cannot_close_on_the_agents_word(pipe: Pipeline):
    """The disqualification condition, caught where it would enter the system.

    The agent is as emphatic as it likes. Volume is not evidence, so the card
    does not move.
    """
    p = _to_demo_booked(pipe)
    for i in range(5):
        pipe.record_evidence(
            p.id, Channel.AGENT_ASSERTION, f"they definitely said yes ({i})"
        )

    assert pipe.close_verdict(p.id) is Verdict.UNVERIFIED
    can, why = pipe.can_close(p.id)
    assert not can
    assert "not enough" in why

    with pytest.raises(UnverifiedClose):
        pipe.advance(p.id, Stage.CLOSED_WON)
    assert pipe.get(p.id).stage is Stage.DEMO_BOOKED


def test_independent_evidence_closes_the_deal(pipe: Pipeline):
    p = _to_demo_booked(pipe)
    pipe.record_evidence(p.id, Channel.AGENT_ASSERTION, "they said yes")
    pipe.record_evidence(
        p.id,
        Channel.INBOUND_EMAIL,
        "countersigned order form, 20 seats, 12 months",
        raw={"from": "jo@acme.example", "subject": "signed"},
    )
    assert pipe.close_verdict(p.id) is Verdict.VERIFIED
    assert pipe.can_close(p.id)[0]

    pipe.advance(p.id, Stage.CLOSED_WON, detail="order form received")
    assert pipe.get(p.id).stage is Stage.CLOSED_WON


def test_a_calendar_event_does_not_prove_a_signature(pipe: Pipeline):
    """Evidence is scoped to what it is evidence *of*.

    A Google Calendar event is independent, supporting, and from a channel we
    do not control - every property the close gate looks for. It proves a
    meeting. Unscoped, it reads as VERIFIED and a prospect who agreed to a demo
    appears on the board as a signed customer, which is fabrication by
    bookkeeping rather than by transcript.
    """
    p = _to_demo_booked(pipe)
    pipe.record_evidence(
        p.id,
        Channel.PROVIDER_API,
        "Calendar event created, both attendees accepted",
        about=EvidenceTopic.MEETING,
    )
    assert pipe.close_verdict(p.id) is Verdict.UNVERIFIED
    assert pipe.evidence_strength(p.id) == "no evidence"
    with pytest.raises(UnverifiedClose):
        pipe.advance(p.id, Stage.CLOSED_WON)

    # It is still on file, and still shown - just not counted for the close.
    assert len(pipe.evidence_for(p.id)) == 1
    assert pipe.evidence_for(p.id, EvidenceTopic.CLOSE) == []
    assert pipe.detail(p.id)["chain"][0]["bears_on_close"] is False


def test_evidence_strength_separates_silence_from_insistence(pipe: Pipeline):
    """Same verdict, very different situations. The board must tell them apart."""
    quiet = pipe.add(_prospect(company="Quiet Co"))
    loud = pipe.add(_prospect(company="Loud Co"))
    pipe.record_evidence(loud.id, Channel.AGENT_ASSERTION, "they definitely signed")

    assert pipe.close_verdict(quiet.id) is pipe.close_verdict(loud.id)
    assert pipe.evidence_strength(quiet.id) == "no evidence"
    assert pipe.evidence_strength(loud.id) == "agent only"


def test_contradicting_evidence_blocks_the_close(pipe: Pipeline):
    """One independent 'no' beats any number of supporting artifacts."""
    p = _to_demo_booked(pipe)
    pipe.record_evidence(p.id, Channel.INBOUND_EMAIL, "signed order form")
    pipe.record_evidence(
        p.id,
        Channel.INBOUND_EMAIL,
        "legal is pulling the order, do not provision",
        supports=False,
    )
    assert pipe.close_verdict(p.id) is Verdict.CONTRADICTED
    with pytest.raises(UnverifiedClose):
        pipe.advance(p.id, Stage.CLOSED_WON)


def test_a_prospect_cannot_be_created_already_closed(pipe: Pipeline):
    with pytest.raises(UnverifiedClose):
        pipe.add(_prospect(stage=Stage.CLOSED_WON))


def test_evidence_hash_is_captured_for_audit(pipe: Pipeline):
    p = pipe.add(_prospect())
    ev = pipe.record_evidence(
        p.id, Channel.INBOUND_SMS, "confirmed", raw={"body": "confirmed"}
    )
    assert ev.content_hash
    assert pipe.evidence_for(p.id)[0].content_hash == ev.content_hash


def test_unknown_prospect_is_a_named_error(pipe: Pipeline):
    with pytest.raises(UnknownProspect):
        pipe.advance("p_nope", Stage.CONTACTED)
    with pytest.raises(UnknownProspect):
        pipe.record_evidence("p_nope", Channel.INBOUND_SMS, "hi")
    assert pipe.detail("p_nope") is None


def test_the_board_keeps_empty_columns(pipe: Pipeline):
    board = pipe.board()
    assert list(board) == list(Stage)
    assert all(v == [] for v in board.values())


def test_state_survives_a_restart(tmp_path: Path):
    db = tmp_path / "pipe.db"
    first = Pipeline(db)
    p = first.add(_prospect(company="Persistent Ltd"))
    first.advance(p.id, Stage.CONTACTED)
    first.propose_terms(p.id, {"seats": 20, "discount_pct": 5})
    first.record_evidence(p.id, Channel.INBOUND_SMS, "call me Tuesday")
    first.close()

    second = Pipeline(db)
    reloaded = second.get(p.id)
    assert reloaded.company == "Persistent Ltd"
    assert reloaded.stage is Stage.CONTACTED
    assert second.terms_for(p.id)["discount_pct"] == 5
    assert len(second.evidence_for(p.id)) == 1
    second.close()


# -- the sample board -----------------------------------------------------


def test_samples_are_seeded_and_labelled(pipe: Pipeline, cfg):
    created = seed_samples(pipe, cfg)
    assert created
    assert all(p.is_sample for p in pipe.all())
    assert pipe.counts()["closed_won"] >= 1


def test_seeding_twice_does_not_duplicate(pipe: Pipeline, cfg):
    seed_samples(pipe, cfg)
    before = len(pipe.all())
    seed_samples(pipe, cfg)
    assert len(pipe.all()) == before


def test_the_seeded_win_is_backed_by_real_evidence(pipe: Pipeline, cfg):
    """Sample data goes through the same gate, so the board cannot demonstrate
    an outcome the code would refuse to produce."""
    seed_samples(pipe, cfg)
    for p in pipe.all(Stage.CLOSED_WON):
        assert pipe.close_verdict(p.id) is Verdict.VERIFIED
        assert any(e.is_independent for e in pipe.evidence_for(p.id))


def test_a_seeded_verbal_yes_does_not_move_a_card(pipe: Pipeline, cfg):
    """The case the board exists to show: agent says closed, board says no."""
    seed_samples(pipe, cfg)
    lattice = pipe.get("sample_lattice")
    assert lattice.stage is Stage.QUALIFIED
    assert pipe.close_verdict(lattice.id) is Verdict.UNVERIFIED
    assert pipe.evidence_strength(lattice.id) == "agent only"


def test_the_seeded_demo_is_not_a_seeded_sale(pipe: Pipeline, cfg):
    """Brightpath has an accepted calendar invite and has not bought anything."""
    seed_samples(pipe, cfg)
    assert pipe.get("sample_brightpath").stage is Stage.DEMO_BOOKED
    assert pipe.evidence_strength("sample_brightpath") == "no evidence"
    assert not pipe.can_close("sample_brightpath")[0]


# -- the app --------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from src.verticals.saas.app import app

    return TestClient(app)


def test_board_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Ledgerline" in r.text
    assert "Sample data" in r.text
    assert "Closed won" in r.text


def test_prospect_detail_shows_the_evidence_chain(client):
    r = client.get("/prospect/sample_northwind")
    assert r.status_code == 200
    assert "Evidence chain" in r.text
    assert "agent-only, cannot verify anything" in r.text
    assert "Countersigned order form" in r.text


def test_unknown_prospect_is_a_404_page(client):
    r = client.get("/prospect/does-not-exist")
    assert r.status_code == 404
    assert "Not found" in r.text


def test_economics_panel_shows_the_stack(client):
    r = client.get("/economics?discount_pct=10&free_months=2&onboarding_fee=0")
    assert r.status_code == 200
    assert "Stacking trap" in r.text
    assert "Rejected" in r.text


def test_economics_api_matches_the_model(client):
    r = client.get("/api/economics?discount_pct=10&free_months=2&onboarding_fee=0")
    body = r.json()
    assert body["check"]["verdict"] == "REJECTED"
    assert body["deal"]["total_contract_value"] == "9000.00"
    assert body["concessions"]["stacking_trap"] is True


def test_junk_query_params_fall_back_rather_than_crash(client):
    r = client.get("/api/economics?seats=abc&discount_pct=999&cac=-5")
    assert r.status_code == 200
    body = r.json()
    assert body["deal"]["seats"] == 25  # list default
    assert Decimal(body["deal"]["discount_pct"]) <= 100
    assert Decimal(body["deal"]["cac"]) >= 0


def test_healthz_reports_campaign_validity(client):
    body = client.get("/healthz").json()
    assert body["ok"]
    assert body["campaign_valid"] is True
