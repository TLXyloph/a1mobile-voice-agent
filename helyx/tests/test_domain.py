"""Pins the anti-fabrication invariant.

If any test in this file goes red, the agent can mark its own work done and the
system is disqualifiable. Treat a failure here as a release blocker.
"""

from __future__ import annotations

import pytest

from helyx.domain import (
    AGREEMENT_CHANNELS,
    INDEPENDENT_CHANNELS,
    Channel,
    Evidence,
    Proposal,
    ProposalStatus,
    Terms,
    tokens_present,
)


@pytest.fixture()
def terms() -> Terms:
    return Terms(
        item="sourdough loaves",
        quantity=120,
        unit_price_cents=425,
        fulfilment_date="2026-08-14",
    )


@pytest.fixture()
def proposal(terms: Terms) -> Proposal:
    return Proposal(terms=terms, counterparty="Kestrel Bakehouse")


# --- the core invariant ----------------------------------------------------


def test_agent_assertion_is_not_an_independent_channel() -> None:
    assert Channel.AGENT_ASSERTION not in INDEPENDENT_CHANNELS
    assert Channel.AGENT_ASSERTION not in AGREEMENT_CHANNELS


def test_proposal_is_born_unconfirmed(proposal: Proposal) -> None:
    assert proposal.status is ProposalStatus.UNCONFIRMED


def test_no_volume_of_agent_assertion_can_confirm(proposal: Proposal) -> None:
    """A thousand confident agent claims still derive UNCONFIRMED."""
    for _ in range(1000):
        proposal.add_evidence(
            Evidence(
                channel=Channel.AGENT_ASSERTION,
                body="CONFIRMED: 120 sourdough loaves at 4.25 each, locked in, done.",
            )
        )
    assert proposal.status is ProposalStatus.UNCONFIRMED
    assert len(proposal.evidence) == 1000


def test_status_has_no_setter(proposal: Proposal) -> None:
    with pytest.raises(AttributeError):
        proposal.status = ProposalStatus.CONFIRMED  # type: ignore[misc]


def test_proposal_cannot_be_filed_by_an_independent_channel(terms: Terms) -> None:
    """Filing is the agent's act; independent facts must arrive as evidence."""
    with pytest.raises(ValueError):
        Proposal(terms=terms, counterparty="X", filed_by=Channel.INBOUND_SMS)


# --- what does confirm -----------------------------------------------------


def test_matching_inbound_sms_confirms(proposal: Proposal) -> None:
    proposal.add_evidence(
        Evidence(
            channel=Channel.INBOUND_SMS,
            body="Kestrel here - confirming 120 loaves at $4.25 each for the 14th.",
            external_ref="tel_abc123",
        )
    )
    assert proposal.status is ProposalStatus.CONFIRMED
    assert "inbound_sms" in proposal.why()


def test_bare_confirmed_does_not_confirm(proposal: Proposal) -> None:
    """The exact failure mode that scores a wrong order as right."""
    proposal.add_evidence(Evidence(channel=Channel.INBOUND_SMS, body="confirmed!"))
    assert proposal.status is ProposalStatus.UNCONFIRMED


def test_wrong_price_in_confirmation_does_not_confirm(proposal: Proposal) -> None:
    proposal.add_evidence(
        Evidence(
            channel=Channel.INBOUND_SMS,
            body="Confirming 120 loaves at $5.75 each.",
        )
    )
    assert proposal.status is ProposalStatus.UNCONFIRMED


def test_wrong_quantity_in_confirmation_does_not_confirm(proposal: Proposal) -> None:
    proposal.add_evidence(
        Evidence(channel=Channel.INBOUND_SMS, body="Confirming 60 loaves at $4.25 each.")
    )
    assert proposal.status is ProposalStatus.UNCONFIRMED


def test_provider_api_receipt_cannot_establish_agreement(proposal: Proposal) -> None:
    """Delivery is not assent. We sent the text; that proves nothing was agreed."""
    proposal.add_evidence(
        Evidence(
            channel=Channel.PROVIDER_API,
            body="sent=true 120 loaves at 4.25 sourdough",
            external_ref="msg_1",
        )
    )
    assert proposal.status is ProposalStatus.UNCONFIRMED


def test_human_review_can_confirm(proposal: Proposal) -> None:
    proposal.add_evidence(
        Evidence(
            channel=Channel.HUMAN_REVIEW,
            body="Operator listened back: 120 sourdough at 4.25 on the 14th. Correct.",
        )
    )
    assert proposal.status is ProposalStatus.CONFIRMED


def test_contradiction_beats_confirmation(proposal: Proposal) -> None:
    proposal.add_evidence(
        Evidence(
            channel=Channel.INBOUND_SMS,
            body="Confirming 120 loaves at $4.25 each.",
        )
    )
    assert proposal.status is ProposalStatus.CONFIRMED
    proposal.add_evidence(
        Evidence(
            channel=Channel.INBOUND_EMAIL,
            body="Sorry - we cannot fill that order after all.",
            supports=False,
        )
    )
    assert proposal.status is ProposalStatus.CONTRADICTED


def test_agent_cannot_undo_a_contradiction(proposal: Proposal) -> None:
    proposal.add_evidence(
        Evidence(channel=Channel.INBOUND_SMS, body="cancelled", supports=False)
    )
    for _ in range(50):
        proposal.add_evidence(
            Evidence(
                channel=Channel.AGENT_ASSERTION,
                body="120 sourdough loaves at 4.25 - it is definitely still on.",
            )
        )
    assert proposal.status is ProposalStatus.CONTRADICTED


# --- token matching --------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected_ok",
    [
        ("120 loaves @ $4.25", True),
        ("  120   sourdough   4.25  ", True),
        ("$4.25 x 120 sourdough", True),
        ("120 sourdough", False),  # no price
        ("4.25 sourdough", False),  # no quantity
        ("120 loaves at 4.25", True),
        ("confirmed", False),
        ("120 widgets at 4.25", False),  # right numbers, wrong product
    ],
)
def test_terms_matching_requires_numbers_and_product(
    terms: Terms, body: str, expected_ok: bool
) -> None:
    ok, _missing = terms.matches(body)
    assert ok is expected_ok


@pytest.mark.parametrize(
    "body",
    [
        "1200 loaves at 4.25",  # 1200 must not satisfy quantity 120
        "120 loaves at 14.25",  # 14.25 must not satisfy price 4.25
        "order 4120 loaves 44.25",
    ],
)
def test_numeric_matching_is_boundary_aware(terms: Terms, body: str) -> None:
    """Substring matching would turn an unrelated number into a confirmation."""
    ok, _ = terms.matches(body)
    assert ok is False


def test_tokens_present_requires_all() -> None:
    assert tokens_present("120 loaves 4.25", ["120", "4.25"])[0] is True
    assert tokens_present("120 loaves", ["120", "4.25"])[0] is False


def test_terms_reject_nonsense() -> None:
    with pytest.raises(ValueError):
        Terms(item="", quantity=1, unit_price_cents=1, fulfilment_date="2026-01-01")
    with pytest.raises(ValueError):
        Terms(item="x", quantity=0, unit_price_cents=1, fulfilment_date="2026-01-01")
    with pytest.raises(ValueError):
        Terms(item="x", quantity=1, unit_price_cents=0, fulfilment_date="2026-01-01")


def test_to_dict_reports_status_and_reason(proposal: Proposal) -> None:
    d = proposal.to_dict()
    assert d["status"] == "unconfirmed"
    assert "no independent agreement evidence" in str(d["why"])
