"""Inbound SMS normalisation, the store's evidence wiring, and the email loop."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from helyx import email_loop
from helyx.domain import ProposalStatus, Proposal, Terms
from helyx.mandate import Mandate
from helyx.negotiator import Negotiation
from helyx.sms import confirmation_request, normalise_inbound
from helyx.store import HelyxStore


def a_store() -> HelyxStore:
    store = HelyxStore()
    m = Mandate(
        item="sourdough loaves",
        quantity=120,
        target_unit_price_cents=425,
        ceiling_unit_price_cents=500,
        needed_by=date(2026, 8, 14),
        counterparty_name="Kestrel Bakehouse",
    )
    store.negotiation = Negotiation(mandate=m)
    store.negotiation.proposals.append(
        Proposal(
            terms=Terms("sourdough loaves", 120, 425, "2026-08-14"),
            counterparty="Kestrel Bakehouse",
        )
    )
    return store


# --- inbound normalisation -------------------------------------------------


def test_normalise_lowercase_a1mobile_payload() -> None:
    msg = normalise_inbound(
        {
            "type": "message.received",
            "from": "+15551230000",
            "to": "+19378608348",
            "text": "confirming 120 loaves at 4.25",
            "media_urls": [],
            "telnyx_id": "abc123",
        }
    )
    assert msg.from_number == "+15551230000"
    assert msg.text == "confirming 120 loaves at 4.25"
    assert msg.provider_id == "abc123"


def test_normalise_tolerates_casing_and_alias_drift() -> None:
    msg = normalise_inbound({"From": "+1555", "To": "+1937", "Body": "hi", "ID": "x1"})
    assert (msg.from_number, msg.text, msg.provider_id) == ("+1555", "hi", "x1")


def test_normalise_rejects_non_object() -> None:
    from helyx.sms import SMSError

    with pytest.raises(SMSError):
        normalise_inbound(["not", "a", "dict"])  # type: ignore[arg-type]


# --- evidence wiring -------------------------------------------------------


def test_matching_inbound_sms_confirms_via_store() -> None:
    store = a_store()
    result = store.attach_inbound_sms(
        normalise_inbound(
            {"from": "+1555", "to": "+1937", "text": "yes - 120 loaves at $4.25", "telnyx_id": "t1"}
        )
    )
    assert result["confirmed"]
    assert store.proposals[0].status is ProposalStatus.CONFIRMED
    assert "CONFIRMED" in store.headline()


def test_vague_inbound_sms_does_not_confirm() -> None:
    store = a_store()
    store.attach_inbound_sms(
        normalise_inbound({"from": "+1555", "to": "+1937", "text": "ok sounds good"})
    )
    assert store.proposals[0].status is ProposalStatus.UNCONFIRMED
    assert "UNCONFIRMED" in store.headline()


def test_refusal_sms_contradicts() -> None:
    store = a_store()
    store.attach_inbound_sms(
        normalise_inbound(
            {"from": "+1555", "to": "+1937", "text": "sorry we cannot fill 120 loaves at 4.25"}
        )
    )
    assert store.proposals[0].status is ProposalStatus.CONTRADICTED
    assert "CONTRADICTED" in store.headline()


def test_headline_is_pessimistic_with_no_evidence() -> None:
    assert "UNCONFIRMED" in a_store().headline()


def test_confirmation_request_restates_the_numbers() -> None:
    """A bare 'yes' must not be able to confirm, so we ask them to echo terms."""
    body = confirmation_request(120, "sourdough loaves", 425, "2026-08-14")
    assert "120" in body and "4.25" in body
    assert "Nothing is booked" in body


# --- email loop ------------------------------------------------------------


def test_email_loop_composes_and_stores_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(email_loop, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(email_loop, "INBOX", tmp_path / "inbox")
    store = a_store()

    record = email_loop.run_email_loop(store)

    assert record["to"] == email_loop.settings().callback_email
    assert record["inbound_found"] is False
    written = list((tmp_path / "outbox").glob("*.eml"))
    assert len(written) == 1
    # The stored .eml is transfer-encoded, so read the decoded payload.
    import email as email_mod
    from email.policy import default as default_policy

    body = email_mod.message_from_string(
        written[0].read_text(), policy=default_policy
    ).get_content()
    assert "UNCONFIRMED" in body
    assert "sourdough loaves" in body
    assert "Kestrel Bakehouse" in body
    # provenance must be visible in the report itself
    assert "agent claim" in body or "no independent agreement evidence" in body


def test_email_loop_reads_inbound_drop_and_can_confirm(
    tmp_path: Path, monkeypatch
) -> None:
    """Inbound email is an independent channel, so it can confirm terms."""
    import json

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.json").write_text(
        json.dumps(
            {
                "from": "orders@kestrel.example",
                "subject": "Order confirmation",
                "body": "Confirmed: 120 sourdough loaves at $4.25 per unit.",
                "message_id": "<m1@kestrel>",
            }
        )
    )
    monkeypatch.setattr(email_loop, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(email_loop, "INBOX", inbox)

    store = a_store()
    record = email_loop.run_email_loop(store)

    assert record["inbound_found"] is True
    assert store.proposals[0].status is ProposalStatus.CONFIRMED
    assert record["confirmed_after_email"]


def test_send_report_refuses_third_parties(tmp_path: Path, monkeypatch) -> None:
    """Helyx must never mail anyone but the operator."""
    monkeypatch.setattr(email_loop, "OUTBOX", tmp_path / "outbox")
    msg = email_loop.compose_summary(a_store().snapshot(), None)
    del msg["To"]
    msg["To"] = "supplier@kestrel.example"
    with pytest.raises(email_loop.EmailRefused):
        email_loop.send_report(msg)


def test_outbox_transport_does_not_claim_delivery(tmp_path: Path, monkeypatch) -> None:
    """Composing is not sending. The record must say so."""
    monkeypatch.setattr(email_loop, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(email_loop, "INBOX", tmp_path / "nope")
    record = email_loop.run_email_loop(a_store())
    assert record["outcome"]["delivered"] is False
    assert "not delivered" in record["outcome"]["detail"]


def test_receipt_is_written_even_with_nothing_confirmed(tmp_path: Path, monkeypatch) -> None:
    from helyx import store as store_mod

    monkeypatch.setattr(store_mod, "VAR_DIR", tmp_path)
    store = a_store()
    path = store.write_receipt()
    assert path.exists()
    assert "UNCONFIRMED" in path.read_text()
