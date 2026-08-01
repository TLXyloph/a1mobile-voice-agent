"""The anti-fabrication invariant is the one thing that must not regress.

If these tests fail, the system can report a success it did not earn - which is
the automatic-disqualification condition in this hackathon's rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verify.receipts import Channel, Evidence, Receipt, Verdict  # noqa: E402


def _receipt() -> Receipt:
    return Receipt(task="Book a table for 4 at 7pm")


def test_claim_starts_unverified():
    c = _receipt().claim("Booked table for 4", "confirmation SMS arrives")
    assert c.verdict is Verdict.UNVERIFIED


def test_agent_assertion_alone_cannot_verify():
    """The core rule: the agent talking about it does not make it true."""
    c = _receipt().claim("Booked table for 4", "confirmation SMS arrives")
    c.attach_evidence(
        Evidence(
            channel=Channel.AGENT_ASSERTION,
            summary="The host told me we're confirmed for 7pm.",
        )
    )
    assert c.verdict is Verdict.UNVERIFIED
    assert c.agent_said is not None


def test_many_agent_assertions_still_cannot_verify():
    """Insistence is not evidence - repetition must not accumulate into proof."""
    c = _receipt().claim("Booked table for 4", "confirmation SMS arrives")
    for i in range(50):
        c.attach_evidence(
            Evidence(channel=Channel.AGENT_ASSERTION, summary=f"Confirmed #{i}")
        )
    assert c.verdict is Verdict.UNVERIFIED


def test_independent_channel_verifies():
    c = _receipt().claim("Booked table for 4", "confirmation SMS arrives")
    c.attach_evidence(
        Evidence(
            channel=Channel.INBOUND_SMS,
            summary="SMS from +14155550123: 'Table for 4 at 7:00 PM confirmed'",
            raw={"from": "+14155550123", "body": "Table for 4 at 7:00 PM confirmed"},
        )
    )
    assert c.verdict is Verdict.VERIFIED


def test_contradiction_beats_confirmation():
    """A later independent denial must override an earlier confirmation."""
    c = _receipt().claim("Booked table for 4", "reservation appears in system")
    c.attach_evidence(Evidence(channel=Channel.INBOUND_SMS, summary="confirmed"))
    c.attach_evidence(
        Evidence(
            channel=Channel.PROVIDER_API,
            summary="No reservation found under that name",
            supports=False,
        )
    )
    assert c.verdict is Verdict.CONTRADICTED


def test_required_channels_must_all_be_present():
    c = _receipt().claim(
        "Booked and paid",
        "SMS confirms booking AND provider API shows payment",
        required_channels=(Channel.INBOUND_SMS, Channel.PROVIDER_API),
    )
    c.attach_evidence(Evidence(channel=Channel.INBOUND_SMS, summary="booking confirmed"))
    assert c.verdict is Verdict.UNVERIFIED, "partial satisfaction must not verify"

    c.attach_evidence(Evidence(channel=Channel.PROVIDER_API, summary="charge settled"))
    assert c.verdict is Verdict.VERIFIED


def test_headline_never_overstates():
    r = _receipt()
    verified = r.claim("Booked table", "SMS arrives")
    verified.attach_evidence(Evidence(channel=Channel.INBOUND_SMS, summary="ok"))
    r.claim("Requested a high chair", "staff note visible in booking")

    assert "PARTIAL" in r.headline
    assert "SUCCESS" not in r.headline


def test_evidence_is_hashed_for_tamper_detection():
    a = Evidence(channel=Channel.INBOUND_SMS, summary="x", raw={"body": "confirmed"})
    b = Evidence(channel=Channel.INBOUND_SMS, summary="x", raw={"body": "cancelled"})
    assert a.content_hash and a.content_hash != b.content_hash


def test_empty_receipt_does_not_claim_success():
    assert "NO CLAIMS MADE" in Receipt(task="do nothing").headline
