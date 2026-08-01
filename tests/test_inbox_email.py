"""Email verification must behave exactly like the SMS channel it replaces."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verify.inbox_email import (  # noqa: E402
    check_via_mcp_payload, tokens_present, verify_claim_via_email,
)
from src.verify.receipts import Channel, Evidence, Receipt, Verdict  # noqa: E402


def _claim():
    return Receipt(task="t").claim("200 muffins for 400.00", "email confirms qty and total")


def _mail(body, sender="host@venue.com"):
    return [{"from": sender, "subject": "Re: catering", "body": body, "date": "now"}]


def test_matching_email_verifies():
    c = _claim()
    assert verify_claim_via_email(c, _mail("Confirmed: 200 pastries, $400, Friday 8am"),
                                  ["200", "400", "Friday"])
    assert c.verdict is Verdict.VERIFIED


def test_partial_match_does_not_verify():
    """'Confirmed' alone must not satisfy a specific order."""
    c = _claim()
    assert not verify_claim_via_email(c, _mail("Confirmed, see you then!"), ["200", "400"])
    assert c.verdict is Verdict.UNVERIFIED


def test_wrong_quantity_does_not_verify():
    c = _claim()
    assert not verify_claim_via_email(c, _mail("Confirmed 150 pastries, $400"), ["200", "400"])
    assert c.verdict is Verdict.UNVERIFIED


def test_currency_and_comma_formatting_still_matches():
    c = _claim()
    assert verify_claim_via_email(c, _mail("Total is $1,400.00 for 200 units"), ["1400", "200"])


def test_sender_filter_blocks_our_own_copy():
    """Our own outbound mail must not satisfy our own claim."""
    c = _claim()
    assert not verify_claim_via_email(
        c, _mail("200 muffins $400", sender="agent@ourcompany.com"),
        ["200", "400"], from_contains="venue.com")
    assert c.verdict is Verdict.UNVERIFIED


def test_no_mail_attaches_nothing():
    """Absence of proof is not proof of failure."""
    c = _claim()
    assert not verify_claim_via_email(c, [], ["200"])
    assert c.verdict is Verdict.UNVERIFIED
    assert c.evidence == []


def test_agent_assertion_still_cannot_verify_alongside_email():
    c = _claim()
    c.attach_evidence(Evidence(channel=Channel.AGENT_ASSERTION, summary="they said yes"))
    assert c.verdict is Verdict.UNVERIFIED


def test_mcp_payload_shape_is_adapted():
    c = _claim()
    assert check_via_mcp_payload(
        c, [{"sender": "host@venue.com", "subject": "Re: order",
             "snippet": "confirming 200 at $400 Friday"}], ["200", "400"])


def test_tokens_present_reports_what_is_missing():
    ok, missing = tokens_present("confirmed 200 units", ["200", "400"])
    assert not ok and missing == ["400"]
