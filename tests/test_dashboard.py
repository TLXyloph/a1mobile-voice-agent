"""The dashboard is a view, and these tests pin that it stays one.

Three things are worth more than the rest of this file:

* `test_stamp_follows_the_claim_not_the_operator` - approving an escalation
  must not colour a receipt green. Approval is permission to offer a price;
  only an independent channel closes the loop. If this goes red, the board is
  capable of showing a judge a VERIFIED stamp on an unverified claim, which is
  the disqualifying failure wearing a nicer font.

* `test_empty_board_renders` - the operator sees an empty board before the
  first call of the day, and a projector is a bad place to discover that path
  raises.

* `test_no_external_urls` - conference wifi. Every asset must be same-origin,
  because a page that needs a CDN is a page that is blank when it matters.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.dashboard import state as dash_state
from src.dashboard.app import HERE, app, use_board
from src.dashboard.state import (
    ApprovalRequest,
    Board,
    LiveCall,
    ReceiptCard,
    seed,
)
from src.verify.receipts import Channel, Evidence, Receipt, Verdict

READ_ROUTES = ("/", "/board", "/api/state", "/healthz")


@pytest.fixture
def seeded() -> Board:
    """A freshly seeded board, wired into the app for one test."""
    return use_board(seed(Board()))


@pytest.fixture
def empty() -> Board:
    """A board that has never seen a call. A supported state, not an error."""
    return use_board(Board())


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# -- routes ---------------------------------------------------------------


@pytest.mark.parametrize("route", READ_ROUTES)
def test_read_routes_ok_when_seeded(client: TestClient, seeded: Board, route: str) -> None:
    assert client.get(route).status_code == 200


@pytest.mark.parametrize("route", READ_ROUTES)
def test_read_routes_ok_when_empty(client: TestClient, empty: Board, route: str) -> None:
    assert client.get(route).status_code == 200


def test_static_assets_are_served(client: TestClient, seeded: Board) -> None:
    for asset in ("/static/dash.css", "/static/dash.js"):
        r = client.get(asset)
        assert r.status_code == 200, asset
        assert r.content, asset


def test_index_embeds_the_same_fragment_the_poller_fetches(
    client: TestClient, seeded: Board
) -> None:
    """One template, so the first paint and every poll agree."""
    page = client.get("/").text
    fragment = client.get("/board").text
    assert "PRINTED RECEIPTS" in fragment
    assert fragment.strip()[:200] in page


#: The poller's change detector, mirrored from `stateKey()` in dash.js.
_CLOCK_TEXT = re.compile(r'(data-elapsed="[^"]*"[^>]*>)[^<]*')


def test_fragment_is_stable_while_nothing_changes(client: TestClient, seeded: Board) -> None:
    """Load-bearing: the poller only writes the DOM when the fragment changes,
    and a DOM write costs the operator whatever they had tabbed to. So the only
    thing allowed to differ between two polls of an idle board is the inside of
    a data-elapsed clock, which JS owns. Anything else time-varying rendered by
    Jinja would make every poll look like a change."""
    first = client.get("/board").text
    time.sleep(1.1)
    second = client.get("/board").text

    assert second != first, "expected the server-rendered clocks to have moved"
    assert _CLOCK_TEXT.sub(r"\1", second) == _CLOCK_TEXT.sub(r"\1", first)


def test_the_poller_ignores_exactly_those_clocks() -> None:
    """If dash.js stops normalising, the test above is checking a property the
    browser no longer relies on."""
    js = (Path(HERE) / "static" / "dash.js").read_text()
    assert 'data-elapsed="[^"]*"' in js
    assert "stateKey" in js


# -- empty data -----------------------------------------------------------


def test_empty_board_renders(client: TestClient, empty: Board) -> None:
    html = client.get("/").text
    for invitation in (
        "RAIL CLEAR",
        "LINE IS OPEN",
        "NO LEDGER REGISTERED",
        "NO LEADS SCREENED",
        "NOTHING PRINTED YET",
    ):
        assert invitation in html, invitation


def test_empty_board_json_is_well_formed(client: TestClient, empty: Board) -> None:
    body = client.get("/api/state").json()
    assert body["counts"] == {
        "pending": 0, "answered": 0, "calls": 0, "leads": 0,
        "receipts": 0, "verified": 0, "contradicted": 0,
    }
    assert body["capacity"] == [] and body["receipts"] == []


def test_empty_board_says_nothing_about_fixtures(client: TestClient, empty: Board) -> None:
    assert "FIXTURE DATA" not in client.get("/").text


def test_seeded_board_admits_it_is_seeded(client: TestClient, seeded: Board) -> None:
    """A board arguing that outcomes must be proven does not get to imply that
    a fixture is a live call."""
    assert "FIXTURE DATA" in client.get("/").text


# -- the approval path ----------------------------------------------------


def _pending_id(client: TestClient) -> str:
    return client.get("/api/state").json()["approvals"]["pending"][0]["id"]


def test_approval_post_changes_state(client: TestClient, seeded: Board) -> None:
    approval_id = _pending_id(client)
    assert seeded.approval(approval_id).is_pending

    r = client.post(f"/approvals/{approval_id}/decide", data={"decision": "approve"})
    assert r.status_code == 200  # 303 followed back to the board

    req = seeded.approval(approval_id)
    assert req.decision == "approve" and req.approved
    assert not req.is_pending
    assert client.get("/api/state").json()["counts"]["pending"] == 0


def test_denial_is_recorded_and_reaches_the_call(client: TestClient, seeded: Board) -> None:
    approval_id = _pending_id(client)
    client.post(f"/approvals/{approval_id}/decide", data={"decision": "deny"})

    req = seeded.approval(approval_id)
    assert req.decision == "deny" and not req.approved
    assert seeded.call(req.call_id).status == "HOLDING FLOOR"


def test_anything_that_is_not_approve_is_a_denial(seeded: Board) -> None:
    """The operator's silence, a truncated form field, a typo - all no."""
    for answer in ("", "maybe", "APPROVE", "yes", "sure"):
        req = ApprovalRequest(business="X", campaign_key="restaurant_catering",
                              ask="?", reason="r")
        req.decide(answer)
        assert not req.approved, answer


def test_a_decision_cannot_be_flipped(client: TestClient, seeded: Board) -> None:
    """A double click, or a second operator a moment later, must not reopen a
    settled question."""
    approval_id = _pending_id(client)
    client.post(f"/approvals/{approval_id}/decide", data={"decision": "deny"})
    client.post(f"/approvals/{approval_id}/decide", data={"decision": "approve"})
    assert seeded.approval(approval_id).decision == "deny"


def test_unknown_approval_id_does_not_crash(client: TestClient, seeded: Board) -> None:
    """A stale tab posting into the void must not take the board down."""
    r = client.post("/approvals/ask_nosuchthing/decide", data={"decision": "approve"})
    assert r.status_code == 200


def test_approve_and_deny_are_real_submit_buttons(client: TestClient, seeded: Board) -> None:
    """Keyboard reachability is not a CSS problem. If these ever become divs
    with click handlers, the fallback for the voice approval path is gone."""
    html = client.get("/board").text
    form = re.search(r'<form class="ticket-act".*?</form>', html, re.S)
    assert form, "no approval form on a board with a pending escalation"
    assert form.group(0).count("<button") == 2
    assert 'name="decision" value="approve"' in form.group(0)
    assert 'name="decision" value="deny"' in form.group(0)
    assert 'method="post"' in form.group(0)


def test_demo_reset_rearms_the_rail(client: TestClient, seeded: Board) -> None:
    client.post(f"/approvals/{_pending_id(client)}/decide", data={"decision": "approve"})
    assert client.get("/api/state").json()["counts"]["pending"] == 0
    assert client.post("/demo/reset").status_code == 200
    assert client.get("/api/state").json()["counts"]["pending"] == 1


# -- the invariant, seen from the dashboard -------------------------------


def _card(*evidence: Evidence) -> ReceiptCard:
    r = Receipt(task="t")
    c = r.claim(description="d", expected_side_effect="s")
    for e in evidence:
        c.attach_evidence(e)
    return ReceiptCard(receipt=r, counterparty="Somebody", campaign_key="restaurant_catering")


def test_stamp_is_derived_from_the_claim() -> None:
    agent = Evidence(channel=Channel.AGENT_ASSERTION, summary="they said yes")
    sms = Evidence(channel=Channel.INBOUND_SMS, summary="confirmed")
    against = Evidence(channel=Channel.PROVIDER_API, summary="no such booking", supports=False)

    assert _card().stamp == "UNCONFIRMED"
    assert _card(agent).stamp == "UNCONFIRMED"
    assert _card(agent, sms).stamp == "VERIFIED"
    assert _card(agent, sms, against).stamp == "CONTRADICTED"


def test_no_volume_of_agent_assertion_promotes_a_stamp() -> None:
    many = [
        Evidence(channel=Channel.AGENT_ASSERTION, summary=f"definitely done {i}")
        for i in range(50)
    ]
    assert _card(*many).stamp == "UNCONFIRMED"


def test_stamp_follows_the_claim_not_the_operator(client: TestClient, seeded: Board) -> None:
    """Approving a below-floor price authorises an offer. It does not make the
    sale, and the receipt must not pretend otherwise until something
    independent lands."""
    approval_id = _pending_id(client)
    client.post(f"/approvals/{approval_id}/decide", data={"decision": "approve"})

    # Force the seeded call to close now rather than waiting out the timer.
    seeded.approval(approval_id).decided_epoch = time.time() - dash_state.RESOLVE_AFTER - 1
    printed = seeded.snapshot()["receipts"][0]

    assert printed.stamp == "UNCONFIRMED"
    assert printed.receipt.claims[0].verdict is Verdict.UNVERIFIED
    assert "VERIFIED" not in printed.receipt.headline


def test_the_independent_channel_is_what_flips_it(client: TestClient, seeded: Board) -> None:
    approval_id = _pending_id(client)
    client.post(f"/approvals/{approval_id}/decide", data={"decision": "approve"})
    seeded.approval(approval_id).decided_epoch = time.time() - dash_state.RESOLVE_AFTER - 1
    card = seeded.snapshot()["receipts"][0]
    assert card.stamp == "UNCONFIRMED"

    card.printed_epoch = time.time() - dash_state.EVIDENCE_AFTER - 1
    seeded.snapshot()

    assert card.stamp == "VERIFIED"
    channels = {e.channel for e in card.receipt.claims[0].evidence}
    assert Channel.INBOUND_SMS in channels


def test_capacity_numbers_come_from_a_real_ledger(seeded: Board) -> None:
    """The bars are arithmetic the ledger agrees with, not numbers we typed."""
    for card in seeded.ledgers:
        s = card.snapshot
        assert s["available"] + s["held"] + s["committed"] == s["total"]
        assert abs(sum(b["pct"] for b in card.bars()) - 100.0) < 1e-6


def test_dashboard_renders_all_five_screens(client: TestClient, seeded: Board) -> None:
    html = client.get("/").text
    for heading in (
        "APPROVAL RAIL", "ON THE LINE", "CAPACITY LEDGER",
        "LEAD SHEET", "PRINTED RECEIPTS",
    ):
        assert heading in html, heading
    assert "Ridgeline Athletics" in html          # live call + approval
    assert "Weekday corporate catering" in html   # campaign name resolved
    assert "muffins/week" in html                 # capacity unit
    assert "Stockton Judo Academy" in html        # lead sheet
    assert "no website found in listing" in html  # qualification reason
    assert "CONTRADICTED" in html                 # the unflattering verdict


def test_no_dataclass_reprs_leaked_into_the_page(client: TestClient, seeded: Board) -> None:
    """A Jinja typo on an object attribute renders the whole repr. It is silent,
    it looks terrible on a projector, and it happened once already."""
    html = client.get("/").text
    for tell in ("Campaign(name=", "Envelope(", "Lead(", "Receipt(", "<src.", " object at 0x"):
        assert tell not in html, tell


# -- offline ---------------------------------------------------------------

_ASSET_REF = re.compile(r"""(?:src|href|action)\s*=\s*["']([^"']+)["']""", re.I)
_CDN_TELLS = (
    "fonts.googleapis", "fonts.gstatic", "cdn.", "cdnjs", "unpkg.com",
    "jsdelivr", "googleapis.com", "bootstrapcdn", "@import",
)


@pytest.mark.parametrize("route", ("/", "/board"))
def test_no_external_urls(client: TestClient, seeded: Board, route: str) -> None:
    html = client.get(route).text
    for ref in _ASSET_REF.findall(html):
        assert not ref.startswith(("http://", "https://", "//")), ref
        assert ref.startswith(("/", "#")), ref
    for tell in _CDN_TELLS:
        assert tell not in html, tell


def _uncommented(source: str) -> str:
    """Block comments stripped. Prose about not fetching things is not a fetch,
    and the assertions below are about declarations."""
    return re.sub(r"/\*.*?\*/", " ", source, flags=re.S)


def test_stylesheet_and_script_are_offline() -> None:
    """Checked on disk as well as over the wire: a url() added to the CSS would
    not show up in the HTML at all."""
    css = _uncommented((Path(HERE) / "static" / "dash.css").read_text())
    js = _uncommented((Path(HERE) / "static" / "dash.js").read_text())
    for blob, name in ((css, "dash.css"), (js, "dash.js")):
        for tell in _CDN_TELLS:
            assert tell not in blob, f"{name}: {tell}"
        assert "http://" not in blob and "https://" not in blob, name
    assert "@font-face" not in css
    # url() is allowed only if it never leaves the origin; simplest is none.
    assert not re.search(r"url\(\s*['\"]?(?!data:)", css)


def test_reduced_motion_and_focus_states_exist() -> None:
    css = (Path(HERE) / "static" / "dash.css").read_text()
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css
    assert ".btn:focus-visible" in css


# -- the integration surface a live run would use --------------------------


def test_a_live_run_can_drive_the_board(client: TestClient, empty: Board) -> None:
    """The board has to be usable by the real agent, not only by fixtures."""
    call = empty.open_call(
        LiveCall(business="Marin Pediatric Dental", phone="+1 415 555 0192",
                 campaign_key="freelance_webdev")
    )
    empty.request_approval(
        ApprovalRequest(business="Marin Pediatric Dental",
                        campaign_key="freelance_webdev",
                        ask="Go to 1,000 for the build?",
                        reason="price 1000.00 USD is below the floor of 1200.00",
                        call_id=call.id)
    )
    html = client.get("/").text
    assert "Marin Pediatric Dental" in html
    assert "Go to 1,000 for the build?" in html
    assert client.get("/api/state").json()["counts"]["pending"] == 1


def test_a_board_with_real_data_does_not_advance_on_a_timer(empty: Board) -> None:
    """`tick()` only moves fixtures. A real run's receipts must never grow an
    inbound SMS the dashboard invented for them."""
    r = Receipt(task="real")
    c = r.claim(description="d", expected_side_effect="s")
    c.attach_evidence(Evidence(channel=Channel.AGENT_ASSERTION, summary="went well"))
    card = empty.add_receipt(
        ReceiptCard(receipt=r, counterparty="Real Co", campaign_key="restaurant_catering",
                    demo_pending_evidence=True, printed_epoch=time.time() - 10_000)
    )
    empty.snapshot()
    assert card.stamp == "UNCONFIRMED"
    assert len(card.receipt.claims[0].evidence) == 1
