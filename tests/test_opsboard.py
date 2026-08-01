"""The ops board is a projector surface, and these tests pin the two ways it
could embarrass the project on one.

The load-bearing ones:

* `test_contradicted_never_counts_toward_proven` and
  `test_lying_receipt_cannot_buy_a_green_stamp` — the board re-derives every
  verdict rather than trusting the file. If these go red, a hand-edited receipt
  can put VERIFIED on a projector, which is the disqualifying failure with a
  bigger font.

* `test_no_dataclass_repr_in_html` — an earlier dashboard shipped a whole
  `Campaign(...)` repr onto the screen. Interpolating an object is a one-token
  mistake and invisible until a judge is reading it aloud.

* `test_empty_board_renders` — the board boots empty every morning and a
  projector is a bad place to discover that path raises.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.opsboard import render
from src.opsboard.app import app, seed, use_evidence, use_registry
from src.opsboard.registry import OpsRegistry, humanise, reachable_from
from src.opsboard.state import build, load_receipts, totals

REAL_EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


@pytest.fixture
def board(tmp_path):
    """A client over an empty evidence dir and a fresh registry.

    Both seams are swapped so a test never reads the live board or the real
    `evidence/`, which another agent may be writing to mid-run.
    """
    use_registry(OpsRegistry())
    use_evidence(tmp_path)
    with TestClient(app) as client:
        yield client
    use_registry(OpsRegistry())


def write_receipt(directory: Path, name: str, claims: list[dict]) -> Path:
    p = directory / f"{name}.json"
    p.write_text(
        json.dumps(
            {
                "id": name,
                "task": f"task for {name}",
                "started_at": "2026-07-31T19:00:00+00:00",
                "ended_at": "2026-07-31T19:05:00+00:00",
                "claims": claims,
            }
        )
    )
    return p


def claim(verdict: str, evidence: list[dict], description: str = "a claim") -> dict:
    return {
        "id": "claim_x",
        "description": description,
        "expected_side_effect": "something observable elsewhere",
        "verdict": verdict,
        "evidence": evidence,
    }


def ev(channel: str, supports: bool = True, summary: str = "s") -> dict:
    return {"channel": channel, "summary": summary, "supports": supports, "independent": True}


# --------------------------------------------------------------------------
# It renders, empty and full
# --------------------------------------------------------------------------


def test_empty_board_renders(board):
    r = board.get("/")
    assert r.status_code == 200
    html = r.text
    assert "<html" in html
    # Empty is a designed state, not a blank one: the thesis and both zeroes
    # are still on screen.
    assert "Booked" in html and "Proven" in html
    assert "No call in flight" in html
    assert "Nothing refused yet." in html
    assert "No receipts with claims yet." in html


def test_renders_with_real_receipts_from_evidence():
    """The real directory, unfiltered. It contains stray non-receipt JSON and
    dozens of zero-claim runs; none of that may raise."""
    if not REAL_EVIDENCE.is_dir():
        pytest.skip("no evidence/ directory in this checkout")
    receipts = load_receipts(REAL_EVIDENCE)
    snap = build(REAL_EVIDENCE, OpsRegistry())
    html = render.page_html(snap)
    assert len(html) > 2000
    assert "<html" in html
    t = totals(receipts)
    assert t["booked"] >= t["proven"] >= 0
    if t["booked"]:
        assert "Receipt wall" in html


def test_stray_json_in_evidence_is_skipped(board, tmp_path):
    (tmp_path / "notes.json").write_text('{"hello": "world"}')
    (tmp_path / "broken.json").write_text("{not json at all")
    (tmp_path / "list.json").write_text("[1,2,3]")
    write_receipt(tmp_path, "receipt_ok", [claim("VERIFIED", [ev("inbound_sms")])])
    assert [r["id"] for r in load_receipts(tmp_path)] == ["receipt_ok"]
    assert board.get("/").status_code == 200


# --------------------------------------------------------------------------
# Booked vs proven
# --------------------------------------------------------------------------


def test_booked_and_proven_differ_correctly(tmp_path):
    write_receipt(
        tmp_path,
        "receipt_a",
        [
            claim("VERIFIED", [ev("agent_assertion"), ev("inbound_sms")]),
            claim("UNVERIFIED", [ev("agent_assertion")]),
        ],
    )
    write_receipt(tmp_path, "receipt_b", [claim("VERIFIED", [ev("provider_api")])])
    write_receipt(tmp_path, "receipt_c", [])

    t = totals(load_receipts(tmp_path))
    assert t["booked"] == 3
    assert t["proven"] == 2
    assert t["unconfirmed"] == 1
    assert t["gap"] == t["booked"] - t["proven"] == 1
    assert t["runs"] == 3
    assert t["silent_runs"] == 1


def test_contradicted_never_counts_toward_proven(tmp_path):
    write_receipt(
        tmp_path,
        "receipt_contra",
        [
            # The agent insists, twice, and an independent channel disagrees once.
            claim(
                "CONTRADICTED",
                [
                    ev("agent_assertion", summary="I booked it"),
                    ev("agent_assertion", summary="definitely booked"),
                    ev("inbound_sms", supports=False, summary="we have no such booking"),
                ],
            ),
            claim("VERIFIED", [ev("inbound_sms")]),
        ],
    )
    t = totals(load_receipts(tmp_path))
    assert t["booked"] == 2
    assert t["proven"] == 1
    assert t["contradicted"] == 1
    assert t["proven"] + t["contradicted"] + t["unconfirmed"] == t["booked"]


def test_contradicting_evidence_beats_a_verified_label(tmp_path):
    """A file may claim VERIFIED; contradicting independent evidence wins."""
    write_receipt(
        tmp_path,
        "receipt_liar",
        [claim("VERIFIED", [ev("inbound_sms", supports=False, summary="never happened")])],
    )
    t = totals(load_receipts(tmp_path))
    assert t["proven"] == 0
    assert t["contradicted"] == 1


def test_lying_receipt_cannot_buy_a_green_stamp(tmp_path):
    """`independent: true` on an agent assertion is a forgery, and the board
    recomputes independence from the channel name rather than reading it."""
    write_receipt(
        tmp_path,
        "receipt_forged",
        [
            {
                "id": "c",
                "description": "booked the table",
                "expected_side_effect": "an SMS arrives",
                "verdict": "VERIFIED",
                "evidence": [
                    {
                        "channel": "agent_assertion",
                        "summary": "I definitely booked it",
                        "supports": True,
                        "independent": True,
                    }
                ],
            }
        ],
    )
    receipts = load_receipts(tmp_path)
    assert receipts[0]["claims"][0]["verdict"] == "UNVERIFIED"
    assert receipts[0]["claims"][0]["evidence"][0]["independent"] is False
    assert totals(receipts)["proven"] == 0
    assert "VERIFIED" not in render.wall_html({"receipts": receipts, "hidden": 0})


def test_call_recording_alone_does_not_prove_anything(tmp_path):
    """CALL_RECORDING is excluded from INDEPENDENT_CHANNELS: the agent is on the
    recording. Only a transcript produced independently counts."""
    write_receipt(tmp_path, "receipt_rec", [claim("UNVERIFIED", [ev("call_recording")])])
    assert totals(load_receipts(tmp_path))["proven"] == 0


def test_unconfirmed_is_not_dressed_as_a_failure(tmp_path):
    receipts = [
        {
            "id": "r",
            "task": "t",
            "started_at": "2026-07-31T19:00:00",
            "ended_at": "",
            "recording": "",
            "verdict": "UNVERIFIED",
            "stamp": "UNCONFIRMED",
            "empty": False,
            "counts": {"VERIFIED": 0, "UNVERIFIED": 1, "CONTRADICTED": 0},
            "claims": [
                {
                    "description": "d",
                    "expected": "e",
                    "verdict": "UNVERIFIED",
                    "stamp": "UNCONFIRMED",
                    "downgraded": False,
                    "independent_count": 0,
                    "evidence": [],
                }
            ],
        }
    ]
    html = render.wall_html({"receipts": receipts, "hidden": 0})
    assert "UNCONFIRMED" in html
    # The word the design must never use for an honest non-answer.
    assert "FAILED" not in html and "ERROR" not in html.upper()


# --------------------------------------------------------------------------
# No object reprs on the projector
# --------------------------------------------------------------------------

#: `Campaign(key='x', name=...)` and friends. Matches a capitalised identifier
#: immediately followed by `(` and a `keyword=` argument.
REPR_RE = re.compile(r"[A-Z][A-Za-z0-9_]*\((?:\s*[a-z_][a-z0-9_]*\s*=)")


def test_no_dataclass_repr_in_html(board, tmp_path):
    write_receipt(
        tmp_path,
        "receipt_x",
        [claim("VERIFIED", [ev("agent_assertion"), ev("inbound_sms")])],
    )
    seed(board.app.state.registry)
    html = board.get("/").text
    found = REPR_RE.findall(html)
    assert not found, f"object repr leaked onto the board: {found[:3]}"
    for bad in ("Campaign(", "Receipt(", "Claim(", "Evidence(", "CallState(", "Refusal(",
                "<object", "object at 0x", "dict_keys", "None None"):
        assert bad not in html


def test_no_external_urls(board):
    """Conference wifi. Every asset same-origin, no CDN, no webfont."""
    html = board.get("/").text
    assert "http://" not in html and "https://" not in html
    css = (Path(render.__file__).parent / "static" / "ops.css").read_text()
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)  # prose about url() is fine
    assert "@import" not in css and "url(" not in css


# --------------------------------------------------------------------------
# The poll
# --------------------------------------------------------------------------


def test_poll_state_key_is_stable_when_nothing_changed(board):
    first = board.get("/api/state").json()
    second = board.get("/api/state").json()
    assert first["state_key"] == second["state_key"]
    assert first["panels"]["call"]["key"] == second["panels"]["call"]["key"]
    assert first["panels"]["wall"]["html"] == second["panels"]["wall"]["html"]


def test_poll_state_key_moves_only_for_the_panel_that_changed(board, tmp_path):
    before = board.get("/api/state").json()
    board.post("/api/refusal", json={"headline": "refused below floor $385.72"})
    after = board.get("/api/state").json()

    assert after["state_key"] != before["state_key"]
    assert after["panels"]["guard"]["key"] != before["panels"]["guard"]["key"]
    # The panels nobody touched keep their keys, so the page never repaints them.
    assert after["panels"]["wall"]["key"] == before["panels"]["wall"]["key"]
    assert after["panels"]["metric"]["key"] == before["panels"]["metric"]["key"]
    assert "refused below floor $385.72" in after["panels"]["guard"]["html"]


def test_state_key_survives_a_ticking_clock(board):
    board.post("/api/call", json={"start": True, "business": "Golden Crumb"})
    first = board.get("/api/state").json()["state_key"]
    import time as _t

    _t.sleep(0.05)
    assert board.get("/api/state").json()["state_key"] == first


def test_poll_payload_has_every_panel(board):
    payload = board.get("/api/state").json()
    assert set(payload["panels"]) == {"metric", "call", "guard", "wall"}
    for p in payload["panels"].values():
        assert p["key"] and isinstance(p["html"], str)


# --------------------------------------------------------------------------
# The live call panel
# --------------------------------------------------------------------------


def test_unreachable_phases_are_marked_unreachable(board):
    board.post("/api/call", json={"start": True, "business": "Golden Crumb"})
    board.post("/api/call", json={"phase": "quoted"})
    rail = board.get("/api/data").json()["panels"]["call"]["rail"]
    by = {n["phase"]: n for n in rail}

    assert by["quoted"]["state"] == "now"
    # No edge leads back from QUOTED, so discovery is gone, not discouraged.
    assert by["discovery"]["sealed"] is True
    assert by["qualified"]["sealed"] is True
    assert by["closing"]["state"] == "next"
    assert "discovery" not in reachable_from("quoted")


def test_closed_call_has_nothing_reachable(board):
    board.post("/api/call", json={"start": True, "business": "X", "phase": "closed"})
    rail = board.get("/api/data").json()["panels"]["call"]["rail"]
    assert all(n["state"] in ("past", "now") for n in rail if not n["spur"])
    assert reachable_from("closed") == frozenset()


def test_floor_binding_is_detected_and_shown(board):
    board.post("/api/call", json={"start": True, "business": "Golden Crumb"})
    board.post("/api/call", json={"phase": "negotiating", "quote": 400.0,
                                  "floor": 385.72, "budget": 280.0})
    data = board.get("/api/data").json()["panels"]["call"]
    assert data["call"]["floor_binding"] is True
    html = render.call_html(data)
    assert "Floor is binding" in html
    assert "$385.72" in html and "$280" in html and "$400" in html


def test_a_quote_above_the_floor_is_not_binding(board):
    board.post("/api/call", json={"start": True, "business": "X"})
    board.post("/api/call", json={"quote": 500.0, "floor": 385.72, "budget": 600.0})
    assert board.get("/api/data").json()["panels"]["call"]["call"]["floor_binding"] is False


def test_unknown_field_does_not_end_the_call(board):
    board.post("/api/call", json={"start": True, "business": "X"})
    r = board.post("/api/call", json={"typo_field": 1, "quote": 12.0})
    assert r.status_code == 200
    assert board.get("/api/data").json()["panels"]["call"]["call"]["quote"] == 12.0


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------


def test_gate_refusals_are_rephrased_for_a_human():
    headline, detail = humanise(
        "BLOCKED. They already offered 385.00; quoting 74.00 discards 311.00. "
        "Offer 385.00 or more."
    )
    assert headline == "refused to quote $74 — buyer already offered $385"
    assert "385.00" in detail  # the machine's own words are kept verbatim

    headline, _ = humanise(
        "BLOCKED. Before reserving 600, confirm whether 600 is the number of ITEMS "
        "or the number of PEOPLE."
    )
    assert headline.startswith("refused to reserve 600")


def test_refusals_read_as_proof_not_as_errors(board):
    board.post("/api/refusal", json={"raw": "BLOCKED. They already offered 385.00; "
                                            "quoting 74.00 discards 311.00."})
    board.post("/api/refusal", json={"headline": "refused 600 units — capacity is 400",
                                     "kind": "capacity"})
    html = board.get("/api/state").json()["panels"]["guard"]["html"]
    assert "refused to quote $74" in html
    assert "refused 600 units" in html
    assert "Held" in html
    for scary in ("Error", "Warning", "Failure", "Exception"):
        assert scary not in html


def test_a_gate_that_allows_records_nothing(board):
    reg = board.app.state.registry
    assert reg.gate_refusal(None) is None
    assert reg.gate_refusal("") is None
    assert reg.snapshot()["refusals"] == []


def test_reset_clears_the_ledger_but_not_the_receipts(board, tmp_path):
    write_receipt(tmp_path, "receipt_keep", [claim("VERIFIED", [ev("inbound_sms")])])
    seed(board.app.state.registry)
    assert board.get("/api/data").json()["panels"]["guard"]["refusals"]
    board.post("/api/reset")
    data = board.get("/api/data").json()
    assert data["panels"]["guard"]["refusals"] == []
    assert data["panels"]["metric"]["proven"] == 1


def test_fixture_data_is_labelled_on_screen(board):
    seed(board.app.state.registry)
    assert "Fixture data" in board.get("/").text


def test_healthz(board):
    r = board.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert "booked" in r.json()["totals"]
