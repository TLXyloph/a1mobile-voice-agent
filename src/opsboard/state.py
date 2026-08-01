"""Reading the evidence directory, pessimistically.

A receipt on disk already carries a `verdict` per claim, written by
`Claim.verdict`, which has no setter. This module still does not trust it.

Two independent re-derivations happen here:

* `independent` is recomputed from the channel name against
  `INDEPENDENT_CHANNELS`, so a file asserting `"channel": "agent_assertion",
  "independent": true` cannot buy itself a green stamp.
* the verdict is recomputed from the evidence list and the board displays
  **whichever of the two is worse**. `required_channels` is not serialised, so
  re-derivation can only ever be too generous; taking the minimum means the
  board can under-report a real success but can never over-report one.

That asymmetry is the same one the rest of the project runs on: the worst thing
this screen can do is claim something is proven when it is not.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.opsboard.registry import OPS, OpsRegistry

try:
    from src.verify.receipts import INDEPENDENT_CHANNELS

    _INDEPENDENT = frozenset(c.value for c in INDEPENDENT_CHANNELS)
except Exception:  # noqa: BLE001  # pragma: no cover - only while receipts.py is mid-edit
    _INDEPENDENT = frozenset(
        {
            "inbound_sms",
            "inbound_email",
            "provider_api",
            "web_check",
            "dtmf_confirmation",
            "independent_transcript",
            "human_callback",
        }
    )

VERIFIED = "VERIFIED"
UNVERIFIED = "UNVERIFIED"
CONTRADICTED = "CONTRADICTED"

#: Worse is lower. Used to pick between the stored verdict and our own.
_SEVERITY = {CONTRADICTED: 0, UNVERIFIED: 1, VERIFIED: 2}

#: What the stamp says. "UNCONFIRMED" rather than "UNVERIFIED" because the
#: former describes the evidence and the latter sounds like a grade.
STAMP = {VERIFIED: "VERIFIED", UNVERIFIED: "UNCONFIRMED", CONTRADICTED: "CONTRADICTED"}

CHANNEL_LABEL = {
    "agent_assertion": "the agent said so",
    "inbound_sms": "inbound SMS",
    "inbound_email": "inbound email",
    "provider_api": "provider API",
    "web_check": "web check",
    "dtmf_confirmation": "keypad confirmation",
    "call_recording": "call recording",
    "independent_transcript": "independent transcript",
    "human_callback": "human callback",
}

EVIDENCE_DIR = Path(os.environ.get("OPSBOARD_EVIDENCE", "evidence"))

#: Cards on the wall. Older runs are counted in the totals but not drawn.
WALL_LIMIT = 24


def _derive_verdict(evidence: list[dict[str, Any]]) -> str:
    if any(e["independent"] and not e["supports"] for e in evidence):
        return CONTRADICTED
    if any(e["independent"] and e["supports"] for e in evidence):
        return VERIFIED
    return UNVERIFIED


def _read_evidence(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in raw if isinstance(raw, list) else []:
        if not isinstance(e, dict):
            continue
        channel = str(e.get("channel", "unknown"))
        out.append(
            {
                "channel": channel,
                "label": CHANNEL_LABEL.get(channel, channel.replace("_", " ")),
                "summary": str(e.get("summary", "")),
                "supports": bool(e.get("supports", True)),
                # Recomputed, never read from the file.
                "independent": channel in _INDEPENDENT,
                "content_hash": e.get("content_hash") or "",
                "artifact_path": e.get("artifact_path") or "",
            }
        )
    return out


def _read_claim(raw: dict[str, Any]) -> dict[str, Any]:
    evidence = _read_evidence(raw.get("evidence"))
    stored = str(raw.get("verdict", UNVERIFIED)).upper()
    if stored not in _SEVERITY:
        stored = UNVERIFIED
    derived = _derive_verdict(evidence)
    verdict = stored if _SEVERITY[stored] <= _SEVERITY[derived] else derived
    return {
        "description": str(raw.get("description", "(no description)")),
        "expected": str(raw.get("expected_side_effect", "")),
        "verdict": verdict,
        "stamp": STAMP[verdict],
        "downgraded": verdict != stored,
        "evidence": evidence,
        "independent_count": sum(1 for e in evidence if e["independent"]),
    }


def _read_receipt(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("claims"), list):
        return None

    claims = [_read_claim(c) for c in raw["claims"] if isinstance(c, dict)]
    counts = {
        VERIFIED: sum(1 for c in claims if c["verdict"] == VERIFIED),
        UNVERIFIED: sum(1 for c in claims if c["verdict"] == UNVERIFIED),
        CONTRADICTED: sum(1 for c in claims if c["verdict"] == CONTRADICTED),
    }
    if counts[CONTRADICTED]:
        verdict = CONTRADICTED
    elif claims and counts[UNVERIFIED] == 0:
        verdict = VERIFIED
    else:
        verdict = UNVERIFIED

    return {
        "id": str(raw.get("id", path.stem)),
        "task": str(raw.get("task", "(untitled run)")),
        "started_at": str(raw.get("started_at", "")),
        "ended_at": str(raw.get("ended_at", "") or ""),
        "recording": str(raw.get("call_recording") or ""),
        "claims": claims,
        "counts": counts,
        "verdict": verdict,
        "stamp": STAMP[verdict] if claims else "NO CLAIMS",
        "empty": not claims,
    }


def load_receipts(directory: Path | str = EVIDENCE_DIR) -> list[dict[str, Any]]:
    """Every parseable receipt in `directory`, newest first.

    Non-receipt JSON is skipped rather than raising: `evidence/` is a working
    directory during a hackathon and picks up stray artifacts.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    found = [r for p in sorted(d.glob("*.json")) if (r := _read_receipt(p))]
    found.sort(key=lambda r: (r["started_at"], r["id"]), reverse=True)
    return found


def totals(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """Booked vs proven.

    BOOKED is every claim the agent filed — everything it believes it did.
    PROVEN is the subset an independent channel confirmed. A CONTRADICTED claim
    is booked and is *not* proven; it is broken out separately so the gap is
    never quietly attributed to slow SMS.
    """
    booked = sum(len(r["claims"]) for r in receipts)
    proven = sum(r["counts"][VERIFIED] for r in receipts)
    contradicted = sum(r["counts"][CONTRADICTED] for r in receipts)
    unconfirmed = sum(r["counts"][UNVERIFIED] for r in receipts)
    return {
        "booked": booked,
        "proven": proven,
        "unconfirmed": unconfirmed,
        "contradicted": contradicted,
        "gap": booked - proven,
        "runs": len(receipts),
        "silent_runs": sum(1 for r in receipts if r["empty"]),
        "proven_pct": round(100 * proven / booked) if booked else 0,
    }


def _key(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build(
    directory: Path | str = EVIDENCE_DIR, registry: OpsRegistry | None = None
) -> dict[str, Any]:
    """The whole board as data, plus a content key per panel.

    The keys are hashes of panel *content* only. Nothing wall-clock goes in, so
    a board nobody has touched returns byte-identical keys forever and the page
    never repaints — which is the whole reason the poll is cheap and the reason
    focus survives it.
    """
    reg = registry if registry is not None else OPS
    receipts = load_receipts(directory)
    live = reg.snapshot()

    # The wall is picked by *substance*, not recency. Sixty-two of the runs in
    # a hackathon evidence directory are smoke tests that filed no claims; if
    # the newest twenty-four win, the one receipt a judge came to see is off
    # the bottom of the screen. Claim-bearing receipts get the cards, everything
    # else is counted honestly in one line underneath.
    claimful = [r for r in receipts if not r["empty"]]
    shown = claimful[:WALL_LIMIT]
    wall = {"receipts": shown, "hidden": len(receipts) - len(shown)}
    metric = totals(receipts)
    call = {"call": live["call"], "rail": _rail(live["call"])}
    guard = {"refusals": live["refusals"]}

    panels = {"metric": metric, "call": call, "guard": guard, "wall": wall}
    keys = {name: _key(data) for name, data in panels.items()}
    return {
        "panels": panels,
        "keys": keys,
        "state_key": _key(keys),
    }


def _rail(call: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The phase graph, resolved against where the call actually is.

    `unreachable` is the state that carries the argument: a phase with no path
    from here, computed by walking the edge list rather than asserted. Once the
    call is QUOTED, DISCOVERY is gone — not discouraged, gone — because no edge
    leads back. `sealed` marks a phase already behind the call that also cannot
    be re-entered, which is the same fact told about the past instead of the
    future, and is what makes the rail read as one-way on the projector.
    """
    from src.opsboard.registry import (
        PHASE_GLOSS,
        PHASE_RAIL,
        SPUR_PHASE,
        TRANSITIONS,
        reachable_from,
    )

    now = (call or {}).get("phase") or "opening"
    if now not in PHASE_RAIL and now != SPUR_PHASE:
        now = "opening"
    ahead = reachable_from(now)
    direct = set(TRANSITIONS.get(now, frozenset()))
    order = {p: i for i, p in enumerate(PHASE_RAIL)}
    here = order.get(now, order["qualified"] if now == SPUR_PHASE else 0)

    rail = []
    for phase in (*PHASE_RAIL, SPUR_PHASE):
        sealed = False
        if phase == now:
            state = "now"
        elif phase in order and order[phase] < here:
            state = "past"
            sealed = phase not in ahead
        elif phase in direct:
            state = "next"
        elif phase in ahead:
            state = "ahead"
        else:
            state = "unreachable"
        rail.append(
            {
                "phase": phase,
                "label": phase.upper(),
                "gloss": PHASE_GLOSS.get(phase, ""),
                "state": state,
                "sealed": sealed,
                "spur": phase == SPUR_PHASE,
            }
        )
    return rail
