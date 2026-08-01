"""Synthesized sample calls, for when the evidence directory is nearly empty.

A dashboard whose thesis is "prove your outcomes" must not show fixtures as if
they were real calls, so every receipt built here is flagged `is_sample` on the
way into the store and badged in the UI, and the money panels report the
real-only totals next to the headline.

The receipts are built as ordinary receipt dicts and pushed through the same
`ingest_receipt` path as real files - no special insert. That means the
verdicts on sample rows are derived by the same `Claim.verdict` as everything
else: the CONTRADICTED sample below is contradicted because it carries an
inbound SMS with `supports: false`, not because a fixture said so.

The scenarios are chosen to make the failure modes visible rather than to make
the numbers look good: a contradicted order, two unconfirmed ones, and a call
where the floor bound and the deal still closed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from src.verticals.restaurant import config as cfg

#: Deterministic ids: re-seeding overwrites the same rows instead of stacking
#: a new set of fake calls on every restart.
_PREFIX = "receipt_sample"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _iso(base: datetime, minutes: int) -> str:
    return (base + timedelta(minutes=minutes)).isoformat()


#: (slug, task, to_number, qty, total, delivery, outcome-shape)
#: outcome-shape drives which evidence gets attached, which is what decides
#: the verdict - see the module docstring.
SCENARIOS: list[dict[str, Any]] = [
    {
        "slug": "northside-clinic",
        "task": "Standing weekly breakfast order - Northside Clinic",
        "to": "+14155550188",
        "qty": 120,
        "total": 318.00,
        "when": "Tuesday 7:30am",
        "shape": "verified_sms",
        "note": "Front desk texted the confirmation while the agent was still on the line.",
    },
    {
        "slug": "harbor-offsite",
        "task": "One-off offsite catering - Harbor Logistics",
        "to": "+14155550143",
        "qty": 240,
        "total": 612.00,
        "when": "Thursday 8am",
        "shape": "verified_email",
        "note": "Office manager replied by email with the PO number.",
    },
    {
        "slug": "pinecrest-school",
        "task": "Fall term breakfast block - Pinecrest School",
        "to": "+14155550119",
        "qty": 300,
        "total": 726.00,
        "when": "Monday 7am",
        "shape": "unverified",
        "note": "Verbal yes from the office, nothing in writing yet. Chase before Friday.",
    },
    {
        "slug": "greenline-coworking",
        "task": "Weekly members breakfast - Greenline Coworking",
        "to": "+14155550164",
        "qty": 80,
        "total": 202.00,
        "when": "Wednesday 8:30am",
        "shape": "unverified",
        "note": "Community manager said she needs to check the budget line.",
    },
    {
        "slug": "vantage-partners",
        "task": "Client breakfast - Vantage Partners",
        "to": "+14155550177",
        "qty": 60,
        "total": 158.00,
        "when": "Friday 8am",
        "shape": "contradicted",
        "note": (
            "Agent believed this closed. The reply text cancelled it. This row is why "
            "booked and proven are shown separately."
        ),
    },
    {
        "slug": "atlas-dental",
        "task": "Monthly staff breakfast - Atlas Dental",
        "to": "+14155550152",
        "qty": 45,
        "when": "2026-08-19",
        "shape": "verified_sms",
        "at_floor": True,
        "note": "Priced at the floor. The agent held there through three pushes.",
    },
    {
        "slug": "riverside-gym",
        "task": "Saturday members morning - Riverside Gym",
        "to": "+14155550131",
        "qty": 150,
        "total": 372.00,
        "when": "Saturday 7am",
        "shape": "verified_sms",
        "note": "Owner texted 'confirmed' with the quantity spelled out.",
    },
    {
        "slug": "summit-clinic",
        "task": "New office opening - Summit Clinic",
        "to": "+14155550196",
        "qty": 200,
        "total": 486.00,
        "when": "Wednesday 8am",
        "shape": "unverified",
        "note": "Practice manager away until Monday; the receptionist would not confirm.",
    },
    {
        "slug": "lakeside-agency",
        "task": "Quarterly all-hands - Lakeside Agency",
        "to": "+14155550107",
        "qty": 90,
        "total": 228.00,
        "when": "Thursday 9am",
        "shape": "verified_email",
        "note": "Ops lead forwarded the confirmation from their calendar invite.",
    },
    {
        "slug": "borough-market",
        "task": "Weekend market stall resupply - Borough Market",
        "to": "+14155550122",
        "qty": 400,
        "when": "Saturday 6am",
        "shape": "unverified",
        "at_floor": True,
        "note": (
            "Volume deal talked all the way down to the floor. Still unconfirmed, so "
            "it is the worst kind of row: thin margin and no proof."
        ),
    },
]


def _total_for(sc: dict[str, Any]) -> float:
    """The agreed total for a scenario.

    Scenarios flagged `at_floor` take their number from the real cost model
    rather than a hand-typed one, so "how often did the floor bind" is a
    measurement of `src/business/pricing.py` and not of a fixture author's
    arithmetic.
    """
    if sc.get("at_floor"):
        return float(cfg.default().cost_model.floor_price(int(sc["qty"])))
    return float(sc["total"])


def _evidence_for(
    shape: str, sc: dict[str, Any], base: datetime, total: float
) -> list[dict[str, Any]]:
    unit = cfg.default().unit
    agent = {
        "id": f"ev_{sc['slug']}_agent",
        "channel": "agent_assertion",
        "summary": f"Caller agreed to {sc['qty']} at {total:.2f} for {sc['when']}",
        "supports": True,
        "independent": False,
        "artifact_path": None,
        "content_hash": None,
        "captured_at": _iso(base, 6),
    }
    if shape == "unverified":
        # No independent channel at all. This is the honest shape of a call
        # that went well and has not been confirmed yet.
        return [agent]

    if shape == "contradicted":
        body = f"Sorry - we have to cancel the {sc['qty']} for {sc['when']}."
        return [
            agent,
            {
                "id": f"ev_{sc['slug']}_sms",
                "channel": "inbound_sms",
                "summary": f"SMS {sc['to']}: {body!r}",
                "supports": False,
                "independent": True,
                "artifact_path": None,
                "content_hash": _hash(body),
                "captured_at": _iso(base, 41),
            },
        ]

    if shape == "verified_email":
        body = (
            f"Confirmed: {sc['qty']} {unit}, ${total:.2f}, delivery {sc['when']}."
        )
        return [
            agent,
            {
                "id": f"ev_{sc['slug']}_email",
                "channel": "inbound_email",
                "summary": f"Email from {sc['to']}: {body!r}",
                "supports": True,
                "independent": True,
                "artifact_path": None,
                "content_hash": _hash(body),
                "captured_at": _iso(base, 18),
            },
        ]

    body = f"Confirmed {sc['qty']} {unit}, ${total:.0f}, {sc['when']}"
    return [
        agent,
        {
            "id": f"ev_{sc['slug']}_sms",
            "channel": "inbound_sms",
            "summary": f"SMS {sc['to']}: {body!r}",
            "supports": True,
            "independent": True,
            "artifact_path": None,
            "content_hash": _hash(body),
            "captured_at": _iso(base, 9),
        },
    ]


def _headline(shape: str) -> str:
    if shape == "contradicted":
        return "FAILED - 1 claim(s) contradicted by independent evidence."
    if shape == "unverified":
        return "PARTIAL - 0/1 verified, 1 could not be confirmed."
    return "SUCCESS - all 1 claim(s) independently verified."


def sample_receipts(count: int = 8, *, now: datetime | None = None) -> list[dict[str, Any]]:
    """Build up to `count` sample receipts, newest first.

    Shaped exactly like a real receipt from `Receipt.to_dict()` so they can go
    through `ingest_receipt` unchanged. The `sample` key is extra and ignored
    by ingest; the `is_sample` flag on the row is what the UI reads.
    """
    unit = cfg.default().unit
    base_now = now or datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for i, sc in enumerate(SCENARIOS[: max(0, count)]):
        started = base_now - timedelta(hours=6 * (i + 1))
        total = _total_for(sc)
        evidence = _evidence_for(sc["shape"], sc, started, total)
        claim_id = f"claim_sample_{sc['slug']}"
        out.append(
            {
                "id": f"{_PREFIX}_{sc['slug']}",
                "task": sc["task"],
                "headline": _headline(sc["shape"]),
                "started_at": started.isoformat(),
                "ended_at": _iso(started, 7),
                "call_recording": None,
                "room": f"sample-{sc['slug']}",
                "to_number": sc["to"],
                "sample": True,
                "totals": {
                    "claims": 1,
                    "verified": 1 if sc["shape"].startswith("verified") else 0,
                    "unverified": 1 if sc["shape"] == "unverified" else 0,
                    "contradicted": 1 if sc["shape"] == "contradicted" else 0,
                },
                "claims": [
                    {
                        "id": claim_id,
                        "description": (
                            f"{sc['qty']} {unit} for {total:.2f}, "
                            f"delivered {sc['when']}"
                        ),
                        "expected_side_effect": (
                            f"written confirmation arrives at {sc['to']} stating "
                            f"quantity {sc['qty']} and total {total:.2f}"
                        ),
                        # Deliberately omitted: no "verdict" key. Ingest derives it.
                        "created_at": started.isoformat(),
                        "evidence": evidence,
                    }
                ],
                "notes": [f"{started.isoformat()} sample scenario: {sc['note']}"],
            }
        )
    return out
