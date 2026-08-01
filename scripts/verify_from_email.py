"""Promote a claim using a confirmation email.

The Python process has no Gmail credential - the mailbox is reachable through
the operator's connected Gmail MCP, which lives in the assistant, not here. So
the email text is passed in and matched here, keeping the matching rules
identical to every other channel.

What matters for the invariant is unchanged: the message originates with the
prospect, arrives through a channel the agent cannot write to, and every
expected token must appear. A vague "confirmed" still verifies nothing.

    .venv/bin/python scripts/verify_from_email.py \
        --sender someone@example.com \
        --body "Confirmed 200 croissants, $400, Friday 8am" \
        --tokens 200 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verify.inbox_email import verify_claim_via_email  # noqa: E402
from src.verify.receipts import (  # noqa: E402
    Channel, Claim, Evidence, Receipt, Verdict,
)

EVIDENCE = ROOT / "evidence"


def load(path: Path) -> Receipt:
    """Rebuild a Receipt from disk, re-deriving verdicts from evidence.

    Stored verdicts are ignored on purpose. A receipt file is data, not
    authority - if it were trusted, editing one line would forge a success.
    """
    d = json.loads(path.read_text())
    r = Receipt(task=d["task"], started_at=d.get("started_at", ""), id=d["id"])
    r.ended_at = d.get("ended_at")
    r.notes = d.get("notes", [])
    for c in d.get("claims", []):
        claim = Claim(
            description=c["description"],
            expected_side_effect=c.get("expected_side_effect", ""),
            created_at=c.get("created_at", r.started_at),
            id=c["id"],
        )
        for e in c.get("evidence", []):
            claim.attach_evidence(Evidence(
                channel=Channel(e["channel"]),
                summary=e["summary"],
                supports=e.get("supports", True),
                artifact_path=e.get("artifact_path"),
                captured_at=e.get("captured_at", ""),
                id=e["id"],
                content_hash=e.get("content_hash"),
            ))
        r.claims.append(claim)
    return r


def latest_with_claims() -> Path | None:
    files = sorted(EVIDENCE.glob("receipt_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            if json.loads(f.read_text()).get("claims"):
                return f
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", help="path; default = newest receipt WITH claims")
    ap.add_argument("--sender", required=True)
    ap.add_argument("--subject", default="")
    ap.add_argument("--body", required=True)
    ap.add_argument("--tokens", nargs="+", required=True,
                    help="every one must appear, e.g. 200 400 Friday")
    args = ap.parse_args()

    path = Path(args.receipt) if args.receipt else latest_with_claims()
    if path is None:
        print("No receipt with claims found. The call filed nothing to verify.")
        return 1

    receipt = load(path)
    print(f"receipt : {path.name}")
    print(f"before  : {receipt.headline}")

    msg = [{"from": args.sender, "subject": args.subject,
            "body": args.body, "date": ""}]

    promoted = 0
    for claim in receipt.claims:
        if claim.verdict is Verdict.VERIFIED:
            continue
        if verify_claim_via_email(claim, msg, args.tokens):
            promoted += 1
            print(f"  VERIFIED: {claim.description[:70]}")

    if not promoted:
        print(f"  no claim matched all of {args.tokens} - nothing promoted.")
        print("  (that is the correct outcome for a partial confirmation)")

    path.write_text(json.dumps(receipt.to_dict(), indent=2))
    print(f"after   : {receipt.headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
