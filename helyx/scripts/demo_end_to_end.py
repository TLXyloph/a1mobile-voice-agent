"""Headless end-to-end run against the real model gateway.

Drives intake -> mandate -> negotiation under planted friction -> inbound SMS
verification -> email loop -> receipt, printing what is actually established at
each step.

    .venv/bin/python helyx/scripts/demo_end_to_end.py

Set HELYX_SEND_SMS=1 to also send a real confirmation text (needs a recipient
in HELYX_SMS_TO). Off by default so the demo contacts nobody.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helyx.email_loop import run_email_loop  # noqa: E402
from helyx.intake import IntakeAgent  # noqa: E402
from helyx.llm import LLMClient  # noqa: E402
from helyx.negotiator import Negotiation, Negotiator  # noqa: E402
from helyx.sms import normalise_inbound  # noqa: E402
from helyx.store import HelyxStore  # noqa: E402

RULE = "=" * 72


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> int:
    store = HelyxStore()
    client = LLMClient()
    print(f"model={client.model}  fallback={client.fallback_model}")

    # --- 1. intake ------------------------------------------------------
    section("1. INTAKE - operator supplies the parameters that bound the deal")
    agent = IntakeAgent(client)
    for line in [
        "I need 120 sourdough loaves from Kestrel Bakehouse for our Saturday service.",
        "I want to pay about $4.25 a loaf and I will not go above $5.00. Need them by 2026-08-14.",
    ]:
        print(f"\n operator> {line}")
        agent.turn(store.intake, line)
        print(f" helyx   > {store.intake.last_reply}")
        print(f" [slots filled: {sorted(store.intake.slots)}]")
        print(f" [missing: {store.intake.missing}  ready: {store.intake.ready}]")

    if not store.intake.ready:
        print("\n intake did not validate; call stays blocked. That is the correct outcome.")
        print(f" reason: {store.intake.validation_error() or store.intake.missing}")
        return 1

    mandate = store.intake.mandate()
    print(f"\n MANDATE: {mandate.quantity} x {mandate.item}")
    print(f"   target  {mandate.target_unit_price_cents / 100:.2f}/unit")
    print(f"   ceiling {mandate.ceiling_unit_price_cents / 100:.2f}/unit  <- enforced in code")

    # --- 2. negotiation under friction ----------------------------------
    section("2. NEGOTIATION - the supplier pushes past the ceiling")
    store.negotiation = Negotiation(mandate=mandate)
    negotiator = Negotiator(client)
    print(f" ladder (authorised offers): "
          f"{[f'${c / 100:.2f}' for c in store.negotiation.guard.ladder.schedule()]}")
    print(f"\n helyx > {negotiator.opening_line(store.negotiation)}")

    friction = [
        "Flour costs are up, so the best I can do is $5.80 a loaf.",
        "Look, everyone else pays $6.50. I can't move much below that.",
        "Fine - meet me at $4.80 a loaf and we have a deal for the 14th.",
    ]
    for said in friction:
        if store.negotiation.finished:
            break
        print(f"\n supplier> {said}")
        turn = negotiator.turn(store.negotiation, said)
        print(f" helyx   > {turn.agent_said}")
        print(f" [authorised this round: ${turn.authorized_offer_cents / 100:.2f}]")
        if turn.violations:
            for v in turn.violations:
                print(f" [BLOCKED] {v.text}: {v.reason}")
            print(" [line was replaced with a mandate-safe reply]")
        if turn.decision:
            print(f" [decision: {turn.decision.move.value} - {turn.decision.reason}]")

    print(f"\n outcome: {store.negotiation.outcome.value}")

    # --- 3. what is actually established --------------------------------
    section("3. EVIDENCE - what the agent's word alone is worth")
    for p in store.proposals:
        print(f"  {p.status.value.upper():13s} {p.terms.quantity} x {p.terms.item} "
              f"@ ${p.terms.unit_price_cents / 100:.2f}")
        print(f"                -> {p.why()}")
    print(f"\n headline: {store.headline()}")

    # --- 4. independent confirmation ------------------------------------
    section("4. INDEPENDENT CHANNEL - inbound SMS arrives")
    if os.getenv("HELYX_SEND_SMS") == "1" and os.getenv("HELYX_SMS_TO"):
        from helyx.sms import confirmation_request, send_sms

        p = store.proposals[-1]
        body = confirmation_request(
            p.terms.quantity, p.terms.item, p.terms.unit_price_cents, p.terms.fulfilment_date
        )
        result = send_sms(os.environ["HELYX_SMS_TO"], body)
        print(f" sent real SMS: {result.to_dict()}")
    else:
        print(" (no real SMS sent; simulating the inbound webhook payload)")

    if store.proposals:
        p = store.proposals[-1]
        vague = normalise_inbound(
            {"type": "message.received", "from": "+15551230000",
             "to": "+19378608348", "text": "yep all good", "telnyx_id": "sim_vague"}
        )
        store.attach_inbound_sms(vague)
        print(f"\n inbound 'yep all good' -> status: {p.status.value}")
        print(f"   {p.why()}")

        exact = normalise_inbound(
            {"type": "message.received", "from": "+15551230000", "to": "+19378608348",
             "text": f"Confirmed: {p.terms.quantity} sourdough loaves at "
                     f"${p.terms.unit_price_cents / 100:.2f} each on {p.terms.fulfilment_date}.",
             "telnyx_id": "sim_exact"}
        )
        store.attach_inbound_sms(exact)
        print(f"\n inbound restating the terms -> status: {p.status.value}")
        print(f"   {p.why()}")

    # --- 5. email loop + receipt ----------------------------------------
    section("5. EMAIL LOOP - check inbound, then report to the operator")
    record = run_email_loop(store)
    print(f" inbound email found : {record['inbound_found']}")
    print(f" report to           : {record['to']}")
    print(f" subject             : {record['subject']}")
    print(f" transport           : {record['outcome']['transport']}")
    print(f" delivered           : {record['outcome']['delivered']}  ({record['outcome']['detail']})")
    if record["outcome"]["path"]:
        print(f" stored at           : {record['outcome']['path']}")

    receipt = store.write_receipt()
    print(f"\n receipt: {receipt}")
    print(f"\n FINAL: {store.headline()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
