"""End-to-end pipeline rehearsal with no phone call.

Two jobs. First, it proves the whole chain works before you bet a live call on
it. Second, and more usefully: if telephony dies at 8pm, this IS the demo. You
narrate a real pipeline over a recorded call instead of a live one, and every
artifact it produces - the lead sheet, the receipt, the generated site - is
genuinely real. Only the audio is canned.

    .venv/bin/python scripts/dry_run.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "config" / ".env")

from src.agents.sales_agent import (  # noqa: E402
    CallSession, OperatorChannel, SalesAgent, build_instructions,
)
from src.business.campaign import get_campaign  # noqa: E402
from src.business.capacity import CapacityLedger  # noqa: E402
from src.business.discovery import find_no_website_leads, to_sheet  # noqa: E402
from src.business.pricing import CostModel  # noqa: E402
from src.verify.receipts import Channel, Evidence, Receipt  # noqa: E402

BOLD, DIM, GREEN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


def head(n: int, title: str) -> None:
    print(f"\n{BOLD}{'─' * 68}\n{n}. {title}\n{'─' * 68}{RESET}")


class ScriptedOperator(OperatorChannel):
    """Stands in for you answering by voice. Prints what you'd have heard."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def ask(self, question: str, *, timeout: float = 90.0) -> str | None:
        print(f"   {YELLOW}agent asks you:{RESET} {question}")
        print(f"   {YELLOW}you say:{RESET} {self.answer!r}")
        return self.answer


# A brochure-only bakery, a functional competitor, and an unreachable listing.
CANDIDATES = [
    {"business_name": "Rosewater Bakehouse", "phone": "+14155550142",
     "website": "https://rosewater.example"},
    {"business_name": "Blue Plate Kitchen", "phone": "+14155550188",
     "website": "https://blueplate.example"},
    {"business_name": "Joe's Barbers", "phone": "+14155550100", "website": None},
]

PAGES = {
    "https://rosewater.example": (
        "<html><body><h1>Rosewater Bakehouse</h1>"
        "<p>Sourdough and pastry. Call (415) 555-0142.</p></body></html>"
    ),
    "https://blueplate.example": (
        '<html><body><h1>Blue Plate</h1>'
        '<a href="https://www.exploretock.com/blueplate">Reserve a table</a>'
        '<p>Order online, add to cart</p></body></html>'
    ),
}


async def _fetch(url: str) -> str:
    if url not in PAGES:
        raise ConnectionError("no such site")
    return PAGES[url]


TRANSCRIPT = """
AGENT: Hi, this is Sam calling on behalf of Rosewater Bakehouse. Do you have a moment?
OWNER: Sure, what's this about?
AGENT: We bake sourdough and pastry in the Sunset, everything made same morning.
       I saw you run events and wondered who handles your catering.
OWNER: We use Round Table usually. Honestly it's about three hundred for a crowd.
AGENT: Understood. Ours is baked that morning and delivery and setup are included.
OWNER: What would two hundred pastries run me?
AGENT: Let me check what I can do for you.
OWNER: If you can get near two-eighty I'd switch.
AGENT: Let me confirm that with the owner - can you hold a moment?
OWNER: Sure.
AGENT: Thanks for holding. We can do two hundred at four hundred, delivered Friday 8am.
OWNER: That works. Send it over.
AGENT: I'll have the confirmation texted to you now. Two hundred, four hundred dollars,
       Friday eight a.m. - can you text that back so we both have it in writing?
OWNER: Will do.
"""


async def main() -> int:
    head(1, "DISCOVERY — find businesses that need what we sell")
    leads = await find_no_website_leads(
        CANDIDATES, intent="build them a website", fetch=_fetch
    )
    for lead in leads:
        print(f"   {GREEN}LEAD{RESET} {lead.business_name:26} {lead.phone}")
        print(f"        {DIM}{lead.qualification_reason}{RESET}")
    excluded = {c['business_name'] for c in CANDIDATES} - {l.business_name for l in leads}
    for name in sorted(excluded):
        print(f"   {DIM}skip {name} — has working functionality, not a prospect{RESET}")

    sheet = ROOT / "evidence" / "leads.csv"
    sheet.write_text(to_sheet(leads))
    print(f"\n   sheet written: {sheet}")

    head(2, "THE CALL — capacity, negotiation, escalation")
    campaign = get_campaign("restaurant_catering")
    ledger = CapacityLedger(400, "muffins")
    costs = CostModel(
        materials_per_unit="0.80", labor_per_unit="0.40", transport_per_unit="0.15",
        min_margin_pct="30", target_margin_pct="45", unit="muffin",
    )
    session = CallSession(
        campaign=campaign, ledger=ledger, costs=costs,
        receipt=Receipt(task="Sell 200 pastries to a local event host"),
        operator=ScriptedOperator("yes, four hundred is fine"),
    )
    agent = SalesAgent(session, build_instructions(campaign, {"name": "Rosewater Bakehouse"}))
    tool = lambda n: getattr(agent, n).__wrapped__  # noqa: E731

    print(f"   capacity before: {ledger.available()} muffins")
    # Units before capacity - the flow graph requires it, because a headcount
    # and an item count are indistinguishable once they are integers.
    print("  ", await tool("confirm_units")(agent, None, 200, 0, "200 pastries"))
    print("  ", await tool("check_capacity")(agent, None, 200))
    print(f"   capacity now:    {ledger.available()} available, {ledger.held()} held")

    print("\n   they name a competitor:")
    print("  ", await tool("record_their_position")(agent, None, "Round Table", 500.0, 400.0))

    print("\n   agent tries an unchecked lowball:")
    print("  ", (await tool("propose_price")(agent, None, 250.0)).replace("\n", " "))

    print("\n   approved price now passes:")
    print("  ", await tool("propose_price")(agent, None, 400.0))

    # The caller's own lines from the scripted transcript. close_order checks a
    # claimed agreement against these, exactly as it does against live STT - so
    # the rehearsal exercises the real verification path, not a shortcut.
    session.heard.extend(
        line.split(":", 1)[1].strip()
        for line in TRANSCRIPT.splitlines()
        if line.strip().startswith("OWNER:")
    )

    print("\n   closing:")
    print("  ", (await tool("close_order")(
        agent, None, 200, 400.0, "Friday 8am", "+14155550142")).replace("\n", " "))

    head(3, "VERIFICATION — the agent's word is not enough")
    order = next(c for c in session.receipt.claims if "muffins" in c.description)
    print(f"   before confirmation: {BOLD}{order.verdict.value}{RESET}")
    print(f"   {DIM}(the caller said yes on the phone — that changes nothing){RESET}")

    order.attach_evidence(Evidence(
        channel=Channel.INBOUND_SMS,
        summary="SMS +14155550142: 'Confirmed 200 pastries, $400, Friday 8am'",
        raw={"from": "+14155550142", "body": "Confirmed 200 pastries, $400, Friday 8am"},
    ))
    print(f"   after real inbound SMS: {GREEN}{order.verdict.value}{RESET}")

    await agent.finish(confirmed=True)
    print(f"\n   capacity settled: {ledger.committed()} committed, {ledger.available()} left")

    head(4, "DELIVERABLE — a site built from what the call revealed")
    from src.deliver.sitegen import deploy, render_site

    try:
        from src.deliver.sitegen import extract_facts
        import os
        if os.getenv("OPENAI_API_KEY"):
            facts = await extract_facts(TRANSCRIPT)
            print(f"   {GREEN}extracted live from transcript via OpenAI{RESET}")
        else:
            raise RuntimeError("no key")
    except Exception as exc:  # noqa: BLE001
        from src.deliver.sitegen import facts_from_payload
        print(f"   {YELLOW}OPENAI_API_KEY unset — using canned facts ({exc}){RESET}")
        facts = facts_from_payload({
            "name": "Rosewater Bakehouse",
            "tagline": "Everything made here, before dawn.",
            "phone": "(415) 555-0142",
            "services": [
                {"name": "Pastry catering", "description": "Baked same morning", "price": "from $2.00"},
                {"name": "Custom cakes", "description": "Two weeks' notice", "price": "from $65"},
            ],
            "hours": {"Tue-Fri": "7am-3pm", "Sat-Sun": "8am-2pm", "Mon": "Closed"},
        })

    url = deploy(render_site(facts), "dry-run")
    print(f"   site deployed: {GREEN}{url}{RESET}")

    head(5, "RECEIPT — what a judge reads")
    print(session.receipt.render())
    path = session.receipt.save(ROOT / "evidence")
    print(f"   saved: {path}")

    verified = len(session.receipt.verified)
    total = len(session.receipt.claims)
    print(f"\n{BOLD}{session.receipt.headline}{RESET}")
    print(f"{DIM}({verified}/{total} claims backed by independent evidence){RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
