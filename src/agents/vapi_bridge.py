"""Vapi bridge - run the same agent on Vapi's telephony instead of LiveKit SIP.

a1mobile's SIP trunk has no Outbound Voice Profile assigned, so outbound INVITEs
are rejected 403 and no call can be placed. Vapi provisions its own numbers and
sidesteps that entirely.

What matters is that this is a *transport* swap, not a rewrite. Vapi runs the
voice loop; every decision that could cost money or credibility still happens
here, in the same modules the LiveKit path uses:

    check_capacity  -> CapacityLedger.hold()       may refuse
    propose_price   -> CostModel.validate_quote()  may refuse
    ask_operator    -> OperatorChannel             defaults to no
    close_order     -> Receipt.claim()             born UNVERIFIED

So the safety properties are identical on both transports, and the 181 tests
covering them still apply. Vapi's LLM can no more fabricate a sale than
LiveKit's could - the tools it is given simply do not permit it.

Mount on the existing public listener:
    uvicorn src.verify.webhooks:app --port 8080
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Request

from src.agents.sales_agent import (
    CallSession,
    ConsoleOperator,
    SalesAgent,
    build_instructions,
)
from src.business.campaign import get_campaign
from src.business.capacity import CapacityLedger
from src.business.pricing import CostModel
from src.business.profile_overlay import apply_business_profile
from src.verify.receipts import Receipt

logger = logging.getLogger("agents.vapi")

router = APIRouter()

#: One live agent per Vapi call id. Vapi is stateless between tool calls, so the
#: capacity hold and negotiation state have to live here or they reset every
#: turn - and a reset hold is an oversold week.
_SESSIONS: dict[str, SalesAgent] = {}


def _new_agent(call_id: str) -> SalesAgent:
    campaign = get_campaign(os.getenv("CAMPAIGN", "restaurant_catering"))
    costs = CostModel(
        materials_per_unit=os.getenv("COST_MATERIALS", "0.80"),
        labor_per_unit=os.getenv("COST_LABOR", "0.40"),
        transport_per_unit=os.getenv("COST_TRANSPORT", "0.15"),
        min_margin_pct=os.getenv("MIN_MARGIN_PCT", "30"),
        target_margin_pct=os.getenv("TARGET_MARGIN_PCT", "45"),
        unit=os.getenv("CAPACITY_UNIT", "muffin"),
    )
    # Same overlay as the LiveKit path. If only one transport honoured the
    # owner's caps, which limits applied would depend on plumbing they cannot
    # see - the discount cap would be 10% or 15% depending on the rail.
    campaign = apply_business_profile(campaign, costs)
    session = CallSession(
        campaign=campaign,
        ledger=CapacityLedger(
            int(os.getenv("CAPACITY_TOTAL", "400")),
            os.getenv("CAPACITY_UNIT", "muffins"),
        ),
        costs=costs,
        receipt=Receipt(task=os.getenv("ERRAND_TASK", "Sell catering to an event host")),
        operator=ConsoleOperator(),
    )
    business = {"name": os.getenv("BUSINESS_NAME", "Rosewater Bakehouse")}
    agent = SalesAgent(session, build_instructions(campaign, business))
    logger.info("new vapi session %s", call_id)
    return agent


def get_agent(call_id: str) -> SalesAgent:
    if call_id not in _SESSIONS:
        _SESSIONS[call_id] = _new_agent(call_id)
    return _SESSIONS[call_id]


#: Tool name -> the SalesAgent coroutine behind it. `__wrapped__` reaches past
#: livekit's @function_tool decorator to the plain coroutine, which is how the
#: test-suite drives these too - so this path is already covered.
def _tool(agent: SalesAgent, name: str):
    attr = getattr(agent, name, None)
    if attr is None:
        return None
    return getattr(attr, "__wrapped__", attr)


TOOL_NAMES = (
    "confirm_units",
    "check_capacity",
    "propose_price",
    "record_their_position",
    "next_move",
    "they_declined",
    "ask_operator",
    "close_order",
)


async def dispatch(agent: SalesAgent, name: str, args: dict[str, Any]) -> str:
    """Run one tool. Never raises - Vapi needs a string back mid-call.

    An exception here would surface to the caller as dead air, so failures are
    converted into an instruction the model can act on instead.
    """
    if name not in TOOL_NAMES:
        return f"Unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}."

    fn = _tool(agent, name)
    if fn is None:
        return f"Tool {name!r} is not available on this agent."

    try:
        return str(await fn(agent, None, **args))
    except TypeError as exc:
        logger.warning("bad args for %s: %s", name, exc)
        return f"Wrong arguments for {name}: {exc}. Check the parameter names and retry."
    except Exception as exc:  # noqa: BLE001
        logger.exception("tool %s failed", name)
        return (
            f"{name} failed ({type(exc).__name__}). Do not assume it succeeded. "
            "Tell the customer you need a moment, and try once more."
        )


@router.post("/vapi/tools")
async def vapi_tools(request: Request) -> dict[str, Any]:
    """Vapi's custom-tool webhook."""
    body = await request.json()
    message = body.get("message", {})
    call_id = (message.get("call") or {}).get("id") or "unknown-call"
    agent = get_agent(call_id)

    # Vapi has shipped both spellings; accept either rather than 500 on a rename.
    calls = message.get("toolCallList") or message.get("toolCalls") or []

    results = []
    for call in calls:
        tool_id = call.get("id") or call.get("toolCallId")
        fn = call.get("function") or call
        name = fn.get("name", "")
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            import json

            try:
                args = json.loads(args)
            except Exception:  # noqa: BLE001
                args = {}

        out = await dispatch(agent, name, args)
        logger.info("vapi tool %s(%s) -> %s", name, args, out[:120])
        results.append({"toolCallId": tool_id, "result": out})

    return {"results": results}


@router.post("/vapi/events")
async def vapi_events(request: Request) -> dict[str, Any]:
    """Call lifecycle. On end-of-call, settle capacity and write the receipt.

    Every call must produce a receipt, including one that dropped - a run with
    no receipt is indistinguishable from a run that lied.
    """
    body = await request.json()
    message = body.get("message", {})
    kind = message.get("type", "")
    call_id = (message.get("call") or {}).get("id") or "unknown-call"

    if kind in ("end-of-call-report", "status-update"):
        status = message.get("status") or message.get("endedReason") or ""
        if kind == "end-of-call-report" or status in ("ended", "completed"):
            agent = _SESSIONS.pop(call_id, None)
            if agent is not None:
                # Confirmed only if a claim actually verified. Vapi's own view of
                # how the call went is not consulted.
                confirmed = bool(agent.s.receipt.verified)
                receipt = await agent.finish(confirmed=confirmed)
                if transcript := message.get("transcript"):
                    receipt.note(f"vapi transcript: {str(transcript)[:4000]}")
                if rec := message.get("recordingUrl"):
                    receipt.call_recording = rec
                receipt.save()
                logger.info("call %s settled: %s", call_id, receipt.headline)
                return {"ok": True, "receipt": receipt.id, "headline": receipt.headline}

    return {"ok": True}


def tool_definitions(server_url: str) -> list[dict[str, Any]]:
    """Vapi tool schemas. Pass to assistant creation."""

    def tool(name: str, desc: str, props: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
            "server": {"url": f"{server_url}/vapi/tools"},
        }

    num = {"type": "number"}
    txt = {"type": "string"}
    return [
        tool("confirm_units",
             "Establish how many ITEMS the order is. Required before check_capacity. "
             "A headcount is not an item count.",
             {"units": {"type": "integer"}, "headcount": {"type": "integer"}, "basis": txt},
             ["units"]),
        tool("check_capacity",
             "Reserve capacity before quoting. Call BEFORE naming any number.",
             {"qty": {"type": "integer", "description": "units requested"}}, ["qty"]),
        tool("propose_price",
             "Check a price before saying it aloud. Never quote an unchecked number.",
             {"total": {**num, "description": "total in dollars"}}, ["total"]),
        tool("record_their_position",
             "Log who they buy from now and any budget they stated.",
             {"current_vendor": txt, "their_budget": num}, []),
        tool("next_move",
             "Ask what to do next when they push back or the conversation stalls.",
             {}, []),
        tool("they_declined",
             "Log a refusal. Two refusals and the call must end politely.", {}, []),
        tool("ask_operator",
             "Ask the business owner for permission while holding the line. "
             "Use whenever the deal would exceed what you may agree alone.",
             {"question": txt,
              "proposed_total": {**num, "description": "exact total to clear"}},
             ["question"]),
        tool("close_order",
             "File the agreed order. Does NOT complete the sale - written "
             "confirmation does.",
             {"qty": {"type": "integer"}, "total": num, "when": txt, "confirm_to": txt},
             ["qty", "total", "when", "confirm_to"]),
    ]
