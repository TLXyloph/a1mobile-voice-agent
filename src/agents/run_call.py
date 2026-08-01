"""Place a real outbound call and run the sales agent on it.

Two entry points, because they run as separate processes:

    # 1. the worker - keep this running in one terminal
    .venv/bin/python -m src.agents.run_call dev

    # 2. place a call - from another terminal
    .venv/bin/python scripts/place_call.py +14155550142

Voice is routed through LiveKit inference rather than OpenAI directly. The
a1mobile gateway is text-only - no /realtime, no /audio - and rejects every
model name we tried, so speech-to-speech is off the table. The cascade through
LiveKit costs us some interruption quality and buys a stack that actually works,
which at this hour is the right trade.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / "config" / ".env")

from livekit import agents  # noqa: E402
from livekit.agents import (  # noqa: E402
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    inference,
)
from livekit.plugins import silero  # noqa: E402

from src.agents.sales_agent import (  # noqa: E402
    CallSession,
    SalesAgent,
    VoiceOperator,
    build_instructions,
)
from src.business.campaign import get_campaign  # noqa: E402
from src.business.capacity import CapacityLedger  # noqa: E402
from src.business.pricing import CostModel  # noqa: E402
from src.business.profile_overlay import apply_business_profile  # noqa: E402
from src.verify.receipts import Receipt  # noqa: E402

logger = logging.getLogger("run-call")

server = AgentServer()


def _direct_openai_key() -> str | None:
    """A genuine OpenAI key, or None.

    a1mobile issues `a1hk_...` keys against a gateway that is text-only: no
    /realtime, no /audio, and every model name rejected. Those 401 against
    api.openai.com, so speech-to-speech needs a real `sk-` key.
    OPENAI_REALTIME_KEY carries one without disturbing the gateway key that
    other parts of the system may still use.
    """
    for var in ("OPENAI_REALTIME_KEY", "OPENAI_API_KEY"):
        key = os.getenv(var, "")
        if key.startswith("sk-"):
            return key
    return None


def build_session() -> AgentSession:
    """Speech-to-speech when a real OpenAI key exists; cascade otherwise.

    Realtime is materially better on a phone call: one model both hears and
    speaks, so a barge-in lands in roughly half the time the STT->LLM->TTS chain
    needs. Measured here, the cascade costs about 1.2s per turn, and that gap is
    most of what "sounds like a bot" actually means.

    Falls back rather than failing: arriving at a demo with REALTIME=1 and no
    valid key should degrade to a working call, not to no call.
    """
    key = _direct_openai_key()
    if os.getenv("REALTIME", "0") == "1":
        if key is None:
            logger.warning(
                "REALTIME=1 but no sk- key found - using cascade. a1mobile's "
                "a1hk_ gateway key cannot do speech-to-speech (no /realtime route)."
            )
        else:
            from livekit.plugins import openai as oai

            logger.info("using OpenAI realtime speech-to-speech")
            return AgentSession(
                llm=oai.realtime.RealtimeModel(
                    model=os.getenv("REALTIME_MODEL", "gpt-realtime"),
                    voice=os.getenv("REALTIME_VOICE", "marin"),
                    api_key=key,
                ),
                vad=silero.VAD.load(),
            )

    return AgentSession(
        stt=inference.STT(os.getenv("STT_MODEL", "deepgram/nova-3")),
        llm=inference.LLM(model=os.getenv("LLM_MODEL", "openai/gpt-5.4-mini")),
        tts=inference.TTS(model=os.getenv("TTS_MODEL", "cartesia/sonic-2")),
        vad=silero.VAD.load(),
    )


WEBAPP_SPEC = ROOT / "config" / "webapp.json"


def _spec_overrides() -> dict:
    """Whatever the webapp intake last launched.

    Without this the intake is theatre: the operator answers questions, /launch
    writes a spec, and the worker builds its session from env vars anyway - so
    nothing they said reaches the call.
    """
    try:
        d = json.loads(WEBAPP_SPEC.read_text())
        return d.get("worker_env") or d.get("spec") or d
    except Exception:  # noqa: BLE001
        return {}


def _cfg(key: str, default: str) -> str:
    spec = _spec_overrides()
    for k in (key, key.lower()):
        if spec.get(k) not in (None, ""):
            return str(spec[k])
    return os.getenv(key, default)


def build_call_session() -> CallSession:
    """Assemble the business context for one call.

    Precedence: webapp launch spec > environment > default. The intake has to
    win, or answering its questions changes nothing about the call."""
    campaign = get_campaign(_cfg("CAMPAIGN", "restaurant_catering"))
    costs = CostModel(
        materials_per_unit=_cfg("COST_MATERIALS", "0.80"),
        labor_per_unit=_cfg("COST_LABOR", "0.40"),
        transport_per_unit=_cfg("COST_TRANSPORT", "0.15"),
        min_margin_pct=_cfg("MIN_MARGIN_PCT", "30"),
        target_margin_pct=_cfg("TARGET_MARGIN_PCT", "45"),
        unit=_cfg("CAPACITY_UNIT", "muffin"),
    )
    # The owner sets their limits once, during intake. Without this the profile
    # is written and then ignored, and the campaign preset silently wins.
    campaign = apply_business_profile(campaign, costs)

    return CallSession(
        campaign=campaign,
        ledger=CapacityLedger(
            int(_cfg("CAPACITY_TOTAL", "400")),
            _cfg("CAPACITY_UNIT", "muffins"),
        ),
        costs=costs,
        receipt=Receipt(task=_cfg("ERRAND_TASK", "Sell catering to a local event host")),
        operator=VoiceOperator(),
    )


@server.rtc_session(agent_name="sales-agent")
async def entrypoint(ctx: JobContext) -> None:
    call = build_call_session()
    business = {"name": _cfg("BUSINESS_NAME", "Rosewater Bakehouse")}
    agent = SalesAgent(call, build_instructions(call.campaign, business))
    session = build_session()

    confirmed = False
    try:
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(close_on_disconnect=False),
        )
        # Feed the caller's own transcribed words to the agent so close_order can
        # check a claimed agreement against what was actually said.
        @session.on("user_input_transcribed")
        def _heard(ev) -> None:  # noqa: ANN001
            text = getattr(ev, "transcript", "") or ""
            if getattr(ev, "is_final", True) and text.strip():
                call.heard.append(text.strip())

        await session.generate_reply(
            instructions=(
                "Introduce yourself by name and business, say in a few words what "
                "you do, then ask a direct question inviting them to buy - naming "
                "one concrete reason to care (freshness, delivery included, "
                "usually cheaper than their current supplier). Two sentences at "
                "most, then stop and let them respond."
            )
        )
        # A call only counts as confirmed when a claim actually verified - the
        # agent's own sense of how it went is explicitly not consulted here.
        confirmed = bool(call.receipt.verified)
    finally:
        receipt = await agent.finish(confirmed=confirmed)
        logger.info("receipt: %s", receipt.headline)
        print("\n" + receipt.render())
        if call.escalations:
            logger.warning("escalations this call: %s", call.escalations)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)
