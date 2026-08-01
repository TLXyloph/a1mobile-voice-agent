"""Outbound errand agent.

Runs a real phone call against an uncontrolled party (a human, an IVR tree, a
hold queue) and produces a Receipt rather than a summary.

Two design choices carry most of the weight:

1. The agent's tools cannot mark anything verified. `record_claim` files an
   assertion; promotion to VERIFIED happens out of band in src/verify/. The
   model is told this explicitly, so it stops trying to reassure and starts
   trying to trigger confirmations it cannot forge.

2. Friction is a first-class path, not an error. The scoring uses planted
   friction - sold-out slots, declined cards, deliberately vague answers - so
   the prompt gives the agent a standing policy for each rather than letting it
   improvise into a hallucinated success.

Run:
    uv run src/agents/errand_agent.py console     # laptop mic, no phone needed
    uv run src/agents/errand_agent.py dev         # connect to LiveKit
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from livekit import agents  # noqa: E402
from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    function_tool,
    room_io,
)
from livekit.plugins import openai, silero  # noqa: E402

from src.verify.receipts import Channel, Evidence, Receipt  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[2] / "config" / ".env")
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger("errand-agent")


INSTRUCTIONS = """
You place real phone calls to real businesses to complete an errand end to end.

## What counts as done
You are scored ONLY on side effects that someone else can independently check -
a confirmation text that actually arrives, a booking that actually appears in
their system. Your own summary of the call is worth nothing. Saying a task
succeeded when it did not is the single worst outcome available to you; it is
worse than plainly reporting failure.

So: do not narrate success. Cause evidence.

## Always try to trigger a confirmation you cannot fake
Before ending any call, push for something that lands outside the call:
  "Could you text the confirmation to this number?"
  "What's the confirmation or reference number?"
  "Can you send that to <email>?"
A reference number you can repeat back, and they can look up, beats a promise.
Call `record_claim` for each concrete outcome, then `request_confirmation` when
you have asked them to send something.

## Handling friction
You will hit deliberate obstacles. Standing policy:

- Option unavailable (time slot gone, item sold out): do NOT accept silently and
  do NOT invent an alternative the user didn't authorise. Ask for the nearest
  options, then call `needs_decision` with the real choices. If the constraints
  you were given already cover the case, proceed and record what changed.
- Payment fails or they demand card details: never read out card details you
  were not explicitly given for this purpose. Ask whether a hold without payment
  is possible, or payment on arrival. If neither, call `needs_decision`.
- Ambiguity ("which Thursday?", "under what name?"): ask one direct clarifying
  question of the person on the phone. Do not guess at facts about your user -
  if you genuinely lack it, call `needs_decision`.
- IVR / phone tree: navigate it. Say the option or use `send_dtmf`. Do not give
  up on hold - stay on unless told the wait exceeds your time budget.
- They refuse or cannot help: ask who can, and whether there is a direct number.
  Record the outcome honestly with `record_claim` and end politely.

## Voice
You are on a phone call with a busy stranger. Short sentences. No lists, no
markdown, no emoji. Give your name and your purpose in the first breath. Speak
numbers as digits: "four" not "a party of four people". If interrupted, stop
talking and listen. Never claim to be a human if asked directly - say you are an
assistant calling on someone's behalf, then continue.
""".strip()


class ErrandAgent(Agent):
    def __init__(self, task: str, constraints: str, callback_number: str) -> None:
        super().__init__(
            instructions=(
                f"{INSTRUCTIONS}\n\n"
                f"## Your errand\n{task}\n\n"
                f"## Constraints you may act on without asking\n{constraints}\n\n"
                f"## Callback number for confirmations\n{callback_number}\n"
            )
        )
        self.receipt = Receipt(task=task)
        self.callback_number = callback_number
        self.pending_decisions: list[str] = []

    # -- tools -----------------------------------------------------------
    # None of these can mark a claim verified. That is the point.

    @function_tool
    async def record_claim(
        self, ctx: RunContext, outcome: str, how_to_check: str
    ) -> str:
        """File a concrete outcome you believe you achieved.

        Args:
            outcome: What happened, specifically. Include numbers, times, names.
            how_to_check: How someone else could independently confirm this,
                e.g. "a text from the restaurant arrives with a party of 4".
        """
        claim = self.receipt.claim(
            description=outcome, expected_side_effect=how_to_check
        )
        claim.attach_evidence(
            Evidence(channel=Channel.AGENT_ASSERTION, summary=outcome)
        )
        logger.info("claim filed: %s", outcome)
        return (
            "Recorded as UNVERIFIED. It stays unverified until an independent "
            "channel confirms it, so if you have not yet asked them to send a "
            "text or given you a reference number, do that now."
        )

    @function_tool
    async def request_confirmation(
        self, ctx: RunContext, channel: str, detail: str
    ) -> str:
        """Note that you asked them to send confirmation out of band.

        Args:
            channel: One of "sms", "email", "reference_number".
            detail: What they agreed to send, and to where.
        """
        logger.info("confirmation requested via %s: %s", channel, detail)
        return (
            f"Noted. Watching for {channel}. Stay on the line until they confirm "
            "they have sent it, then read the reference number back to check it."
        )

    @function_tool
    async def needs_decision(self, ctx: RunContext, question: str, options: str) -> str:
        """Escalate a choice you are not authorised to make.

        Args:
            question: The decision needed, in one sentence.
            options: The real options offered, verbatim.
        """
        self.pending_decisions.append(f"{question} | options: {options}")
        logger.warning("DECISION NEEDED: %s -> %s", question, options)
        return (
            "Escalated. Tell the person you need to check with the person you "
            "are calling for, and ask if you can hold the option briefly or call "
            "back. Do not pick one yourself."
        )

    @function_tool
    async def send_dtmf(self, ctx: RunContext, digits: str) -> str:
        """Press keys on a phone menu.

        Args:
            digits: The digits to press, e.g. "1" or "0" or "1234#".
        """
        room = ctx.session._room_io._room if ctx.session else None
        if room is None:
            return "No room available; say the option aloud instead."
        try:
            await room.local_participant.publish_dtmf(
                code=int(digits[0]) if digits[0].isdigit() else 0, digit=digits
            )
            return f"Pressed {digits}."
        except Exception as exc:  # noqa: BLE001
            logger.warning("dtmf failed: %s", exc)
            return "Keypad failed; say the option aloud instead."


def build_session() -> AgentSession:
    """Realtime speech-to-speech by default; cascade if REALTIME=0.

    Realtime wins on interruption handling and sounds materially more human,
    which matters for the "Most Human" category. The cascade path exists because
    it lets you swap in a stronger reasoning model for gnarly friction, and
    because having a second path saved is cheap insurance.
    """
    if os.getenv("REALTIME", "1") == "1":
        return AgentSession(
            llm=openai.realtime.RealtimeModel(
                model=os.getenv("REALTIME_MODEL", "gpt-realtime"),
                voice=os.getenv("REALTIME_VOICE", "marin"),
            ),
            vad=silero.VAD.load(),
        )

    from livekit.plugins import deepgram, elevenlabs

    return AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model=os.getenv("LLM_MODEL", "gpt-4.1")),
        tts=elevenlabs.TTS(),
        vad=silero.VAD.load(),
    )


server = AgentServer()


@server.rtc_session(agent_name="errand-agent")
async def entrypoint(ctx: JobContext) -> None:
    task = os.getenv("ERRAND_TASK", "Introduce yourself and ask what hours they keep.")
    constraints = os.getenv("ERRAND_CONSTRAINTS", "None given - escalate any choice.")
    callback = os.getenv("CALLBACK_NUMBER", "(not configured)")

    agent = ErrandAgent(task=task, constraints=constraints, callback_number=callback)
    session = build_session()

    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(close_on_disconnect=False),
        ),
    )

    try:
        await session.generate_reply(
            instructions="Greet whoever answers and state your purpose in one sentence."
        )
    finally:
        # Always emit a receipt, even on a dropped call. A crashed run that
        # writes no receipt is indistinguishable from a run that lied.
        path = agent.receipt.save(
            Path(__file__).resolve().parents[2] / "evidence"
        )
        logger.info("receipt written: %s", path)
        if agent.pending_decisions:
            logger.warning("unresolved decisions: %s", agent.pending_decisions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)
