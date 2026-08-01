"""Dial a number and drop the sales agent onto the call.

    .venv/bin/python scripts/place_call.py +14155550142

The worker (`python -m src.agents.run_call dev`) must already be running in
another terminal - this script only creates the room, dispatches the agent into
it, and then bridges the phone leg in. Order matters: the agent has to be in the
room before the callee answers, or the first thing they hear is silence and they
hang up.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "config" / ".env")

from livekit import api  # noqa: E402

AGENT_NAME = "sales-agent"


async def place(to_number: str, *, room: str, task: str) -> int:
    trunk = os.getenv("LIVEKIT_SIP_TRUNK_ID")
    if not trunk:
        print("LIVEKIT_SIP_TRUNK_ID unset - create a trunk first.")
        return 1

    lk = api.LiveKitAPI()
    try:
        print(f"room:    {room}")
        print(f"trunk:   {trunk}")
        print(f"calling: {to_number}")

        # Agent first, phone second.
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME, room=room, metadata=task
            )
        )
        print("agent dispatched, bridging phone leg...")

        participant = await lk.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=trunk,
                sip_call_to=to_number,
                room_name=room,
                participant_identity=f"callee-{to_number}",
                participant_name=to_number,
                wait_until_answered=True,
                krisp_enabled=True,
            )
        )
        print(f"ANSWERED - participant {participant.participant_identity}")
        print("\nWatch the worker terminal for the transcript and receipt.")
        return 0

    except api.TwirpError as exc:
        print(f"\nSIP FAILED: {exc.code} - {exc.message}")
        if exc.metadata:
            print(f"  sip status: {exc.metadata}")
        print(
            "\nCommon causes: the trunk address is wrong, the SIP credentials "
            "were rejected upstream, or the number is not permitted on this "
            "trunk. Check `lk sip outbound list`."
        )
        return 1
    finally:
        await lk.aclose()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("to_number", help="E.164, e.g. +14155550142")
    p.add_argument("--room", default=None)
    p.add_argument("--task", default=os.getenv("ERRAND_TASK", "catering outreach"))
    args = p.parse_args()

    if not args.to_number.startswith("+"):
        print("Number must be E.164 and start with '+' (e.g. +14155550142)")
        return 1

    room = args.room or f"call-{args.to_number.lstrip('+')}"
    return asyncio.run(place(args.to_number, room=room, task=args.task))


if __name__ == "__main__":
    raise SystemExit(main())
