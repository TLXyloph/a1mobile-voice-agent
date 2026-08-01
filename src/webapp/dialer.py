"""Placing the phone leg. One seam, two implementations, no surprises.

The real path is the one `scripts/place_call.py` already proves out: create the
room, dispatch the agent into it, *then* bridge the SIP participant. The order
is not stylistic - if the phone answers before the agent is in the room, the
first thing a human hears is silence, and they hang up.

Everything here is behind `Dialer` because the alternative is a test suite that
rings a real phone. `FakeDialer` records what would have happened and is what
`tests/test_webapp.py` uses; `app.use_dialer()` is the swap.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("webapp.dialer")

#: Must match the `agent_name` the worker registers with. See
#: `src/agents/run_call.py` - `@server.rtc_session(agent_name="sales-agent")`.
AGENT_NAME = os.getenv("WEBAPP_AGENT_NAME", "sales-agent")


@dataclass
class DialResult:
    """What happened when we tried. `answered` is the only claim of fact here."""

    room: str
    to_number: str
    dispatched: bool = False
    answered: bool = False
    detail: str = ""
    participant: str | None = None

    @property
    def ok(self) -> bool:
        return self.dispatched and self.answered

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


class Dialer(Protocol):
    async def dial(self, *, to_number: str, room: str, metadata: str) -> DialResult: ...


class LiveKitDialer:
    """The real rails: LiveKit agent dispatch plus a SIP outbound participant."""

    async def dial(self, *, to_number: str, room: str, metadata: str) -> DialResult:
        from livekit import api

        trunk = os.getenv("LIVEKIT_SIP_TRUNK_ID", "")
        if not trunk:
            return DialResult(
                room, to_number,
                detail="LIVEKIT_SIP_TRUNK_ID is unset - no trunk to dial out on",
            )

        result = DialResult(room, to_number)
        lk = api.LiveKitAPI()
        try:
            await lk.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=AGENT_NAME, room=room, metadata=metadata
                )
            )
            result.dispatched = True
            logger.info("dispatched %s into %s", AGENT_NAME, room)

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
            result.answered = True
            result.participant = participant.participant_identity
            result.detail = "answered"
            return result

        except api.TwirpError as exc:
            result.detail = f"SIP failed: {exc.code} - {exc.message}"
            logger.error("%s", result.detail)
            return result
        except Exception as exc:
            result.detail = f"dial failed: {type(exc).__name__}: {exc}"
            logger.exception("dial failed")
            return result
        finally:
            await lk.aclose()


@dataclass
class FakeDialer:
    """Records the attempt and rings nothing. The default under pytest."""

    calls: list[dict[str, str]] = field(default_factory=list)
    answered: bool = True
    detail: str = "fake dialer - no phone was rung"

    async def dial(self, *, to_number: str, room: str, metadata: str) -> DialResult:
        self.calls.append({"to": to_number, "room": room, "metadata": metadata})
        return DialResult(
            room, to_number,
            dispatched=True,
            answered=self.answered,
            detail=self.detail,
            participant=f"callee-{to_number}" if self.answered else None,
        )


def default_dialer() -> Dialer:
    """Real rails unless something says otherwise.

    `WEBAPP_DIAL=0` forces the fake, which is how you demo the UI end to end
    without spending credits or ringing someone's actual desk.
    """
    if os.getenv("WEBAPP_DIAL", "1") == "0" or os.getenv("PYTEST_CURRENT_TEST"):
        return FakeDialer()
    return LiveKitDialer()
