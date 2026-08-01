"""Swappable telephony backends.

You will not see a1mobile's starter kit until kickoff. Everything above this
module is written against `TelephonyProvider`, so adopting their rails at 9am is
a matter of filling in `A1MobileProvider` and changing one env var - not
rewriting the agent.

Pick a backend with TELEPHONY_PROVIDER=livekit|a1mobile|twilio.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallHandle:
    """A placed call. `room` is where the agent and caller meet."""

    call_id: str
    room: str
    to_number: str
    provider: str
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TelephonyProvider(ABC):
    """Minimum surface the agent needs. Keep this small so ports stay cheap."""

    name: str = "abstract"

    @abstractmethod
    async def place_call(
        self, to_number: str, *, room: str, task: str, **kwargs: Any
    ) -> CallHandle: ...

    @abstractmethod
    async def hangup(self, handle: CallHandle) -> None: ...

    async def send_sms(self, to_number: str, body: str) -> str:
        raise NotImplementedError(f"{self.name} has no SMS support wired up")

    async def preflight(self) -> tuple[bool | None, str]:
        """Cheap credential check. Run this before you trust the backend.

        Returns True (ready), False (broken - fix it), or None (not ready yet,
        but legitimately so). The tri-state matters: a red failure for something
        you deliberately deferred teaches you to ignore preflight output, and
        then you miss the real failure.
        """
        return True, f"{self.name}: no preflight implemented"


class LiveKitSIPProvider(TelephonyProvider):
    """LiveKit Cloud SIP. Needs an outbound trunk and LIVEKIT_* credentials.

    Requires: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_SIP_TRUNK_ID
    """

    name = "livekit"

    def __init__(self) -> None:
        self.trunk_id = os.getenv("LIVEKIT_SIP_TRUNK_ID", "")

    async def place_call(
        self, to_number: str, *, room: str, task: str, **kwargs: Any
    ) -> CallHandle:
        from livekit import api

        if not self.trunk_id:
            raise RuntimeError(
                "LIVEKIT_SIP_TRUNK_ID is unset. Create an outbound trunk first: "
                "lk sip outbound create config/sip-outbound-trunk.json"
            )

        lkapi = api.LiveKitAPI()
        try:
            # wait_until_answered lets us fail fast on a dead number instead of
            # dropping the agent into a room nobody ever joins.
            participant = await lkapi.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=self.trunk_id,
                    sip_call_to=to_number,
                    room_name=room,
                    participant_identity=f"caller-{to_number}",
                    participant_name=to_number,
                    participant_metadata=task,
                    wait_until_answered=kwargs.get("wait_until_answered", True),
                    krisp_enabled=kwargs.get("krisp_enabled", True),
                    ringing_timeout=kwargs.get("ringing_timeout", None),
                    max_call_duration=kwargs.get("max_call_duration", None),
                )
            )
            return CallHandle(
                call_id=getattr(participant, "sip_call_id", "") or participant.participant_id,
                room=room,
                to_number=to_number,
                provider=self.name,
                raw=participant,
            )
        finally:
            await lkapi.aclose()

    async def hangup(self, handle: CallHandle) -> None:
        from livekit import api

        lkapi = api.LiveKitAPI()
        try:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=handle.room))
        finally:
            await lkapi.aclose()

    async def preflight(self) -> tuple[bool | None, str]:
        missing = [
            k
            for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
            if not os.getenv(k)
        ]
        if missing:
            return False, f"livekit: missing {', '.join(missing)}"
        if not self.trunk_id:
            # Expected until a1mobile hands over a trunk at kickoff. Not a fault.
            return None, "livekit: creds OK; no SIP trunk yet (expected until kickoff)"
        return True, "livekit: credentials + trunk configured"


class A1MobileProvider(TelephonyProvider):
    """a1mobile rails. FILL THIS IN AT KICKOFF.

    Their starter kit is not public, so this is a deliberate stub rather than a
    guess. When you get their docs, implement the three methods below and set
    TELEPHONY_PROVIDER=a1mobile. Nothing else in the codebase changes.

    What to look for in their kit:
      - outbound call creation endpoint + how audio is attached (SIP? WebRTC?
        their own SDK?). If SIP, you may be able to point LiveKit at their trunk
        and reuse LiveKitSIPProvider wholesale - check that first, it is the
        cheapest path.
      - inbound SMS webhook format -> this feeds src/verify/webhooks.py and is
        how confirmations get verified. Wire this up early; it is worth more
        points than the call itself.
      - call recording retrieval -> feeds independent transcription.
    """

    name = "a1mobile"

    def __init__(self) -> None:
        self.api_key = os.getenv("A1MOBILE_API_KEY", "")
        self.base_url = os.getenv("A1MOBILE_BASE_URL", "")
        self.from_number = os.getenv("A1MOBILE_FROM_NUMBER", "")

    async def place_call(
        self, to_number: str, *, room: str, task: str, **kwargs: Any
    ) -> CallHandle:
        raise NotImplementedError(
            "Fill in A1MobileProvider.place_call from the kickoff starter kit. "
            "First check whether they expose a SIP trunk - if so, set "
            "LIVEKIT_SIP_TRUNK_ID to it and use TELEPHONY_PROVIDER=livekit instead."
        )

    async def hangup(self, handle: CallHandle) -> None:
        raise NotImplementedError("Fill in from the kickoff starter kit.")

    async def preflight(self) -> tuple[bool | None, str]:
        if not self.api_key:
            return None, "a1mobile: A1MOBILE_API_KEY unset (expected until kickoff)"
        return True, "a1mobile: key present - verify endpoints against their docs"


class TwilioProvider(TelephonyProvider):
    """Fallback rails. Only useful if you buy a number.

    Kept as insurance: if a1mobile's kit is broken or oversubscribed at 9am, a
    Twilio number is ~$1.15 and unblocks you in about ten minutes.
    """

    name = "twilio"

    def __init__(self) -> None:
        self.sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER", "")

    def _client(self):
        from twilio.rest import Client

        return Client(self.sid, self.token)

    async def place_call(
        self, to_number: str, *, room: str, task: str, **kwargs: Any
    ) -> CallHandle:
        call = self._client().calls.create(
            to=to_number,
            from_=self.from_number,
            url=kwargs.get("twiml_url") or os.getenv("TWILIO_TWIML_URL", ""),
            record=kwargs.get("record", True),
        )
        return CallHandle(
            call_id=call.sid, room=room, to_number=to_number, provider=self.name, raw=call
        )

    async def hangup(self, handle: CallHandle) -> None:
        self._client().calls(handle.call_id).update(status="completed")

    async def send_sms(self, to_number: str, body: str) -> str:
        msg = self._client().messages.create(
            to=to_number, from_=self.from_number, body=body
        )
        return msg.sid

    async def preflight(self) -> tuple[bool | None, str]:
        if not (self.sid and self.token):
            return None, "twilio: not configured (fallback rails only)"
        try:
            self._client().api.accounts(self.sid).fetch()
        except Exception as exc:  # noqa: BLE001
            return False, f"twilio: credentials rejected ({exc})"
        if not self.from_number:
            return False, "twilio: authenticated but TWILIO_FROM_NUMBER unset"
        return True, f"twilio: authenticated as {self.from_number}"


_PROVIDERS: dict[str, type[TelephonyProvider]] = {
    "livekit": LiveKitSIPProvider,
    "a1mobile": A1MobileProvider,
    "twilio": TwilioProvider,
}


def get_provider(name: str | None = None) -> TelephonyProvider:
    key = (name or os.getenv("TELEPHONY_PROVIDER") or "livekit").lower()
    if key not in _PROVIDERS:
        raise ValueError(f"Unknown provider {key!r}. Options: {sorted(_PROVIDERS)}")
    return _PROVIDERS[key]()
