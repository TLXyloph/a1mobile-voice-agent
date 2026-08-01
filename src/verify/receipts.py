"""Claims, evidence, and receipts.

The judging rule for this hackathon is that an agent is scored on verifiable
side effects, never on what it says it did, and a fabricated success is an
automatic disqualification. That rule is load-bearing for this whole design.

The invariant enforced here: **an agent can only ever create a Claim.** A Claim
is UNVERIFIED at birth and there is no code path by which the agent's own words
promote it. Only `attach_evidence` - fed from a channel the agent does not
control (an inbound SMS webhook, a provider API, an independent transcription of
the call audio) - can move a Claim to VERIFIED.

This means the worst honest failure mode is reporting UNVERIFIED, which loses
points. Fabrication, which is disqualifying, is not reachable through the API.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Verdict(str, Enum):
    """The only three honest answers about a claim."""

    VERIFIED = "VERIFIED"
    """An independent channel confirmed it. Safe to report as done."""

    UNVERIFIED = "UNVERIFIED"
    """No independent confirmation. Report as 'attempted, unconfirmed'."""

    CONTRADICTED = "CONTRADICTED"
    """An independent channel disagrees with the agent. Report as failed."""


class Channel(str, Enum):
    """Where a piece of evidence came from.

    `AGENT_ASSERTION` is deliberately included and deliberately powerless: we
    record what the agent believed so we can later measure its honesty rate,
    but it can never satisfy a verification requirement.
    """

    AGENT_ASSERTION = "agent_assertion"
    INBOUND_SMS = "inbound_sms"
    INBOUND_EMAIL = "inbound_email"
    PROVIDER_API = "provider_api"
    WEB_CHECK = "web_check"
    DTMF_CONFIRMATION = "dtmf_confirmation"
    CALL_RECORDING = "call_recording"
    INDEPENDENT_TRANSCRIPT = "independent_transcript"
    HUMAN_CALLBACK = "human_callback"


#: Channels the agent cannot influence by talking. Only these can verify.
INDEPENDENT_CHANNELS: frozenset[Channel] = frozenset(
    {
        Channel.INBOUND_SMS,
        Channel.INBOUND_EMAIL,
        Channel.PROVIDER_API,
        Channel.WEB_CHECK,
        Channel.DTMF_CONFIRMATION,
        Channel.INDEPENDENT_TRANSCRIPT,
        Channel.HUMAN_CALLBACK,
    }
)


@dataclass
class Evidence:
    """A single artifact supporting or contradicting a claim.

    `content_hash` exists so a judge can confirm the artifact was not edited
    after the fact. Hash the raw bytes at capture time, not at report time.
    """

    channel: Channel
    summary: str
    raw: Any = None
    artifact_path: str | None = None
    supports: bool = True
    captured_at: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _uid("ev"))
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if self.content_hash is None and self.raw is not None:
            blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
            self.content_hash = hashlib.sha256(blob).hexdigest()[:16]

    @property
    def is_independent(self) -> bool:
        return self.channel in INDEPENDENT_CHANNELS


@dataclass
class Claim:
    """Something the agent asserts is true about the world.

    Do not construct these with a verdict. Construct, then attach evidence.
    """

    description: str
    expected_side_effect: str
    """What must be observably true elsewhere if this claim holds.

    Write this as a checkable sentence, e.g. 'an SMS from the restaurant
    arrives at +1555... containing a party size of 4'. If you cannot write
    this sentence, the claim is not verifiable and should not be made.
    """

    required_channels: tuple[Channel, ...] = ()
    """Channels that would satisfy this claim. Empty means 'any independent'."""

    evidence: list[Evidence] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _uid("claim"))

    def attach_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)

    @property
    def verdict(self) -> Verdict:
        """Derived, never stored. There is no setter, by design."""
        # Contradiction wins over confirmation: if an independent channel says
        # the agent is wrong, that is the answer regardless of other support.
        if any(e.is_independent and not e.supports for e in self.evidence):
            return Verdict.CONTRADICTED

        supporting = [e for e in self.evidence if e.is_independent and e.supports]
        if not supporting:
            return Verdict.UNVERIFIED

        if self.required_channels:
            have = {e.channel for e in supporting}
            if not have.issuperset(self.required_channels):
                return Verdict.UNVERIFIED

        return Verdict.VERIFIED

    @property
    def agent_said(self) -> str | None:
        for e in self.evidence:
            if e.channel is Channel.AGENT_ASSERTION:
                return e.summary
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "expected_side_effect": self.expected_side_effect,
            "verdict": self.verdict.value,
            "agent_said": self.agent_said,
            "evidence": [
                {
                    "id": e.id,
                    "channel": e.channel.value,
                    "summary": e.summary,
                    "supports": e.supports,
                    "independent": e.is_independent,
                    "artifact_path": e.artifact_path,
                    "content_hash": e.content_hash,
                    "captured_at": e.captured_at,
                }
                for e in self.evidence
            ],
        }


@dataclass
class Receipt:
    """The deliverable you hand a judge.

    `headline` is intentionally pessimistic: it reports success only when every
    claim is verified, so reading it aloud can never overstate the outcome.
    """

    task: str
    claims: list[Claim] = field(default_factory=list)
    call_recording: str | None = None
    started_at: str = field(default_factory=_now)
    ended_at: str | None = None
    id: str = field(default_factory=lambda: _uid("receipt"))

    notes: list[str] = field(default_factory=list)
    """Audit trail: what the system *did*, as opposed to what it claims is true.

    Operator escalations belong here, not in `claims`. An escalation is internal
    control flow with no external side effect to verify, so filing it as a claim
    adds a permanently UNVERIFIED row and makes a flawless call read as PARTIAL.
    """

    def note(self, message: str) -> None:
        self.notes.append(f"{_now()} {message}")

    def claim(
        self,
        description: str,
        expected_side_effect: str,
        required_channels: tuple[Channel, ...] = (),
    ) -> Claim:
        c = Claim(
            description=description,
            expected_side_effect=expected_side_effect,
            required_channels=required_channels,
        )
        self.claims.append(c)
        return c

    @property
    def verified(self) -> list[Claim]:
        return [c for c in self.claims if c.verdict is Verdict.VERIFIED]

    @property
    def unverified(self) -> list[Claim]:
        return [c for c in self.claims if c.verdict is Verdict.UNVERIFIED]

    @property
    def contradicted(self) -> list[Claim]:
        return [c for c in self.claims if c.verdict is Verdict.CONTRADICTED]

    @property
    def headline(self) -> str:
        if not self.claims:
            return "NO CLAIMS MADE - nothing was attempted."
        if self.contradicted:
            return (
                f"FAILED - {len(self.contradicted)} claim(s) contradicted by "
                f"independent evidence."
            )
        if self.unverified:
            return (
                f"PARTIAL - {len(self.verified)}/{len(self.claims)} verified, "
                f"{len(self.unverified)} could not be confirmed."
            )
        return f"SUCCESS - all {len(self.claims)} claim(s) independently verified."

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "headline": self.headline,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "call_recording": self.call_recording,
            "totals": {
                "claims": len(self.claims),
                "verified": len(self.verified),
                "unverified": len(self.unverified),
                "contradicted": len(self.contradicted),
            },
            "claims": [c.to_dict() for c in self.claims],
        }

    def save(self, directory: str | Path = "evidence") -> Path:
        self.ended_at = self.ended_at or _now()
        out = Path(directory) / f"{self.id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2))
        return out

    def render(self) -> str:
        """Plain-text summary for reading aloud during a demo."""
        mark = {
            Verdict.VERIFIED: "[VERIFIED]",
            Verdict.UNVERIFIED: "[UNCONFIRMED]",
            Verdict.CONTRADICTED: "[CONTRADICTED]",
        }
        lines = [f"TASK: {self.task}", f"RESULT: {self.headline}", ""]
        for c in self.claims:
            lines.append(f"{mark[c.verdict]} {c.description}")
            lines.append(f"    expected: {c.expected_side_effect}")
            for e in c.evidence:
                tag = "independent" if e.is_independent else "agent-only"
                sign = "+" if e.supports else "-"
                lines.append(f"    {sign} [{tag}] {e.channel.value}: {e.summary}")
            lines.append("")
        return "\n".join(lines)
