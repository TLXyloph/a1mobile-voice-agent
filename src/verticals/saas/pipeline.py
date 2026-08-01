"""The pipeline: prospects moving through stages, in stdlib sqlite3.

A CRM is where fabricated success usually hides. Not in a transcript - nobody
reads those - but in a board where somebody dragged a card to Closed Won
because the call "went really well". The same rule that governs
`src/verify/receipts.py` therefore governs this table:

    **A prospect reaches CLOSED_WON only when an independent channel says so.**

That is not re-implemented here. `close_verdict()` builds a real
`src.verify.receipts.Claim` out of the evidence rows and reads its derived
`verdict`, so the SaaS board and the receipt agree by construction rather than
by discipline. If `INDEPENDENT_CHANNELS` changes, this changes with it. An
`AGENT_ASSERTION` row is stored - we want to measure how often the agent was
right - and it can never move a card.

Transitions are a whitelist and strictly linear. QUALIFIED cannot be skipped
even when qualification and the demo booking happened in the same breath on the
same call: recording it is one extra call to `advance()`, and the alternative
is a board where "qualified" means "somebody thought so".

CLOSED_LOST is reachable from every open stage and needs no evidence. Walking
away is always allowed - the asymmetry is deliberate, and it is the same one as
in triage: the expensive error is the one that overstates.

Everything is synchronous and one connection per `Pipeline`. `check_same_thread`
is off because FastAPI serves reads from a threadpool; a `threading.Lock` around
writes covers the rest. This is a hackathon board, not a multi-tenant CRM.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.verify.receipts import (
    INDEPENDENT_CHANNELS,
    Channel,
    Claim,
    Evidence,
    Verdict,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class EvidenceTopic(str, Enum):
    """What a piece of evidence is evidence *of*.

    Without this the board tells a specific lie. A Google Calendar event is a
    real independent artifact from a real independent channel, and it proves a
    meeting exists - it says nothing whatever about whether anyone signed a
    subscription. Filed against a close claim it reads as VERIFIED, and a
    prospect who has agreed to a demo shows up on the board as a closed deal.

    Evidence is therefore scoped, and `close_claim()` only counts CLOSE rows.
    Everything else is still stored and still shown in the chain - it is
    genuine evidence, just of something else.
    """

    CLOSE = "close"
    """They signed. Order form, billing record, written yes."""

    MEETING = "meeting"
    """A demo or call exists on a calendar both sides can read."""

    CONTACT = "contact"
    """We reached a human, or learned something qualifying about them."""


class Stage(str, Enum):
    """Where a prospect is. Ordered; the order is enforced."""

    TARGETED = "targeted"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    DEMO_BOOKED = "demo_booked"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


#: Board order, left to right.
BOARD_ORDER: tuple[Stage, ...] = (
    Stage.TARGETED,
    Stage.CONTACTED,
    Stage.QUALIFIED,
    Stage.DEMO_BOOKED,
    Stage.CLOSED_WON,
    Stage.CLOSED_LOST,
)

OPEN_STAGES: frozenset[Stage] = frozenset(
    {Stage.TARGETED, Stage.CONTACTED, Stage.QUALIFIED, Stage.DEMO_BOOKED}
)

TERMINAL_STAGES: frozenset[Stage] = frozenset({Stage.CLOSED_WON, Stage.CLOSED_LOST})

#: One step forward, or out. No skipping, no going back, nothing after a close.
#: Reopening a lost deal is a new prospect row, because the second attempt has
#: its own evidence and conflating them makes the first one's failure invisible.
ALLOWED_TRANSITIONS: dict[Stage, frozenset[Stage]] = {
    Stage.TARGETED: frozenset({Stage.CONTACTED, Stage.CLOSED_LOST}),
    Stage.CONTACTED: frozenset({Stage.QUALIFIED, Stage.CLOSED_LOST}),
    Stage.QUALIFIED: frozenset({Stage.DEMO_BOOKED, Stage.CLOSED_LOST}),
    Stage.DEMO_BOOKED: frozenset({Stage.CLOSED_WON, Stage.CLOSED_LOST}),
    Stage.CLOSED_WON: frozenset(),
    Stage.CLOSED_LOST: frozenset(),
}


class PipelineError(RuntimeError):
    """Base for every refusal this pipeline makes."""


class UnknownProspect(PipelineError):
    pass


class InvalidTransition(PipelineError):
    """Not a legal move on this board."""


class UnverifiedClose(PipelineError):
    """CLOSED_WON was attempted with no independent evidence behind it.

    This is the disqualification condition, caught at the only place it could
    have entered the system.
    """


SCHEMA = """
CREATE TABLE IF NOT EXISTS prospects (
    id           TEXT PRIMARY KEY,
    company      TEXT NOT NULL,
    contact      TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    phone        TEXT NOT NULL DEFAULT '',
    email        TEXT NOT NULL DEFAULT '',
    seats        INTEGER NOT NULL DEFAULT 0,
    stage        TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT '',
    notes        TEXT NOT NULL DEFAULT '',
    is_sample    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    prospect_id  TEXT NOT NULL,
    at           TEXT NOT NULL,
    from_stage   TEXT,
    to_stage     TEXT NOT NULL,
    detail       TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS evidence (
    id            TEXT PRIMARY KEY,
    prospect_id   TEXT NOT NULL,
    channel       TEXT NOT NULL,
    about         TEXT NOT NULL DEFAULT 'close',
    summary       TEXT NOT NULL,
    supports      INTEGER NOT NULL DEFAULT 1,
    artifact_path TEXT,
    content_hash  TEXT,
    captured_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deals (
    prospect_id  TEXT PRIMARY KEY,
    terms        TEXT NOT NULL,
    proposed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_evidence_prospect ON evidence(prospect_id);
CREATE INDEX IF NOT EXISTS ix_events_prospect ON stage_events(prospect_id);
"""


@dataclass
class Prospect:
    """One company we are trying to sell to."""

    company: str
    contact: str = ""
    title: str = ""
    phone: str = ""
    email: str = ""
    seats: int = 0
    stage: Stage = Stage.TARGETED
    source: str = ""
    notes: str = ""
    is_sample: bool = False
    id: str = field(default_factory=lambda: _uid("p"))
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Prospect:
        return cls(
            id=row["id"],
            company=row["company"],
            contact=row["contact"],
            title=row["title"],
            phone=row["phone"],
            email=row["email"],
            seats=row["seats"],
            stage=Stage(row["stage"]),
            source=row["source"],
            notes=row["notes"],
            is_sample=bool(row["is_sample"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "company": self.company,
            "contact": self.contact,
            "title": self.title,
            "phone": self.phone,
            "email": self.email,
            "seats": self.seats,
            "stage": self.stage.value,
            "source": self.source,
            "notes": self.notes,
            "is_sample": self.is_sample,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class StageEvent:
    at: str
    from_stage: str | None
    to_stage: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "from": self.from_stage,
            "to": self.to_stage,
            "detail": self.detail,
        }


class Pipeline:
    """Prospects, their evidence, and their proposed terms.

    `Pipeline(":memory:")` for tests, a path for the app. The schema is created
    on construction, so there is no migration step to forget.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- writes ---------------------------------------------------------

    def add(self, prospect: Prospect) -> Prospect:
        """Insert a prospect at its stage, recording how it got there.

        A prospect seeded directly into a late stage still has to satisfy the
        close rule: `add()` refuses CLOSED_WON outright, because a card cannot
        arrive already closed with no evidence trail behind it. Sample data
        goes in at TARGETED and is walked forward through `advance()` like
        everything else.
        """
        if prospect.stage is Stage.CLOSED_WON:
            raise UnverifiedClose(
                "a prospect cannot be created already closed-won; add it, attach "
                "the evidence, then advance() - the trail is the point"
            )
        with self._lock:
            self._conn.execute(
                "INSERT INTO prospects (id, company, contact, title, phone, email,"
                " seats, stage, source, notes, is_sample, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    prospect.id,
                    prospect.company,
                    prospect.contact,
                    prospect.title,
                    prospect.phone,
                    prospect.email,
                    int(prospect.seats),
                    prospect.stage.value,
                    prospect.source,
                    prospect.notes,
                    int(prospect.is_sample),
                    prospect.created_at,
                    prospect.updated_at,
                ),
            )
            self._conn.execute(
                "INSERT INTO stage_events (prospect_id, at, from_stage, to_stage, detail)"
                " VALUES (?,?,?,?,?)",
                (prospect.id, prospect.created_at, None, prospect.stage.value, "added"),
            )
            self._conn.commit()
        return prospect

    def advance(self, prospect_id: str, to: Stage, detail: str = "") -> Prospect:
        """Move a prospect one legal step, or refuse and say why.

        Two refusals, in this order:

        1. `InvalidTransition` - not a legal move on this board.
        2. `UnverifiedClose` - a legal move to CLOSED_WON with nothing but the
           agent's word behind it. This is the one that matters.
        """
        to = Stage(to)
        current = self.get(prospect_id)
        if current is None:
            raise UnknownProspect(f"no prospect {prospect_id!r}")

        allowed = ALLOWED_TRANSITIONS[current.stage]
        if to not in allowed:
            options = ", ".join(sorted(s.value for s in allowed)) or "nothing"
            raise InvalidTransition(
                f"{current.company} is at {current.stage.value}; "
                f"{to.value} is not reachable from there (only: {options})"
            )

        if to is Stage.CLOSED_WON:
            verdict = self.close_verdict(prospect_id)
            if verdict is not Verdict.VERIFIED:
                raise UnverifiedClose(
                    f"cannot close {current.company}: the close claim is "
                    f"{verdict.value}. An agent saying the deal is done is not "
                    "evidence - attach an independent artifact (countersigned "
                    "order form, inbound email or SMS, billing provider record) "
                    "and try again."
                )

        stamp = _now()
        with self._lock:
            self._conn.execute(
                "UPDATE prospects SET stage = ?, updated_at = ? WHERE id = ?",
                (to.value, stamp, prospect_id),
            )
            self._conn.execute(
                "INSERT INTO stage_events (prospect_id, at, from_stage, to_stage, detail)"
                " VALUES (?,?,?,?,?)",
                (prospect_id, stamp, current.stage.value, to.value, detail),
            )
            self._conn.commit()
        current.stage = to
        current.updated_at = stamp
        return current

    def record_evidence(
        self,
        prospect_id: str,
        channel: Channel | str,
        summary: str,
        *,
        about: EvidenceTopic | str = EvidenceTopic.CLOSE,
        supports: bool = True,
        artifact_path: str | None = None,
        raw: Any = None,
    ) -> Evidence:
        """Attach an artifact to a prospect. Stored whatever channel it came from.

        `about` defaults to CLOSE because that is the claim this board tracks,
        and a default that quietly widened what counts as proof of a sale would
        be the wrong default. Pass MEETING or CONTACT for anything that is real
        evidence of something else.

        Agent assertions are stored deliberately: we want the honesty rate, and
        a claim with no agent assertion beside it is harder to audit, not
        easier. They just cannot verify anything.
        """
        if self.get(prospect_id) is None:
            raise UnknownProspect(f"no prospect {prospect_id!r}")
        topic = EvidenceTopic(about)
        ev = Evidence(
            channel=Channel(channel),
            summary=summary,
            raw=raw,
            artifact_path=artifact_path,
            supports=supports,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO evidence (id, prospect_id, channel, about, summary,"
                " supports, artifact_path, content_hash, captured_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ev.id,
                    prospect_id,
                    ev.channel.value,
                    topic.value,
                    ev.summary,
                    int(ev.supports),
                    ev.artifact_path,
                    ev.content_hash,
                    ev.captured_at,
                ),
            )
            self._conn.commit()
        return ev

    def propose_terms(self, prospect_id: str, terms: dict[str, Any]) -> None:
        """Record the deal on the table. Proposing is not agreeing."""
        if self.get(prospect_id) is None:
            raise UnknownProspect(f"no prospect {prospect_id!r}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO deals (prospect_id, terms, proposed_at) VALUES (?,?,?)"
                " ON CONFLICT(prospect_id) DO UPDATE SET terms = excluded.terms,"
                " proposed_at = excluded.proposed_at",
                (prospect_id, json.dumps(terms), _now()),
            )
            self._conn.commit()

    # -- reads ----------------------------------------------------------

    def get(self, prospect_id: str) -> Prospect | None:
        row = self._conn.execute(
            "SELECT * FROM prospects WHERE id = ?", (prospect_id,)
        ).fetchone()
        return Prospect.from_row(row) if row else None

    def all(self, stage: Stage | None = None) -> list[Prospect]:
        if stage is None:
            rows = self._conn.execute(
                "SELECT * FROM prospects ORDER BY company"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM prospects WHERE stage = ? ORDER BY company",
                (Stage(stage).value,),
            ).fetchall()
        return [Prospect.from_row(r) for r in rows]

    def board(self) -> dict[Stage, list[Prospect]]:
        """Every stage as a column, empty ones included - a column that
        vanishes when it empties makes the board move under the operator."""
        out: dict[Stage, list[Prospect]] = {s: [] for s in BOARD_ORDER}
        for p in self.all():
            out[p.stage].append(p)
        return out

    def evidence_for(
        self, prospect_id: str, about: EvidenceTopic | str | None = None
    ) -> list[Evidence]:
        """Evidence rows, optionally narrowed to one topic."""
        return [ev for _, ev in self.evidence_ledger(prospect_id, about)]

    def evidence_ledger(
        self, prospect_id: str, about: EvidenceTopic | str | None = None
    ) -> list[tuple[EvidenceTopic, Evidence]]:
        """Everything on file, each row paired with what it is evidence of."""
        sql = "SELECT * FROM evidence WHERE prospect_id = ?"
        args: list[Any] = [prospect_id]
        if about is not None:
            sql += " AND about = ?"
            args.append(EvidenceTopic(about).value)
        rows = self._conn.execute(sql + " ORDER BY captured_at, id", args).fetchall()
        return [
            (
                EvidenceTopic(r["about"]),
                Evidence(
                    channel=Channel(r["channel"]),
                    summary=r["summary"],
                    artifact_path=r["artifact_path"],
                    supports=bool(r["supports"]),
                    captured_at=r["captured_at"],
                    id=r["id"],
                    content_hash=r["content_hash"],
                ),
            )
            for r in rows
        ]

    def events_for(self, prospect_id: str) -> list[StageEvent]:
        rows = self._conn.execute(
            "SELECT * FROM stage_events WHERE prospect_id = ? ORDER BY id",
            (prospect_id,),
        ).fetchall()
        return [
            StageEvent(
                at=r["at"],
                from_stage=r["from_stage"],
                to_stage=r["to_stage"],
                detail=r["detail"],
            )
            for r in rows
        ]

    def terms_for(self, prospect_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT terms FROM deals WHERE prospect_id = ?", (prospect_id,)
        ).fetchone()
        return json.loads(row["terms"]) if row else None

    # -- the invariant --------------------------------------------------

    def close_claim(self, prospect_id: str) -> Claim:
        """The close, expressed as a `Claim` with this prospect's evidence.

        Built rather than stored, so it cannot drift from the receipts module.
        `Claim.verdict` does the actual work; this function only supplies rows.
        """
        p = self.get(prospect_id)
        if p is None:
            raise UnknownProspect(f"no prospect {prospect_id!r}")
        claim = Claim(
            description=f"{p.company} signed a subscription ({p.seats} seats)",
            expected_side_effect=(
                f"a countersigned order form, billing record, or written yes "
                f"from {p.email or p.company} exists and names the seat count "
                f"and term"
            ),
        )
        # CLOSE rows only. A calendar event is independent, supporting, and
        # about a different question entirely.
        for ev in self.evidence_for(prospect_id, EvidenceTopic.CLOSE):
            claim.attach_evidence(ev)
        return claim

    def close_verdict(self, prospect_id: str) -> Verdict:
        """VERIFIED / UNVERIFIED / CONTRADICTED for this prospect's close."""
        return self.close_claim(prospect_id).verdict

    def can_close(self, prospect_id: str) -> tuple[bool, str]:
        """Advisory. `advance()` is still the only thing that can move a card."""
        p = self.get(prospect_id)
        if p is None:
            return False, "no such prospect"
        if p.stage is Stage.CLOSED_WON:
            return True, (
                "closed, and it was closed by independent evidence - the "
                "transition could not have happened otherwise"
            )
        if p.stage is Stage.CLOSED_LOST:
            return False, "closed lost; a second attempt is a new prospect"
        if Stage.CLOSED_WON not in ALLOWED_TRANSITIONS[p.stage]:
            return False, (
                f"still at {p.stage.value} - a deal has to reach demo booked "
                "before it can close"
            )
        verdict = self.close_verdict(prospect_id)
        if verdict is Verdict.VERIFIED:
            return True, "independent evidence supports the close"
        if verdict is Verdict.CONTRADICTED:
            return False, "an independent channel contradicts the close"
        return False, "no independent evidence yet - the agent's word is not enough"

    # -- reporting ------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {s.value: len(v) for s, v in self.board().items()}

    def evidence_strength(self, prospect_id: str) -> str:
        """One word for the board: what stands behind this prospect's close.

        Distinct from the verdict because "no evidence at all" and "the agent
        insists" are the same verdict and very different situations - the
        second one is the failure mode worth putting on a wall.
        """
        verdict = self.close_verdict(prospect_id)
        if verdict is Verdict.CONTRADICTED:
            return "contradicted"
        if verdict is Verdict.VERIFIED:
            return "verified"
        rows = self.evidence_for(prospect_id, EvidenceTopic.CLOSE)
        return "agent only" if rows else "no evidence"

    def detail(self, prospect_id: str) -> dict[str, Any] | None:
        p = self.get(prospect_id)
        if p is None:
            return None
        claim = self.close_claim(prospect_id)
        can, why = self.can_close(prospect_id)
        chain = [
            {
                "about": topic.value,
                "bears_on_close": topic is EvidenceTopic.CLOSE,
                "channel": ev.channel.value,
                "summary": ev.summary,
                "supports": ev.supports,
                "independent": ev.is_independent,
                "content_hash": ev.content_hash,
                "captured_at": ev.captured_at,
            }
            for topic, ev in self.evidence_ledger(prospect_id)
        ]
        return {
            "prospect": p.to_dict(),
            "terms": self.terms_for(prospect_id),
            "events": [e.to_dict() for e in self.events_for(prospect_id)],
            "claim": claim.to_dict(),
            "chain": chain,
            "verdict": claim.verdict.value,
            "strength": self.evidence_strength(prospect_id),
            "can_close": can,
            "can_close_reason": why,
            "independent_evidence": [
                e.summary for e in claim.evidence if e.is_independent
            ],
        }

    def as_json(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "board": {
                s.value: [p.to_dict() for p in ps] for s, ps in self.board().items()
            },
            "independent_channels": sorted(c.value for c in INDEPENDENT_CHANNELS),
        }


# -- sample data ----------------------------------------------------------


SAMPLE_BANNER = "SAMPLE DATA - seeded from config/saas.json, not a real prospect"


def seed_samples(pipeline: Pipeline, cfg: dict[str, Any]) -> list[Prospect]:
    """Put the demo prospects on the board.

    They are walked forward through `advance()` one stage at a time, with their
    evidence attached first, rather than inserted at their final stage. That
    costs a few lines and buys something worth more than the lines: the seeded
    CLOSED_WON row is closed for the same reason a real one would be, so the
    board cannot demonstrate an outcome the code would not actually permit.
    """
    created: list[Prospect] = []
    existing = {p.id for p in pipeline.all()}

    for spec in cfg.get("sample_prospects", []):
        if spec["id"] in existing:
            continue
        target = Stage(spec["stage"])
        p = Prospect(
            id=spec["id"],
            company=spec["company"],
            contact=spec.get("contact", ""),
            title=spec.get("title", ""),
            phone=spec.get("phone", ""),
            email=spec.get("email", ""),
            seats=int(spec.get("seats", 0)),
            stage=Stage.TARGETED,
            source=spec.get("source", ""),
            notes=spec.get("notes", ""),
            is_sample=True,
        )
        pipeline.add(p)
        for ev in spec.get("evidence", []):
            pipeline.record_evidence(
                p.id,
                ev["channel"],
                ev["summary"],
                about=ev.get("about", EvidenceTopic.CLOSE),
                supports=bool(ev.get("supports", True)),
            )
        if spec.get("terms"):
            pipeline.propose_terms(p.id, spec["terms"])
        for stage in _path_to(target):
            pipeline.advance(p.id, stage, detail=SAMPLE_BANNER)
        created.append(pipeline.get(p.id))  # type: ignore[arg-type]
    return created


def _path_to(target: Stage) -> Iterable[Stage]:
    """The legal sequence of steps from TARGETED to `target`."""
    if target is Stage.TARGETED:
        return ()
    linear = [Stage.CONTACTED, Stage.QUALIFIED, Stage.DEMO_BOOKED, Stage.CLOSED_WON]
    if target is Stage.CLOSED_LOST:
        # Lost from wherever it was reached; contacted is the honest minimum.
        return (Stage.CONTACTED, Stage.CLOSED_LOST)
    return tuple(linear[: linear.index(target) + 1])
