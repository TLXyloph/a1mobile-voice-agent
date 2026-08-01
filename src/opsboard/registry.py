"""In-memory live-call registry. Other components push; the board only reads.

Nothing here can mark anything done. A `Refusal` is a record that the system
*declined* to act, and the live call state is a description of a conversation
in flight — neither can promote a claim, and neither is ever written to a
receipt. The board is downstream of the truth, never a source of it.

Usage from a calling agent::

    from src.opsboard import OPS

    OPS.start_call(business="Golden Crumb", phone="+1415...", task="200 pastries")
    OPS.update_call(phase="qualified", units=200, capacity_total=400, capacity_held=200)
    OPS.update_call(phase="quoted", quote=400.0, floor=385.72, budget=280.0)
    OPS.gate_refusal(gate.allow_quote(74.0), phase=gate.phase)   # auto-phrased
    OPS.refusal("refused 600 units - capacity is 400", kind="capacity")
    OPS.end_call()

`gate_refusal` accepts the raw `BLOCKED ...` string the Gate hands back to the
model and turns it into the sentence a judge can read from across the room. It
tolerates `None` (the Gate's "proceed") so call sites can stay one-liners.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# The rail we draw, in order. ESCALATED is deliberately absent: it is a spur off
# the main line, not a station on it, and it is rendered as one.
PHASE_RAIL: tuple[str, ...] = (
    "opening",
    "discovery",
    "qualified",
    "quoted",
    "negotiating",
    "closing",
    "closed",
)
SPUR_PHASE = "escalated"

# One-line gloss per phase, for the projector.
PHASE_GLOSS: dict[str, str] = {
    "opening": "greeting; nothing learned",
    "discovery": "what, from whom, how much",
    "qualified": "units known, capacity held",
    "quoted": "a validated price is on the table",
    "negotiating": "pushback; concessions, floor enforced",
    "escalated": "outside the envelope — waiting on the operator",
    "closing": "terms agreed, getting it in writing",
    "closed": "filed as a claim, or hung up",
}


def _load_transitions() -> dict[str, frozenset[str]]:
    """The real graph from `src.agents.flow`, with a frozen copy as a parachute.

    The board is a projector surface during a live demo. If the flow module is
    mid-edit and will not import, dimming the wrong node is a far better outcome
    than a blank screen, so the fallback exists and is never silently preferred.
    """
    try:
        from src.agents.flow import TRANSITIONS as _T

        return {k.value: frozenset(v.value for v in vs) for k, vs in _T.items()}
    except Exception:  # noqa: BLE001  # pragma: no cover - only while flow.py is broken
        return {
            "opening": frozenset({"discovery", "closed"}),
            "discovery": frozenset({"qualified", "discovery", "closed"}),
            "qualified": frozenset({"quoted", "escalated", "discovery", "closed"}),
            "quoted": frozenset({"negotiating", "closing", "escalated", "closed"}),
            "negotiating": frozenset({"quoted", "escalated", "closing", "closed"}),
            "escalated": frozenset({"quoted", "negotiating", "closing", "closed"}),
            "closing": frozenset({"closed", "negotiating"}),
            "closed": frozenset(),
        }


TRANSITIONS: dict[str, frozenset[str]] = _load_transitions()


def reachable_from(phase: str) -> frozenset[str]:
    """Every phase still attainable, following edges. Not including `phase`."""
    seen: set[str] = set()
    stack = list(TRANSITIONS.get(phase, frozenset()))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(TRANSITIONS.get(p, frozenset()))
    seen.discard(phase)
    return frozenset(seen)


# --------------------------------------------------------------------------
# Turning a Gate refusal into a sentence a judge can read
# --------------------------------------------------------------------------

_BUDGET_RE = re.compile(
    r"already offered ([\d.]+); quoting ([\d.]+)", re.IGNORECASE
)
_UNITS_RE = re.compile(r"Before reserving (\d+), confirm whether", re.IGNORECASE)
_EARLY_RE = re.compile(r"too early to quote\.\s*Still needed:\s*(.+?)\.\s*Do not", re.IGNORECASE | re.DOTALL)
_GAPS_RE = re.compile(r"still needed:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _money(raw: str) -> str:
    v = float(raw)
    return f"${v:,.0f}" if abs(v - round(v)) < 0.005 else f"${v:,.2f}"


def humanise(raw: str) -> tuple[str, str]:
    """`(headline, detail)` for one raw gate refusal.

    The headline is the sentence on the projector; the detail is the machine's
    own words, kept verbatim underneath so nobody has to trust the paraphrase.
    """
    text = " ".join(raw.split())

    if m := _BUDGET_RE.search(text):
        offered, quoted = _money(m.group(1)), _money(m.group(2))
        return f"refused to quote {quoted} — buyer already offered {offered}", text
    if m := _UNITS_RE.search(text):
        n = m.group(1)
        return (
            f"refused to reserve {n} — nobody has said whether {n} is items or people",
            text,
        )
    if m := _EARLY_RE.search(text):
        return f"refused to quote — too early; still needs {_first_gap(m.group(1))}", text
    if "This call is over" in text:
        return "refused to act — the call is already closed", text
    if "No validated price" in text:
        return "refused to close — no validated price was ever agreed", text
    if "waiting on the owner" in text:
        return "refused to close — the operator has not answered yet", text
    if "already filed" in text:
        return "refused to close — this order was filed once already", text
    if m := _GAPS_RE.search(text):
        return f"refused to quote — still needs {_first_gap(m.group(1))}", text

    return text.removeprefix("BLOCKED.").removeprefix("BLOCKED -").strip() or text, text


def _first_gap(blob: str) -> str:
    """The first missing precondition, trimmed of its parenthetical coaching."""
    gap = blob.split(";")[0].strip().rstrip(".")
    return re.sub(r"\s*\(.*", "", gap).strip() or gap


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    """One time the system said no. Product working, not error."""

    headline: str
    detail: str = ""
    kind: str = "gate"
    phase: str | None = None
    seq: int = 0
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "headline": self.headline,
            "detail": self.detail,
            "kind": self.kind,
            "phase": self.phase,
        }


@dataclass
class CallState:
    """A conversation in flight, as far as the board is allowed to know."""

    business: str = ""
    phone: str = ""
    task: str = ""
    phase: str = "opening"
    quote: float | None = None
    floor: float | None = None
    budget: float | None = None
    units: int | None = None
    unit_label: str = "units"
    capacity_total: int | None = None
    capacity_held: int | None = None
    started_epoch: float = field(default_factory=time.time)
    ended_epoch: float | None = None

    fixture: bool = False
    """Seeded demo data. The board says so on screen, loudly, the whole time —
    an unlabelled fixture on a projector is a fabricated success with a nicer
    excuse."""

    @property
    def live(self) -> bool:
        return self.ended_epoch is None

    @property
    def floor_binding(self) -> bool:
        """True when the floor, not the buyer, is setting the price."""
        if self.floor is None:
            return False
        if self.budget is not None and self.budget < self.floor:
            return True
        return self.quote is not None and self.quote <= self.floor + 0.005

    def to_dict(self) -> dict[str, Any]:
        return {
            "business": self.business,
            "phone": self.phone,
            "task": self.task,
            "phase": self.phase,
            "quote": self.quote,
            "floor": self.floor,
            "budget": self.budget,
            "units": self.units,
            "unit_label": self.unit_label,
            "capacity_total": self.capacity_total,
            "capacity_held": self.capacity_held,
            "live": self.live,
            "fixture": self.fixture,
            "floor_binding": self.floor_binding,
            # Wall-clock deliberately excluded from the payload the state key is
            # hashed over; the page ticks its own clock from `started_epoch`.
            "started_epoch": self.started_epoch,
        }


class OpsRegistry:
    """Thread-safe. Uvicorn serves the board off a threadpool; agents push from
    their own tasks. One lock, held only for field assignment."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._call: CallState | None = None
        self._refusals: list[Refusal] = []
        self._seq = 0

    # -- writes ------------------------------------------------------

    def start_call(self, business: str, phone: str = "", task: str = "") -> CallState:
        with self._lock:
            self._call = CallState(business=business, phone=phone, task=task)
            self._refusals = []
            self._seq = 0
            return self._call

    def update_call(self, **fields: Any) -> None:
        """Patch the live call. Unknown keys are ignored rather than raising —
        a typo in an instrumentation call must not end a phone call."""
        with self._lock:
            if self._call is None:
                self._call = CallState(business=str(fields.get("business", "")))
            for key, value in fields.items():
                if key == "phase" and value is not None:
                    value = getattr(value, "value", value)
                if hasattr(self._call, key) and key not in ("started_epoch",):
                    setattr(self._call, key, value)

    def refusal(
        self, headline: str, *, detail: str = "", kind: str = "gate", phase: Any = None
    ) -> Refusal:
        with self._lock:
            self._seq += 1
            r = Refusal(
                headline=headline,
                detail=detail,
                kind=kind,
                phase=getattr(phase, "value", phase),
                seq=self._seq,
            )
            self._refusals.append(r)
            return r

    def gate_refusal(self, raw: str | None, *, phase: Any = None, kind: str = "gate") -> Refusal | None:
        """Record a `BLOCKED ...` string from the Gate. `None` means it allowed
        the action, so `None` records nothing — call sites stay one-liners."""
        if not raw:
            return None
        headline, detail = humanise(raw)
        return self.refusal(headline, detail=detail, kind=kind, phase=phase)

    def end_call(self) -> None:
        with self._lock:
            if self._call is not None and self._call.ended_epoch is None:
                self._call.ended_epoch = time.time()

    def reset(self) -> None:
        with self._lock:
            self._call = None
            self._refusals = []
            self._seq = 0

    # -- reads -------------------------------------------------------

    @property
    def call(self) -> CallState | None:
        return self._call

    def refusals(self) -> list[Refusal]:
        with self._lock:
            return list(self._refusals)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            call = self._call.to_dict() if self._call else None
            refusals = [r.to_dict() for r in self._refusals]
        return {"call": call, "refusals": refusals}


#: The process-wide board. Import this, do not build your own.
OPS = OpsRegistry()
