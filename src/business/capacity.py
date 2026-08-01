"""Capacity ledger with expiring holds.

The failure mode this exists to make impossible: three sales calls run at once,
each asks "can you do 200 muffins this week?" against a 400/week capacity, each
reads 400 available, each says yes - and the bakery is now on the hook for 600.
Nobody lied. The check simply was not atomic with the reservation.

This mirrors the invariant in `src/verify/receipts.py`: **an agent can only ever
propose.** `hold()` takes units out of circulation for the length of a call but
commits the business to nothing. Only `commit()` - driven by the same kind of
external confirmation that promotes a Claim - moves units to COMMITTED. A hold
nobody confirms expires and the capacity comes back on its own, so the worst
honest failure is a lost sale, never an oversold week.

`Hold.state` is derived from timestamps and has no setter, so there is no code
path by which an agent marks its own reservation as committed capacity.

Locking is `threading.Lock`, not `asyncio.Lock`, deliberately: no critical
section here contains an `await`, so a plain lock is correct both for asyncio
tasks sharing one loop and for worker threads, and it keeps the API synchronous
so it can be called from inside a voice-agent tool callback.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

#: Monotonic seconds. Wall-clock is never used for expiry - an NTP correction
#: mid-call must not resurrect or kill a hold.
Clock = Callable[[], float]


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class CapacityState(str, Enum):
    """What the units behind a hold are currently doing."""

    AVAILABLE = "available"
    """Back in the pool. Either the hold expired or it was released."""

    HELD = "held"
    """Reserved for a live conversation. Not sellable, not sold."""

    COMMITTED = "committed"
    """Confirmed order. Never returns to the pool on its own."""


class CapacityError(RuntimeError):
    """Base for every refusal this ledger makes."""


class UnknownHold(CapacityError):
    """No such hold id - this ledger has never issued it."""


class ExpiredHold(CapacityError):
    """The hold's units went back in the pool and may already be resold.

    Committing anyway is exactly the oversell this module exists to prevent, so
    the caller has to take a fresh hold and find out whether it still fits.
    """


@dataclass
class Hold:
    """A reservation on `qty` units that expires unless it is committed.

    Build these with `CapacityLedger.hold()`, never by hand: a Hold constructed
    directly belongs to no ledger and therefore reserves nothing.
    """

    qty: int
    unit: str
    expires_at: float
    created_at: float = field(default_factory=time.monotonic)
    id: str = field(default_factory=lambda: _uid("hold"))
    committed_at: float | None = None
    released_at: float | None = None
    clock: Clock = time.monotonic

    @property
    def state(self) -> CapacityState:
        """Derived, never stored. There is no setter, by design.

        Order matters: COMMITTED is tested first so that a passing deadline can
        never take capacity away from an order the customer already confirmed.
        """
        if self.committed_at is not None:
            return CapacityState.COMMITTED
        if self.released_at is not None or self.clock() >= self.expires_at:
            return CapacityState.AVAILABLE
        return CapacityState.HELD

    @property
    def is_expired(self) -> bool:
        return self.committed_at is None and self.clock() >= self.expires_at

    @property
    def seconds_remaining(self) -> float:
        """0.0 once the deadline has passed, so this is safe to read aloud."""
        return max(0.0, self.expires_at - self.clock())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "qty": self.qty,
            "unit": self.unit,
            "state": self.state.value,
            "seconds_remaining": round(self.seconds_remaining, 1),
            "committed": self.committed_at is not None,
        }


class CapacityLedger:
    """Units of one resource, tracked through AVAILABLE -> HELD -> COMMITTED.

    The resource is whatever the operator sells by the unit - muffins/week,
    crew-hours/month, site-builds/month. `unit` is a label for the transcript
    and the receipt; the ledger never interprets it.

    Expired holds are reclaimed lazily, on any read or write, so there is no
    background task to start, stop, or forget to start.
    """

    def __init__(
        self,
        total: int,
        unit: str = "units",
        *,
        default_ttl_seconds: float = 180.0,
        clock: Clock = time.monotonic,
    ) -> None:
        if total < 0:
            raise ValueError("total capacity cannot be negative")
        self.unit = unit
        self._total = int(total)
        self._default_ttl = float(default_ttl_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._holds: dict[str, Hold] = {}  # live: HELD or COMMITTED
        self._closed: dict[str, Hold] = {}  # expired or released, kept to explain refusals

    # -- writes ---------------------------------------------------------

    def hold(self, qty: int, ttl_seconds: float | None = None) -> Hold | None:
        """Reserve `qty` units for `ttl_seconds`, or return None if they do not fit.

        The read and the reservation happen inside one lock, which is the whole
        point: there is no window between "is 200 available?" and "200 are mine"
        for a second call to slip through.

        Never partially fills. Selling 120 of a requested 200 is a different
        conversation and the agent has to be told to go have it.
        """
        if qty <= 0:
            raise ValueError("hold qty must be positive")
        ttl = self._default_ttl if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")

        with self._lock:
            self._reclaim_locked()
            if qty > self._available_locked():
                return None
            now = self._clock()
            h = Hold(
                qty=int(qty),
                unit=self.unit,
                expires_at=now + ttl,
                created_at=now,
                clock=self._clock,
            )
            self._holds[h.id] = h
            return h

    def commit(self, hold_id: str) -> Hold:
        """Promote HELD -> COMMITTED. Idempotent, because webhooks retry.

        Raises `ExpiredHold` if the deadline passed first: those units are back
        in the pool and may belong to somebody else now.
        """
        with self._lock:
            self._reclaim_locked()
            h = self._holds.get(hold_id)
            if h is None:
                raise self._explain_missing_locked(hold_id, "commit")
            if h.state is CapacityState.COMMITTED:
                return h
            h.committed_at = self._clock()
            return h

    def extend(self, hold_id: str, ttl_seconds: float | None = None) -> Hold:
        """Push a live hold's deadline back. Call this on every turn of a call.

        Without this, the TTL is a bet on how long a negotiation takes, and the
        bet is lost in the worst possible place: the agent quotes, takes a hold,
        negotiates, escalates to the operator by voice, waits for a human to
        actually hear the question and answer - and three minutes is gone. The
        hold expires and `commit()` fails at the exact moment the deal closes.

        The fix is not a longer TTL. A long TTL means a dropped call sterilises
        capacity for its whole duration. Instead keep the TTL short and treat
        this as a heartbeat: the hold lives as long as the call is alive, and
        dies quickly once it is not.

        An expired hold is deliberately NOT extendable - those units went back
        in the pool and may already belong to someone else, so the caller has to
        take a fresh hold and find out whether they still fit.
        """
        with self._lock:
            self._reclaim_locked()
            h = self._holds.get(hold_id)
            if h is None:
                raise self._explain_missing_locked(hold_id, "extend")
            if h.state is CapacityState.COMMITTED:
                raise CapacityError(
                    f"hold {hold_id!r} is COMMITTED; there is nothing to extend"
                )
            ttl = self._default_ttl if ttl_seconds is None else float(ttl_seconds)
            if ttl <= 0:
                raise ValueError("ttl_seconds must be positive")
            h.expires_at = self._clock() + ttl
            return h

    def release(self, hold_id: str) -> bool:
        """Hand a live hold's units back early. True if this call freed them.

        Returns False for a hold that already expired or was already released -
        the capacity is back either way, so a retry is not an error.
        """
        with self._lock:
            self._reclaim_locked()
            h = self._holds.get(hold_id)
            if h is None:
                if hold_id in self._closed:
                    return False
                raise UnknownHold(f"no hold {hold_id!r} in this ledger")
            if h.state is CapacityState.COMMITTED:
                # Cancelling a confirmed order is an operator decision with its
                # own paperwork, not something a call in progress may do.
                raise CapacityError(
                    f"hold {hold_id!r} is COMMITTED; committed capacity is not "
                    "released by release()"
                )
            h.released_at = self._clock()
            self._closed[hold_id] = self._holds.pop(hold_id)
            return True

    # -- reads ----------------------------------------------------------

    @property
    def total(self) -> int:
        return self._total

    def available(self) -> int:
        """Units sellable right now. Expired holds are not counted as held."""
        with self._lock:
            self._reclaim_locked()
            return self._available_locked()

    def held(self) -> int:
        with self._lock:
            self._reclaim_locked()
            return self._sum_locked(CapacityState.HELD)

    def committed(self) -> int:
        with self._lock:
            self._reclaim_locked()
            return self._sum_locked(CapacityState.COMMITTED)

    def get(self, hold_id: str) -> Hold | None:
        """The hold record, live or closed, for reporting on a call afterwards."""
        with self._lock:
            self._reclaim_locked()
            return self._holds.get(hold_id) or self._closed.get(hold_id)

    def open_holds(self) -> list[Hold]:
        with self._lock:
            self._reclaim_locked()
            return [h for h in self._holds.values() if h.state is CapacityState.HELD]

    def can_fulfil(self, qty: int) -> bool:
        """Advisory only - true here is not a reservation.

        Useful for phrasing ("we're tight that week"), useless for promising.
        The only safe way to answer a customer is to take a `hold()` first.
        """
        return 0 < qty <= self.available()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._reclaim_locked()
            return {
                "unit": self.unit,
                "total": self._total,
                "available": self._available_locked(),
                "held": self._sum_locked(CapacityState.HELD),
                "committed": self._sum_locked(CapacityState.COMMITTED),
                "open_holds": [
                    h.to_dict()
                    for h in self._holds.values()
                    if h.state is CapacityState.HELD
                ],
            }

    # -- internals (callers already hold the lock) ----------------------

    def _reclaim_locked(self) -> None:
        """Move expired holds out of the live set. Their units are already free.

        This is bookkeeping, not accounting: `_sum_locked` ignores them anyway
        because their derived state is AVAILABLE. Keeping the record lets
        `commit()` say "that expired" instead of "never heard of it".
        """
        dead = [
            hid
            for hid, h in self._holds.items()
            if h.state is CapacityState.AVAILABLE
        ]
        for hid in dead:
            self._closed[hid] = self._holds.pop(hid)

    def _sum_locked(self, state: CapacityState) -> int:
        return sum(h.qty for h in self._holds.values() if h.state is state)

    def _available_locked(self) -> int:
        spoken_for = sum(
            h.qty
            for h in self._holds.values()
            if h.state in (CapacityState.HELD, CapacityState.COMMITTED)
        )
        return self._total - spoken_for

    def _explain_missing_locked(self, hold_id: str, action: str) -> CapacityError:
        closed = self._closed.get(hold_id)
        if closed is None:
            return UnknownHold(f"no hold {hold_id!r} in this ledger")
        if closed.released_at is not None:
            return ExpiredHold(f"hold {hold_id!r} was released; cannot {action}")
        return ExpiredHold(
            f"hold {hold_id!r} expired before {action}; take a fresh hold - "
            f"those {closed.qty} {closed.unit} may already be resold"
        )
