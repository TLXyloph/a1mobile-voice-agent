"""Overselling must be structurally impossible, not merely unlikely.

The scenario these pin down: several sales calls in flight at once, each asking
about the same week's capacity. If a check and a reservation can ever come apart,
the business gets committed to work it cannot do - and unlike a fabricated claim,
that failure only surfaces on delivery day.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.business.capacity import (  # noqa: E402
    CapacityError,
    CapacityLedger,
    CapacityState,
    ExpiredHold,
    UnknownHold,
)


class _FakeClock:
    """Monotonic clock we can advance, so expiry tests do not sleep."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _ledger(total: int = 400, clock=None) -> CapacityLedger:
    return CapacityLedger(total, "muffins/week", clock=clock or _FakeClock())


# -- basics -------------------------------------------------------------


def test_fresh_ledger_is_all_available():
    led = _ledger()
    assert led.available() == 400
    assert led.held() == 0 and led.committed() == 0


def test_hold_reserves_capacity():
    led = _ledger()
    h = led.hold(200, ttl_seconds=60)
    assert h is not None
    assert h.state is CapacityState.HELD
    assert led.available() == 200
    assert led.held() == 200


def test_hold_beyond_capacity_returns_none():
    led = _ledger()
    assert led.hold(300, 60) is not None
    assert led.hold(300, 60) is None  # would take the week to 600
    assert led.available() == 100


def test_hold_never_partially_fills():
    led = _ledger(total=100)
    assert led.hold(150, 60) is None
    assert led.available() == 100


def test_hold_qty_must_be_positive():
    led = _ledger()
    for bad in (0, -5):
        with pytest.raises(ValueError):
            led.hold(bad, 60)


def test_commit_promotes_held_to_committed():
    led = _ledger()
    h = led.hold(200, 60)
    committed = led.commit(h.id)
    assert committed.state is CapacityState.COMMITTED
    assert led.committed() == 200 and led.held() == 0
    assert led.available() == 200


def test_commit_is_idempotent_because_webhooks_retry():
    led = _ledger()
    h = led.hold(100, 60)
    led.commit(h.id)
    led.commit(h.id)
    assert led.committed() == 100


def test_release_returns_capacity():
    led = _ledger()
    h = led.hold(400, 60)
    assert led.available() == 0
    assert led.release(h.id) is True
    assert led.available() == 400
    assert h.state is CapacityState.AVAILABLE


def test_release_is_idempotent():
    led = _ledger()
    h = led.hold(50, 60)
    assert led.release(h.id) is True
    assert led.release(h.id) is False


def test_unknown_hold_ids_are_rejected():
    led = _ledger()
    with pytest.raises(UnknownHold):
        led.commit("hold_deadbeef")
    with pytest.raises(UnknownHold):
        led.release("hold_deadbeef")


# -- expiry -------------------------------------------------------------


def test_expired_hold_frees_capacity():
    clock = _FakeClock()
    led = _ledger(clock=clock)
    h = led.hold(400, ttl_seconds=30)
    assert led.available() == 0

    clock.advance(31)
    assert led.available() == 400  # reclaimed lazily, on the read itself
    assert led.held() == 0
    assert h.state is CapacityState.AVAILABLE


def test_available_never_counts_expired_holds_as_held():
    clock = _FakeClock()
    led = _ledger(clock=clock)
    led.hold(100, 10)
    led.hold(100, 600)
    clock.advance(11)
    assert led.held() == 100
    assert led.available() == 300


def test_expired_hold_does_not_block_a_new_one():
    clock = _FakeClock()
    led = _ledger(total=200, clock=clock)
    led.hold(200, 30)
    assert led.hold(200, 30) is None
    clock.advance(31)
    assert led.hold(200, 30) is not None


def test_commit_after_expiry_raises():
    """Those units may already belong to the next caller. Re-hold, do not assume."""
    clock = _FakeClock()
    led = _ledger(clock=clock)
    h = led.hold(200, 30)
    clock.advance(31)
    with pytest.raises(ExpiredHold):
        led.commit(h.id)
    assert led.committed() == 0


def test_committed_capacity_survives_hold_expiry():
    """The one thing expiry must never do is un-sell a confirmed order."""
    clock = _FakeClock()
    led = _ledger(clock=clock)
    h = led.hold(250, ttl_seconds=30)
    led.commit(h.id)

    clock.advance(10_000)
    assert led.committed() == 250
    assert led.available() == 150
    assert h.state is CapacityState.COMMITTED
    assert h.is_expired is False


def test_committed_hold_cannot_be_released():
    led = _ledger()
    h = led.hold(100, 60)
    led.commit(h.id)
    with pytest.raises(CapacityError):
        led.release(h.id)
    assert led.committed() == 100


def test_hold_state_has_no_setter():
    """Mirrors Verdict in receipts.py: state is derived, never assigned."""
    led = _ledger()
    h = led.hold(10, 60)
    with pytest.raises(AttributeError):
        h.state = CapacityState.COMMITTED  # type: ignore[misc]


# -- concurrency: the actual point of the module ------------------------


def test_concurrent_holds_cannot_oversell():
    """32 calls, all asking for 200 against 400. Exactly two may win."""
    led = CapacityLedger(400, "muffins/week")
    attempts = 32
    barrier = threading.Barrier(attempts, timeout=15)  # collide on purpose
    won: list[object] = []
    guard = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        h = led.hold(200, 60)
        if h is not None:
            with guard:
                won.append(h)

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        list(pool.map(lambda _: attempt(), range(attempts)))

    assert len(won) == 2
    assert led.held() == 400
    assert led.available() == 0


def test_concurrent_mixed_sizes_never_exceed_capacity():
    """Repeat with ragged sizes; held + committed must never top the total."""
    for _ in range(20):
        led = CapacityLedger(100, "crew-hours/month")
        sizes = [7, 13, 40, 25, 60, 5, 33, 19, 50, 11, 44, 2]
        barrier = threading.Barrier(len(sizes), timeout=15)

        def attempt(qty: int) -> int:
            barrier.wait()
            h = led.hold(qty, 60)
            return qty if h is not None else 0

        with ThreadPoolExecutor(max_workers=len(sizes)) as pool:
            granted = sum(pool.map(attempt, sizes))

        assert granted == led.held()
        assert granted <= 100
        assert led.available() == 100 - granted


def test_concurrent_holds_and_commits_stay_consistent():
    led = CapacityLedger(500, "site-builds/month")
    barrier = threading.Barrier(25, timeout=15)

    def attempt(_: int) -> None:
        barrier.wait()
        h = led.hold(20, 60)
        if h is not None:
            led.commit(h.id)

    with ThreadPoolExecutor(max_workers=25) as pool:
        list(pool.map(attempt, range(25)))

    assert led.committed() == 500
    assert led.available() == 0
    assert led.hold(1, 60) is None


async def test_concurrent_asyncio_holds_cannot_oversell():
    """Same guarantee from the event loop the voice agent actually runs on."""
    led = CapacityLedger(400, "muffins/week")
    results = await asyncio.gather(
        *(asyncio.to_thread(led.hold, 150, 60) for _ in range(10))
    )
    granted = [h for h in results if h is not None]
    assert len(granted) == 2
    assert sum(h.qty for h in granted) <= 400
    assert led.available() == 100


# -- reporting ----------------------------------------------------------


def test_can_fulfil_is_advisory_only():
    led = _ledger(total=10)
    assert led.can_fulfil(10) is True
    led.hold(10, 60)
    assert led.can_fulfil(1) is False


def test_snapshot_reports_the_three_states():
    clock = _FakeClock()
    led = _ledger(clock=clock)
    sold = led.hold(100, 60)
    led.commit(sold.id)
    led.hold(50, 60)
    snap = led.snapshot()
    assert snap["total"] == 400
    assert snap["committed"] == 100
    assert snap["held"] == 50
    assert snap["available"] == 250
    assert snap["unit"] == "muffins/week"
    assert len(snap["open_holds"]) == 1


# -- heartbeat extension --------------------------------------------------
# A live call routinely outlives the default TTL: quote -> negotiate ->
# escalate to the operator by voice -> wait for a human answer. Without
# extend(), commit() fails at the exact moment the deal closes.

def test_extend_keeps_a_hold_alive_across_a_long_negotiation():
    clock = _FakeClock()
    led = CapacityLedger(400, "muffins", default_ttl_seconds=180.0, clock=clock)
    h = led.hold(200)

    # Five minutes of negotiation, heartbeating every 60s.
    for _ in range(5):
        clock.advance(60)
        led.extend(h.id)

    assert led.commit(h.id).state is CapacityState.COMMITTED
    assert led.committed() == 200


def test_hold_still_dies_when_the_call_drops():
    """Extension is a heartbeat, not immortality."""
    clock = _FakeClock()
    led = CapacityLedger(400, "muffins", default_ttl_seconds=180.0, clock=clock)
    h = led.hold(200)
    led.extend(h.id)

    clock.advance(181)  # nobody heartbeat: the call is gone
    assert led.available() == 400
    with pytest.raises(ExpiredHold):
        led.commit(h.id)


def test_expired_hold_cannot_be_resurrected_by_extend():
    """Those units may already belong to someone else."""
    clock = _FakeClock()
    led = CapacityLedger(400, "muffins", default_ttl_seconds=10.0, clock=clock)
    h = led.hold(200)
    clock.advance(11)
    with pytest.raises((ExpiredHold, UnknownHold)):
        led.extend(h.id)


def test_extend_does_not_let_a_hold_exceed_capacity():
    clock = _FakeClock()
    led = CapacityLedger(400, "muffins", default_ttl_seconds=60.0, clock=clock)
    a = led.hold(300)
    led.extend(a.id, 600)
    assert led.hold(200) is None, "extended hold must still occupy its units"
    assert led.available() == 100
