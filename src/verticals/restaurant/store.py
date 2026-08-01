"""SQLite persistence for calls, orders, claims and evidence.

Stdlib `sqlite3`, no ORM: the schema is five tables and the queries are the
product, so a mapping layer would only hide them.

## The one invariant this file exists to protect

`claims.verdict` is a *derived* column. It is computed at write time by handing
the evidence rows to the real `src.verify.receipts.Claim` and reading its
`verdict` property - the same code path the receipts and the tests rely on -
and it is then immutable. Three things enforce that:

1. No function here takes a `verdict` argument. There is nothing to pass.
   `tests/test_restaurant.py` asserts this by introspection, so adding one
   turns a test red rather than quietly reopening the fabrication path.
2. A `BEFORE UPDATE OF verdict` trigger aborts the statement. Even raw SQL
   against the file cannot promote a claim.
3. `read()` refuses anything that is not a SELECT, so the query and UI layers
   physically cannot write.

Re-ingesting a receipt whose evidence has grown is therefore a delete of the
claim and its evidence followed by a fresh insert, not an update. The verdict
that lands is always the one `Claim.verdict` derives from the evidence present
at that moment. There is no code path in which a caller's opinion of a verdict
is written to disk.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

from src.verify.receipts import Channel, Claim, Evidence, Verdict

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id              TEXT PRIMARY KEY,
    room            TEXT,
    to_number       TEXT,
    started         TEXT,
    ended           TEXT,
    outcome         TEXT,
    transcript_path TEXT,
    recording_path  TEXT,
    task            TEXT,
    headline        TEXT,
    is_sample       INTEGER NOT NULL DEFAULT 0,
    ingested_at     TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id          TEXT PRIMARY KEY,
    call_id     TEXT NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    claim_id    TEXT,
    qty         INTEGER NOT NULL,
    unit        TEXT NOT NULL,
    total       REAL NOT NULL,
    delivery_at TEXT,
    status      TEXT NOT NULL,
    list_total  REAL,
    floor_total REAL,
    is_sample   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS claims (
    id                   TEXT PRIMARY KEY,
    call_id              TEXT NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    description          TEXT NOT NULL,
    verdict              TEXT NOT NULL,
    expected_side_effect TEXT,
    created_at           TEXT,
    is_sample            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS evidence (
    id            TEXT PRIMARY KEY,
    claim_id      TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    channel       TEXT NOT NULL,
    summary       TEXT,
    independent   INTEGER NOT NULL,
    supports      INTEGER NOT NULL DEFAULT 1,
    content_hash  TEXT,
    artifact_path TEXT,
    captured_at   TEXT
);

CREATE TABLE IF NOT EXISTS capacity_log (
    date      TEXT NOT NULL,
    unit      TEXT NOT NULL,
    total     INTEGER NOT NULL,
    committed INTEGER NOT NULL DEFAULT 0,
    held      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date, unit)
);

CREATE INDEX IF NOT EXISTS idx_orders_call    ON orders(call_id);
CREATE INDEX IF NOT EXISTS idx_claims_call    ON claims(call_id);
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_calls_started  ON calls(started);

-- Defence in depth. `record_claim` never offers a verdict argument, but the
-- database file outlives this process and can be opened by anything. A verdict
-- is a function of the evidence rows; the only honest way to change one is to
-- change the evidence and let it be re-derived.
CREATE TRIGGER IF NOT EXISTS claims_verdict_is_immutable
BEFORE UPDATE OF verdict ON claims
BEGIN
    SELECT RAISE(ABORT, 'claims.verdict is derived from evidence and cannot be updated - attach evidence and re-ingest instead');
END;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReadOnlyViolation(RuntimeError):
    """`Store.read()` was handed something that is not a SELECT."""


@dataclass
class EvidenceRow:
    """One evidence artifact, on its way into the database.

    `independent` is not accepted from the caller: it is decided by looking the
    channel up in `INDEPENDENT_CHANNELS`. A receipt file that claimed an agent
    assertion was independent would otherwise be believed.
    """

    channel: str
    summary: str = ""
    supports: bool = True
    content_hash: str | None = None
    artifact_path: str | None = None
    captured_at: str | None = None
    id: str | None = None

    @property
    def channel_enum(self) -> Channel | None:
        try:
            return Channel(self.channel)
        except ValueError:
            return None

    @property
    def is_independent(self) -> bool:
        ch = self.channel_enum
        return bool(ch and Evidence(channel=ch, summary="").is_independent)


@dataclass
class CallRow:
    """A call as the store keeps it. `id` is the receipt id when there is one."""

    id: str
    room: str | None = None
    to_number: str | None = None
    started: str | None = None
    ended: str | None = None
    outcome: str | None = None
    transcript_path: str | None = None
    recording_path: str | None = None
    task: str | None = None
    headline: str | None = None
    is_sample: bool = False


@dataclass
class OrderRow:
    """An order extracted from a claim.

    `status` mirrors the claim's derived verdict where one exists. It is a
    label for the owner, never an input to revenue: `query.revenue()` joins
    back to `claims.verdict` so a mislabelled status cannot inflate proven
    revenue.
    """

    id: str
    call_id: str
    qty: int
    unit: str
    total: float
    delivery_at: str | None = None
    status: str = "pending"
    claim_id: str | None = None
    list_total: float | None = None
    floor_total: float | None = None
    is_sample: bool = False


def derive_verdict(evidence: Iterable[EvidenceRow]) -> Verdict:
    """Compute a verdict the only legitimate way: through `Claim.verdict`.

    Deliberately not a reimplementation of the rules. If `src/verify/receipts.py`
    changes what counts as verification, this follows, and the two can never
    disagree about the same evidence.
    """
    claim = Claim(description="", expected_side_effect="")
    for row in evidence:
        ch = row.channel_enum
        if ch is None:
            # Unknown channel: recorded elsewhere, but never allowed to verify.
            continue
        claim.attach_evidence(
            Evidence(
                channel=ch,
                summary=row.summary,
                supports=row.supports,
                content_hash=row.content_hash,
                artifact_path=row.artifact_path,
            )
        )
    return claim.verdict


class Store:
    """A SQLite database of calls, orders, claims and evidence.

    Not thread-affine: `check_same_thread=False` plus a per-statement commit is
    enough for a read-mostly dashboard, and the alternative - a connection pool
    - is complexity nobody is going to debug at 8pm.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.create_schema()

    # -- lifecycle --------------------------------------------------------

    def create_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reads ------------------------------------------------------------

    def read(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run a SELECT and return plain dicts.

        Refuses anything else. The query and UI layers import this and nothing
        that writes, so no amount of string-building upstream can mutate a
        verdict.
        """
        head = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
        if head not in {"SELECT", "WITH"}:
            raise ReadOnlyViolation(f"read() accepts SELECT/WITH only, got {head!r}")
        cur = self._conn.execute(sql, tuple(params))
        return [dict(r) for r in cur.fetchall()]

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        rows = self.read(sql, params)
        if not rows:
            return default
        value = next(iter(rows[0].values()))
        return default if value is None else value

    def counts(self) -> dict[str, int]:
        return {
            table: int(self.scalar(f"SELECT COUNT(*) FROM {table}", default=0) or 0)
            for table in ("calls", "orders", "claims", "evidence", "capacity_log")
        }

    # -- writes -----------------------------------------------------------

    def record_call(self, call: CallRow) -> str:
        """Insert or refresh a call. Idempotent on `call.id`."""
        self._conn.execute(
            """
            INSERT INTO calls (id, room, to_number, started, ended, outcome,
                               transcript_path, recording_path, task, headline,
                               is_sample, ingested_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                room = excluded.room,
                to_number = excluded.to_number,
                started = excluded.started,
                ended = excluded.ended,
                outcome = excluded.outcome,
                transcript_path = excluded.transcript_path,
                recording_path = excluded.recording_path,
                task = excluded.task,
                headline = excluded.headline,
                is_sample = excluded.is_sample,
                ingested_at = excluded.ingested_at
            """,
            (
                call.id,
                call.room,
                call.to_number,
                call.started,
                call.ended,
                call.outcome,
                call.transcript_path,
                call.recording_path,
                call.task,
                call.headline,
                int(call.is_sample),
                _now(),
            ),
        )
        self._conn.commit()
        return call.id

    def record_claim(
        self,
        *,
        claim_id: str,
        call_id: str,
        description: str,
        expected_side_effect: str | None = None,
        evidence: Sequence[EvidenceRow] = (),
        created_at: str | None = None,
        is_sample: bool = False,
    ) -> Verdict:
        """Write a claim and its evidence, deriving the verdict here.

        Note the signature: there is no `verdict` parameter and there is no
        overload that adds one. The returned `Verdict` is what was stored, so
        callers can report it without being able to choose it.

        Idempotent on `claim_id` by delete-then-insert. That is not an
        optimisation dodge - it is what keeps the verdict a pure function of
        the evidence present at write time even when a receipt is re-ingested
        after an SMS finally lands.
        """
        rows = list(evidence)
        verdict = derive_verdict(rows)

        with self._conn:  # one transaction: never a claim without its evidence
            self._conn.execute("DELETE FROM evidence WHERE claim_id = ?", (claim_id,))
            self._conn.execute("DELETE FROM claims WHERE id = ?", (claim_id,))
            self._conn.execute(
                """
                INSERT INTO claims (id, call_id, description, verdict,
                                    expected_side_effect, created_at, is_sample)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    claim_id,
                    call_id,
                    description,
                    verdict.value,
                    expected_side_effect,
                    created_at or _now(),
                    int(is_sample),
                ),
            )
            for i, row in enumerate(rows):
                self._conn.execute(
                    """
                    INSERT INTO evidence (id, claim_id, channel, summary, independent,
                                          supports, content_hash, artifact_path, captured_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row.id or f"{claim_id}_ev{i}",
                        claim_id,
                        row.channel,
                        row.summary,
                        int(row.is_independent),
                        int(row.supports),
                        row.content_hash,
                        row.artifact_path,
                        row.captured_at or _now(),
                    ),
                )
        return verdict

    def record_order(self, order: OrderRow) -> str:
        """Insert or refresh an order. Idempotent on `order.id`."""
        self._conn.execute(
            """
            INSERT INTO orders (id, call_id, claim_id, qty, unit, total,
                                delivery_at, status, list_total, floor_total, is_sample)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                call_id = excluded.call_id,
                claim_id = excluded.claim_id,
                qty = excluded.qty,
                unit = excluded.unit,
                total = excluded.total,
                delivery_at = excluded.delivery_at,
                status = excluded.status,
                list_total = excluded.list_total,
                floor_total = excluded.floor_total,
                is_sample = excluded.is_sample
            """,
            (
                order.id,
                order.call_id,
                order.claim_id,
                int(order.qty),
                order.unit,
                float(order.total),
                order.delivery_at,
                order.status,
                order.list_total,
                order.floor_total,
                int(order.is_sample),
            ),
        )
        self._conn.commit()
        return order.id

    def log_capacity(
        self, *, date: str, unit: str, total: int, committed: int = 0, held: int = 0
    ) -> None:
        """Snapshot capacity for one day. Idempotent on (date, unit)."""
        self._conn.execute(
            """
            INSERT INTO capacity_log (date, unit, total, committed, held)
            VALUES (?,?,?,?,?)
            ON CONFLICT(date, unit) DO UPDATE SET
                total = excluded.total,
                committed = excluded.committed,
                held = excluded.held
            """,
            (date, unit, int(total), int(committed), int(held)),
        )
        self._conn.commit()

    def clear_capacity_log(self) -> None:
        """Capacity snapshots are recomputed wholesale after every ingest."""
        self._conn.execute("DELETE FROM capacity_log")
        self._conn.commit()

    def delete_call(self, call_id: str) -> None:
        """Remove a call and everything hanging off it. Used to refresh samples."""
        with self._conn:
            self._conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
            # Cascades are on, but be explicit for rows written before a
            # foreign-key-enabled connection touched the file.
            self._conn.execute(
                "DELETE FROM evidence WHERE claim_id IN "
                "(SELECT id FROM claims WHERE call_id = ?)",
                (call_id,),
            )
            self._conn.execute("DELETE FROM claims WHERE call_id = ?", (call_id,))
            self._conn.execute("DELETE FROM orders WHERE call_id = ?", (call_id,))

    def delete_claim_cascade(self, claim_id: str) -> None:
        """Drop a claim, its evidence and any order derived from it.

        Used when a re-ingested receipt no longer carries a claim it used to.
        Deleting is the only legitimate way to un-say something: there is no
        edit that could leave a stale verdict behind.
        """
        with self._conn:
            self._conn.execute("DELETE FROM orders WHERE claim_id = ?", (claim_id,))
            self._conn.execute("DELETE FROM evidence WHERE claim_id = ?", (claim_id,))
            self._conn.execute("DELETE FROM claims WHERE id = ?", (claim_id,))

    def delete_samples(self) -> int:
        """Drop every synthesized row. Real calls are untouched."""
        ids = [r["id"] for r in self.read("SELECT id FROM calls WHERE is_sample = 1")]
        for call_id in ids:
            self.delete_call(call_id)
        return len(ids)


def open_store(path: str | Path | None = None) -> Store:
    """Open the configured database, creating it if needed."""
    from src.verticals.restaurant import config as cfg

    return Store(path or cfg.default().database_path)
