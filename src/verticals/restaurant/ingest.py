"""Load `evidence/receipt_*.json` into the store, idempotently.

Receipts are the source of truth and this is a projection of them, so ingest
has to be safe to run on a loop: re-running over the same directory must leave
the row counts unchanged. Every write is keyed - calls on the receipt id,
claims on the claim id, orders on `{claim_id}_order` - so the second pass
overwrites the first rather than doubling it.

Two things are deliberately *not* copied from the file:

* **The verdict.** `store.record_claim` re-derives it from the evidence rows.
  A hand-edited receipt that says VERIFIED with no independent evidence lands
  in the database as UNVERIFIED, which is the whole point of the exercise.
* **`evidence.independent`.** Also re-derived, from the channel.

Order rows are parsed out of claim descriptions because that is where the
sales agent puts them (`close_order` in `src/agents/sales_agent.py` writes
"200 muffins for 400.00, delivered Friday 8am"). Parsing fails toward *not*
creating an order: a claim we cannot read is still stored as a claim, it just
does not become a line of revenue. Inventing an order from an unparseable
string would be the same class of mistake as inventing a verdict.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.verticals.restaurant import config as cfg
from src.verticals.restaurant.store import CallRow, EvidenceRow, OrderRow, Store

#: "200 muffins for 400.00, delivered Friday 8am"
_ORDER_RE = re.compile(
    r"^\s*(?P<qty>\d[\d,]*)\s+(?P<unit>[A-Za-z][A-Za-z /-]*?)\s+for\s+"
    r"\$?(?P<total>[\d,]+(?:\.\d+)?)"
    r"(?:\s*,\s*(?:delivered|delivery|for delivery)\s+(?P<when>.+?))?\s*$",
    re.IGNORECASE,
)

#: "$400 for 200 muffins on Friday" - the other order the agent sometimes uses.
_ORDER_RE_ALT = re.compile(
    r"^\s*\$?(?P<total>[\d,]+(?:\.\d+)?)\s+for\s+(?P<qty>\d[\d,]*)\s+"
    r"(?P<unit>[A-Za-z][A-Za-z /-]*?)"
    r"(?:\s*,?\s*(?:on|delivered|delivery)\s+(?P<when>.+?))?\s*$",
    re.IGNORECASE,
)

#: A phone number or email inside an expected side effect.
_TARGET_RE = re.compile(r"(?:arrives at|to)\s+(?P<target>\+?[\d][\d\- ]{6,}|\S+@\S+)", re.IGNORECASE)

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class IngestReport:
    """What one ingest pass did. Printed by the CLI, shown in the UI footer."""

    files_seen: int = 0
    files_skipped: int = 0
    calls: int = 0
    claims: int = 0
    orders: int = 0
    evidence: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_seen": self.files_seen,
            "files_skipped": self.files_skipped,
            "calls": self.calls,
            "claims": self.claims,
            "orders": self.orders,
            "evidence": self.evidence,
            "errors": list(self.errors),
        }


# -- parsing helpers ------------------------------------------------------


def parse_order(description: str) -> dict[str, Any] | None:
    """Pull qty / unit / total / delivery text out of a claim description.

    Returns None when the description is not an order - an escalation note, a
    callback promise, anything else a claim can legitimately be.
    """
    for pattern in (_ORDER_RE, _ORDER_RE_ALT):
        m = pattern.match(description or "")
        if not m:
            continue
        try:
            qty = int(m.group("qty").replace(",", ""))
            total = float(m.group("total").replace(",", ""))
        except (TypeError, ValueError):
            continue
        if qty <= 0 or total <= 0:
            continue
        unit = (m.group("unit") or "").strip().lower()
        when = (m.group("when") or "").strip() or None
        return {"qty": qty, "unit": unit, "total": total, "delivery_at": when}
    return None


def parse_target(expected_side_effect: str | None) -> str | None:
    """The number or address the confirmation was supposed to arrive at."""
    if not expected_side_effect:
        return None
    m = _TARGET_RE.search(expected_side_effect)
    if not m:
        return None
    return m.group("target").strip().rstrip(".,")


def _as_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def resolve_delivery_date(text: str | None, reference: date | None = None) -> date | None:
    """Best-effort delivery date from free text like "Friday 8am".

    Returns None rather than guessing when the text carries no date at all -
    a capacity row on the wrong week is worse than a missing one, because the
    owner reads the weekly number as a commitment.
    """
    if not text:
        return None
    ref = reference or date.today()  # noqa: DTZ011 - a bakery's day is local, not UTC
    t = text.strip().lower()

    if iso := re.search(r"(\d{4})-(\d{2})-(\d{2})", t):
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None

    if md := re.search(r"\b(" + "|".join(_MONTHS) + r")[a-z]*\.?\s+(\d{1,2})\b", t):
        try:
            return date(ref.year, _MONTHS[md.group(1)], int(md.group(2)))
        except ValueError:
            return None

    if dm := re.search(r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")[a-z]*\b", t):
        try:
            return date(ref.year, _MONTHS[dm.group(2)], int(dm.group(1)))
        except ValueError:
            return None

    for name, idx in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", t):
            ahead = (idx - ref.weekday()) % 7
            if ahead == 0 or "next" in t:
                ahead = 7 if "next" in t or ahead == 0 else ahead
            return ref + timedelta(days=ahead)

    if "tomorrow" in t:
        return ref + timedelta(days=1)
    if "today" in t:
        return ref
    return None


def _outcome(headline: str | None) -> str:
    """First token of the receipt headline: SUCCESS / PARTIAL / FAILED / NO."""
    if not headline:
        return "UNKNOWN"
    head = headline.split(" ", 1)[0].strip().upper()
    return "NO CLAIMS" if head == "NO" else head or "UNKNOWN"


def _status_for(verdict: str) -> str:
    return {
        "VERIFIED": "confirmed",
        "UNVERIFIED": "unconfirmed",
        "CONTRADICTED": "contradicted",
    }.get(verdict, "unknown")


# -- the ingest itself ----------------------------------------------------


def ingest_receipt(
    store: Store,
    receipt: dict[str, Any],
    *,
    source: Path | None = None,
    is_sample: bool = False,
    report: IngestReport | None = None,
) -> str | None:
    """Load one receipt dict. Returns the call id, or None if unusable."""
    rep = report or IngestReport()
    receipt_id = str(receipt.get("id") or "").strip()
    if not receipt_id:
        rep.errors.append(f"{source or '<dict>'}: receipt has no id")
        return None

    claims = receipt.get("claims") or []
    to_number = None
    for c in claims:
        to_number = to_number or parse_target(c.get("expected_side_effect"))

    store.record_call(
        CallRow(
            id=receipt_id,
            room=receipt.get("room") or receipt.get("call_room"),
            to_number=to_number or receipt.get("to_number"),
            started=receipt.get("started_at"),
            ended=receipt.get("ended_at"),
            outcome=_outcome(receipt.get("headline")),
            transcript_path=receipt.get("transcript") or receipt.get("transcript_path"),
            recording_path=receipt.get("call_recording"),
            task=receipt.get("task"),
            headline=receipt.get("headline"),
            is_sample=is_sample,
        )
    )
    rep.calls += 1

    costs = cfg.default().cost_model

    # A re-ingest of a receipt that lost a claim must not leave the old row
    # behind pretending to be revenue.
    keep = {str(c.get("id")) for c in claims if c.get("id")}
    for row in store.read("SELECT id FROM claims WHERE call_id = ?", (receipt_id,)):
        if row["id"] not in keep:
            store.delete_claim_cascade(row["id"])

    for c in claims:
        claim_id = str(c.get("id") or "").strip()
        if not claim_id:
            rep.errors.append(f"{receipt_id}: claim without id skipped")
            continue

        ev_rows = [
            EvidenceRow(
                id=str(e.get("id") or ""),
                channel=str(e.get("channel") or ""),
                summary=str(e.get("summary") or ""),
                supports=bool(e.get("supports", True)),
                content_hash=e.get("content_hash"),
                artifact_path=e.get("artifact_path"),
                captured_at=e.get("captured_at"),
            )
            for e in (c.get("evidence") or [])
        ]

        verdict = store.record_claim(
            claim_id=claim_id,
            call_id=receipt_id,
            description=str(c.get("description") or ""),
            expected_side_effect=c.get("expected_side_effect"),
            evidence=ev_rows,
            created_at=c.get("created_at") or receipt.get("started_at"),
            is_sample=is_sample,
        )
        rep.claims += 1
        rep.evidence += len(ev_rows)

        parsed = parse_order(str(c.get("description") or ""))
        if not parsed:
            continue
        qty = parsed["qty"]
        store.record_order(
            OrderRow(
                id=f"{claim_id}_order",
                call_id=receipt_id,
                claim_id=claim_id,
                qty=qty,
                unit=parsed["unit"] or cfg.default().unit,
                total=parsed["total"],
                delivery_at=parsed["delivery_at"],
                status=_status_for(verdict.value),
                list_total=float(costs.target_price(qty)),
                floor_total=float(costs.floor_price(qty)),
                is_sample=is_sample,
            )
        )
        rep.orders += 1

    return receipt_id


def ingest_file(
    store: Store, path: Path, *, is_sample: bool = False, report: IngestReport | None = None
) -> str | None:
    rep = report or IngestReport()
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        rep.files_skipped += 1
        rep.errors.append(f"{path.name}: {exc}")
        return None
    if not isinstance(data, dict):
        rep.files_skipped += 1
        rep.errors.append(f"{path.name}: not a receipt object")
        return None
    return ingest_receipt(store, data, source=path, is_sample=is_sample, report=rep)


def ingest_directory(
    store: Store,
    directory: str | Path | None = None,
    *,
    pattern: str = "receipt_*.json",
    is_sample: bool = False,
    rebuild_capacity: bool = True,
) -> IngestReport:
    """Load every receipt in `directory`. Safe to run repeatedly."""
    rep = IngestReport()
    root = Path(directory) if directory else cfg.default().evidence_dir
    if not Path(root).exists():
        return rep
    for path in sorted(Path(root).glob(pattern)):
        rep.files_seen += 1
        ingest_file(store, path, is_sample=is_sample, report=rep)
    if rebuild_capacity:
        rebuild_capacity_log(store)
    return rep


def rebuild_capacity_log(store: Store, *, weekly_capacity: int | None = None) -> int:
    """Recompute the daily capacity snapshot from the orders table.

    Wholesale, not incremental: the log is a derived view and an incremental
    update that drifts is worse than a rebuild that takes a millisecond.

    Only VERIFIED orders count as `committed`. UNVERIFIED ones are `held` -
    the same distinction `src/business/capacity.py` draws between a hold and a
    commit, and for the same reason: an unconfirmed order is not capacity the
    owner should plan around, but it is capacity they should not resell either.
    """
    conf = cfg.default()
    total = int(weekly_capacity if weekly_capacity is not None else conf.weekly_capacity)
    daily = max(1, round(total / 7))

    rows = store.read(
        """
        SELECT o.qty, o.unit, o.delivery_at, c.verdict, ca.started
          FROM orders o
          JOIN claims c ON c.id = o.claim_id
          JOIN calls  ca ON ca.id = o.call_id
        """
    )
    store.clear_capacity_log()
    buckets: dict[tuple[str, str], dict[str, int]] = {}
    for r in rows:
        ref = _as_date(r["started"])
        when = resolve_delivery_date(r["delivery_at"], ref) or ref
        if when is None:
            continue
        key = (when.isoformat(), r["unit"] or conf.unit)
        b = buckets.setdefault(key, {"committed": 0, "held": 0})
        if r["verdict"] == "VERIFIED":
            b["committed"] += int(r["qty"])
        elif r["verdict"] == "UNVERIFIED":
            b["held"] += int(r["qty"])
        # CONTRADICTED consumes no capacity: the order did not happen.

    for (day, unit), b in sorted(buckets.items()):
        store.log_capacity(
            date=day, unit=unit, total=daily, committed=b["committed"], held=b["held"]
        )
    return len(buckets)


def refresh(
    store: Store | None = None,
    *,
    directory: str | Path | None = None,
    with_samples: bool | None = None,
) -> tuple[Store, IngestReport]:
    """Ingest real receipts, then top up with clearly-marked sample calls.

    The topping-up is the part to be careful about. A demo dashboard with four
    rows reads as broken, but a dashboard that shows fixtures as if they were
    calls is exactly the fabrication this project is built to refuse. So sample
    rows carry `is_sample = 1` all the way to the UI, which badges them and
    reports the real-only totals alongside.
    """
    from src.verticals.restaurant import seed

    conf = cfg.default()
    st = store or Store(conf.database_path)
    rep = ingest_directory(st, directory, rebuild_capacity=False)

    want_samples = conf.sample_enabled if with_samples is None else with_samples
    if want_samples:
        st.delete_samples()
        real_orders = int(
            st.scalar("SELECT COUNT(*) FROM orders WHERE is_sample = 0", default=0) or 0
        )
        # Top up on *orders*, not calls: forty receipts with no claims still
        # leave every money panel empty, which is the thing samples exist for.
        #
        # All-or-nothing rather than "enough to reach the threshold". The
        # scenario list is curated so the awkward cases - a contradicted
        # order, a deal that landed on the floor - are in it, and a partial
        # load quietly drops exactly those: they are at the back.
        if real_orders < conf.sample_min_orders:
            for receipt in seed.sample_receipts(len(seed.SCENARIOS)):
                ingest_receipt(st, receipt, is_sample=True, report=rep)

    rebuild_capacity_log(st)
    return st, rep


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Ingest receipts into the restaurant store")
    ap.add_argument("--db", default=None, help="database path (default: from config)")
    ap.add_argument("--dir", default=None, help="evidence directory")
    ap.add_argument("--no-samples", action="store_true", help="real receipts only")
    args = ap.parse_args(list(argv) if argv is not None else None)

    st = Store(args.db or cfg.default().database_path)
    _, rep = refresh(st, directory=args.dir, with_samples=not args.no_samples)
    print(json.dumps({**rep.to_dict(), "rows": st.counts()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
