"""The questions a bakery owner actually asks, as SQL.

Everything here returns plain dicts and lists of dicts. No dataclasses, no
lazy cursors, nothing that needs the store to still be open when a template
renders it - so the same function feeds the FastAPI page, the CSV export and
a test assertion without adapters.

## Booked vs proven

`revenue()` is the headline, and the gap it reports is the product.

* **Booked** is what the agent believes it sold: every order whose claim is not
  contradicted. It is the number a normal CRM would show, and it is the number
  a fabricating agent would show as complete.
* **Proven** is the subset whose claim is VERIFIED - an independent channel
  confirmed it. That is the only money an owner can plan around.

The two differ by exactly the amount of optimism in the system. Reporting only
booked is what gets a team disqualified; reporting only proven hides work in
flight. Both, side by side, with the gap named, is the honest answer.

`revenue()` joins to `claims.verdict` rather than reading `orders.status`. The
status column is a convenience label written at ingest; the verdict is derived
from evidence. If the two ever disagree, the verdict wins, and no revenue
number in this module can be moved by editing a status.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from src.verticals.restaurant import config as cfg
from src.verticals.restaurant.store import Store

VERDICTS = ("VERIFIED", "UNVERIFIED", "CONTRADICTED")

#: Verdict -> the word the owner should read. "UNCONFIRMED" rather than
#: "UNVERIFIED" because the first is a state of the world and the second
#: sounds like an accusation about the agent.
LABEL = {
    "VERIFIED": "PROVEN",
    "UNVERIFIED": "UNCONFIRMED",
    "CONTRADICTED": "CONTRADICTED",
}


def _sample_clause(include_samples: bool, alias: str = "o") -> str:
    return "" if include_samples else f" AND {alias}.is_sample = 0"


def _round(value: Any, places: int = 2) -> float:
    try:
        return round(float(value or 0.0), places)
    except (TypeError, ValueError):
        return 0.0


# -- money ----------------------------------------------------------------


def revenue(store: Store, *, include_samples: bool = True) -> dict[str, Any]:
    """Booked vs proven revenue, and the gap between them.

    Contradicted orders are excluded from `booked` entirely rather than
    subtracted at the end: an order an independent channel says did not happen
    was never revenue, and showing it inside booked invites someone to quote
    the bigger number.
    """
    rows = store.read(
        f"""
        SELECT c.verdict AS verdict,
               COUNT(*)  AS orders,
               SUM(o.total) AS total,
               SUM(o.qty)   AS units
          FROM orders o
          JOIN claims c ON c.id = o.claim_id
         WHERE 1=1{_sample_clause(include_samples)}
         GROUP BY c.verdict
        """
    )
    by = {r["verdict"]: r for r in rows}

    def part(verdict: str, key: str) -> float:
        return _round((by.get(verdict) or {}).get(key))

    proven = part("VERIFIED", "total")
    unconfirmed = part("UNVERIFIED", "total")
    contradicted = part("CONTRADICTED", "total")
    booked = _round(proven + unconfirmed)

    return {
        "currency": cfg.default().currency,
        "booked": booked,
        "proven": proven,
        "unconfirmed": unconfirmed,
        "contradicted": contradicted,
        "gap": _round(booked - proven),
        "proven_pct": _round((proven / booked * 100) if booked else 0.0, 1),
        "orders": {
            "booked": int((by.get("VERIFIED") or {}).get("orders") or 0)
            + int((by.get("UNVERIFIED") or {}).get("orders") or 0),
            "proven": int((by.get("VERIFIED") or {}).get("orders") or 0),
            "unconfirmed": int((by.get("UNVERIFIED") or {}).get("orders") or 0),
            "contradicted": int((by.get("CONTRADICTED") or {}).get("orders") or 0),
        },
        "units": {
            "proven": int(part("VERIFIED", "units")),
            "unconfirmed": int(part("UNVERIFIED", "units")),
            "contradicted": int(part("CONTRADICTED", "units")),
        },
        "includes_samples": include_samples,
    }


def revenue_split(store: Store) -> dict[str, Any]:
    """Both revenue views: everything on screen, and real calls only.

    The UI shows the first and footnotes the second, so a judge can see at a
    glance how much of the headline is synthesized.
    """
    return {
        "all": revenue(store, include_samples=True),
        "real_only": revenue(store, include_samples=False),
        "sample_orders": int(
            store.scalar("SELECT COUNT(*) FROM orders WHERE is_sample = 1", default=0) or 0
        ),
        "real_orders": int(
            store.scalar("SELECT COUNT(*) FROM orders WHERE is_sample = 0", default=0) or 0
        ),
    }


# -- orders ---------------------------------------------------------------


def orders_by_verdict(store: Store, *, include_samples: bool = True) -> dict[str, Any]:
    """Every order bucketed VERIFIED / UNVERIFIED / CONTRADICTED."""
    rows = store.read(
        f"""
        SELECT o.id, o.call_id, o.claim_id, o.qty, o.unit, o.total,
               o.delivery_at, o.status, o.list_total, o.floor_total, o.is_sample,
               c.verdict, c.description, c.expected_side_effect,
               ca.to_number, ca.started, ca.task
          FROM orders o
          JOIN claims c ON c.id = o.claim_id
          JOIN calls ca ON ca.id = o.call_id
         WHERE 1=1{_sample_clause(include_samples)}
         ORDER BY ca.started DESC, o.id
        """
    )
    buckets: dict[str, list[dict[str, Any]]] = {v: [] for v in VERDICTS}
    for r in rows:
        r["label"] = LABEL.get(r["verdict"], r["verdict"])
        r["is_sample"] = bool(r["is_sample"])
        r["total"] = _round(r["total"])
        buckets.setdefault(r["verdict"], []).append(r)
    return {
        "buckets": buckets,
        "counts": {v: len(buckets.get(v, [])) for v in VERDICTS},
        "totals": {v: _round(sum(x["total"] for x in buckets.get(v, []))) for v in VERDICTS},
        "rows": rows,
    }


def orders_flat(store: Store, *, include_samples: bool = True) -> list[dict[str, Any]]:
    """Same data, one flat list. This is what the CSV export writes."""
    return orders_by_verdict(store, include_samples=include_samples)["rows"]


# -- capacity -------------------------------------------------------------


def _week_bounds(week_of: date | str | None = None) -> tuple[date, date]:
    if isinstance(week_of, str):
        try:
            week_of = datetime.fromisoformat(week_of).date()
        except ValueError:
            week_of = None
    ref = week_of or date.today()  # noqa: DTZ011 - a bakery's week is local, not UTC
    start = ref - timedelta(days=ref.weekday())
    return start, start + timedelta(days=6)


def _default_week(store: Store) -> tuple[date, str]:
    """This week, unless it is empty and a later one is not.

    Catering is booked forward: on a Friday afternoon every open commitment
    can legitimately sit in next week, and a panel that reads 0/600 because of
    a calendar boundary is worse than useless - it says "nothing sold".
    """
    today = date.today()  # noqa: DTZ011 - a bakery's week is local, not UTC
    this_start, this_end = _week_bounds(today)
    has_now = store.scalar(
        "SELECT COUNT(*) FROM capacity_log WHERE date >= ? AND date <= ? "
        "AND (committed > 0 OR held > 0)",
        (this_start.isoformat(), this_end.isoformat()),
        default=0,
    )
    if int(has_now or 0):
        return today, "current week"

    nxt = store.scalar(
        "SELECT MIN(date) FROM capacity_log WHERE date > ? AND (committed > 0 OR held > 0)",
        (this_end.isoformat(),),
    )
    if nxt:
        try:
            return date.fromisoformat(str(nxt)), "next week with commitments"
        except ValueError:
            pass
    return today, "current week (no commitments booked)"


def weekly_commitments(
    store: Store, *, week_of: date | str | None = None, include_samples: bool = True
) -> dict[str, Any]:
    """What we committed to this week, and how much capacity is left.

    Committed counts VERIFIED orders only. Held counts UNVERIFIED ones and is
    reported separately rather than folded in: the owner needs to know the
    difference between "buy flour for this" and "chase this".

    `remaining` subtracts both, because an unconfirmed order still occupies an
    oven slot until it is chased and either confirmed or dropped. Over-selling
    against pending orders is how the same week gets promised twice.
    """
    conf = cfg.default()
    if week_of is None:
        week_of, why = _default_week(store)
    else:
        why = "requested week"
    start, end = _week_bounds(week_of)

    log = store.read(
        "SELECT date, unit, total, committed, held FROM capacity_log "
        "WHERE date >= ? AND date <= ? ORDER BY date",
        (start.isoformat(), end.isoformat()),
    )
    committed = sum(int(r["committed"]) for r in log)
    held = sum(int(r["held"]) for r in log)
    total = conf.weekly_capacity

    orders = store.read(
        f"""
        SELECT o.id, o.call_id, o.qty, o.unit, o.total, o.delivery_at,
               c.verdict, ca.to_number, ca.task, o.is_sample
          FROM orders o
          JOIN claims c ON c.id = o.claim_id
          JOIN calls ca ON ca.id = o.call_id
         WHERE c.verdict != 'CONTRADICTED'{_sample_clause(include_samples)}
         ORDER BY ca.started DESC
        """
    )
    for o in orders:
        o["is_sample"] = bool(o["is_sample"])
        o["total"] = _round(o["total"])
        o["label"] = LABEL.get(o["verdict"], o["verdict"])

    return {
        "week_start": start.isoformat(),
        "week_end": end.isoformat(),
        "week_reason": why,
        "unit": conf.unit,
        "total_capacity": total,
        "committed": committed,
        "held": held,
        "remaining": max(0, total - committed - held),
        "utilisation_pct": _round(((committed + held) / total * 100) if total else 0.0, 1),
        "days": log,
        "orders": orders,
    }


def capacity_rows(store: Store) -> list[dict[str, Any]]:
    """The raw daily capacity log. Exported as-is."""
    return store.read(
        "SELECT date, unit, total, committed, held, "
        "(total - committed - held) AS remaining "
        "FROM capacity_log ORDER BY date"
    )


# -- calls ----------------------------------------------------------------


def calls_table(store: Store, *, include_samples: bool = True) -> list[dict[str, Any]]:
    """One row per call, with its claim and money counts. The landing page."""
    clause = "" if include_samples else " WHERE ca.is_sample = 0"
    rows = store.read(
        f"""
        SELECT ca.id, ca.room, ca.to_number, ca.started, ca.ended, ca.outcome,
               ca.task, ca.headline, ca.transcript_path, ca.recording_path,
               ca.is_sample,
               (SELECT COUNT(*) FROM claims c WHERE c.call_id = ca.id) AS claims,
               (SELECT COUNT(*) FROM claims c WHERE c.call_id = ca.id
                  AND c.verdict = 'VERIFIED') AS verified,
               (SELECT COUNT(*) FROM claims c WHERE c.call_id = ca.id
                  AND c.verdict = 'CONTRADICTED') AS contradicted,
               (SELECT COALESCE(SUM(o.total), 0) FROM orders o
                  JOIN claims c ON c.id = o.claim_id
                 WHERE o.call_id = ca.id AND c.verdict != 'CONTRADICTED') AS booked,
               (SELECT COALESCE(SUM(o.total), 0) FROM orders o
                  JOIN claims c ON c.id = o.claim_id
                 WHERE o.call_id = ca.id AND c.verdict = 'VERIFIED') AS proven
          FROM calls ca{clause}
         ORDER BY ca.started DESC
        """
    )
    for r in rows:
        r["is_sample"] = bool(r["is_sample"])
        r["booked"] = _round(r["booked"])
        r["proven"] = _round(r["proven"])
        r["duration_s"] = _duration(r["started"], r["ended"])
    return rows


def _duration(started: str | None, ended: str | None) -> float | None:
    try:
        a = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(ended).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return round((b - a).total_seconds(), 1)


def call_detail(store: Store, call_id: str) -> dict[str, Any] | None:
    """One call, with every claim and the full evidence chain under each.

    This is the drill-down a judge asks for: not "the agent says it booked
    200 muffins" but "here is the claim, here is the SMS that confirmed it,
    here is its hash and the time it arrived".
    """
    calls = store.read("SELECT * FROM calls WHERE id = ?", (call_id,))
    if not calls:
        return None
    call = calls[0]
    call["is_sample"] = bool(call["is_sample"])
    call["duration_s"] = _duration(call["started"], call["ended"])

    claims = store.read(
        "SELECT id, description, verdict, expected_side_effect, created_at "
        "FROM claims WHERE call_id = ? ORDER BY created_at, id",
        (call_id,),
    )
    for c in claims:
        c["label"] = LABEL.get(c["verdict"], c["verdict"])
        c["evidence"] = [
            {**e, "independent": bool(e["independent"]), "supports": bool(e["supports"])}
            for e in store.read(
                "SELECT id, channel, summary, independent, supports, content_hash, "
                "artifact_path, captured_at FROM evidence WHERE claim_id = ? "
                "ORDER BY captured_at, id",
                (c["id"],),
            )
        ]
        c["independent_evidence"] = sum(1 for e in c["evidence"] if e["independent"])

    orders = store.read(
        "SELECT id, claim_id, qty, unit, total, delivery_at, status, "
        "list_total, floor_total FROM orders WHERE call_id = ? ORDER BY id",
        (call_id,),
    )
    for o in orders:
        o["total"] = _round(o["total"])
        o["discount_pct"] = _discount_pct(o["list_total"], o["total"])
        o["floor_bound"] = _floor_bound(o["floor_total"], o["total"])

    booked = _round(
        sum(
            o["total"]
            for o in orders
            if _verdict_of(claims, o["claim_id"]) != "CONTRADICTED"
        )
    )
    proven = _round(
        sum(o["total"] for o in orders if _verdict_of(claims, o["claim_id"]) == "VERIFIED")
    )
    return {
        "call": call,
        "claims": claims,
        "orders": orders,
        "booked": booked,
        "proven": proven,
        "gap": _round(booked - proven),
        "currency": cfg.default().currency,
    }


def _verdict_of(claims: list[dict[str, Any]], claim_id: str | None) -> str:
    for c in claims:
        if c["id"] == claim_id:
            return str(c["verdict"])
    return "UNVERIFIED"


# -- negotiation quality --------------------------------------------------


def _discount_pct(list_total: Any, total: Any) -> float | None:
    try:
        lt = float(list_total)
        t = float(total)
    except (TypeError, ValueError):
        return None
    if lt <= 0:
        return None
    return round((lt - t) / lt * 100, 2)


def _floor_bound(floor_total: Any, total: Any) -> bool:
    """True when the agent ended up at the floor, within a cent.

    "At the floor" and "below the floor" are different stories - the second is
    a bug in `src/business/pricing.py` and would be worth an alarm - but for
    counting how often the floor did the work, at-or-under is the question.
    """
    try:
        return float(total) <= float(floor_total) + 0.01
    except (TypeError, ValueError):
        return False


def discount_stats(store: Store, *, include_samples: bool = True) -> dict[str, Any]:
    """Average discount given, and how often the price floor was the binding
    constraint.

    A high average discount with a low floor-bound rate means the agent is
    conceding when it did not have to. A low average with a high floor-bound
    rate means the floor is set where the market is, and the agent is holding
    the line. Either way the pair says more than either number alone.
    """
    rows = store.read(
        f"""
        SELECT o.id, o.call_id, o.qty, o.total, o.list_total, o.floor_total,
               c.verdict, o.is_sample
          FROM orders o
          JOIN claims c ON c.id = o.claim_id
         WHERE o.list_total IS NOT NULL{_sample_clause(include_samples)}
        """
    )
    discounts: list[float] = []
    floor_hits = 0
    for r in rows:
        d = _discount_pct(r["list_total"], r["total"])
        if d is not None:
            discounts.append(d)
        if _floor_bound(r["floor_total"], r["total"]):
            floor_hits += 1

    n = len(rows)
    return {
        "orders_priced": n,
        "avg_discount_pct": _round(sum(discounts) / len(discounts), 2) if discounts else 0.0,
        "max_discount_pct": _round(max(discounts), 2) if discounts else 0.0,
        "min_discount_pct": _round(min(discounts), 2) if discounts else 0.0,
        "floor_bound_orders": floor_hits,
        "floor_bound_pct": _round((floor_hits / n * 100) if n else 0.0, 1),
        "note": (
            "Discount is measured against the target price from the operator's own "
            "cost model in config/restaurant.json, not against a list price the "
            "agent chose."
        ),
    }


def discount_rows(store: Store, *, include_samples: bool = True) -> list[dict[str, Any]]:
    """Per-order discount detail. The CSV behind `discount_stats`."""
    rows = store.read(
        f"""
        SELECT o.id, o.call_id, o.qty, o.unit, o.total, o.list_total, o.floor_total,
               c.verdict, o.is_sample
          FROM orders o
          JOIN claims c ON c.id = o.claim_id
         WHERE 1=1{_sample_clause(include_samples)}
         ORDER BY o.id
        """
    )
    for r in rows:
        r["discount_pct"] = _discount_pct(r["list_total"], r["total"])
        r["floor_bound"] = _floor_bound(r["floor_total"], r["total"])
        r["is_sample"] = bool(r["is_sample"])
    return rows


# -- evidence -------------------------------------------------------------


def evidence_chain(store: Store, *, include_samples: bool = True) -> list[dict[str, Any]]:
    """Every evidence row, joined to its claim and call. The audit export."""
    clause = "" if include_samples else " WHERE c.is_sample = 0"
    return store.read(
        f"""
        SELECT e.id, e.claim_id, c.call_id, c.description, c.verdict,
               e.channel, e.independent, e.supports, e.summary,
               e.content_hash, e.captured_at
          FROM evidence e
          JOIN claims c ON c.id = e.claim_id{clause}
         ORDER BY e.captured_at, e.id
        """
    )


# -- the top of the page --------------------------------------------------


def overview(store: Store) -> dict[str, Any]:
    """Everything the dashboard header needs, in one call."""
    conf = cfg.default()
    money = revenue_split(store)
    return {
        "business": conf.business_name,
        "currency": conf.currency,
        "unit": conf.unit,
        "revenue": money["all"],
        "revenue_real_only": money["real_only"],
        "sample_orders": money["sample_orders"],
        "real_orders": money["real_orders"],
        "capacity": weekly_commitments(store),
        "discounts": discount_stats(store),
        "counts": store.counts(),
        "calls_real": int(
            store.scalar("SELECT COUNT(*) FROM calls WHERE is_sample = 0", default=0) or 0
        ),
        "calls_sample": int(
            store.scalar("SELECT COUNT(*) FROM calls WHERE is_sample = 1", default=0) or 0
        ),
    }


# -- export registry ------------------------------------------------------

#: name -> (human title, rows callable). Anything in here gets a CSV endpoint
#: and a Sheets tab for free, so adding a query to the UI is one line.
EXPORTS: dict[str, tuple[str, Callable[[Store], list[dict[str, Any]]]]] = {
    "calls": ("Calls", lambda s: calls_table(s)),
    "orders": ("Orders by verdict", lambda s: orders_flat(s)),
    "evidence": ("Evidence chain", lambda s: evidence_chain(s)),
    "capacity": ("Capacity log", lambda s: capacity_rows(s)),
    "discounts": ("Discounts per order", lambda s: discount_rows(s)),
    "revenue": ("Booked vs proven", lambda s: _revenue_rows(s)),
}


def _revenue_rows(store: Store) -> list[dict[str, Any]]:
    """The headline as two rows, so it survives a spreadsheet."""
    money = revenue_split(store)
    out = []
    for scope, data in (("all rows", money["all"]), ("real calls only", money["real_only"])):
        out.append(
            {
                "scope": scope,
                "currency": data["currency"],
                "booked": data["booked"],
                "proven": data["proven"],
                "gap": data["gap"],
                "proven_pct": data["proven_pct"],
                "unconfirmed": data["unconfirmed"],
                "contradicted": data["contradicted"],
                "orders_booked": data["orders"]["booked"],
                "orders_proven": data["orders"]["proven"],
            }
        )
    return out


def run_export(store: Store, name: str) -> list[dict[str, Any]]:
    """Rows for a named export, or KeyError naming the valid options."""
    try:
        _, fn = EXPORTS[name]
    except KeyError:
        raise KeyError(f"unknown export {name!r}; known: {', '.join(sorted(EXPORTS))}") from None
    return fn(store)
