"""Tests for the restaurant data layer.

The load-bearing ones are the last two classes. Everything else checks that
the numbers add up; `TestVerdictIsImmutable` and `TestNoDirectVerdictPath`
check that the numbers cannot be *made* to add up by anything other than
evidence. If those go red, the disqualification condition in CLAUDE.md is
reachable through the database instead of through the agent, which is the same
failure wearing a different hat.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.verify.receipts import Verdict
from src.verticals.restaurant import export as ex
from src.verticals.restaurant import ingest, seed
from src.verticals.restaurant import query as q
from src.verticals.restaurant import store as store_mod
from src.verticals.restaurant.store import (
    CallRow,
    EvidenceRow,
    ReadOnlyViolation,
    Store,
    derive_verdict,
)


def _now(offset_minutes: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)).isoformat()


def _receipt(
    receipt_id: str,
    *,
    qty: int = 100,
    total: float = 250.0,
    when: str = "Friday 8am",
    evidence: list[dict] | None = None,
    task: str = "Sell pastries",
) -> dict:
    """A receipt in exactly the shape `Receipt.to_dict()` writes."""
    return {
        "id": receipt_id,
        "task": task,
        "headline": "PARTIAL - 0/1 verified, 1 could not be confirmed.",
        "started_at": _now(-10),
        "ended_at": _now(-4),
        "call_recording": None,
        "claims": [
            {
                "id": f"claim_{receipt_id}",
                "description": f"{qty} muffins for {total:.2f}, delivered {when}",
                "expected_side_effect": (
                    f"written confirmation arrives at +14155550100 stating quantity "
                    f"{qty} and total {total:.2f}"
                ),
                "evidence": evidence
                if evidence is not None
                else [
                    {
                        "id": f"ev_{receipt_id}_agent",
                        "channel": "agent_assertion",
                        "summary": "Caller agreed on the phone",
                        "supports": True,
                        "independent": False,
                        "captured_at": _now(-5),
                    }
                ],
            }
        ],
        "notes": [],
    }


def _sms(receipt_id: str, *, supports: bool = True) -> dict:
    return {
        "id": f"ev_{receipt_id}_sms",
        "channel": "inbound_sms",
        "summary": "SMS from the customer",
        "supports": supports,
        "independent": True,
        "content_hash": "deadbeefdeadbeef",
        "captured_at": _now(-2),
    }


def _agent(receipt_id: str) -> dict:
    return {
        "id": f"ev_{receipt_id}_agent",
        "channel": "agent_assertion",
        "summary": "Caller agreed on the phone",
        "supports": True,
        "independent": False,
        "captured_at": _now(-5),
    }


@pytest.fixture
def store(tmp_path):
    st = Store(tmp_path / "test.db")
    yield st
    st.close()


@pytest.fixture
def evidence_dir(tmp_path):
    d = tmp_path / "evidence"
    d.mkdir()
    return d


def _write(directory, receipt: dict) -> None:
    (directory / f"{receipt['id']}.json").write_text(json.dumps(receipt))


# -- schema ---------------------------------------------------------------


class TestSchema:
    def test_creates_cleanly(self, store):
        tables = {
            r["name"]
            for r in store.read("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"calls", "orders", "claims", "evidence", "capacity_log"} <= tables

    def test_create_schema_is_idempotent(self, store):
        store.create_schema()
        store.create_schema()
        assert store.counts()["calls"] == 0

    def test_required_columns_exist(self, store):
        expected = {
            "calls": {"id", "room", "to_number", "started", "ended", "outcome",
                      "transcript_path", "recording_path"},
            "orders": {"id", "call_id", "qty", "unit", "total", "delivery_at", "status"},
            "claims": {"id", "call_id", "description", "verdict", "expected_side_effect"},
            "evidence": {"id", "claim_id", "channel", "summary", "independent",
                         "content_hash", "captured_at"},
            "capacity_log": {"date", "unit", "total", "committed", "held"},
        }
        for table, cols in expected.items():
            have = {r["name"] for r in store.read(f"SELECT name FROM pragma_table_info('{table}')")}
            assert cols <= have, f"{table} is missing {cols - have}"

    def test_read_refuses_writes(self, store):
        with pytest.raises(ReadOnlyViolation):
            store.read("DELETE FROM claims")
        with pytest.raises(ReadOnlyViolation):
            store.read("UPDATE claims SET verdict = 'VERIFIED'")


# -- derived verdicts -----------------------------------------------------


class TestDerivedVerdict:
    def test_agent_assertion_alone_is_unverified(self):
        assert derive_verdict([EvidenceRow(channel="agent_assertion")]) is Verdict.UNVERIFIED

    def test_independent_support_verifies(self):
        rows = [EvidenceRow(channel="agent_assertion"), EvidenceRow(channel="inbound_sms")]
        assert derive_verdict(rows) is Verdict.VERIFIED

    def test_independent_dissent_contradicts(self):
        rows = [
            EvidenceRow(channel="inbound_sms", supports=True),
            EvidenceRow(channel="provider_api", supports=False),
        ]
        assert derive_verdict(rows) is Verdict.CONTRADICTED

    def test_no_pile_of_agent_assertions_verifies(self):
        rows = [EvidenceRow(channel="agent_assertion") for _ in range(50)]
        assert derive_verdict(rows) is Verdict.UNVERIFIED

    def test_unknown_channel_cannot_verify(self):
        rows = [EvidenceRow(channel="totally_legit_channel", supports=True)]
        assert derive_verdict(rows) is Verdict.UNVERIFIED

    def test_stored_verdict_ignores_the_file(self, store, evidence_dir):
        """A receipt claiming VERIFIED with only agent evidence lands UNVERIFIED."""
        r = _receipt("receipt_liar")
        r["claims"][0]["verdict"] = "VERIFIED"
        r["headline"] = "SUCCESS - all 1 claim(s) independently verified."
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)
        assert store.scalar("SELECT verdict FROM claims") == "UNVERIFIED"

    def test_evidence_independence_ignores_the_file(self, store, evidence_dir):
        """`independent: true` on an agent assertion is not believed."""
        r = _receipt("receipt_fibber")
        r["claims"][0]["evidence"] = [
            {
                "id": "ev_fib",
                "channel": "agent_assertion",
                "summary": "trust me",
                "supports": True,
                "independent": True,
                "captured_at": _now(),
            }
        ]
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)
        assert store.scalar("SELECT independent FROM evidence") == 0
        assert store.scalar("SELECT verdict FROM claims") == "UNVERIFIED"


# -- ingest ---------------------------------------------------------------


class TestIngest:
    def test_loads_a_receipt(self, store, evidence_dir):
        _write(evidence_dir, _receipt("receipt_aaa"))
        rep = ingest.ingest_directory(store, evidence_dir)
        assert rep.files_seen == 1
        counts = store.counts()
        assert counts["calls"] == 1 and counts["claims"] == 1 and counts["orders"] == 1

    def test_is_idempotent(self, store, evidence_dir):
        for i in range(3):
            _write(evidence_dir, _receipt(f"receipt_{i}"))
        ingest.ingest_directory(store, evidence_dir)
        first = store.counts()
        ingest.ingest_directory(store, evidence_dir)
        ingest.ingest_directory(store, evidence_dir)
        assert store.counts() == first

    def test_reingest_picks_up_late_evidence(self, store, evidence_dir):
        """The SMS arrives after the call. Re-ingest must promote the claim."""
        r = _receipt("receipt_late")
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)
        assert store.scalar("SELECT verdict FROM claims") == "UNVERIFIED"

        r["claims"][0]["evidence"].append(_sms("receipt_late"))
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)

        assert store.scalar("SELECT verdict FROM claims") == "VERIFIED"
        assert store.counts()["claims"] == 1, "re-ingest duplicated the claim"

    def test_dropped_claim_is_removed(self, store, evidence_dir):
        r = _receipt("receipt_shrink")
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)
        assert store.counts()["orders"] == 1

        r["claims"] = []
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)
        assert store.counts()["claims"] == 0
        assert store.counts()["orders"] == 0

    def test_unparseable_claim_makes_no_order(self, store, evidence_dir):
        r = _receipt("receipt_note")
        r["claims"][0]["description"] = "Operator asked: can we do 200 on Friday?"
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)
        assert store.counts()["claims"] == 1
        assert store.counts()["orders"] == 0, "invented an order from an unreadable claim"

    def test_bad_json_is_skipped_not_fatal(self, store, evidence_dir):
        (evidence_dir / "receipt_broken.json").write_text("{not json")
        _write(evidence_dir, _receipt("receipt_good"))
        rep = ingest.ingest_directory(store, evidence_dir)
        assert rep.files_skipped == 1
        assert store.counts()["calls"] == 1

    def test_parses_the_agents_order_phrasing(self):
        parsed = ingest.parse_order("200 muffins for 400.00, delivered Friday 8am")
        assert parsed == {
            "qty": 200, "unit": "muffins", "total": 400.0, "delivery_at": "Friday 8am"
        }

    def test_parses_target_number(self):
        assert (
            ingest.parse_target("written confirmation arrives at +14155550142 stating ...")
            == "+14155550142"
        )

    def test_capacity_log_separates_committed_from_held(self, store, evidence_dir):
        verified = _receipt("receipt_v", qty=100, when="2026-08-12")
        verified["claims"][0]["evidence"].append(_sms("receipt_v"))
        unverified = _receipt("receipt_u", qty=60, when="2026-08-12")
        _write(evidence_dir, verified)
        _write(evidence_dir, unverified)
        ingest.ingest_directory(store, evidence_dir)

        row = store.read("SELECT * FROM capacity_log WHERE date = '2026-08-12'")[0]
        assert row["committed"] == 100
        assert row["held"] == 60

    def test_contradicted_order_consumes_no_capacity(self, store, evidence_dir):
        r = _receipt("receipt_dead", qty=500, when="2026-08-13")
        r["claims"][0]["evidence"].append(_sms("receipt_dead", supports=False))
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)
        rows = store.read("SELECT * FROM capacity_log WHERE date = '2026-08-13'")
        assert rows == [] or (rows[0]["committed"] == 0 and rows[0]["held"] == 0)


# -- money ----------------------------------------------------------------


class TestBookedVsProven:
    def test_booked_and_proven_differ_when_unconfirmed(self, store, evidence_dir):
        proven = _receipt("receipt_p", qty=100, total=300.0)
        proven["claims"][0]["evidence"].append(_sms("receipt_p"))
        _write(evidence_dir, proven)
        _write(evidence_dir, _receipt("receipt_q", qty=50, total=150.0))
        ingest.ingest_directory(store, evidence_dir)

        rev = q.revenue(store)
        assert rev["booked"] == 450.0
        assert rev["proven"] == 300.0
        assert rev["gap"] == 150.0
        assert rev["unconfirmed"] == 150.0

    def test_contradicted_never_counts_toward_proven(self, store, evidence_dir):
        bad = _receipt("receipt_bad", qty=80, total=999.0)
        bad["claims"][0]["evidence"].append(_sms("receipt_bad", supports=False))
        _write(evidence_dir, bad)
        ingest.ingest_directory(store, evidence_dir)

        rev = q.revenue(store)
        assert rev["proven"] == 0.0
        assert rev["booked"] == 0.0, "a contradicted order is not booked revenue either"
        assert rev["contradicted"] == 999.0

    def test_contradiction_beats_supporting_evidence(self, store, evidence_dir):
        r = _receipt("receipt_mixed", qty=10, total=100.0)
        r["claims"][0]["evidence"].append(_sms("receipt_mixed", supports=True))
        r["claims"][0]["evidence"].append(
            {
                "id": "ev_mixed_api",
                "channel": "provider_api",
                "summary": "order not found in POS",
                "supports": False,
                "independent": True,
                "captured_at": _now(),
            }
        )
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)
        assert q.revenue(store)["proven"] == 0.0
        assert store.scalar("SELECT verdict FROM claims") == "CONTRADICTED"

    def test_proven_equals_booked_when_everything_confirms(self, store, evidence_dir):
        r = _receipt("receipt_clean", qty=20, total=88.0)
        r["claims"][0]["evidence"].append(_sms("receipt_clean"))
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)
        rev = q.revenue(store)
        assert rev["booked"] == rev["proven"] == 88.0
        assert rev["gap"] == 0.0
        assert rev["proven_pct"] == 100.0

    def test_editing_order_status_cannot_inflate_proven(self, store, evidence_dir):
        """`orders.status` is a label. Revenue reads the verdict, not the label."""
        _write(evidence_dir, _receipt("receipt_status", qty=10, total=500.0))
        ingest.ingest_directory(store, evidence_dir)
        store._conn.execute("UPDATE orders SET status = 'confirmed'")
        store._conn.commit()
        assert q.revenue(store)["proven"] == 0.0


# -- queries --------------------------------------------------------------


class TestQueries:
    def test_orders_by_verdict_buckets_everything(self, store, evidence_dir):
        good = _receipt("receipt_g", qty=10, total=100.0)
        good["claims"][0]["evidence"].append(_sms("receipt_g"))
        bad = _receipt("receipt_b", qty=10, total=100.0)
        bad["claims"][0]["evidence"].append(_sms("receipt_b", supports=False))
        for r in (good, bad, _receipt("receipt_m", qty=10, total=100.0)):
            _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)

        by = q.orders_by_verdict(store)
        assert by["counts"] == {"VERIFIED": 1, "UNVERIFIED": 1, "CONTRADICTED": 1}

    def test_weekly_commitments_reports_capacity_left(self, store, evidence_dir):
        when = "2026-08-12"
        r = _receipt("receipt_w", qty=100, when=when)
        r["claims"][0]["evidence"].append(_sms("receipt_w"))
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)

        week = q.weekly_commitments(store, week_of=when)
        assert week["committed"] == 100
        assert week["remaining"] == week["total_capacity"] - 100
        assert week["week_start"] == "2026-08-10"

    def test_call_detail_returns_the_full_evidence_chain(self, store, evidence_dir):
        r = _receipt("receipt_chain", qty=30, total=120.0)
        r["claims"][0]["evidence"].append(_sms("receipt_chain"))
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)

        detail = q.call_detail(store, "receipt_chain")
        assert detail is not None
        assert detail["call"]["id"] == "receipt_chain"
        assert len(detail["claims"]) == 1

        claim = detail["claims"][0]
        assert claim["verdict"] == "VERIFIED"
        assert len(claim["evidence"]) == 2, "evidence chain is truncated"
        channels = {e["channel"] for e in claim["evidence"]}
        assert channels == {"agent_assertion", "inbound_sms"}
        assert claim["independent_evidence"] == 1
        sms = next(e for e in claim["evidence"] if e["channel"] == "inbound_sms")
        assert sms["content_hash"] == "deadbeefdeadbeef"
        assert sms["independent"] is True
        assert len(detail["orders"]) == 1
        assert detail["proven"] == 120.0

    def test_call_detail_is_none_for_unknown_call(self, store):
        assert q.call_detail(store, "receipt_nope") is None

    def test_discount_stats_average_and_floor(self, store, evidence_dir):
        from src.verticals.restaurant import config as cfg

        costs = cfg.default().cost_model
        at_floor = float(costs.floor_price(100))
        at_target = float(costs.target_price(100))

        _write(evidence_dir, _receipt("receipt_floor", qty=100, total=at_floor))
        _write(evidence_dir, _receipt("receipt_target", qty=100, total=at_target))
        ingest.ingest_directory(store, evidence_dir)

        stats = q.discount_stats(store)
        assert stats["orders_priced"] == 2
        assert stats["floor_bound_orders"] == 1
        assert stats["floor_bound_pct"] == 50.0
        # One order at target (0% off), one at the floor. The average is half
        # the floor discount.
        floor_discount = (at_target - at_floor) / at_target * 100
        assert stats["avg_discount_pct"] == pytest.approx(floor_discount / 2, abs=0.05)

    def test_overview_has_everything_the_page_needs(self, store, evidence_dir):
        _write(evidence_dir, _receipt("receipt_o"))
        ingest.ingest_directory(store, evidence_dir)
        o = q.overview(store)
        for key in ("revenue", "revenue_real_only", "capacity", "discounts", "counts"):
            assert key in o

    def test_every_registered_export_runs(self, store, evidence_dir):
        _write(evidence_dir, _receipt("receipt_e"))
        ingest.ingest_directory(store, evidence_dir)
        for name in q.EXPORTS:
            assert isinstance(q.run_export(store, name), list)

    def test_unknown_export_names_the_valid_ones(self, store):
        with pytest.raises(KeyError, match="calls"):
            q.run_export(store, "nonsense")


# -- sample data ----------------------------------------------------------


class TestSampleData:
    def test_samples_are_flagged(self, store, evidence_dir):
        ingest.refresh(store, directory=evidence_dir, with_samples=True)
        assert store.scalar("SELECT COUNT(*) FROM calls WHERE is_sample = 1") > 0
        assert store.scalar("SELECT COUNT(*) FROM calls WHERE is_sample = 0") == 0

    def test_real_only_revenue_excludes_samples(self, store, evidence_dir):
        real = _receipt("receipt_real", qty=10, total=42.0)
        real["claims"][0]["evidence"].append(_sms("receipt_real"))
        _write(evidence_dir, real)
        ingest.refresh(store, directory=evidence_dir, with_samples=True)

        split = q.revenue_split(store)
        assert split["real_only"]["proven"] == 42.0
        assert split["all"]["proven"] > 42.0
        assert split["sample_orders"] > 0

    def test_samples_can_be_switched_off(self, store, evidence_dir):
        ingest.refresh(store, directory=evidence_dir, with_samples=False)
        assert store.counts()["calls"] == 0

    def test_reseeding_does_not_duplicate(self, store, evidence_dir):
        ingest.refresh(store, directory=evidence_dir, with_samples=True)
        first = store.counts()
        ingest.refresh(store, directory=evidence_dir, with_samples=True)
        assert store.counts() == first

    def test_sample_verdicts_are_derived_not_declared(self):
        """No sample receipt carries a verdict key - ingest computes them."""
        for r in seed.sample_receipts(len(seed.SCENARIOS)):
            for c in r["claims"]:
                assert "verdict" not in c

    def test_sample_set_includes_a_contradiction(self, store, evidence_dir):
        ingest.refresh(store, directory=evidence_dir, with_samples=True)
        assert store.scalar("SELECT COUNT(*) FROM claims WHERE verdict='CONTRADICTED'") >= 1


# -- export ---------------------------------------------------------------


class TestExport:
    def test_csv_round_trips(self, store, evidence_dir, tmp_path):
        r = _receipt("receipt_csv", qty=25, total=99.5)
        r["claims"][0]["evidence"].append(_sms("receipt_csv"))
        _write(evidence_dir, r)
        ingest.ingest_directory(store, evidence_dir)

        rows = q.run_export(store, "orders")
        path = ex.write_csv(rows, tmp_path / "orders.csv")
        back = ex.read_csv(path)

        assert len(back) == len(rows) == 1
        assert set(back[0]) == set(rows[0])
        assert back[0]["id"] == rows[0]["id"]
        assert float(back[0]["total"]) == rows[0]["total"]
        assert int(back[0]["qty"]) == rows[0]["qty"]
        assert back[0]["verdict"] == "VERIFIED"

    def test_csv_handles_ragged_rows(self, tmp_path):
        rows = [{"a": 1}, {"b": 2, "c": None}]
        back = ex.read_csv(ex.write_csv(rows, tmp_path / "ragged.csv"))
        assert set(back[0]) == {"a", "b", "c"}
        assert back[1]["b"] == "2"

    def test_empty_export_round_trips_to_empty(self, tmp_path):
        assert ex.read_csv(ex.write_csv([], tmp_path / "empty.csv")) == []

    def test_export_writes_csv_even_when_sheets_unavailable(
        self, store, evidence_dir, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("GOOGLE_SHEETS_TOKEN", raising=False)
        _write(evidence_dir, _receipt("receipt_x"))
        ingest.ingest_directory(store, evidence_dir)

        result = ex.export(store, "orders", directory=tmp_path, to_sheets=True)
        assert result.csv_path.exists()
        assert result.rows == 1
        assert result.sheets.ok is False
        assert result.sheets.skipped is True
        assert "Sheets skipped" in result.summary

    def test_sheets_failure_never_raises(self, store, tmp_path, monkeypatch):
        monkeypatch.setenv("GOOGLE_SHEETS_TOKEN", "not-a-real-token")

        class Boom:
            def __init__(self, *a, **k) -> None:
                raise RuntimeError("network is on fire")

        import httpx

        monkeypatch.setattr(httpx, "Client", Boom)
        result = ex.export(store, "calls", directory=tmp_path, to_sheets=True)
        assert result.csv_path.exists()
        assert result.sheets.ok is False
        assert "network is on fire" in result.sheets.reason

    def test_all_exports_produce_files(self, store, evidence_dir, tmp_path):
        _write(evidence_dir, _receipt("receipt_all"))
        ingest.ingest_directory(store, evidence_dir)
        for res in ex.export_all(store, directory=tmp_path, to_sheets=False):
            assert res.csv_path.exists()


# -- the invariant --------------------------------------------------------


class TestVerdictIsImmutable:
    """Not even raw SQL against the file may promote a claim."""

    def _one_claim(self, store) -> str:
        store.record_call(CallRow(id="call_1"))
        store.record_claim(
            claim_id="claim_1",
            call_id="call_1",
            description="100 muffins for 250.00, delivered Friday",
            evidence=[EvidenceRow(channel="agent_assertion", summary="they said yes")],
        )
        return "claim_1"

    def test_born_unverified(self, store):
        self._one_claim(store)
        assert store.scalar("SELECT verdict FROM claims WHERE id='claim_1'") == "UNVERIFIED"

    def test_direct_update_is_aborted(self, store):
        self._one_claim(store)
        with pytest.raises(sqlite3.IntegrityError, match="derived from evidence"):
            store._conn.execute("UPDATE claims SET verdict = 'VERIFIED' WHERE id = 'claim_1'")
        assert store.scalar("SELECT verdict FROM claims WHERE id='claim_1'") == "UNVERIFIED"

    def test_blanket_update_touching_verdict_is_aborted(self, store):
        self._one_claim(store)
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "UPDATE claims SET description = 'x', verdict = 'VERIFIED'"
            )

    def test_update_survives_a_reopened_file(self, tmp_path):
        """The trigger lives in the file, so it protects the file."""
        path = tmp_path / "reopened.db"
        with Store(path) as st:
            st.record_call(CallRow(id="c"))
            st.record_claim(claim_id="cl", call_id="c", description="d")

        raw = sqlite3.connect(path)
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute("UPDATE claims SET verdict = 'VERIFIED'")
        raw.close()

    def test_rewriting_a_claim_rederives_rather_than_sets(self, store):
        self._one_claim(store)
        verdict = store.record_claim(
            claim_id="claim_1",
            call_id="call_1",
            description="100 muffins for 250.00, delivered Friday",
            evidence=[
                EvidenceRow(channel="agent_assertion"),
                EvidenceRow(channel="inbound_sms", summary="Confirmed"),
            ],
        )
        assert verdict is Verdict.VERIFIED
        assert store.scalar("SELECT verdict FROM claims WHERE id='claim_1'") == "VERIFIED"
        assert store.counts()["claims"] == 1

    def test_removing_evidence_demotes(self, store):
        store.record_call(CallRow(id="call_1"))
        store.record_claim(
            claim_id="claim_1",
            call_id="call_1",
            description="d",
            evidence=[EvidenceRow(channel="inbound_sms")],
        )
        assert store.scalar("SELECT verdict FROM claims") == "VERIFIED"
        store.record_claim(claim_id="claim_1", call_id="call_1", description="d", evidence=[])
        assert store.scalar("SELECT verdict FROM claims") == "UNVERIFIED"
        assert store.counts()["evidence"] == 0


class TestNoDirectVerdictPath:
    """There is no API that accepts a verdict. This is checked, not asserted."""

    def test_no_public_callable_takes_a_verdict_argument(self):
        offenders: list[str] = []
        for module in (store_mod, ingest, q, ex, seed):
            for name, obj in vars(module).items():
                if name.startswith("_"):
                    continue
                targets = []
                if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    targets.append((f"{module.__name__}.{name}", obj))
                elif inspect.isclass(obj) and obj.__module__ == module.__name__:
                    targets += [
                        (f"{module.__name__}.{name}.{m}", fn)
                        for m, fn in vars(obj).items()
                        if inspect.isfunction(fn) and not m.startswith("_")
                    ]
                for label, fn in targets:
                    params = inspect.signature(fn).parameters
                    if any("verdict" in p.lower() for p in params):
                        offenders.append(f"{label}({', '.join(params)})")
        assert not offenders, (
            "these accept a verdict from the caller; a verdict must be derived "
            f"from evidence: {offenders}"
        )

    def test_no_sql_in_the_package_assigns_a_verdict(self):
        """Grep is a blunt instrument, and that is the point: the only place
        `verdict` may be written is the single INSERT in `record_claim`."""
        import re
        from pathlib import Path as P

        pkg = P(store_mod.__file__).parent
        bad: list[str] = []
        for path in pkg.glob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"(?i)\bSET\b[^\n]*\bverdict\s*=", line):
                    bad.append(f"{path.name}:{i}: {line.strip()}")
        assert not bad, f"UPDATE ... SET verdict found: {bad}"

    def test_record_claim_returns_but_does_not_accept(self):
        sig = inspect.signature(store_mod.Store.record_claim)
        assert "verdict" not in sig.parameters
        assert sig.return_annotation in (Verdict, "Verdict")


# -- the app --------------------------------------------------------------


class TestApp:
    @pytest.fixture
    def client(self, store, evidence_dir, monkeypatch):
        monkeypatch.setenv("RESTAURANT_INGEST_ON_BOOT", "0")
        from fastapi.testclient import TestClient

        from src.verticals.restaurant import app as app_mod

        r = _receipt("receipt_ui", qty=40, total=160.0)
        r["claims"][0]["evidence"].append(_sms("receipt_ui"))
        _write(evidence_dir, r)
        _write(evidence_dir, _receipt("receipt_ui2", qty=10, total=60.0))
        ingest.ingest_directory(store, evidence_dir)

        previous = app_mod.store()
        app_mod.use_store(store)
        yield TestClient(app_mod.app)
        app_mod.use_store(previous)

    def test_index_shows_booked_and_proven(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "booked vs revenue proven" in r.text.lower()
        assert "$220.00" in r.text  # booked
        assert "$160.00" in r.text  # proven

    def test_call_page_shows_the_evidence_chain(self, client):
        r = client.get("/call/receipt_ui")
        assert r.status_code == 200
        assert "inbound_sms" in r.text
        assert "deadbeefdeadbeef" in r.text

    def test_unknown_call_is_not_a_500(self, client):
        assert client.get("/call/nope").status_code == 200

    def test_csv_endpoint_serves_a_download(self, client):
        r = client.get("/export/orders.csv")
        assert r.status_code == 200
        assert "attachment" in r.headers["content-disposition"]
        assert r.text.splitlines()[0].startswith("id,")

    def test_unknown_csv_is_404(self, client):
        assert client.get("/export/wat.csv").status_code == 404

    def test_sheets_endpoint_reports_skip_rather_than_failing(self, client, monkeypatch):
        monkeypatch.delenv("GOOGLE_SHEETS_TOKEN", raising=False)
        r = client.post("/export/orders/sheets")
        assert r.status_code == 200
        body = r.json()
        assert body["sheets"]["ok"] is False
        assert body["rows"] == 2

    def test_api_revenue_exposes_both_views(self, client):
        body = client.get("/api/revenue").json()
        assert body["all"]["booked"] == 220.0
        assert body["all"]["proven"] == 160.0

    def test_healthz(self, client):
        assert client.get("/healthz").json()["ok"] is True
