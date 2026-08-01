"""What the intake conversation is trying to fill in, and what it becomes.

A `TaskSpec` is the operator's job description in structured form. It is filled
by a chat, so every field has to survive being extracted by a language model
from a sentence a busy person typed one-handed. Three rules follow from that:

1. **Nothing is invented.** A field the operator never mentioned stays `None`
   and shows up in `missing_fields()`. The most expensive failure available
   here is a plausible number nobody said - a fabricated cost floor produces a
   fabricated price on a live call, which is the same class of error as a
   fabricated receipt.

2. **A booking errand is not a sale.** "Call three dentists and book the
   earliest cleaning" has no materials cost and no margin. `kind` splits the
   required-field set so the interview does not demand unit economics that do
   not exist, and `to_cost_model()` returns `None` rather than zeros dressed up
   as data.

3. **Units are not headcount.** `units_basis` is a required field with its own
   question because a headcount and an item count are both just integers by the
   time they reach a tool. See `src/agents/flow.py` - the $74 call. Three
   dentists is not three appointments either, so the field applies to errands
   too.

The spec is not the runtime. `to_campaign()`, `to_envelope()`, `to_cost_model()`
and `to_call_session()` convert it into the objects the existing engine already
enforces, and every safety property stays where it was: the envelope bounds the
agent, the cost model owns the floor, the receipt owns the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.agents.sales_agent import CallSession, OperatorChannel
from src.business.campaign import (
    Campaign,
    CloseCondition,
    DiscoveryStrategy,
    Envelope,
)
from src.business.capacity import CapacityLedger
from src.business.pricing import CostModel
from src.verify.receipts import Receipt
from src.webapp.fields import (
    ERRAND,
    FIELDS,
    FIELDS_BY_ID,
    GROUPS,
    KINDS,
    SALE,
    SpecField,
)
from src.webapp.values import (
    E164,
    Economics,
    Target,
    as_float,
    as_int,
    merge_targets,
    parse_date,
    singular,
    to_close,
    to_discovery,
)

#: Re-exported so callers have one import for "the shape of a brief".
__all__ = [
    "E164",
    "ERRAND",
    "FIELDS",
    "FIELDS_BY_ID",
    "GROUPS",
    "KINDS",
    "NOMINAL_MIN_PRICE",
    "SALE",
    "Economics",
    "NoOperator",
    "SpecField",
    "Target",
    "TaskSpec",
]

#: The smallest positive floor that keeps an `Envelope` coherent when the task
#: has no unit economics at all. Mirrors `STARTUP_OUTBOUND` in
#: `src/business/campaign.py`, which uses the same trick for a free demo call.
#: Paired with `max_discount_pct=0` it reads as what it is: no pricing
#: authority, rather than a price somebody chose.
NOMINAL_MIN_PRICE = 0.01


class NoOperator(OperatorChannel):
    """There is no human to ask. Always answers None, which is read as 'no'.

    Every session this module builds sets `allow_escalation=False`, so
    `ask_operator` short-circuits before reaching a channel at all. This exists
    so that if that flag is ever flipped by accident, the fallback is a refusal
    rather than a hang or an unbounded wait.
    """

    async def ask(self, question: str, *, timeout: float = 90.0) -> str | None:
        return None



# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Everything one outbound job needs, as gathered from a conversation."""

    kind: str | None = None
    objective: str = ""
    business_name: str = ""
    vertical: str = ""

    targets: list[Target] = field(default_factory=list)

    offer: str = ""
    unit_label: str = ""
    units_basis: str = ""
    capacity_total: int | None = None

    economics: Economics | None = None
    pricing_note: str = ""
    """Why there is no cost model, when there is none. Written by the intake
    agent so the UI can say 'no pricing - booking errand' instead of showing
    an empty economics panel that looks like a bug."""

    max_discount_pct: float | None = None
    earliest_date: str = ""
    latest_date: str = ""
    max_qty: int | None = None

    close_condition: str = ""
    done_definition: str = ""
    confirm_to: str = ""

    discovery: str = DiscoveryStrategy.SEEDED_LIST.value
    """The operator hands us the list in this product. We never invent targets."""

    # -- completeness ---------------------------------------------------

    def _has(self, fid: str) -> bool:
        if fid == "kind":
            return self.kind in KINDS
        if fid == "targets":
            return any(t.usable for t in self.targets)
        if fid == "economics":
            return self.economics is not None and self.economics.is_complete
        if fid == "capacity_total":
            return bool(self.capacity_total and self.capacity_total > 0)
        if fid == "max_qty":
            return bool(self.max_qty and self.max_qty > 0)
        if fid == "max_discount_pct":
            return self.max_discount_pct is not None
        if fid == "date_window":
            return bool(parse_date(self.earliest_date) and parse_date(self.latest_date))
        if fid == "close_condition":
            return self.close_condition in {c.value for c in CloseCondition}
        return bool(str(getattr(self, fid, "") or "").strip())

    @property
    def effective_kind(self) -> str:
        """The kind to reason about before the operator has told us. Errand is
        the cheaper wrong guess: it asks for fewer fields, so a sale mislabelled
        as an errand surfaces as a missing-economics question, while an errand
        mislabelled as a sale demands costs that do not exist."""
        return self.kind if self.kind in KINDS else ERRAND

    def required_fields(self) -> list[SpecField]:
        kind = self.effective_kind
        return [f for f in FIELDS if f.applies_to(kind)]

    def missing_fields(self) -> list[str]:
        return [f.id for f in self.required_fields() if not self._has(f.id)]

    @property
    def is_complete(self) -> bool:
        return not self.missing_fields() and not self.problems()

    @property
    def dialable_targets(self) -> list[Target]:
        return [t for t in self.targets if t.dialable]

    @property
    def can_launch(self) -> bool:
        """Complete *and* actually reachable. A description is not a number."""
        return self.is_complete and bool(self.dialable_targets)

    def blockers(self) -> list[str]:
        """Human-readable reasons launch is refused. Empty means go."""
        out: list[str] = []
        for fid in self.missing_fields():
            out.append(f"still need: {FIELDS_BY_ID[fid].label.lower()}")
        out.extend(self.problems())
        if not self.missing_fields() and not self.dialable_targets:
            out.append(
                "no target has a dialable +E.164 number - I can describe who to "
                "find but I cannot dial a description"
            )
        return out

    def problems(self) -> list[str]:
        """Incoherence that a complete-looking spec can still contain."""
        out: list[str] = []
        early, late = parse_date(self.earliest_date), parse_date(self.latest_date)
        if early and late and late < early:
            out.append(f"date window is inverted: {early} .. {late}")
        if (
            self.capacity_total is not None
            and self.max_qty is not None
            and self.max_qty > self.capacity_total
        ):
            out.append(
                f"max quantity {self.max_qty} exceeds the {self.capacity_total} "
                f"{self.unit_label or 'units'} you said you can supply"
            )
        if self.economics is not None and self.economics.is_complete:
            tgt = self.economics.target_margin_pct
            mn = self.economics.min_margin_pct
            if tgt is not None and mn is not None and tgt < mn:
                out.append("target margin is below the minimum margin")
            elif self.to_cost_model() is None:
                out.append(
                    "unit economics do not make a usable cost model - margins "
                    "must be within 0-100% and costs cannot be negative"
                )
        return out

    # -- conversion -----------------------------------------------------

    def to_envelope(self) -> Envelope:
        """The standing authorisation. Derived, never guessed.

        `min_price` comes from the cost model's own per-unit floor when there is
        one. When there is not, it is `NOMINAL_MIN_PRICE` with
        `max_discount_pct` pinned to 0, which is the campaign-level way of
        saying the agent has no pricing authority at all.
        """
        costs = self.to_cost_model()
        if costs is not None:
            floor = float(costs.floor_price(1))
            min_price = floor if floor > 0 else NOMINAL_MIN_PRICE
            discount = float(self.max_discount_pct or 0.0)
        else:
            min_price = NOMINAL_MIN_PRICE
            discount = 0.0

        early = parse_date(self.earliest_date) or date.today()
        late = parse_date(self.latest_date) or early
        return Envelope(
            min_price=min_price,
            max_qty=int(self.max_qty or self.capacity_total or 1),
            earliest_date=early,
            latest_date=late,
            max_discount_pct=discount,
        )

    def to_cost_model(self) -> CostModel | None:
        """The cost model, or None when the task has no unit economics.

        Returning None is the whole point. A booking errand that got handed a
        `CostModel(0, 0, 0)` would look identical, in every log and every
        receipt, to a sale whose costs the agent made up.
        """
        e = self.economics
        if e is None or not e.is_complete:
            return None
        try:
            return CostModel(
                materials_per_unit=str(e.materials_per_unit),
                labor_per_unit=str(e.labor_per_unit),
                transport_per_unit=str(e.transport_per_unit),
                min_margin_pct=str(e.min_margin_pct),
                target_margin_pct=(
                    None if e.target_margin_pct is None else str(e.target_margin_pct)
                ),
                unit=singular(self.unit_label or "unit"),
            )
        except ValueError:
            # Incoherent numbers (a target margin under the floor, a margin at
            # or over 100%). `problems()` reports these in words; returning None
            # here keeps `to_dict()` renderable so the UI can show the operator
            # what it read, instead of 500ing on their typo.
            return None

    def to_campaign(self) -> Campaign:
        who = "; ".join(
            filter(
                None,
                (
                    ", ".join(t.name for t in self.targets if t.name),
                    "; ".join(t.find_hint for t in self.targets if t.find_hint),
                ),
            )
        )
        limits = [
            f"max quantity {self.max_qty}",
            f"window {self.earliest_date} .. {self.latest_date}",
        ]
        if self.max_discount_pct is not None:
            limits.append(f"max discount {self.max_discount_pct}%")
        if self.economics is None:
            limits.append("no pricing authority: " + (self.pricing_note or "not a sale"))
        return Campaign(
            name=(self.objective or "Outbound task")[:120],
            vertical=self.vertical or self.effective_kind,
            icp=who or "the operator's own list",
            offer=self.offer,
            discovery=to_discovery(self.discovery),
            close_condition=to_close(self.close_condition),
            capacity_units=self.unit_label or "units",
            envelope=self.to_envelope(),
            notes=(
                f"Definition of done: {self.done_definition or self.close_condition}. "
                f"Confirmation must reach {self.confirm_to}. "
                f"Units basis: {self.units_basis}. "
                f"Hard limits: {'; '.join(limits)}. "
                "There is no operator to escalate to on this call."
            ),
            tags=tuple(t for t in (self.effective_kind, self.vertical) if t),
        )

    def to_call_session(self) -> CallSession:
        """One call's worth of runtime, with escalation off.

        `allow_escalation=False` is passed explicitly rather than relying on the
        default: the limits gathered in the interview are the whole mandate, and
        a reader of this line should not have to go and check what the default
        is to know that.
        """
        costs = self.to_cost_model() or _no_pricing_model(self.unit_label)
        return CallSession(
            campaign=self.to_campaign(),
            ledger=CapacityLedger(
                int(self.capacity_total or self.max_qty or 1),
                self.unit_label or "units",
            ),
            costs=costs,
            receipt=Receipt(task=self.objective or "outbound task"),
            operator=NoOperator(),
            allow_escalation=False,
        )

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        missing = self.missing_fields()
        return {
            "kind": self.kind,
            "objective": self.objective,
            "business_name": self.business_name,
            "vertical": self.vertical,
            "targets": [t.to_dict() for t in self.targets],
            "offer": self.offer,
            "unit_label": self.unit_label,
            "units_basis": self.units_basis,
            "capacity_total": self.capacity_total,
            "economics": None if self.economics is None else self.economics.to_dict(),
            "pricing_note": self.pricing_note,
            "has_pricing": self.to_cost_model() is not None,
            "max_discount_pct": self.max_discount_pct,
            "earliest_date": self.earliest_date,
            "latest_date": self.latest_date,
            "max_qty": self.max_qty,
            "close_condition": self.close_condition,
            "done_definition": self.done_definition,
            "confirm_to": self.confirm_to,
            "discovery": self.discovery,
            "missing": missing,
            "required": [f.id for f in self.required_fields()],
            "complete": self.is_complete,
            "can_launch": self.can_launch,
            "blockers": self.blockers(),
        }

    # -- mutation -------------------------------------------------------

    def apply(self, patch: dict[str, Any]) -> list[str]:
        """Merge a patch from the intake agent. Returns the ids that changed.

        Deliberately forgiving about junk and deliberately strict about types:
        a model that answers with `"capacity_total": "about 400"` must not end
        up quietly setting capacity to zero, so an uncoercible value is dropped
        and the field stays missing - which puts the question back in the
        interview instead of into a live call.
        """
        changed: list[str] = []
        for key, value in (patch or {}).items():
            if value is None or value == "" or value == []:
                continue
            before = self.to_dict().get(key)
            try:
                if not self._set(key, value):
                    continue
            except (TypeError, ValueError):
                continue
            if self.to_dict().get(key) != before:
                changed.append(key)
        return changed

    def _set(self, key: str, value: Any) -> bool:
        if key == "kind":
            v = str(value).strip().lower()
            if v not in KINDS:
                return False
            self.kind = v
        elif key in ("objective", "business_name", "vertical", "offer",
                     "unit_label", "units_basis", "done_definition",
                     "confirm_to", "pricing_note"):
            setattr(self, key, str(value).strip())
        elif key == "targets":
            self.targets = merge_targets(self.targets, value)
        elif key == "capacity_total":
            self.capacity_total = as_int(value)
        elif key == "max_qty":
            self.max_qty = as_int(value)
        elif key == "max_discount_pct":
            self.max_discount_pct = as_float(value)
        elif key in ("earliest_date", "latest_date"):
            d = parse_date(str(value))
            if d is None:
                return False
            setattr(self, key, d.isoformat())
        elif key == "date_window":
            ok = False
            if isinstance(value, dict):
                ok |= self._set("earliest_date", value.get("earliest_date", ""))
                ok |= self._set("latest_date", value.get("latest_date", ""))
            return ok
        elif key == "close_condition":
            v = str(value).strip().lower()
            if v not in {c.value for c in CloseCondition}:
                return False
            self.close_condition = v
        elif key == "discovery":
            v = str(value).strip().lower()
            if v not in {d.value for d in DiscoveryStrategy}:
                return False
            self.discovery = v
        elif key == "economics":
            if not isinstance(value, dict):
                return False
            e = self.economics or Economics()
            for k in (
                "materials_per_unit",
                "labor_per_unit",
                "transport_per_unit",
                "min_margin_pct",
                "target_margin_pct",
            ):
                if k in value and value[k] is not None:
                    f = as_float(value[k])
                    if f is not None:
                        setattr(e, k, f)
            self.economics = e
        else:
            return False
        return True




def _no_pricing_model(unit_label: str) -> CostModel:
    """An explicitly empty cost model for a task with no unit economics.

    Not a guess and not a default: every number is zero, so `floor_price` is
    zero and the pricing ladder has nothing to say. Only reached from
    `to_call_session()`, because `CallSession` requires a cost model; anything
    that wants to *ask* whether the task has economics must call
    `to_cost_model()`, which answers None.
    """
    return CostModel(
        materials_per_unit="0",
        labor_per_unit="0",
        transport_per_unit="0",
        min_margin_pct="0",
        unit=singular(unit_label or "unit"),
    )
