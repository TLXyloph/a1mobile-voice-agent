"""The typed values a brief is made of, and the coercion that guards them.

Everything here exists at the boundary between a language model's JSON and a
real cost floor, so the bias is uniform: an uncoercible value is dropped, not
guessed. `as_int("about four hundred")` returning 400 would be convenient and
would eventually put a number nobody said into a live negotiation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from src.business.campaign import CloseCondition, DiscoveryStrategy

#: E.164. Anything else cannot be handed to a SIP trunk, so it is not a number.
E164 = re.compile(r"^\+[1-9]\d{6,14}$")


@dataclass
class Target:
    """Someone to call. A description is allowed; a description cannot be dialled."""

    name: str = ""
    phone: str = ""
    find_hint: str = ""
    notes: str = ""

    @property
    def dialable(self) -> bool:
        return bool(E164.match(self.phone or ""))

    @property
    def usable(self) -> bool:
        return bool(self.name.strip()) and bool(self.phone or self.find_hint)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "dialable": self.dialable}


@dataclass
class Economics:
    """Per-unit cost and the two margin lines. Absent fields stay None.

    `is_complete` requires the three costs and the hard floor. The target
    margin is optional because `CostModel` already defaults it to the floor.
    """

    materials_per_unit: float | None = None
    labor_per_unit: float | None = None
    transport_per_unit: float | None = None
    min_margin_pct: float | None = None
    target_margin_pct: float | None = None

    @property
    def is_complete(self) -> bool:
        return all(
            v is not None
            for v in (
                self.materials_per_unit,
                self.labor_per_unit,
                self.transport_per_unit,
                self.min_margin_pct,
            )
        )

    @property
    def missing(self) -> list[str]:
        return [
            k
            for k in (
                "materials_per_unit",
                "labor_per_unit",
                "transport_per_unit",
                "min_margin_pct",
            )
            if getattr(self, k) is None
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_targets(existing: list[Target], value: Any) -> list[Target]:
    """Merge by name, so a later turn can add a phone to an earlier name."""
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise TypeError("targets must be a list")
    out = list(existing)
    index = {t.name.strip().lower(): t for t in out if t.name}
    for raw in value:
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        phone = clean_phone(str(raw.get("phone", "")))
        hit = index.get(name.lower()) if name else None
        if hit is None:
            hit = Target(name=name)
            out.append(hit)
            if name:
                index[name.lower()] = hit
        if phone:
            hit.phone = phone
        for k in ("find_hint", "notes"):
            if raw.get(k):
                setattr(hit, k, str(raw[k]).strip())
    return [t for t in out if t.name or t.phone or t.find_hint]


def clean_phone(raw: str) -> str:
    """Strip formatting; add +1 only for a bare 10-digit North American number.

    Anything else keeps whatever it had, which will fail `E164` and show as
    not-dialable rather than being silently mangled into the wrong country.
    """
    s = re.sub(r"[^\d+]", "", raw or "")
    if not s:
        return ""
    if s.startswith("+"):
        return s
    if len(s) == 10:
        return "+1" + s
    if len(s) == 11 and s.startswith("1"):
        return "+" + s
    return s


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    m = re.search(r"-?\d[\d,]*", str(value))
    return int(m.group().replace(",", "")) if m else None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d*\.?\d+", str(value).replace(",", ""))
    return float(m.group()) if m else None


def parse_date(raw: str) -> date | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def singular(label: str) -> str:
    label = label.strip()
    if len(label) > 3 and label.endswith("ies"):
        return label[:-3] + "y"
    if len(label) > 2 and label.endswith("s") and not label.endswith("ss"):
        return label[:-1]
    return label or "unit"


def to_discovery(raw: str) -> DiscoveryStrategy:
    try:
        return DiscoveryStrategy(raw)
    except ValueError:
        return DiscoveryStrategy.SEEDED_LIST


def to_close(raw: str) -> CloseCondition:
    try:
        return CloseCondition(raw)
    except ValueError:
        # The pessimistic default: written confirmation is the hardest of the
        # three to satisfy by talking, so guessing it never makes a call easier
        # to declare closed than the operator intended.
        return CloseCondition.WRITTEN_CONFIRMATION
