"""Loading `config/restaurant.json`, and the cost model it describes.

The cost model matters more than it looks. A stored order total is a fact; the
*discount* on it is only meaningful relative to a list price, and if that list
price is invented per-query the discount statistics drift. So it is derived
once, here, from the same `CostModel` the sales agent negotiated inside.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.business.pricing import CostModel

#: Repo root, resolved from this file so nothing depends on the cwd.
ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG_PATH = ROOT / "config" / "restaurant.json"

FALLBACK: dict[str, Any] = {
    "business": {
        "name": "Golden Crumb Bakery",
        "phone": "",
        "unit": "muffins",
        "unit_singular": "muffin",
        "currency": "USD",
    },
    "weekly_capacity": 600,
    "cost_model": {
        "materials_per_unit": "0.62",
        "labor_per_unit": "0.72",
        "transport_per_unit": "0.11",
        "min_margin_pct": "25",
        "target_margin_pct": "40",
        "unit": "muffin",
        "currency": "USD",
    },
    "database": "evidence/restaurant.db",
    "evidence_dir": "evidence",
    "exports_dir": "evidence/exports",
    "sample_data": {"enabled": True, "min_calls": 8},
    "sheets": {"spreadsheet_id": "", "credentials_path": "", "token_env": "GOOGLE_SHEETS_TOKEN"},
}


@dataclass(frozen=True)
class RestaurantConfig:
    """Everything the vertical needs, with a usable default for each field."""

    raw: dict[str, Any]
    source: Path | None = None

    # -- business ---------------------------------------------------------

    @property
    def business_name(self) -> str:
        return str(self.raw.get("business", {}).get("name", "Restaurant"))

    @property
    def unit(self) -> str:
        return str(self.raw.get("business", {}).get("unit", "units"))

    @property
    def currency(self) -> str:
        return str(self.raw.get("business", {}).get("currency", "USD"))

    @property
    def weekly_capacity(self) -> int:
        return int(self.raw.get("weekly_capacity", 0))

    # -- paths ------------------------------------------------------------

    def _path(self, key: str, default: str) -> Path:
        value = str(self.raw.get(key, default))
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    @property
    def database_path(self) -> Path:
        return self._path("database", "evidence/restaurant.db")

    @property
    def evidence_dir(self) -> Path:
        return self._path("evidence_dir", "evidence")

    @property
    def exports_dir(self) -> Path:
        return self._path("exports_dir", "evidence/exports")

    # -- pricing ----------------------------------------------------------

    @property
    def cost_model(self) -> CostModel:
        cm = dict(FALLBACK["cost_model"])
        cm.update(self.raw.get("cost_model", {}))
        return CostModel(
            materials_per_unit=cm["materials_per_unit"],
            labor_per_unit=cm["labor_per_unit"],
            transport_per_unit=cm["transport_per_unit"],
            min_margin_pct=cm["min_margin_pct"],
            target_margin_pct=cm.get("target_margin_pct"),
            unit=cm.get("unit", "unit"),
            currency=cm.get("currency", "USD"),
        )

    # -- sample data ------------------------------------------------------

    @property
    def sample_enabled(self) -> bool:
        return bool(self.raw.get("sample_data", {}).get("enabled", True))

    @property
    def sample_min_orders(self) -> int:
        """Below this many *real* orders, load the sample set as well."""
        block = self.raw.get("sample_data", {})
        return int(block.get("min_orders", block.get("min_calls", 8)))

    # -- sheets -----------------------------------------------------------

    @property
    def sheets(self) -> dict[str, Any]:
        out = dict(FALLBACK["sheets"])
        out.update(self.raw.get("sheets", {}))
        return out

    def sheets_credentials_path(self) -> Path | None:
        raw = str(self.sheets.get("credentials_path") or "").strip()
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p


def load(path: str | Path | None = None) -> RestaurantConfig:
    """Read the config, falling back to defaults rather than failing.

    A missing config file must not take the dashboard down mid-demo. Every key
    has a working default; the file only ever overrides.
    """
    p = Path(path) if path else Path(os.environ.get("RESTAURANT_CONFIG", DEFAULT_CONFIG_PATH))
    if not p.is_absolute():
        p = ROOT / p
    merged: dict[str, Any] = json.loads(json.dumps(FALLBACK))
    if p.exists():
        try:
            loaded = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return RestaurantConfig(raw=merged, source=None)
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
        return RestaurantConfig(raw=merged, source=p)
    return RestaurantConfig(raw=merged, source=None)


@lru_cache(maxsize=1)
def default() -> RestaurantConfig:
    """Process-wide config. Cached so the UI does not re-read it per request."""
    return load()
