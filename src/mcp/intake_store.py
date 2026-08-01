"""Where the profile is written, and the rules about not destroying anything.

Split out of `intake_server.py` because the .env writer is the riskiest code in
this package and deserves to be readable on its own screen. It reaches into a
file that holds every credential the project has, to change eight unrelated
lines.

Three properties, in order of how bad the alternative is:

**A credential is never a write target and never a rewrite target.** Keys
matching `PROTECTED_KEY` are refused outright at the top of `write_env`, and any
existing line whose key matches is copied through verbatim even if something
upstream asks for it. Two independent guards, because one of them being subtly
wrong is a leaked or destroyed API key.

**Unrelated lines are copied, not re-serialised.** Parsing the file to a dict
and dumping it back would work and would also silently delete every comment
explaining what LIVEKIT_SIP_TRUNK_ID is, and reorder a file people read by eye.
Lines are matched, replaced or kept, in place, including the `export ` prefix.

**Writes are atomic.** Temp file plus rename, so a crash halfway cannot leave
the project with half a .env and no way to make a call.

Answers round-trip through the same coercers a spoken answer meets, so a
hand-edited `business_profile.json` fails at load rather than mid-call.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.mcp.intake_profile import BY_FIELD, transport_per_unit

#: Any env key matching this is a credential. It is never written and never
#: rewritten, whatever the rest of this package thinks it is doing. The list is
#: wider than the four names the profile could collide with, deliberately: the
#: cost of an extra pattern is nothing, and the cost of a missed one is a
#: rewritten API key.
PROTECTED_KEY = re.compile(
    r"API_KEY|SECRET|PASSWORD|TEAM_KEY|TOKEN|PRIVATE_KEY|CREDENTIAL", re.IGNORECASE
)

#: Captures the `export ` prefix so rewriting a value does not silently change
#: whether the line is exported.
_ENV_LINE = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)\s*=")

_SECTION = "# --- Business profile (written by src/mcp/intake_server.py) ------------"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def serialise_answers(answers: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in answers.items():
        if isinstance(value, (tuple, list)):
            out[name] = ",".join(str(v) for v in value)
        elif isinstance(value, date):
            out[name] = value.isoformat()
        elif isinstance(value, Decimal):
            out[name] = str(value)
        else:
            out[name] = value
    return out


def deserialise_answers(raw: dict[str, Any]) -> dict[str, Any]:
    """Re-coerce a stored profile. Raises `Refusal` on a value we would not accept."""
    out: dict[str, Any] = {}
    for name, value in raw.items():
        question = BY_FIELD.get(name)
        out[name] = question.coerce(value) if question else value
    return out


# ---------------------------------------------------------------------------
# config/.env
# ---------------------------------------------------------------------------


def env_updates(answers: dict[str, Any], *, business_name: str, vertical: str) -> dict[str, str]:
    """The env lines this profile implies, in the names run_call.py already reads.

    The first eight are live today - `src/agents/run_call.py` and
    `src/agents/vapi_bridge.py` read them directly. The rest are written so the
    profile is complete and inspectable; nothing consumes them yet.
    """
    unit = str(answers["unit"])
    return {
        "BUSINESS_NAME": business_name,
        "BUSINESS_VERTICAL": vertical,
        # One key feeds two consumers: CapacityLedger's label and CostModel.unit.
        # The singular is written because CostModel's sentences are read aloud.
        "CAPACITY_UNIT": unit,
        "CAPACITY_TOTAL": str(int(answers["capacity_total"])),
        "CAPACITY_PERIOD": str(answers["capacity_period"]),
        "ITEMS_PER_PERSON": str(int(answers["items_per_person"])),
        "COST_MATERIALS": str(answers["materials_per_unit"]),
        "COST_LABOR": str(answers["labor_per_unit"]),
        "COST_TRANSPORT": str(transport_per_unit(answers)),
        "MIN_MARGIN_PCT": str(answers["min_margin_pct"]),
        "TARGET_MARGIN_PCT": str(answers["target_margin_pct"]),
        "MAX_DISCOUNT_PCT": str(answers["max_discount_pct"]),
        "EARLIEST_DATE": answers["earliest_date"].isoformat(),
        "LATEST_DATE": answers["latest_date"].isoformat(),
        "BLACKOUT_DAYS": ",".join(answers["blackout_days"]),
        "APPROVAL_MODE": str(answers["approval_mode"]),
    }


def _quote(value: str) -> str:
    """Quote only when a bare value would not survive a dotenv reader."""
    if value == "" or re.search(r"[\s#'\"]", value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def write_env(updates: dict[str, str], path: Path) -> dict[str, list[str]]:
    """Update keys in place. Everything else in the file is copied byte for byte."""
    refused = [key for key in updates if PROTECTED_KEY.search(key)]
    if refused:
        # Unreachable from the tools; this is the guard that keeps it unreachable.
        raise RuntimeError(f"refusing to write credential-shaped keys: {sorted(refused)}")

    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = dict(updates)
    updated: list[str] = []
    out: list[str] = []

    for line in existing:
        match = _ENV_LINE.match(line)
        key = match.group(2) if match else None
        if key is None or key not in pending or PROTECTED_KEY.search(key):
            out.append(line)  # comment, blank, unrelated key, or a credential
            continue
        out.append(f"{match.group(1)}{key}={_quote(pending.pop(key))}")
        updated.append(key)

    added = sorted(pending)
    if added:
        if out and out[-1].strip():
            out.append("")
        out.append(_SECTION)
        out.extend(f"{key}={_quote(pending[key])}" for key in added)

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, "\n".join(out) + "\n")
    return {"updated_in_place": updated, "appended": added}


def atomic_write(path: Path, text: str) -> None:
    """Temp file plus rename, so a crash mid-write cannot leave half a .env."""
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
