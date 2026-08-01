"""MCP server that interviews a business owner and writes the config we already run on.

The gap this closes: `src/agents/run_call.py` reads COST_MATERIALS, CAPACITY_TOTAL,
MIN_MARGIN_PCT and five more out of the environment. The bakery owner who knows
those eight numbers is the one person who will never edit a dotfile to supply
them, so the agent goes out negotiating on behalf of a business that is really
just whatever the demo was seeded with.

Six tools, spoken-language in, real objects out:

    start_intake -> answer -> answer -> ... -> save_profile
                      ^                              |
                      +-- parse_document ------------+-- load_profile

Three properties are deliberate.

**A bad answer returns an instruction, never an exception.** Exactly the shape
of the Gate in `src/agents/flow.py`: a refusal is the next thing to say, not a
tool failure, because the caller here is a voice model mid-conversation.

**Nothing is written until it builds.** `save_profile` runs `to_config()` first,
so a profile that would blow up in the middle of a live call blows up at intake
instead, where the owner is still on the line to fix it.

**The .env writer cannot touch a credential.** Keys are rewritten in place, one
line at a time, and any line whose key contains API_KEY, SECRET, PASSWORD or
TEAM_KEY is copied through verbatim and is refused as a write target outright.
`tests/test_intake.py` pins that with a fixture full of fake credentials.

Transport is stdio, which is what VoiceOS asks for. Their guideline is a single
file run as `python3 /path/to/server.py`, with `mcp.run(transport="stdio")` at
the bottom - both of which this file satisfies, and the sys.path bootstrap below
is what makes the first one true for a module that lives inside a package.

Their published template imports `FastMCP` from `mcp.server.fastmcp`, which is
the MCP SDK 1.x spelling. SDK 2.x renamed that class to `MCPServer` and dropped
the old module, so the import below tries both rather than pinning either: the
same file runs whether VoiceOS ships 1.x or 2.x. Nothing else about their host
is assumed, so it is equally launchable from Claude Desktop or Claude Code -
see `config/mcp-intake.json`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# VoiceOS starts a server with `python3 <abs path to this file>`, which puts
# this file's own directory on sys.path and not the repo root - so the `src.`
# imports below would not resolve and the process would die before the
# handshake. Putting the root back makes one file work under all three of
# `python3 <path>`, `python -m src.mcp.intake_server`, and a plain import.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mcp import intake_docs  # noqa: E402
from src.mcp.intake_profile import (  # noqa: E402
    BY_FIELD,
    QUESTIONS,
    BusinessConfig,
    Refusal,
    missing_fields,
    next_question,
    required_fields,
    to_config,
)
from src.mcp.intake_store import (  # noqa: E402
    atomic_write,
    deserialise_answers,
    env_updates,
    serialise_answers,
    write_env,
)

try:  # SDK 2.x. VoiceOS's template uses the 1.x name, handled just below.
    from mcp.server import MCPServer as _Server  # noqa: E402
except ImportError:  # pragma: no cover - SDK 1.x, the spelling VoiceOS documents
    from mcp.server.fastmcp import FastMCP as _Server  # noqa: E402

REPO_ROOT = _ROOT
PROFILE_PATH = REPO_ROOT / "config" / "business_profile.json"
ENV_PATH = REPO_ROOT / "config" / ".env"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class IntakeSession:
    """One interview. A single session per server process, as MCP stdio is 1:1."""

    business_name: str
    vertical: str
    answers: dict[str, Any] = dc_field(default_factory=dict)
    sources: dict[str, str] = dc_field(default_factory=dict)
    started_at: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


_session: IntakeSession | None = None


def _need_session() -> dict[str, Any] | None:
    if _session is None:
        return {
            "ok": False,
            "instruction": (
                "No interview is open. Call start_intake with the business name and "
                "what kind of business it is before recording answers."
            ),
        }
    return None


def _ask(session: IntakeSession) -> dict[str, Any]:
    """The next question as a payload, or the completion summary."""
    question = next_question(session.answers)
    if question is not None:
        return {
            "ok": True,
            "done": False,
            "field": question.field,
            "question": question.ask(session.answers),
            "why": question.why,
            "remaining": len(missing_fields(session.answers)),
        }
    return _completion(session)


def _completion(session: IntakeSession) -> dict[str, Any]:
    """Read the derived numbers back before anything is saved.

    The floor price is the number the owner most needs to hear out loud: it is
    what their own costs and margin actually imply, and it is often not what
    they expected.
    """
    try:
        config = to_config(session.answers)
    except ValueError as exc:
        return {"ok": False, "done": False, "instruction": str(exc)}

    unit = config.costs.unit
    per_person = config.items_per_person
    return {
        "ok": True,
        "done": True,
        "summary": (
            f"Got it. {session.business_name} sells {unit}, "
            f"{per_person} per person, up to {config.ledger.total} a "
            f"{config.capacity_period}. One {unit} costs you "
            f"{config.costs.unit_cost} to make, so the lowest price that clears your "
            f"{config.costs.min_margin_pct}% floor is "
            f"{config.envelope.min_price:.2f} each, and your target is "
            f"{config.costs.target_price(1)}. A 30-person order means "
            f"{30 * per_person} {unit}, not 30."
        ),
        "derived": config.to_dict(),
        "next": "Call save_profile to write this to config/business_profile.json and config/.env.",
    }


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------

server = _Server(
    name="a1mobile-intake",
    instructions=(
        "Interview a small-business owner and turn their answers into the pricing, "
        "capacity and negotiating-envelope config the voice agent runs on. Ask one "
        "question at a time, in the order the server gives them, in the owner's own "
        "words. When a tool returns an 'instruction', say that instruction - it is "
        "the next thing to ask, not an error. Never invent a cost, a capacity or a "
        "margin: a missing number stays missing until the owner says it."
    ),
)


@server.tool()
def start_intake(business_name: str, vertical: str) -> dict[str, Any]:
    """Open an interview for one business. Returns the first question only."""
    global _session
    name = str(business_name or "").strip()
    kind = str(vertical or "").strip()
    if not name:
        return {
            "ok": False,
            "instruction": "I need the name of the business first. What is it called?",
        }
    if not kind:
        return {
            "ok": False,
            "instruction": (
                f"And what kind of business is {name} - bakery, contractor, agency?"
            ),
        }

    _session = IntakeSession(business_name=name, vertical=kind)
    first = QUESTIONS[0]
    return {
        "ok": True,
        "done": False,
        "business_name": name,
        "vertical": kind,
        "total_questions": len(required_fields({})),
        "field": first.field,
        "question": first.ask({}),
        "why": first.why,
    }


@server.tool()
def answer(field: str, value: str) -> dict[str, Any]:
    """Record one answer. Returns the next question, or the completion summary.

    A value that makes no sense comes back as an instruction to say to the owner,
    with the same question repeated - never as an exception.
    """
    if (problem := _need_session()) is not None:
        return problem
    assert _session is not None

    name = str(field or "").strip()
    question = BY_FIELD.get(name)
    if question is None:
        pending = next_question(_session.answers)
        return {
            "ok": False,
            "instruction": (
                f"I have no field called {name!r}. "
                + (
                    f"The one I am asking about is {pending.field!r}: {pending.ask(_session.answers)}"
                    if pending
                    else "Everything is answered - call save_profile."
                )
            ),
            "known_fields": list(BY_FIELD),
        }

    try:
        parsed = question.parse(value, _session.answers)
    except Refusal as refusal:
        return {
            "ok": False,
            "rejected": {"field": name, "value": value},
            "instruction": str(refusal),
            "field": name,
            "question": question.ask(_session.answers),
        }

    _session.answers[name] = parsed
    _session.sources[name] = "interview"
    result = _ask(_session)
    result["recorded"] = {"field": name, "value": serialise_answers({name: parsed})[name]}
    if not question.applies(_session.answers):
        result["note"] = (
            f"{name} was recorded but is not used for this profile - "
            "the earlier answers made it irrelevant."
        )
    return result


@server.tool()
def intake_status() -> dict[str, Any]:
    """What is known, where each value came from, and what is still missing."""
    if (problem := _need_session()) is not None:
        return problem
    assert _session is not None

    gaps = missing_fields(_session.answers)
    return {
        "ok": True,
        "business_name": _session.business_name,
        "vertical": _session.vertical,
        "started_at": _session.started_at,
        "known": serialise_answers(_session.answers),
        "sources": dict(_session.sources),
        "missing": list(gaps),
        "missing_questions": [
            {"field": f, "question": BY_FIELD[f].ask(_session.answers)} for f in gaps
        ],
        "complete": not gaps,
        "ready_to_save": not gaps,
    }


@server.tool()
def parse_document(path: str) -> dict[str, Any]:
    """Read a CSV/XLSX/TXT/MD menu or price sheet into the open interview.

    Reports what it found, what it could not find, and what it refused. Nothing
    is defaulted: a cost that is not in the file is still a question for the
    owner, because a cost defaulted to zero makes every price look profitable.
    """
    try:
        doc = intake_docs.parse_document(path, root=REPO_ROOT)
    except intake_docs.DocumentError as exc:
        return {"ok": False, "instruction": str(exc)}

    payload = doc.to_dict()
    payload["ok"] = True
    payload["applied"] = {}
    payload["rejected"] = {}

    if _session is None:
        payload["note"] = (
            "No interview is open, so nothing was applied. Call start_intake, then "
            "parse_document again to fill these in."
        )
        return payload

    for name, finding in doc.found.items():
        question = BY_FIELD.get(name)
        if question is None:
            continue
        try:
            parsed = question.parse(finding["value"], _session.answers)
        except Refusal as refusal:
            # A spreadsheet gets no more benefit of the doubt than a person does.
            payload["rejected"][name] = {
                "value": finding["value"],
                "evidence": finding["evidence"],
                "instruction": str(refusal),
            }
            continue
        _session.answers[name] = parsed
        _session.sources[name] = f"document:{Path(doc.path).name}"
        payload["applied"][name] = {
            "value": serialise_answers({name: parsed})[name],
            "evidence": finding["evidence"],
        }

    payload["still_missing"] = list(missing_fields(_session.answers))
    payload["next"] = _ask(_session)
    return payload


@server.tool()
def save_profile() -> dict[str, Any]:
    """Validate, then write config/business_profile.json and update config/.env.

    Refuses a partial profile. Refuses a profile that does not build into a real
    CostModel and Envelope. Only after both does anything touch disk.
    """
    if (problem := _need_session()) is not None:
        return problem
    assert _session is not None

    if gaps := missing_fields(_session.answers):
        pending = next_question(_session.answers)
        return {
            "ok": False,
            "saved": False,
            "missing": list(gaps),
            "instruction": (
                f"Not enough to save yet - still missing {', '.join(gaps)}. "
                + (f"Next: {pending.ask(_session.answers)}" if pending else "")
            ),
        }

    try:
        config = to_config(_session.answers)
    except ValueError as exc:
        return {"ok": False, "saved": False, "instruction": str(exc)}

    profile = {
        "business_name": _session.business_name,
        "vertical": _session.vertical,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "answers": serialise_answers(_session.answers),
        "sources": dict(_session.sources),
        "derived": config.to_dict(),
    }
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(PROFILE_PATH, json.dumps(profile, indent=2, sort_keys=True) + "\n")

    updates = env_updates(
        _session.answers,
        business_name=_session.business_name,
        vertical=_session.vertical,
    )
    written = write_env(updates, ENV_PATH)

    return {
        "ok": True,
        "saved": True,
        "profile_path": str(PROFILE_PATH),
        "env_path": str(ENV_PATH),
        "env_keys": sorted(updates),
        "env_written": written,
        "derived": config.to_dict(),
        "summary": _completion(_session).get("summary", ""),
    }


@server.tool()
def load_profile() -> dict[str, Any]:
    """Read config/business_profile.json back and re-check that it still builds."""
    if not PROFILE_PATH.exists():
        return {
            "ok": False,
            "instruction": (
                f"No saved profile at {PROFILE_PATH}. Run start_intake to make one."
            ),
        }
    try:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "instruction": f"Could not read the saved profile: {exc}"}

    raw = profile.get("answers", {})
    try:
        answers = deserialise_answers(raw)
        config: BusinessConfig | None = to_config(answers)
    except (Refusal, ValueError) as exc:
        return {
            "ok": False,
            "profile": profile,
            "instruction": (
                f"The saved profile no longer builds: {exc} Re-run the interview for "
                "the fields involved."
            ),
        }

    return {
        "ok": True,
        "business_name": profile.get("business_name"),
        "vertical": profile.get("vertical"),
        "saved_at": profile.get("saved_at"),
        "answers": serialise_answers(answers),
        "sources": profile.get("sources", {}),
        "derived": config.to_dict(),
        "profile_path": str(PROFILE_PATH),
    }


def reset_session() -> None:
    """Drop the open interview. For tests and for starting over."""
    global _session
    _session = None


#: VoiceOS's template names the server `mcp`. Same object, their spelling, so
#: a copy-pasted instruction from their docs finds what it expects.
mcp = server


if __name__ == "__main__":  # pragma: no cover
    # Explicit rather than defaulted: this is the line VoiceOS asks for.
    server.run(transport="stdio")
