"""The meta-layer, as a web app.

    .venv/bin/python -m uvicorn src.generator.app:app --port 8140

Describe a goal in one sentence. The page comes back with the intake questions
*that goal* needs - not the union of every field the three hand-written
verticals wanted - and you approve or edit them before anything is built. Then
it generates a dashboard fitted to the same profile and shows it to you inline.

Three things this file is responsible for:

* **Edits are re-hardened server-side, not trusted.** Whatever the browser
  sends back goes through `harden()` again, so deleting the units-vs-headcount
  question puts it straight back and says so in `repairs`. The edit box can
  remove that question from the screen; it cannot remove it from the form. The
  409 behind it is the backstop for a set that hardening genuinely cannot
  repair - an unfillable form is not worth a dashboard.

* **The preview is a file, not a string.** `/preview/{slug}` reads what was
  actually written to `config/generated/<slug>/dashboard.html`. If the disk
  copy is broken the preview is broken, which is the honest behaviour.

* **The source of every artifact is visible.** Whether the questions came from
  the model or the offline fallback, and whether the dashboard came from
  headless Claude or the built-in template, is shown in the UI. A demo where
  nobody can tell which path ran is a demo that proves nothing.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / "config" / ".env")

from src.generator import dashboard_gen  # noqa: E402
from src.generator.questions import (  # noqa: E402
    LiveKitPlanner,
    Planner,
    Question,
    QuestionSet,
    canonical_set,
    generate,
    harden,
)
from src.generator.spec import TaskProfile, heuristic_profile, slugify  # noqa: E402

logger = logging.getLogger("generator.app")

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
OUT_ROOT = ROOT / "config" / "generated"

app = FastAPI(title="Task generator", docs_url=None, redoc_url=None)
app.state.planner: Planner = LiveKitPlanner()
app.state.runner = None
"""None means the real `claude -p`. Tests and offline rehearsal swap it."""


def use_planner(replacement: Planner) -> Planner:
    app.state.planner = replacement
    return replacement


def use_runner(replacement) -> Any:
    app.state.runner = replacement
    return replacement


def _rebuild(payload: dict[str, Any]) -> QuestionSet:
    """Rebuild a set from whatever the browser edited, then re-harden it.

    Hardening again is the point. The user may have deleted a required field
    or a pricing filter may now apply differently because they changed the
    exchange kind; re-running it means the server's answer does not depend on
    the browser having been well behaved.
    """
    profile = TaskProfile.from_dict(payload.get("profile") or {})
    drafted: list[Question] = []
    for item in payload.get("questions") or []:
        if isinstance(item, dict) and (q := Question.from_dict(item)):
            drafted.append(q)
    out = harden(profile, drafted) if drafted else canonical_set(profile)
    out.source = str(payload.get("source") or "edited")
    return out


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "claude_cli": dashboard_gen.claude_available(),
            "generated": sorted(p.name for p in OUT_ROOT.glob("*") if p.is_dir()),
        }
    )


@app.post("/api/plan")
async def plan(body: dict[str, Any]) -> JSONResponse:
    """Goal in, question set out. Falls back rather than failing."""
    goal = str(body.get("goal") or "").strip()
    if not goal:
        return JSONResponse({"error": "describe the goal first"}, status_code=400)
    if len(goal) > 2000:
        goal = goal[:2000]

    qs = await generate(goal, app.state.planner)
    return JSONResponse(qs.to_dict())


@app.post("/api/questions/validate")
async def validate_questions(body: dict[str, Any]) -> JSONResponse:
    """Re-check an edited set. The edit box's conscience."""
    return JSONResponse(_rebuild(body).to_dict())


@app.post("/api/dashboard")
async def dashboard(body: dict[str, Any]) -> JSONResponse:
    """Generate the dashboard for an approved set.

    Refuses on an invalid set: a dashboard built from a form that never asks
    the units question would render a number nobody can interpret.
    """
    qs = _rebuild(body)
    if not qs.is_valid:
        return JSONResponse(
            {"error": "question set is not valid", "problems": qs.problems()},
            status_code=409,
        )

    result = await asyncio.to_thread(
        dashboard_gen.generate_dashboard,
        qs.profile,
        qs,
        runner=app.state.runner,
        out_root=OUT_ROOT,
    )
    payload = result.to_dict()
    payload["slug"] = qs.profile.slug
    payload["preview"] = f"/preview/{qs.profile.slug}"
    payload["repairs"] = qs.repairs
    payload["questions"] = [q.to_dict() for q in qs.questions]
    return JSONResponse(payload)


@app.get("/preview/{slug}", response_class=HTMLResponse)
async def preview(slug: str) -> HTMLResponse:
    """Serve the file that was actually written. Traversal-proof by slugify."""
    safe = slugify(slug, fallback="")
    path = OUT_ROOT / safe / "dashboard.html" if safe else None
    if path is None or not path.is_file():
        return HTMLResponse(
            "<p style='font:14px system-ui;padding:40px'>Nothing generated for "
            f"<code>{safe or '(empty)'}</code> yet.</p>",
            status_code=404,
        )
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post("/api/answers/check")
async def check_answers(body: dict[str, Any]) -> JSONResponse:
    """Run the generated validation rules over real answers.

    Here so the rules are provably not decoration: the same `Rule.check` the
    form renders is the one the server runs.
    """
    qs = _rebuild(body)
    errors = qs.check_answers(body.get("answers") or {})
    return JSONResponse({"errors": errors, "ok": not errors})


def offline_set(goal: str) -> QuestionSet:
    """No model, no network. The rehearsal path."""
    return canonical_set(heuristic_profile(goal))
