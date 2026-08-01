"""The operator dashboard.

    .venv/bin/python -m uvicorn src.dashboard.app:app --port 8095

Deliberately boring plumbing. The page polls one HTML fragment every two
seconds and swaps it in; there is no websocket, no build step, no CDN. A
conference network that drops a poll costs the operator two seconds, and a
network that drops entirely still leaves the last render on the projector.

The one route that is not a read is `POST /approvals/{id}/decide`. It is a real
form post with two submit buttons, so Approve and Deny work with the keyboard,
work with JavaScript off, and work if the fetch interceptor throws. That path
is the fallback for the voice approval channel, so it does not get to depend on
anything clever.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.dashboard.state import BOARD, Board, seed

HERE = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))

app = FastAPI(title="Expeditor - operator dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


def board() -> Board:
    """The board this process serves. One seam, so tests can swap it."""
    return app.state.board


def use_board(replacement: Board) -> Board:
    """Point the app at a different board. The seam tests pull on."""
    app.state.board = replacement
    return replacement


# Bound at import rather than in a startup hook: an ASGI app that only works
# once someone has run its lifespan is an app with two behaviours, and the one
# that shows up under a test client would not be the one on the projector.
app.state.board = BOARD
if os.environ.get("DASHBOARD_SEED", "1") != "0":
    # An empty board is a supported state - it is what the operator sees
    # before the first call - but it is not what a demo should boot into.
    seed(BOARD)


def _context(request: Request) -> dict[str, Any]:
    b = board()
    return {"request": request, "demo": b.demo, **b.snapshot()}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(request, "index.html", _context(request))


@app.get("/board", response_class=HTMLResponse)
def board_fragment(request: Request) -> HTMLResponse:
    """The polled region. Same template the full page embeds, so there is
    exactly one description of what the board looks like."""
    return TEMPLATES.TemplateResponse(request, "board.html", _context(request))


@app.get("/api/state")
def api_state() -> JSONResponse:
    return JSONResponse(board().as_json())


@app.post("/approvals/{approval_id}/decide")
def decide(approval_id: str, decision: str = Form(...)) -> RedirectResponse:
    """Answer an escalation.

    Anything that is not exactly 'approve' is recorded as a denial, and an
    unknown id is a no-op rather than a 500 - a stale tab posting against a
    decided request must not take the board down mid-demo.
    """
    board().decide(approval_id, decision)
    return RedirectResponse(url="/", status_code=303)


@app.post("/demo/reset")
def demo_reset() -> RedirectResponse:
    """Re-arm the fixtures.

    An approved escalation is spent - the call closes and the ticket leaves the
    rail - so the second judge of the day would otherwise walk up to an empty
    board. This puts the seeded moment back. It only ever touches fixture data;
    the board says FIXTURE DATA the whole time it is holding any.
    """
    seed(board())
    return RedirectResponse(url="/", status_code=303)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "counts": board().snapshot()["counts"]}
