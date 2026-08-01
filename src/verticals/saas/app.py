"""The SaaS pipeline app.

    .venv/bin/python -m uvicorn src.verticals.saas.app:app --port 8120

Three things an operator needs while an agent is on the phone: where every
prospect is, what evidence stands behind the ones that claim to be closed, and
what a proposed set of terms is actually worth.

All reads except the economics form, and the form is a plain GET - state lives
in the query string, so a judge can copy a URL of a bad deal and paste it into
Slack. No JavaScript at all: the page has to survive a conference network, and
a board that needs a bundle to render is a board that does not render.

Everything HTML lives in `render.py`. This file is routing and coercion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.verticals.saas import render
from src.verticals.saas.campaign import (
    build_playbook,
    concessions_from,
    load_config,
    model_from_terms,
)
from src.verticals.saas.economics import SubscriptionModel
from src.verticals.saas.pipeline import Pipeline, Stage, seed_samples

DEFAULT_DB = Path("evidence") / "saas_pipeline.db"

app = FastAPI(title="Ledgerline - SaaS pipeline", docs_url=None, redoc_url=None)


def _boot() -> None:
    """Bind config, playbook and pipeline at import.

    Same reasoning as the operator dashboard: an ASGI app that only works once
    somebody has run its lifespan has two behaviours, and the one under a test
    client would not be the one on the projector.
    """
    cfg = load_config()
    app.state.cfg = cfg
    app.state.playbook = build_playbook(cfg)
    app.state.pipeline = Pipeline(os.environ.get("SAAS_DB", str(DEFAULT_DB)))
    if os.environ.get("SAAS_SEED", "1") != "0":
        seed_samples(app.state.pipeline, cfg)


_boot()


def pipeline() -> Pipeline:
    return app.state.pipeline


def use_pipeline(replacement: Pipeline) -> Pipeline:
    """Point the app at a different pipeline. The seam tests pull on."""
    app.state.pipeline = replacement
    return replacement


# -- coercion -------------------------------------------------------------

#: Query params the economics panel accepts, and how to read them. Anything
#: absent falls back to the configured list terms rather than to zero: a blank
#: field must not silently produce a free deal that "clears".
_INT_FIELDS = ("seats", "term_months", "free_months")
_DEC_FIELDS = (
    "price_per_seat_month",
    "discount_pct",
    "onboarding_fee",
    "monthly_cost_to_serve_per_seat",
    "cac",
)


def _terms_from_query(params: Any) -> dict[str, Any]:
    """Read proposed terms off the query string, tolerantly.

    Validation at the boundary: a junk value is dropped in favour of the list
    default rather than crashing the page, because this form is going to be
    typed into live in front of a judge.
    """
    out: dict[str, Any] = {}
    for name in _INT_FIELDS:
        raw = params.get(name)
        if raw not in (None, ""):
            try:
                out[name] = max(0, int(float(raw)))
            except (TypeError, ValueError):
                pass
    for name in _DEC_FIELDS:
        raw = params.get(name)
        if raw not in (None, ""):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                out[name] = value
    if "discount_pct" in out:
        out["discount_pct"] = min(100.0, out["discount_pct"])
    return out


def _analyse(terms: dict[str, Any]) -> tuple[SubscriptionModel, dict[str, Any], dict[str, Any]]:
    """Model, floor verdict, and the per-lever concession report for `terms`."""
    cfg = app.state.cfg
    playbook = app.state.playbook
    model = model_from_terms(terms, cfg)
    check = playbook.floor.evaluate(model)
    concessions = concessions_from(model.to_dict(), cfg)
    report = playbook.concession_report(*concessions)
    # Report the caps against the *actual* proposed model, not the list model
    # the playbook would rebuild - a seat change moves every number.
    report["combined"] = check.to_dict()
    report["stacking_trap"] = (
        bool(concessions)
        and all(r["within_cap"] for r in report["levers"])
        and not check.approved
    )
    return model, check.to_dict(), report


# -- routes ---------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def board(request: Request) -> HTMLResponse:
    p = pipeline()
    cols = p.board()
    strengths: dict[str, str] = {}
    value: dict[str, Any] = {}
    open_value = won_value = 0.0
    open_count = seats_sold = 0

    for stage, prospects in cols.items():
        for prospect in prospects:
            strengths[prospect.id] = p.evidence_strength(prospect.id)
            terms = p.terms_for(prospect.id)
            if not terms:
                continue
            model = model_from_terms(terms, app.state.cfg)
            tcv = float(model.total_contract_value)
            value[prospect.id] = tcv
            if stage is Stage.CLOSED_WON:
                won_value += tcv
                seats_sold += model.seats
            elif stage is not Stage.CLOSED_LOST:
                open_value += tcv
                open_count += 1

    stats = {
        "prospects": sum(len(v) for v in cols.values()),
        "open_value": open_value,
        "open_count": open_count,
        "won_value": won_value,
        "seats_sold": seats_sold,
        "capacity": app.state.playbook.seats_onboardable_per_month,
    }
    return HTMLResponse(
        render.page(
            "Pipeline",
            render.board_page(cols, strengths, value, stats),
            active="board",
        )
    )


@app.get("/prospect/{prospect_id}", response_class=HTMLResponse)
def prospect(prospect_id: str) -> HTMLResponse:
    detail = pipeline().detail(prospect_id)
    if detail is None:
        return HTMLResponse(
            render.page("Not found", render.not_found(f"no prospect {prospect_id}")),
            status_code=404,
        )
    deal = check = None
    if detail["terms"]:
        model, check, _ = _analyse(detail["terms"])
        deal = model.to_dict()
    return HTMLResponse(
        render.page(
            detail["prospect"]["company"],
            render.prospect_page(detail, deal, check),
            active="board",
        )
    )


@app.get("/economics", response_class=HTMLResponse)
def economics(request: Request) -> HTMLResponse:
    model, check, report = _analyse(_terms_from_query(request.query_params))
    return HTMLResponse(
        render.page(
            "Deal economics",
            render.economics_page(model.to_dict(), check, report),
            active="economics",
        )
    )


@app.get("/api/economics")
def api_economics(request: Request) -> JSONResponse:
    model, check, report = _analyse(_terms_from_query(request.query_params))
    return JSONResponse({"deal": model.to_dict(), "check": check, "concessions": report})


@app.get("/api/pipeline")
def api_pipeline() -> JSONResponse:
    p = pipeline()
    payload = p.as_json()
    payload["verdicts"] = {
        prospect.id: p.close_verdict(prospect.id).value for prospect in p.all()
    }
    return JSONResponse(payload)


@app.get("/api/prospect/{prospect_id}")
def api_prospect(prospect_id: str) -> JSONResponse:
    detail = pipeline().detail(prospect_id)
    if detail is None:
        return JSONResponse({"error": "unknown prospect"}, status_code=404)
    return JSONResponse(detail)


@app.get("/api/campaign")
def api_campaign() -> JSONResponse:
    return JSONResponse(app.state.playbook.to_dict())


@app.get("/healthz")
def healthz(seats: int = Query(default=0)) -> dict[str, Any]:
    return {
        "ok": True,
        "counts": pipeline().counts(),
        "campaign_valid": app.state.playbook.is_valid,
    }
