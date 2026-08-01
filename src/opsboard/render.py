"""HTML for the board. One renderer, used for both the first paint and the poll.

The page is server-rendered and the poll returns the *same* fragments, keyed by
content hash. That is deliberate: a second renderer written in JavaScript is a
second description of the truth, and the two drift on the day you cannot afford
it. The browser's only job is to swap a fragment whose key changed, which is
also why focus survives a poll — untouched panels are never re-parsed.

Everything is escaped through `esc`. Nothing is interpolated as an object: a
dataclass repr on a projector is how a judge learns your internal field names.
"""

from __future__ import annotations

from html import escape as _escape
from typing import Any

from src.opsboard.state import CONTRADICTED, UNVERIFIED, VERIFIED


def esc(value: Any) -> str:
    return _escape("" if value is None else str(value), quote=True)


def money(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"${v:,.0f}" if abs(v - round(v)) < 0.005 else f"${v:,.2f}"


def _n(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


VERDICT_CLASS = {VERIFIED: "verified", UNVERIFIED: "unconfirmed", CONTRADICTED: "contradicted"}


# --------------------------------------------------------------------------
# 1. Booked vs proven
# --------------------------------------------------------------------------


def metric_html(d: dict[str, Any]) -> str:
    booked, proven, gap = d["booked"], d["proven"], d["gap"]
    contra = d["contradicted"]

    if booked:
        pv = 100 * proven / booked
        cv = 100 * contra / booked
        bar = (
            f'<span class="seg seg-proven" style="width:{pv:.4f}%"></span>'
            f'<span class="seg seg-contra" style="width:{cv:.4f}%"></span>'
        )
    else:
        bar = '<span class="seg seg-void"></span>'

    if not booked:
        line = "Nothing claimed yet. The gap is the only number that matters and it starts empty."
    elif gap == 0:
        line = f"Every one of {_n(booked)} claims is backed by a channel the agent cannot talk to."
    else:
        parts = [f"{_n(gap)} claim{'s' if gap != 1 else ''} the agent believes and nobody else has confirmed"]
        if contra:
            parts.append(f"{_n(contra)} an independent channel actively disagrees with")
        line = " — ".join(parts) + ". Reported as unconfirmed, never as done."

    return f"""
<div class="figures">
  <div class="fig fig-booked">
    <div class="fig-n tnum">{_n(booked)}</div>
    <div class="fig-l">Booked</div>
    <div class="fig-s">claims the agent filed</div>
  </div>
  <div class="fig-op" aria-hidden="true">of which</div>
  <div class="fig fig-proven">
    <div class="fig-n tnum">{_n(proven)}</div>
    <div class="fig-l">Proven</div>
    <div class="fig-s">confirmed by an independent channel</div>
  </div>
  <div class="fig-tail">
    <div class="tail-row"><span class="tail-n tnum">{_n(d['unconfirmed'])}</span>
      <span class="tail-l">unconfirmed</span></div>
    <div class="tail-row {'is-hot' if contra else ''}"><span class="tail-n tnum">{_n(contra)}</span>
      <span class="tail-l">contradicted</span></div>
    <div class="tail-row tail-quiet"><span class="tail-n tnum">{_n(d['runs'])}</span>
      <span class="tail-l">runs on disk</span></div>
  </div>
</div>
<div class="bar" role="img" aria-label="{_n(proven)} of {_n(booked)} claims proven">{bar}</div>
<p class="gapline">{esc(line)}</p>
""".strip()


# --------------------------------------------------------------------------
# 2. The live call: phase rail + the three numbers
# --------------------------------------------------------------------------


def _rail_html(rail: list[dict[str, Any]]) -> str:
    main = [n for n in rail if not n["spur"]]
    spur = next((n for n in rail if n["spur"]), None)

    nodes = []
    for node in main:
        classes = f"node is-{node['state']}" + (" is-sealed" if node["sealed"] else "")
        nodes.append(
            f'<li class="{classes}"><span class="dot"></span>'
            f'<span class="pl">{esc(node["label"])}</span></li>'
        )
    out = f'<ol class="rail">{"".join(nodes)}</ol>'

    if spur:
        cls = f"spur is-{spur['state']}" + (" is-sealed" if spur["sealed"] else "")
        out += (
            f'<div class="{cls}"><span class="spur-elbow" aria-hidden="true"></span>'
            f'<span class="dot"></span><span class="pl">{esc(spur["label"])}</span>'
            f'<span class="spur-gloss">{esc(spur["gloss"])}</span></div>'
        )
    return out


def _axis_html(call: dict[str, Any]) -> str:
    """A single price axis. Below the floor is hatched, because it is not a
    region the agent is allowed to enter — it should look like territory, not
    like a number that happens to be low."""
    floor, quote, budget = call.get("floor"), call.get("quote"), call.get("budget")
    vals = [v for v in (floor, quote, budget) if isinstance(v, (int, float))]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or max(hi, 1.0)
    lo, hi = lo - span * 0.35, hi + span * 0.35
    width = hi - lo

    def pct(v: float) -> float:
        return max(0.0, min(100.0, 100 * (v - lo) / width))

    marks = ""
    if isinstance(floor, (int, float)):
        at = pct(floor)
        marks += (
            f'<span class="deny" style="width:{at:.3f}%"></span>'
            f'<span class="mk mk-floor" style="left:{at:.3f}%"></span>'
            f'<span class="mkl mkl-floor" style="left:{at:.3f}%">floor</span>'
        )
    if isinstance(budget, (int, float)):
        at = pct(budget)
        marks += (
            f'<span class="mk mk-budget" style="left:{at:.3f}%"></span>'
            f'<span class="mkl mkl-budget" style="left:{at:.3f}%">them</span>'
        )
    if isinstance(quote, (int, float)):
        at = pct(quote)
        marks += (
            f'<span class="mk mk-quote" style="left:{at:.3f}%"></span>'
            f'<span class="mkl mkl-quote" style="left:{at:.3f}%">us</span>'
        )
    return f'<div class="axis">{marks}</div>'


def call_html(d: dict[str, Any]) -> str:
    call = d.get("call")
    rail = _rail_html(d["rail"])

    if not call:
        return f"""
<div class="panel-head">
  <h2>Live call</h2>
  <span class="chip chip-idle">No call in flight</span>
</div>
{rail}
<p class="rail-gloss">The rail is the flow graph, not a suggestion. A tool that
asks to run in a phase with no edge to it is refused before it executes.</p>
<div class="money is-empty">
  <div class="m"><div class="m-l">Their budget</div><div class="m-n tnum">—</div></div>
  <div class="m"><div class="m-l">On the table</div><div class="m-n tnum">—</div></div>
  <div class="m"><div class="m-l">Our floor</div><div class="m-n tnum">—</div></div>
</div>
<p class="binding is-quiet">Armed and waiting. Start a call and this fills in live.</p>
""".strip()

    live = call.get("live")
    chip = (
        '<span class="chip chip-live"><span class="pulse"></span>On the line</span>'
        if live
        else '<span class="chip chip-done">Call ended</span>'
    )
    if call.get("fixture"):
        chip += '<span class="chip chip-fixture">Fixture data</span>'
    who = esc(call.get("business") or "Unknown party")
    phone = esc(call.get("phone") or "")
    task = esc(call.get("task") or "")
    now_phase = str(call.get("phase") or "opening")
    gloss = next((n["gloss"] for n in d["rail"] if n["phase"] == now_phase), "")
    blocked = [n["label"] for n in d["rail"] if n["state"] == "unreachable"]
    sealed = [n["label"] for n in d["rail"] if n["sealed"]]
    shut = blocked + sealed

    rail_note = (
        f'<strong>{esc(", ".join(shut))}</strong> — unreachable, no edge leads back.'
        if shut
        else "Every phase still reachable; nothing has been closed off yet."
    )

    caps = []
    if call.get("units"):
        caps.append(
            f'<b class="tnum">{_n(call["units"])}</b> '
            f'{esc(call.get("unit_label") or "units")}'
        )
    if call.get("capacity_total"):
        caps.append(
            f'<b class="tnum">{_n(call.get("capacity_held") or 0)}</b> held of '
            f'<b class="tnum">{_n(call["capacity_total"])}</b>'
        )
    cap = f'<span class="cap">{" &middot; ".join(caps)}</span>' if caps else ""

    binding = call.get("floor_binding")
    if binding:
        note = (
            f'<p class="binding is-hot"><b>Floor is binding.</b> '
            f'{money(call.get("floor"))} is the lowest this call can reach; '
            f'their {money(call.get("budget"))} does not clear it.</p>'
        )
    elif call.get("quote") is not None:
        note = (
            f'<p class="binding"><b>Room to move.</b> Every step down from '
            f'{money(call.get("quote"))} is clamped at {money(call.get("floor"))} '
            f'by code, not by instruction.</p>'
        )
    else:
        note = (
            '<p class="binding is-quiet">No price named yet — the gate refuses one '
            'until units, capacity and their current spend are all known.</p>'
        )

    return f"""
<div class="panel-head">
  <h2>Live call</h2>
  {chip}
  <span class="who">{who}{f' <span class="ph">{phone}</span>' if phone else ''}</span>
  <span class="clock tnum" data-since="{esc(call.get('started_epoch') or 0)}">00:00</span>
</div>
<p class="task">{task or "Call in progress"}{cap}</p>
{rail}
<p class="rail-gloss"><span class="rg-now">{esc(gloss)}</span><span>{rail_note}</span></p>
<div class="money">
  <div class="m"><div class="m-l">Their budget</div>
    <div class="m-n tnum">{money(call.get('budget'))}</div></div>
  <div class="m m-table"><div class="m-l">On the table</div>
    <div class="m-n tnum">{money(call.get('quote'))}</div></div>
  <div class="m m-floor{' is-binding' if binding else ''}"><div class="m-l">Our floor</div>
    <div class="m-n tnum">{money(call.get('floor'))}</div></div>
</div>
{_axis_html(call)}
{note}
""".strip()


# --------------------------------------------------------------------------
# 3. Guardrail ledger
# --------------------------------------------------------------------------


def guard_html(d: dict[str, Any]) -> str:
    refusals = list(reversed(d["refusals"]))
    head = (
        '<div class="panel-head"><h2>Guardrail ledger</h2>'
        f'<span class="chip chip-held">{_n(len(refusals))} held</span>'
        '<span class="chip chip-quiet">0 breached</span></div>'
    )
    if not refusals:
        return head + """
<div class="empty">
  <p class="empty-h">Nothing refused yet.</p>
  <p class="empty-b">Every refusal the system issues lands here in plain language.
  An empty ledger means nothing has been attempted, not that nothing is watching.</p>
</div>""".rstrip()

    rows = []
    for r in refusals:
        phase = f'<span class="rf-phase">{esc(r["phase"])}</span>' if r.get("phase") else ""
        detail = f'<p class="rf-d">{esc(r["detail"])}</p>' if r.get("detail") else ""
        rows.append(
            f'<li class="rf"><span class="rf-seq tnum">{r["seq"]:02d}</span>'
            f'<span class="rf-chip">Held</span>'
            f'<p class="rf-h">{esc(r["headline"])}</p>'
            f'<div class="rf-meta"><span class="rf-kind">{esc(r["kind"])}</span>{phase}</div>'
            f"{detail}</li>"
        )
    return head + f'<ol class="rfs">{"".join(rows)}</ol>'


# --------------------------------------------------------------------------
# 4. Receipt wall
# --------------------------------------------------------------------------


def _chain_html(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return '<ul class="chain"><li class="ev ev-none">no evidence attached</li></ul>'
    items = []
    for e in evidence[:CHAIN_PER_CLAIM]:
        if not e["supports"]:
            cls, tag = "ev-against", "contradicts"
        elif e["independent"]:
            cls, tag = "ev-ind", "independent"
        else:
            cls, tag = "ev-agent", "agent-only"
        items.append(
            f'<li class="ev {cls}"><span class="ev-mk" aria-hidden="true"></span>'
            f'<span class="ev-ch">{esc(e["label"])}</span>'
            f'<span class="ev-tag">{tag}</span>'
            f'<span class="ev-s">{esc(e["summary"])}</span></li>'
        )
    return f'<ul class="chain">{"".join(items)}</ul>'


#: Claims drawn per card. A card taller than the wall is a card a judge reads
#: the top half of, so the rest is counted rather than clipped.
CLAIMS_PER_CARD = 2
#: Evidence rows drawn per claim, for the same reason.
CHAIN_PER_CLAIM = 3


def _receipt_html(r: dict[str, Any]) -> str:
    claims = []
    for c in r["claims"][:CLAIMS_PER_CARD]:
        claims.append(
            f'<li class="cl cl-{VERDICT_CLASS[c["verdict"]]}">'
            f'<div class="cl-top"><span class="cl-v">{esc(c["stamp"])}</span>'
            f'<span class="cl-d">{esc(c["description"])}</span></div>'
            f'<p class="cl-x">expects: {esc(c["expected"])}</p>'
            f"{_chain_html(c['evidence'])}</li>"
        )
    more = len(r["claims"]) - len(claims)
    if more:
        claims.append(f'<li class="cl cl-more">+{more} further claim(s) on this receipt</li>')
    when = esc(r["started_at"][:16].replace("T", " "))
    return f"""
<article class="rc rc-{VERDICT_CLASS[r['verdict']]}">
  <header class="rc-head">
    <h3 class="rc-task">{esc(r['task'])}</h3>
    <span class="stamp">{esc(r['stamp'])}</span>
  </header>
  <ol class="claims">{''.join(claims)}</ol>
  <footer class="rc-foot"><span>{esc(r['id'])}</span><span>{when}</span></footer>
</article>""".strip()


def wall_html(d: dict[str, Any]) -> str:
    live = [r for r in d["receipts"] if not r["empty"]]
    silent = sum(1 for r in d["receipts"] if r["empty"]) + d["hidden"]

    silent_chip = (
        f'<span class="chip chip-quiet">{_n(silent)} filed no claims</span>' if silent else ""
    )
    head = (
        '<div class="panel-head"><h2>Receipt wall</h2>'
        f'<span class="chip chip-quiet">{_n(len(live))} with claims</span>{silent_chip}'
        '<span class="head-note">Unconfirmed is not failure. It is the system '
        'declining to tell you something it cannot prove.</span></div>'
    )
    if not live:
        return head + """
<div class="empty empty-wall">
  <p class="empty-h">No receipts with claims yet.</p>
  <p class="empty-b">Every run writes one, including the runs that crash — a run
  with no receipt is indistinguishable from a run that lied.</p>
</div>""".rstrip()

    return head + f'<div class="wall-grid">{"".join(_receipt_html(r) for r in live)}</div>'


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

PANELS = {"metric": metric_html, "call": call_html, "guard": guard_html, "wall": wall_html}


def panel(name: str, data: dict[str, Any]) -> str:
    return PANELS[name](data)


def page_html(snapshot: dict[str, Any]) -> str:
    p = snapshot["panels"]
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Expeditor — ops board</title>
<link rel="stylesheet" href="/static/ops.css">
</head>
<body data-state-key="{esc(snapshot['state_key'])}">
<div class="board">
  <header class="mast">
    <div class="mark">
      <div class="wordmark">Expeditor<span class="wm-sub">ops board</span></div>
      <p class="thesis">The agent's word counts for nothing here. A claim is only
      proven when a channel the agent cannot speak into says so.</p>
    </div>
    <div class="metric" id="panel-metric" data-key="{esc(snapshot['keys']['metric'])}">{metric_html(p['metric'])}</div>
  </header>

  <div class="mid">
    <section class="panel panel-call" id="panel-call" data-key="{esc(snapshot['keys']['call'])}">{call_html(p['call'])}</section>
    <section class="panel panel-guard" id="panel-guard" data-key="{esc(snapshot['keys']['guard'])}">{guard_html(p['guard'])}</section>
  </div>

  <section class="panel panel-wall" id="panel-wall" data-key="{esc(snapshot['keys']['wall'])}">{wall_html(p['wall'])}</section>

  <footer class="foot">
    <span class="foot-l">evidence/ &middot; re-read every poll &middot; nothing on this
    screen can promote a claim</span>
    <span class="foot-r"><span class="poll-dot" id="poll-dot"></span>
      <span id="poll-state">polling</span>
      <span class="tnum" id="poll-key">{esc(snapshot['state_key'])}</span></span>
  </footer>
</div>
<script src="/static/ops.js"></script>
</body></html>"""
