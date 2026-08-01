"""HTML for the SaaS pipeline app. Hand-written, self-contained, no CDN.

Split from `app.py` so routing stays readable and neither file goes past the
500-line rule. No template engine: there is one stylesheet and five fragments,
and a Jinja directory would be more machinery than markup.

Every value that reaches the page goes through `esc()`. Prospect notes and
company names arrive from a config file today and from a call transcript
tomorrow, and a transcript is untrusted text.

Design notes, since "no bootstrap-purple" was the brief: warm paper, near-black
ink, one deep-green accent, and clay red reserved exclusively for a floor
breach or a missing verification. Numbers are tabular-figure and right-aligned
so columns of money line up. Nothing is a gradient. The only shadow on the page
is a 1px hairline.
"""

from __future__ import annotations

import html
from collections.abc import Iterable
from typing import Any

from src.verticals.saas.pipeline import BOARD_ORDER, Prospect, Stage

STAGE_LABEL: dict[Stage, str] = {
    Stage.TARGETED: "Targeted",
    Stage.CONTACTED: "Contacted",
    Stage.QUALIFIED: "Qualified",
    Stage.DEMO_BOOKED: "Demo booked",
    Stage.CLOSED_WON: "Closed won",
    Stage.CLOSED_LOST: "Closed lost",
}


#: One word per close-evidence state, and the colour it earns. "agent only" is
#: amber rather than grey on purpose: a prospect the agent believes it closed,
#: with nothing behind it, is the single most interesting row on the board.
STRENGTH_CLASS: dict[str, str] = {
    "verified": "ok",
    "contradicted": "bad",
    "agent only": "warn",
    "no evidence": "flat",
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def money(amount: Any, currency: str = "USD") -> str:
    """Compact money for a narrow card. $14.9k beats a number that wraps."""
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return str(amount)
    symbol = "$" if currency == "USD" else f"{currency} "
    if value >= 10_000:
        return f"{symbol}{value / 1000:.1f}k"
    return f"{symbol}{value:,.0f}"


def plain(amount: Any) -> str:
    """Drop trailing zeros so a lever reads '10% off', not '10.0% off'."""
    text = str(amount)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


CSS = """
:root {
  --paper: #f6f4ef;
  --surface: #fffdf9;
  --ink: #1b1d1a;
  --muted: #6d6f68;
  --line: #ddd8cd;
  --accent: #1f5340;
  --accent-soft: #e4ece7;
  --clay: #93392c;
  --clay-soft: #f6e6e2;
  --amber: #8a6314;
  --amber-soft: #f6eeda;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font: 15px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, sans-serif;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1240px; margin: 0 auto; padding: 28px 24px 64px; }
header.top {
  border-bottom: 1px solid var(--line); background: var(--surface);
}
header.top .wrap { padding-bottom: 18px; padding-top: 22px; }
.brand { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.brand h1 { font-size: 20px; margin: 0; letter-spacing: -0.01em; font-weight: 620; }
.brand .sub { color: var(--muted); font-size: 13.5px; }
nav.tabs { margin-top: 14px; display: flex; gap: 18px; font-size: 13.5px; }
nav.tabs a { color: var(--muted); padding-bottom: 6px; border-bottom: 2px solid transparent; }
nav.tabs a.on { color: var(--ink); border-bottom-color: var(--accent); }

.banner {
  margin: 18px 0 24px; padding: 10px 14px; border-radius: 3px;
  background: var(--amber-soft); border-left: 3px solid var(--amber);
  color: #5c430e; font-size: 13px;
}
.num { font-variant-numeric: tabular-nums; }

.strip {
  display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 1px;
  background: var(--line); border: 1px solid var(--line); border-radius: 4px;
  margin-bottom: 26px; overflow: hidden;
}
@media (max-width: 760px) { .strip { grid-template-columns: repeat(2, minmax(0,1fr)); } }
.stat { background: var(--surface); padding: 13px 16px; display: flex; flex-direction: column; gap: 2px; }
.stat .lbl { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.stat .val { font-size: 21px; font-weight: 600; letter-spacing: -0.02em; }
.stat .hint { font-size: 11.5px; color: var(--muted); line-height: 1.4; }

.board { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 1080px) { .board { grid-template-columns: repeat(3, minmax(0,1fr)); } }
@media (max-width: 620px)  { .board { grid-template-columns: 1fr; } }
.col h2 {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--muted); margin: 0 0 10px; font-weight: 600;
  display: flex; justify-content: space-between; align-items: center;
}
.col h2 .n { color: var(--ink); font-variant-numeric: tabular-nums; }
.card {
  display: block; background: var(--surface); border: 1px solid var(--line);
  border-radius: 4px; padding: 11px 12px; margin-bottom: 9px; color: inherit;
}
.card:hover { border-color: var(--accent); text-decoration: none; }
.card .co { font-weight: 600; font-size: 14px; letter-spacing: -0.005em; }
.card .who { color: var(--muted); font-size: 12.5px; margin-top: 2px; }
.card .row { display: flex; justify-content: space-between; align-items: center;
  margin-top: 9px; gap: 6px; }
.card .tcv { font-size: 12.5px; white-space: nowrap; font-variant-numeric: tabular-nums; }
.card .tcv.muted { color: var(--muted); font-size: 11.5px; }
.card.won { border-left: 3px solid var(--accent); }
.card.lost { border-left: 3px solid var(--line); opacity: 0.72; }
.empty { color: var(--muted); font-size: 12.5px; font-style: italic; padding: 6px 2px; }

.pill {
  display: inline-block; font-size: 10.5px; letter-spacing: 0.05em;
  text-transform: uppercase; padding: 2px 7px; border-radius: 2px; font-weight: 600;
}
.pill.ok    { background: var(--accent-soft); color: var(--accent); }
.pill.bad   { background: var(--clay-soft);   color: var(--clay); }
.pill.warn  { background: var(--amber-soft);  color: var(--amber); }
.pill.flat  { background: #eceae4;            color: var(--muted); }

.panels { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 22px; }
@media (max-width: 900px) { .panels { grid-template-columns: 1fr; } }
.panel {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 4px; padding: 18px 20px; margin-bottom: 20px;
}
.panel h3 {
  margin: 0 0 14px; font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.09em; color: var(--muted); font-weight: 600;
}
.kv { display: grid; grid-template-columns: auto 1fr; gap: 6px 18px; font-size: 14px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
.kv dd.big { font-size: 17px; font-weight: 600; }

table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: none; }

.chain { list-style: none; margin: 0; padding: 0; }
.chain li { padding: 10px 0 10px 16px; border-left: 2px solid var(--line); position: relative; }
.chain li::before {
  content: ""; position: absolute; left: -5px; top: 16px; width: 8px; height: 8px;
  border-radius: 50%; background: var(--line);
}
.chain li.independent { border-left-color: var(--accent); }
.chain li.independent::before { background: var(--accent); }
.chain li.against { border-left-color: var(--clay); }
.chain li.against::before { background: var(--clay); }
.chain .meta { font-size: 11.5px; color: var(--muted); letter-spacing: 0.02em; }
.chain .body { margin-top: 2px; }

form.terms { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; }
@media (max-width: 760px) { form.terms { grid-template-columns: repeat(2, minmax(0,1fr)); } }
/* Bottom-aligned so a two-line label does not push its input out of the row. */
form.terms > div { display: flex; flex-direction: column; justify-content: flex-end; }
form.terms label { display: block; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted); margin-bottom: 4px; line-height: 1.3; }
form.terms input {
  width: 100%; padding: 7px 9px; font: inherit; font-variant-numeric: tabular-nums;
  border: 1px solid var(--line); border-radius: 3px; background: #fff; color: var(--ink);
}
form.terms input:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
form.terms .actions { grid-column: 1 / -1; align-items: flex-start; margin-top: 2px; }
button {
  font: inherit; font-weight: 600; padding: 8px 18px; border-radius: 3px;
  border: 1px solid var(--accent); background: var(--accent); color: #fff; cursor: pointer;
}
button:hover { background: #17402f; }

.verdict { padding: 14px 16px; border-radius: 4px; margin-bottom: 16px; font-size: 14px; }
.verdict.ok  { background: var(--accent-soft); border-left: 3px solid var(--accent); }
.verdict.bad { background: var(--clay-soft);   border-left: 3px solid var(--clay); }
.verdict.flatbox { background: #eceae4; border-left: 3px solid var(--line); color: var(--muted); }
.verdict b { display: block; font-size: 12px; letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 4px; }
.verdict ul { margin: 8px 0 0; padding-left: 18px; }
.verdict li { margin: 3px 0; }
.note { color: var(--muted); font-size: 12.5px; margin-top: 10px; line-height: 1.55; }
.back { font-size: 13px; color: var(--muted); }
"""


def page(title: str, body: str, *, active: str = "board") -> str:
    tabs = [("board", "/", "Pipeline"), ("economics", "/economics", "Deal economics")]
    nav = "".join(
        f'<a class="{"on" if key == active else ""}" href="{href}">{esc(label)}</a>'
        for key, href, label in tabs
    )
    # Inline SVG favicon: a data URI, so the page stays one request and the
    # console stays clean without shipping a binary asset.
    icon = (
        "data:image/svg+xml,"
        "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
        "%3Crect width='32' height='32' rx='6' fill='%231f5340'/%3E"
        "%3Cpath d='M8 20h4V9H8zm6 0h4v-7h-4zm6 0h4V15h-4z' fill='%23f6f4ef'/%3E"
        "%3C/svg%3E"
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<link rel="icon" href="{icon}">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>"
        '<header class="top"><div class="wrap"><div class="brand">'
        "<h1>Ledgerline &mdash; outbound pipeline</h1>"
        '<span class="sub">a deal is closed when evidence says so, '
        "not when the agent does</span></div>"
        f'<nav class="tabs">{nav}</nav></div></header>'
        f'<div class="wrap">{body}</div></body></html>'
    )


SAMPLE_NOTICE = (
    '<div class="banner"><strong>Sample data.</strong> Prospects marked '
    "<em>sample</em> are seeded from <code>config/saas.json</code> so the board is "
    "not empty. They were walked through the same transitions as a real prospect, "
    "which is why the closed-won row has an independent artifact behind it.</div>"
)


def _verdict_pill(verdict: str) -> str:
    cls = {"VERIFIED": "ok", "CONTRADICTED": "bad"}.get(verdict, "warn")
    return f'<span class="pill {cls}">{esc(verdict.lower())}</span>'


def _strength_pill(strength: str) -> str:
    return (
        f'<span class="pill {STRENGTH_CLASS.get(strength, "flat")}">'
        f"{esc(strength)}</span>"
    )


def summary_strip(stats: dict[str, Any]) -> str:
    """Four numbers an operator glances at between calls."""
    cells = [
        ("Prospects", str(stats["prospects"]), ""),
        (
            "Open pipeline",
            money(stats["open_value"]),
            f'{stats["open_count"]} deals with terms on the table',
        ),
        (
            "Closed won",
            money(stats["won_value"]),
            "every one backed by an independent artifact",
        ),
        (
            "Onboarding load",
            f'{stats["seats_sold"]} / {stats["capacity"]} seats',
            "implementation bandwidth this month, not inventory",
        ),
    ]
    inner = "".join(
        f'<div class="stat"><span class="lbl">{esc(label)}</span>'
        f'<span class="val num">{esc(value)}</span>'
        f'<span class="hint">{esc(hint)}</span></div>'
        for label, value, hint in cells
    )
    return f'<div class="strip">{inner}</div>'


def board_page(
    board: dict[Stage, list[Prospect]],
    strengths: dict[str, str],
    value: dict[str, Any],
    stats: dict[str, Any],
) -> str:
    cols = []
    for stage in BOARD_ORDER:
        prospects = board.get(stage, [])
        cards = []
        for p in prospects:
            extra = (
                " won"
                if p.stage is Stage.CLOSED_WON
                else " lost"
                if p.stage is Stage.CLOSED_LOST
                else ""
            )
            tcv = value.get(p.id)
            right = (
                f'<span class="tcv num">{esc(money(tcv))}</span>'
                if tcv
                else '<span class="tcv muted">no terms</span>'
            )
            cards.append(
                f'<a class="card{extra}" href="/prospect/{esc(p.id)}">'
                f'<div class="co">{esc(p.company)}</div>'
                f'<div class="who">{esc(p.contact)} &middot; {esc(p.seats)} seats</div>'
                f'<div class="row">{_strength_pill(strengths.get(p.id, "no evidence"))}'
                f"{right}</div></a>"
            )
        inner = "".join(cards) or '<div class="empty">nothing here yet</div>'
        cols.append(
            f'<div class="col"><h2>{esc(STAGE_LABEL[stage])}'
            f'<span class="n">{len(prospects)}</span></h2>{inner}</div>'
        )
    legend = (
        '<p class="note" style="margin-top:20px">The pill on each card is what '
        "stands behind that prospect&rsquo;s <em>close</em>, not how the call felt. "
        "<strong>agent only</strong> means the agent believes it closed and nothing "
        "independent agrees yet &mdash; those cards do not move, however confident "
        "the transcript sounds.</p>"
    )
    return (
        SAMPLE_NOTICE
        + summary_strip(stats)
        + f'<div class="board">{"".join(cols)}</div>'
        + legend
    )


def economics_panel(deal: dict[str, Any], check: dict[str, Any]) -> str:
    """Contract value, margin and payback for one set of terms."""
    payback = deal["months_to_cac_payback"]
    margin = deal["gross_margin_pct"]
    effective = deal["effective_discount_pct"]
    cur = esc(deal["currency"])

    rows = [
        ("Total contract value", f'{cur} {deal["total_contract_value"]}', True),
        ("Effective monthly rate", f'{cur} {deal["effective_monthly_rate"]}', False),
        (
            "Effective rate / seat / month",
            f'{cur} {deal["effective_rate_per_seat_month"]}',
            False,
        ),
        (
            "Effective discount (all rate levers)",
            "n/a" if effective is None else f"{effective}%",
            True,
        ),
        ("Cost to serve, full term", f'{cur} {deal["cost_to_serve_total"]}', False),
        ("Gross profit over term", f'{cur} {deal["gross_profit"]}', False),
        ("Gross margin", "n/a" if margin is None else f"{margin}%", True),
        (
            "CAC payback",
            "never" if payback is None else f"{payback} months",
            True,
        ),
        ("Billable months", f'{deal["billable_months"]} of {deal["term_months"]}', False),
    ]
    kv = "".join(
        f"<dt>{esc(label)}</dt><dd class=\"{'big' if big else ''}\">{esc(v)}</dd>"
        for label, v, big in rows
    )

    if check["approved"]:
        verdict = (
            '<div class="verdict ok"><b>Clears the floor</b>'
            f'{esc(check["headline"])}</div>'
        )
    else:
        items = "".join(f"<li>{esc(b)}</li>" for b in check["breaches"])
        verdict = (
            '<div class="verdict bad"><b>Rejected</b>'
            f'{esc(check["headline"])}<ul>{items}</ul></div>'
        )

    floor = check["floor"]
    floor_note = (
        f'<p class="note">Floor: at least {cur} {esc(floor["min_contract_value"])} of '
        f'contract value, CAC repaid inside {esc(floor["max_payback_months"])} months, '
        f'gross margin at or above {esc(floor["min_gross_margin_pct"])}%. '
        "Either money floor failing rejects the deal on its own - margin is a "
        "backstop, not the binding constraint.</p>"
    )
    return f'{verdict}<dl class="kv">{kv}</dl>{floor_note}'


def _terms_form(deal: dict[str, Any]) -> str:
    fields = [
        ("price_per_seat_month", "Price / seat / mo", deal["price_per_seat_month"]),
        ("seats", "Seats", deal["seats"]),
        ("term_months", "Term (months)", deal["term_months"]),
        ("discount_pct", "Discount %", deal["discount_pct"]),
        ("free_months", "Free months", deal["free_months"]),
        ("onboarding_fee", "Onboarding fee", deal["onboarding_fee"]),
        (
            "monthly_cost_to_serve_per_seat",
            "Cost to serve / seat / mo",
            deal["monthly_cost_to_serve_per_seat"],
        ),
        ("cac", "CAC", deal["cac"]),
    ]
    inputs = "".join(
        f'<div><label for="{esc(name)}">{esc(label)}</label>'
        f'<input id="{esc(name)}" name="{esc(name)}" value="{esc(plain(value))}"></div>'
        for name, label, value in fields
    )
    return (
        '<form class="terms" method="get" action="/economics">'
        f'{inputs}<div class="actions"><button type="submit">Recalculate</button>'
        "</div></form>"
    )


def economics_page(
    deal: dict[str, Any], check: dict[str, Any], concessions: dict[str, Any]
) -> str:
    body = (
        '<div class="panels"><div>'
        f'<div class="panel"><h3>Proposed terms</h3>{_terms_form(deal)}</div>'
        f'<div class="panel"><h3>Concessions against list</h3>'
        f"{_concession_table(concessions)}</div></div>"
        f'<div><div class="panel"><h3>Deal economics</h3>'
        f"{economics_panel(deal, check)}</div></div></div>"
    )
    return body


def _concession_table(report: dict[str, Any]) -> str:
    if not report["levers"]:
        return (
            '<p class="note">These are the list terms. Change a field and '
            "recalculate to see what each concession costs.</p>"
        )
    rows = "".join(
        f"<tr><td>{esc(row['described'])}</td>"
        f"<td>{'<span class=\"pill ok\">within cap</span>' if row['within_cap'] else '<span class=\"pill bad\">over cap</span>'}</td>"
        "</tr>"
        for row in report["levers"]
    )
    combined = report["combined"]
    tail = (
        '<p class="note" style="color:var(--clay)"><strong>Stacking trap.</strong> '
        "Every concession above is inside its own cap and the combined deal is "
        "still rejected. This is the case a per-lever check cannot see, and the "
        "reason only the combined evaluation may clear a deal.</p>"
        if report["stacking_trap"]
        else '<p class="note">Per-lever caps answer "may I say yes to this one '
        'question". Only the combined evaluation on the right may clear a deal.</p>'
    )
    verdict = (
        '<span class="pill ok">clears</span>'
        if combined["approved"]
        else '<span class="pill bad">rejected</span>'
    )
    return (
        f'<table><thead><tr><th>Concession</th><th class="n">Own cap</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
        f'<p class="note" style="margin-top:12px">Combined verdict: {verdict}</p>{tail}'
    )


def evidence_chain(chain: list[dict[str, Any]]) -> str:
    """Every artifact on file, marked by channel and by what it is evidence of.

    The `about` label is load-bearing rather than decorative. A calendar event
    is independent and supporting and proves a meeting, not a signature, so it
    shows up here in full and is greyed out of the close.
    """
    if not chain:
        return (
            '<p class="note">No evidence yet. Until an independent channel says '
            "otherwise, nothing about this prospect is confirmed.</p>"
        )
    items = []
    for e in chain:
        bears = e["bears_on_close"]
        if not e["supports"]:
            cls = "against"
        elif e["independent"] and bears:
            cls = "independent"
        else:
            cls = ""
        tag = (
            "independent"
            if e["independent"]
            else "agent-only, cannot verify anything"
        )
        sign = "supports" if e["supports"] else "contradicts"
        digest = f" &middot; sha {esc(e['content_hash'])}" if e["content_hash"] else ""
        scope = (
            ""
            if bears
            else f' <span class="pill flat">about the {esc(e["about"])}</span>'
        )
        note = (
            ""
            if bears
            else '<div class="meta" style="margin-top:3px">Real evidence of '
            "something else. Does not bear on whether they signed.</div>"
        )
        items.append(
            f'<li class="{cls}"><div class="meta">{esc(e["channel"])} &middot; '
            f"{esc(tag)} &middot; {esc(sign)}{digest}{scope}</div>"
            f'<div class="body">{esc(e["summary"])}</div>{note}</li>'
        )
    return f'<ul class="chain">{"".join(items)}</ul>'


def prospect_page(
    detail: dict[str, Any],
    deal: dict[str, Any] | None,
    check: dict[str, Any] | None,
) -> str:
    p = detail["prospect"]
    sample = (
        ' <span class="pill flat">sample</span>' if p["is_sample"] else ""
    )
    head = (
        f'<p class="back"><a href="/">&larr; pipeline</a></p>'
        f'<div class="brand" style="margin:6px 0 18px"><h1>{esc(p["company"])}</h1>'
        f'{_strength_pill(detail["strength"])}{sample}</div>'
        f'<p class="note" style="margin-top:-10px">{esc(p["contact"])}'
        f'{" &middot; " + esc(p["title"]) if p["title"] else ""} &middot; '
        f'{esc(p["phone"])} &middot; {esc(p["email"])} &middot; '
        f'{esc(p["seats"])} seats &middot; stage <strong>'
        f'{esc(STAGE_LABEL[Stage(p["stage"])])}</strong></p>'
    )

    stage = Stage(p["stage"])
    if stage is Stage.CLOSED_WON:
        close_cls, close_head = "ok", "Closed"
    elif stage is Stage.CLOSED_LOST:
        close_cls, close_head = "flatbox", "Closed lost"
    else:
        close_cls = "ok" if detail["can_close"] else "bad"
        close_head = "Close gate"
    close_box = (
        f'<div class="verdict {close_cls}"><b>{esc(close_head)}</b>'
        f'{esc(detail["can_close_reason"])}</div>'
    )

    econ = (
        economics_panel(deal, check)
        if deal and check
        else '<p class="note">No terms proposed yet.</p>'
    )

    events = "".join(
        f'<tr><td>{esc(e["at"][:19].replace("T", " "))}</td>'
        f'<td>{esc(e["from"] or "-")} &rarr; {esc(e["to"])}</td>'
        f'<td>{esc(e["detail"])}</td></tr>'
        for e in detail["events"]
    )

    return (
        head
        + '<div class="panels"><div>'
        + close_box
        + f'<div class="panel"><h3>Evidence chain</h3>'
        f'<p class="note" style="margin:-4px 0 12px">Expected side effect: '
        f'{esc(detail["claim"]["expected_side_effect"])}</p>'
        f'{evidence_chain(detail["chain"])}</div>'
        f'<div class="panel"><h3>Stage history</h3><table><thead><tr>'
        f"<th>When</th><th>Move</th><th>Detail</th></tr></thead>"
        f"<tbody>{events}</tbody></table></div>"
        "</div><div>"
        f'<div class="panel"><h3>Deal economics</h3>{econ}</div>'
        f'<div class="panel"><h3>Notes</h3><p class="note">'
        f'{esc(p["notes"]) or "none"}</p>'
        f'<p class="note">Source: {esc(p["source"]) or "unrecorded"}</p></div>'
        "</div></div>"
    )


def not_found(what: str) -> str:
    return (
        f'<p class="back"><a href="/">&larr; pipeline</a></p>'
        f'<div class="verdict bad"><b>Not found</b>{esc(what)}</div>'
    )


def stage_labels(stages: Iterable[Stage]) -> list[str]:
    return [STAGE_LABEL[s] for s in stages]
