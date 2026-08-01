"""The owner-facing dashboard.

    .venv/bin/python -m uvicorn src.verticals.restaurant.app:app --port 8110

Self-contained: the CSS is inline, there is no CDN, no build step and no
JavaScript framework. The only script on the page is a dozen lines that turn
the Sheets button into a fetch so it can report back without a navigation, and
the page works with it disabled - the CSV links are plain anchors and the
refresh is a real form post.

Layout follows the argument the product is making, top to bottom:

1. **Booked vs proven.** The two numbers side by side, with the gap named.
   Every other dashboard in this building shows one number; the gap is the
   thing worth looking at.
2. **Capacity.** What the week is already spoken for.
3. **Calls.** One row each, drilling into the evidence chain.

Sample rows carry a badge everywhere they appear, and the header states how
much of the headline is synthesized. A page whose thesis is "prove your
outcomes" cannot quietly show fixtures as outcomes.
"""

from __future__ import annotations

import html
import os
from collections.abc import Iterable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from src.verticals.restaurant import config as cfg
from src.verticals.restaurant import export as ex
from src.verticals.restaurant import ingest
from src.verticals.restaurant import query as q
from src.verticals.restaurant.store import Store

app = FastAPI(title="Restaurant call ledger", docs_url=None, redoc_url=None)


def _open_store() -> Store:
    return Store(os.environ.get("RESTAURANT_DB") or cfg.default().database_path)


def store() -> Store:
    """The store this process serves. One seam, so tests can swap it."""
    return app.state.store


def use_store(replacement: Store) -> Store:
    app.state.store = replacement
    return replacement


# Bound at import, not in a startup hook: an app that only works once someone
# has run its lifespan has two behaviours, and the one a test sees would not be
# the one on the projector.
app.state.store = _open_store()
if os.environ.get("RESTAURANT_INGEST_ON_BOOT", "1") != "0":
    try:
        ingest.refresh(app.state.store)
    except Exception as exc:  # noqa: BLE001 - an empty dashboard beats no dashboard
        app.state.boot_error = f"{type(exc).__name__}: {exc}"


# -- rendering ------------------------------------------------------------

E = html.escape


def _s(value: Any) -> str:
    return E("" if value is None else str(value))


def _money(value: Any, currency: str = "USD") -> str:
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "")
    try:
        return f"{symbol}{float(value):,.2f}"
    except (TypeError, ValueError):
        return f"{symbol}0.00"


def _short_time(value: Any) -> str:
    text = str(value or "")
    return E(text[:16].replace("T", " ")) if text else "-"


CSS = """
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1d212a;--line:#282d38;--ink:#e8eaf0;
--dim:#8f97a8;--good:#35d07f;--warn:#f5b301;--bad:#ff5c5c;--accent:#6ea8ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
header.top{display:flex;justify-content:space-between;align-items:baseline;
gap:16px;flex-wrap:wrap;margin-bottom:20px}
h1{font-size:20px;margin:0;letter-spacing:.2px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.9px;color:var(--dim);
margin:0 0 12px}
.sub{color:var(--dim);font-size:13px}
.banner{background:#2a2410;border:1px solid #5a4a12;color:#ffd980;padding:10px 14px;
border-radius:8px;font-size:13px;margin-bottom:20px}
.hero{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:22px;margin-bottom:18px}
.heronums{display:flex;gap:34px;flex-wrap:wrap;align-items:flex-end}
.num{font-size:40px;font-weight:650;letter-spacing:-1px;line-height:1.05}
.num.proven{color:var(--good)}
.num.gap{color:var(--warn)}
.num.bad{color:var(--bad)}
.numlabel{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);
margin-bottom:6px}
.bar{height:12px;border-radius:6px;background:var(--panel2);overflow:hidden;
display:flex;margin-top:18px;border:1px solid var(--line)}
.bar span{display:block;height:100%}
.bar .p{background:var(--good)}
.bar .u{background:var(--warn)}
.bar .c{background:var(--bad)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--dim);margin-top:10px}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:16px;
margin-bottom:18px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}
.kv{display:flex;justify-content:space-between;padding:6px 0;
border-bottom:1px solid var(--line);font-size:14px}
.kv:last-child{border-bottom:0}
.kv .k{color:var(--dim)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;
background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:760px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.8px;
color:var(--dim);padding:11px 14px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:11px 14px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
tr:hover td{background:#1b1f28}
.right{text-align:right}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
font-weight:600;letter-spacing:.4px;border:1px solid transparent;white-space:nowrap}
.pill.PROVEN,.pill.SUCCESS{background:#0f3524;color:var(--good);border-color:#1d6440}
.pill.UNCONFIRMED,.pill.PARTIAL{background:#33290a;color:var(--warn);border-color:#6b5410}
.pill.CONTRADICTED,.pill.FAILED{background:#3a1414;color:var(--bad);border-color:#7a2626}
.pill.sample{background:#1b2740;color:#9dc0ff;border-color:#2f4675}
.pill.plain{background:var(--panel2);color:var(--dim);border-color:var(--line)}
.btns{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 0}
.btn{display:inline-block;background:var(--panel2);border:1px solid var(--line);
color:var(--ink);padding:7px 13px;border-radius:8px;font-size:13px;cursor:pointer;
font-family:inherit}
.btn:hover{border-color:var(--accent);text-decoration:none}
.claim{border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px;
background:var(--panel)}
.claim h3{margin:0 0 4px;font-size:15px;font-weight:600}
.chain{list-style:none;margin:12px 0 0;padding:0;border-left:2px solid var(--line);
padding-left:16px}
.chain li{margin-bottom:12px;font-size:13px}
.chain li:last-child{margin-bottom:0}
.chain .meta{color:var(--dim);font-size:11.5px;margin-top:3px;font-family:ui-monospace,
SFMono-Regular,Menlo,monospace}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--dim)}
.note{color:var(--dim);font-size:12.5px;margin-top:10px;line-height:1.55}
#exportmsg{font-size:13px;color:var(--dim);margin-top:10px;min-height:18px}
footer{color:var(--dim);font-size:12px;margin-top:34px;line-height:1.7}
"""

SCRIPT = """
document.addEventListener('click', async (e) => {
  const b = e.target.closest('[data-sheets]');
  if (!b) return;
  e.preventDefault();
  const out = document.getElementById('exportmsg');
  out.textContent = 'Pushing ' + b.dataset.sheets + ' to Google Sheets...';
  try {
    const r = await fetch('/export/' + b.dataset.sheets + '/sheets', {method:'POST'});
    const j = await r.json();
    out.textContent = j.summary || JSON.stringify(j);
  } catch (err) { out.textContent = 'Export failed: ' + err; }
});
"""


def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{E(title)}</title><style>{CSS}</style></head><body>"
        f"<div class=wrap>{body}</div><script>{SCRIPT}</script></body></html>"
    )


def _pill(text: str, extra: str = "") -> str:
    cls = str(text).replace(" ", "")
    return f"<span class='pill {cls} {extra}'>{E(str(text))}</span>"


def _sample_pill(is_sample: Any) -> str:
    return " " + _pill("SAMPLE", "sample") if is_sample else ""


def _rows(items: Iterable[str]) -> str:
    return "".join(items)


# -- pages ----------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(all: int = 0) -> HTMLResponse:
    st = store()
    o = q.overview(st)
    rev = o["revenue"]
    real = o["revenue_real_only"]
    cur = o["currency"]
    cap = o["capacity"]
    disc = o["discounts"]
    all_calls = q.calls_table(st)
    # A rehearsal directory is mostly dial tests: receipts with no claims, no
    # money and nothing to verify. They are real calls and they stay in the
    # database and the CSV, but leading with sixty of them buries the four that
    # matter. Hidden by default, counted out loud, one click away.
    calls = all_calls if all else [c for c in all_calls if c["claims"]]
    hidden = len(all_calls) - len(calls)

    span = max(rev["booked"] + rev["contradicted"], 0.01)
    w_p = rev["proven"] / span * 100
    w_u = rev["unconfirmed"] / span * 100
    w_c = rev["contradicted"] / span * 100

    banner = ""
    if o["calls_sample"]:
        banner = (
            f"<div class=banner><b>{o['calls_sample']} of "
            f"{o['calls_sample'] + o['calls_real']} calls on this page are synthesized "
            f"sample data</b>, badged SAMPLE in every table. Real calls only: "
            f"{_money(real['booked'], cur)} booked / {_money(real['proven'], cur)} proven "
            f"across {o['real_orders']} order(s). Sample rows exist so the layout is "
            f"legible before the demo call - they are not evidence of anything.</div>"
        )

    hero = f"""
    <div class=hero>
      <h2>Revenue booked vs revenue proven</h2>
      <div class=heronums>
        <div><div class=numlabel>Booked (agent believes)</div>
             <div class=num>{_money(rev['booked'], cur)}</div></div>
        <div><div class=numlabel>Proven (independent evidence)</div>
             <div class='num proven'>{_money(rev['proven'], cur)}</div></div>
        <div><div class=numlabel>Unproven gap</div>
             <div class='num gap'>{_money(rev['gap'], cur)}</div></div>
        <div><div class=numlabel>Contradicted</div>
             <div class='num {"bad" if rev["contradicted"] else ""}'>
             {_money(rev['contradicted'], cur)}</div></div>
      </div>
      <div class=bar><span class=p style='width:{w_p:.2f}%'></span>
        <span class=u style='width:{w_u:.2f}%'></span>
        <span class=c style='width:{w_c:.2f}%'></span></div>
      <div class=legend>
        <span><i class='dot' style='background:var(--good)'></i>
          proven {rev['orders']['proven']} order(s) &middot; {rev['proven_pct']}% of booked</span>
        <span><i class='dot' style='background:var(--warn)'></i>
          unconfirmed {rev['orders']['unconfirmed']} order(s)</span>
        <span><i class='dot' style='background:var(--bad)'></i>
          contradicted {rev['orders']['contradicted']} order(s)</span>
      </div>
      <div class=note>Booked counts every order the agent filed. Proven counts only the
      ones an independent channel confirmed - inbound SMS, inbound email, a provider API.
      Nothing the agent said about its own call can move the green number.</div>
    </div>"""

    over = cap["committed"] + cap["held"] > cap["total_capacity"]
    capacity_panel = f"""
    <div class=panel>
      <h2>Capacity &mdash; week of {_s(cap['week_start'])}</h2>
      <div class=kv><span class=k>Weekly capacity</span>
        <span>{cap['total_capacity']} {_s(cap['unit'])}</span></div>
      <div class=kv><span class=k>Committed (proven orders)</span>
        <span style='color:var(--good)'>{cap['committed']}</span></div>
      <div class=kv><span class=k>Held (unconfirmed orders)</span>
        <span style='color:var(--warn)'>{cap['held']}</span></div>
      <div class=kv><span class=k>Left to sell</span>
        <span class='{"" if not over else "num bad"}' style='font-weight:600'>
        {cap['remaining']}</span></div>
      <div class=kv><span class=k>Utilisation</span>
        <span>{cap['utilisation_pct']}%</span></div>
      <div class=note>{'<b style="color:var(--bad)">Over-committed.</b> ' if over else ''}
      Unconfirmed orders are counted against capacity, not ignored: an oven slot promised
      on a call is unavailable until the order is either confirmed or dropped.
      Showing {_s(cap['week_reason'])}.</div>
    </div>"""

    discount_panel = f"""
    <div class=panel>
      <h2>Negotiation</h2>
      <div class=kv><span class=k>Orders priced</span><span>{disc['orders_priced']}</span></div>
      <div class=kv><span class=k>Average discount off target</span>
        <span>{disc['avg_discount_pct']}%</span></div>
      <div class=kv><span class=k>Deepest discount</span>
        <span>{disc['max_discount_pct']}%</span></div>
      <div class=kv><span class=k>Floor was binding</span>
        <span>{disc['floor_bound_orders']} order(s) &middot;
        {disc['floor_bound_pct']}%</span></div>
      <div class=note>{E(disc['note'])}</div>
    </div>"""

    ledger_panel = f"""
    <div class=panel>
      <h2>Ledger</h2>
      <div class=kv><span class=k>Calls</span><span>{o['counts']['calls']}</span></div>
      <div class=kv><span class=k>Orders</span><span>{o['counts']['orders']}</span></div>
      <div class=kv><span class=k>Claims</span><span>{o['counts']['claims']}</span></div>
      <div class=kv><span class=k>Evidence artifacts</span>
        <span>{o['counts']['evidence']}</span></div>
      <div class=btns>
        <form method=post action=/refresh style='display:inline'>
          <button class=btn type=submit>Re-ingest receipts</button></form>
      </div>
      <div class=note>SQLite at
      <span class=mono>{_s(cfg.default().database_path.name)}</span>. Every row here is a
      projection of a receipt JSON in <span class=mono>evidence/</span>; verdicts are
      re-derived from evidence on the way in, never copied from the file.</div>
    </div>"""

    exports = "".join(
        f"<a class=btn href='/export/{name}.csv'>CSV: {E(title)}</a>"
        f"<button class=btn data-sheets='{name}'>&rarr; Sheets</button>"
        for name, (title, _) in q.EXPORTS.items()
    )

    rows = _rows(
        f"<tr>"
        f"<td class=mono>{_short_time(c['started'])}</td>"
        f"<td><a href='/call/{_s(c['id'])}'>{_s(c['task'] or c['id'])}</a>"
        f"{_sample_pill(c['is_sample'])}</td>"
        f"<td class=mono>{_s(c['to_number'] or '-')}</td>"
        f"<td>{_pill(c['outcome'] or 'UNKNOWN')}</td>"
        f"<td class=right>{c['verified']}/{c['claims']}</td>"
        f"<td class=right>{_money(c['booked'], cur)}</td>"
        f"<td class='right' style='color:var(--good)'>{_money(c['proven'], cur)}</td>"
        f"</tr>"
        for c in calls
    ) or "<tr><td colspan=7 class=note>No calls ingested yet.</td></tr>"

    toggle = (
        f"<a class=btn href='/?all=1'>Show {hidden} call(s) with no claims</a>"
        if hidden and not all
        else ("<a class=btn href='/'>Hide calls with no claims</a>" if all else "")
    )

    body = f"""
    <header class=top>
      <div><h1>{E(o['business'])} &mdash; call ledger</h1>
        <div class=sub>Every call, order, claim and piece of evidence, queryable.</div></div>
      <div class=sub>{o['calls_real']} real call(s) &middot; {o['calls_sample']} sample
        &middot; <a href=/api/overview>JSON</a></div>
    </header>
    {banner}
    {hero}
    <div class=grid>{capacity_panel}{discount_panel}{ledger_panel}</div>
    <h2>Calls &mdash; {len(calls)} shown{f", {hidden} with no claims hidden" if hidden and not all else ""}</h2>
    <div class=tablewrap><table>
      <tr><th>Started</th><th>Task</th><th>Called</th><th>Outcome</th>
          <th class=right>Verified</th><th class=right>Booked</th>
          <th class=right>Proven</th></tr>
      {rows}
    </table></div>
    <div class=btns>{toggle}</div>
    <h2 style='margin-top:26px'>Export</h2>
    <div class=btns>{exports}</div>
    <div id=exportmsg></div>
    <footer>
      Verdicts come from <span class=mono>src/verify/receipts.py</span> and are recomputed
      from the evidence rows at write time. <span class=mono>claims.verdict</span> is
      immutable in the database - a trigger aborts any UPDATE - so nothing on this page,
      and no SQL against this file, can promote a claim without new evidence.
    </footer>"""
    return _page(f"{o['business']} - call ledger", body)


@app.get("/call/{call_id}", response_class=HTMLResponse)
def call_page(call_id: str) -> HTMLResponse:
    st = store()
    detail = q.call_detail(st, call_id)
    if detail is None:
        return _page("Not found", "<h1>No such call</h1><p><a href=/>Back</a></p>")

    call = detail["call"]
    cur = detail["currency"]

    orders = _rows(
        f"<tr><td class=mono>{_s(o['id'])}</td><td class=right>{o['qty']}</td>"
        f"<td>{_s(o['unit'])}</td><td class=right>{_money(o['total'], cur)}</td>"
        f"<td class=right>{'-' if o['discount_pct'] is None else str(o['discount_pct']) + '%'}"
        f"{' ' + _pill('AT FLOOR', 'plain') if o['floor_bound'] else ''}</td>"
        f"<td>{_s(o['delivery_at'] or '-')}</td><td>{_pill(o['status'].upper())}</td></tr>"
        for o in detail["orders"]
    ) or "<tr><td colspan=7 class=note>No order parsed from this call's claims.</td></tr>"

    claims = ""
    for c in detail["claims"]:
        chain = _rows(
            f"<li><b>{_s(e['channel'])}</b> "
            f"{_pill('INDEPENDENT' if e['independent'] else 'AGENT-ONLY', 'plain')} "
            f"{_pill('SUPPORTS' if e['supports'] else 'CONTRADICTS', 'plain')}<br>"
            f"{_s(e['summary'])}"
            f"<div class=meta>{_s(e['captured_at'])}"
            f"{' &middot; sha256:' + _s(e['content_hash']) if e['content_hash'] else ''}"
            f"{' &middot; ' + _s(e['artifact_path']) if e['artifact_path'] else ''}</div></li>"
            for e in c["evidence"]
        ) or "<li class=note>No evidence attached. This claim is the agent's word alone.</li>"
        claims += f"""
        <div class=claim>
          <h3>{_pill(c['label'])} {_s(c['description'])}</h3>
          <div class=note>Expected side effect: {_s(c['expected_side_effect'] or '-')}</div>
          <div class=note>{c['independent_evidence']} independent artifact(s) of
            {len(c['evidence'])} total.</div>
          <ul class=chain>{chain}</ul>
        </div>"""
    claims = claims or "<div class=note>This call filed no claims.</div>"

    body = f"""
    <header class=top>
      <div><h1>{_s(call['task'] or call['id'])}{_sample_pill(call['is_sample'])}</h1>
        <div class=sub><a href=/>&larr; All calls</a> &middot;
          <span class=mono>{_s(call['id'])}</span> &middot;
          <a href='/api/call/{_s(call['id'])}'>JSON</a></div></div>
      <div class=sub>{_pill(call['outcome'] or 'UNKNOWN')}</div>
    </header>
    <div class=grid>
      <div class=panel><h2>Call</h2>
        <div class=kv><span class=k>Called</span>
          <span class=mono>{_s(call['to_number'] or '-')}</span></div>
        <div class=kv><span class=k>Room</span>
          <span class=mono>{_s(call['room'] or '-')}</span></div>
        <div class=kv><span class=k>Started</span>
          <span class=mono>{_short_time(call['started'])}</span></div>
        <div class=kv><span class=k>Duration</span>
          <span>{_s(call['duration_s'])}s</span></div>
        <div class=kv><span class=k>Recording</span>
          <span class=mono>{_s(call['recording_path'] or 'none')}</span></div>
        <div class=kv><span class=k>Transcript</span>
          <span class=mono>{_s(call['transcript_path'] or 'none')}</span></div>
      </div>
      <div class=panel><h2>This call</h2>
        <div class=kv><span class=k>Booked</span><span>{_money(detail['booked'], cur)}</span></div>
        <div class=kv><span class=k>Proven</span>
          <span style='color:var(--good)'>{_money(detail['proven'], cur)}</span></div>
        <div class=kv><span class=k>Unproven gap</span>
          <span style='color:var(--warn)'>{_money(detail['gap'], cur)}</span></div>
        <div class=note>{_s(call['headline'])}</div>
      </div>
    </div>
    <h2>Orders</h2>
    <div class=tablewrap><table>
      <tr><th>Order</th><th class=right>Qty</th><th>Unit</th><th class=right>Total</th>
          <th class=right>Discount</th><th>Delivery</th><th>Status</th></tr>
      {orders}
    </table></div>
    <h2 style='margin-top:26px'>Claims and evidence chain</h2>
    {claims}"""
    return _page(f"{call['task'] or call['id']}", body)


# -- actions --------------------------------------------------------------


@app.post("/refresh")
def refresh(request: Request) -> Any:
    from fastapi.responses import RedirectResponse

    _, rep = ingest.refresh(store())
    if "application/json" in (request.headers.get("accept") or ""):
        return JSONResponse(rep.to_dict())
    return RedirectResponse("/", status_code=303)


@app.get("/export/{name}.csv")
def export_csv(name: str) -> Any:
    try:
        rows = q.run_export(store(), name)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return PlainTextResponse(
        ex.to_csv_string(rows),
        media_type="text/csv",
        headers={"content-disposition": f'attachment; filename="{name}.csv"'},
    )


@app.post("/export/{name}/sheets")
def export_sheets(name: str) -> JSONResponse:
    """Write the CSV, then try Sheets. Never fails because Sheets is missing."""
    try:
        result = ex.export(store(), name, to_sheets=True)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return JSONResponse(result.to_dict())


# -- JSON -----------------------------------------------------------------


@app.get("/api/overview")
def api_overview() -> JSONResponse:
    return JSONResponse(q.overview(store()))


@app.get("/api/calls")
def api_calls() -> JSONResponse:
    return JSONResponse(q.calls_table(store()))


@app.get("/api/call/{call_id}")
def api_call(call_id: str) -> JSONResponse:
    detail = q.call_detail(store(), call_id)
    if detail is None:
        return JSONResponse({"error": "no such call"}, status_code=404)
    return JSONResponse(detail)


@app.get("/api/orders")
def api_orders() -> JSONResponse:
    return JSONResponse(q.orders_by_verdict(store()))


@app.get("/api/revenue")
def api_revenue() -> JSONResponse:
    return JSONResponse(q.revenue_split(store()))


@app.get("/api/capacity")
def api_capacity() -> JSONResponse:
    return JSONResponse(q.weekly_commitments(store()))


@app.get("/api/discounts")
def api_discounts() -> JSONResponse:
    return JSONResponse(
        {"stats": q.discount_stats(store()), "orders": q.discount_rows(store())}
    )


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "rows": store().counts()})


def main() -> int:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8110")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
