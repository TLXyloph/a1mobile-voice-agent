"""A page to talk to the intake agent before deciding whether to keep it.

The MCP server in `src/mcp/intake_server.py` only speaks JSON-RPC over stdio,
which is the right transport for a host and a useless one for a human trying to
judge whether the questions are any good. This serves the same six tools over
HTTP with a chat UI in front of them.

It calls the real functions. Nothing is reimplemented here - if a question reads
badly on this page, it reads badly on the call.

**Saving is a dry run by default.** `save_profile` is pointed at a sandbox under
`evidence/intake_preview/`, seeded with the real `config/.env`'s keys, comments
and line order but with every credential value redacted. You get the exact lines
it would rewrite and a diff proving it left every protected line alone, without
a second copy of your API keys existing on disk. Nothing under `config/` is
touched unless you pass --live.

    .venv/bin/python scripts/intake_playground.py          # dry run, port 8765
    .venv/bin/python scripts/intake_playground.py --live    # write config/ for real
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mcp import intake_server as tools  # noqa: E402
from src.mcp.intake_store import PROTECTED_KEY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ENV = REPO_ROOT / "config" / ".env"
SANDBOX = REPO_ROOT / "evidence" / "intake_preview"

app = FastAPI(title="intake playground")
STATE: dict[str, Any] = {"live": False}


def _redacted(text: str) -> str:
    """The real .env's shape - every key, comment and line position - no values.

    The sandbox has to look like the real file for the in-place update to prove
    anything, but copying live credentials to a second path just to demonstrate
    that we do not overwrite them would be its own small leak. Keys and layout
    are what the writer navigates by; the values are not needed.
    """
    out = []
    for line in text.splitlines():
        key = line.split("=", 1)[0]
        if "=" in line and PROTECTED_KEY.search(key):
            out.append(f"{key}=<redacted-for-sandbox>")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def use_sandbox() -> None:
    """Point the server's two write targets at a redacted copy, not at config/."""
    SANDBOX.mkdir(parents=True, exist_ok=True)
    sandbox_env = SANDBOX / ".env"
    seed = _redacted(REAL_ENV.read_text()) if REAL_ENV.exists() else "# no config/.env found\n"
    sandbox_env.write_text(seed)
    STATE["seed"] = seed
    tools.ENV_PATH = sandbox_env
    tools.PROFILE_PATH = SANDBOX / "business_profile.json"


def credential_audit() -> dict[str, Any]:
    """Did the write touch a single protected line? Answered by comparison.

    Seed versus result, which is the honest test of `write_env`: every line it
    was handed and did not mean to change must come back byte-identical.
    """
    if STATE["live"] or not STATE.get("seed"):
        return {"checked": False}
    before = STATE["seed"].splitlines()
    after = tools.ENV_PATH.read_text().splitlines()

    def secrets(lines: list[str]) -> set[str]:
        return {ln for ln in lines if "=" in ln and PROTECTED_KEY.search(ln.split("=")[0])}

    lost = secrets(before) - secrets(after)
    return {
        "checked": True,
        "protected_lines": len(secrets(before)),
        "protected_lines_intact": not lost,
        "altered": sorted(lost),
        "diff": [
            line
            for line in difflib.unified_diff(before, after, "config/.env", "after save", n=1)
        ][:80],
    }


@app.get("/api/state")
def state() -> JSONResponse:
    status = tools.intake_status()
    return JSONResponse(
        {
            "status": status,
            "live": STATE["live"],
            "env_path": str(tools.ENV_PATH),
            "profile_path": str(tools.PROFILE_PATH),
        }
    )


@app.post("/api/start")
async def start(body: dict) -> JSONResponse:
    return JSONResponse(tools.start_intake(body.get("business_name", ""), body.get("vertical", "")))


@app.post("/api/answer")
async def answer(body: dict) -> JSONResponse:
    return JSONResponse(tools.answer(body.get("field", ""), body.get("value", "")))


@app.post("/api/document")
async def document(body: dict) -> JSONResponse:
    return JSONResponse(tools.parse_document(body.get("path", "")))


@app.post("/api/save")
async def save(_: dict | None = None) -> JSONResponse:
    result = tools.save_profile()
    if result.get("saved"):
        result["credential_audit"] = credential_audit()
        result["env_preview"] = tools.ENV_PATH.read_text().splitlines()[-24:]
        result["profile_json"] = json.loads(tools.PROFILE_PATH.read_text())
    return JSONResponse(result)


@app.post("/api/reset")
async def reset(_: dict | None = None) -> JSONResponse:
    tools.reset_session()
    return JSONResponse({"ok": True})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


PAGE = """
<!doctype html><meta charset=utf-8><title>Intake agent - try it</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--card:#161b22;--line:#30363d;
--ok:#3fb950;--bad:#f85149;--accent:#58a6ff}
@media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#1f2328;--dim:#59636e;
--card:#f6f8fa;--line:#d1d9e0;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px;display:grid;
grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:20px}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
h1{font-size:19px;margin:0 0 2px}.sub{color:var(--dim);font-size:13px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.chat{min-height:340px;max-height:56vh;overflow-y:auto;display:flex;flex-direction:column;gap:10px}
.msg{max-width:88%;padding:9px 12px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
.bot{background:rgba(88,166,255,.12);border:1px solid rgba(88,166,255,.3);align-self:flex-start}
.me{background:rgba(63,185,80,.13);border:1px solid rgba(63,185,80,.3);align-self:flex-end}
.err{background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.35);align-self:flex-start}
.why{font-size:12px;color:var(--dim);margin-top:5px;font-style:italic}
.row{display:flex;gap:8px;margin-top:12px}
input,button{font:inherit;border-radius:8px;border:1px solid var(--line);padding:9px 11px}
input{flex:1;background:var(--bg);color:var(--fg)}
button{background:var(--accent);color:#fff;border:0;cursor:pointer;font-weight:600}
button.ghost{background:transparent;color:var(--fg);border:1px solid var(--line);font-weight:500}
button:disabled{opacity:.45;cursor:default}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;
overflow-x:auto;font-size:12px;margin:0}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:3px 0;vertical-align:top}td:first-child{color:var(--dim);padding-right:10px;white-space:nowrap}
.pill{display:inline-block;font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--dim)}
.ok{color:var(--ok)}.bad{color:var(--bad)}h2{font-size:13px;text-transform:uppercase;
letter-spacing:.05em;color:var(--dim);margin:0 0 9px}
.add{color:var(--ok)}.del{color:var(--bad)}
</style>
<div class=wrap>
 <div>
  <h1>Intake agent</h1>
  <div class=sub>Talks to the real MCP tools in <code>src/mcp/intake_server.py</code>.
   <span class=pill id=mode>dry run</span></div>
  <div class=card><div class=chat id=chat></div>
   <div class=row><input id=box placeholder="Answer here, then press Enter" autofocus>
    <button id=send>Send</button></div>
   <div class=row><button class=ghost id=save disabled>Save profile</button>
    <button class=ghost id=reset>Start over</button>
    <input id=doc placeholder="or a price-sheet path to parse (.csv/.xlsx/.md)" style="flex:1">
    <button class=ghost id=parse>Parse</button></div>
  </div>
 </div>
 <div>
  <div class=card><h2>What it knows</h2><table id=known><tr><td>nothing yet</td></tr></table></div>
  <div class=card><h2>Derived config</h2><pre id=derived>Finish the interview to see the
CostModel / CapacityLedger / Envelope it builds.</pre></div>
  <div class=card><h2>Would write to config/.env</h2><pre id=env>Not saved yet.</pre></div>
 </div>
</div>
<script>
const $=s=>document.querySelector(s), chat=$('#chat');
let field=null, started=false;

function say(text,cls,why){
  const d=document.createElement('div'); d.className='msg '+cls; d.textContent=text;
  if(why){const w=document.createElement('div');w.className='why';w.textContent='why: '+why;d.appendChild(w);}
  chat.appendChild(d); chat.scrollTop=chat.scrollHeight;
}
const post=(u,b)=>fetch(u,{method:'POST',headers:{'content-type':'application/json'},
  body:JSON.stringify(b||{})}).then(r=>r.json());

function render(r){
  if(r.instruction) say(r.instruction,'err');
  if(r.summary) say(r.summary,'bot');
  if(r.question){ field=r.field; say(r.question,'bot',r.why); }
  if(r.done){ field=null; $('#save').disabled=false;
    $('#derived').textContent=JSON.stringify(r.derived,null,2); }
  refresh();
}
async function refresh(){
  const s=await fetch('/api/state').then(r=>r.json());
  $('#mode').textContent = s.live ? 'LIVE - writes config/' : 'dry run - sandbox only';
  $('#mode').className = 'pill '+(s.live?'bad':'ok');
  const k=s.status.known||{}; const rows=Object.entries(k);
  $('#known').innerHTML = rows.length
    ? rows.map(([a,b])=>`<tr><td>${a}</td><td>${b===''?'<i>none</i>':b}</td></tr>`).join('')
    : '<tr><td>nothing yet</td></tr>';
}
async function send(){
  const v=$('#box').value.trim(); if(!v) return; $('#box').value='';
  say(v,'me');
  if(!started){
    const parts=v.split(',');
    const r=await post('/api/start',{business_name:parts[0].trim(),
      vertical:(parts[1]||'small business').trim()});
    if(r.ok) started=true;
    return render(r);
  }
  if(!field){ say('Interview is done - press Save profile.','bot'); return; }
  render(await post('/api/answer',{field:field,value:v}));
}
$('#send').onclick=send;
$('#box').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
$('#parse').onclick=async()=>{
  const p=$('#doc').value.trim(); if(!p) return;
  const r=await post('/api/document',{path:p});
  if(!r.ok){ say(r.instruction,'err'); return; }
  say('Read '+p+'\\n\\nApplied: '+(Object.keys(r.applied||{}).join(', ')||'nothing')
    +'\\nStill missing: '+(r.still_missing||r.missing).join(', ')
    +'\\n\\n'+(r.notes||[]).join('\\n'),'bot');
  if(r.next) render(r.next); else refresh();
};
$('#save').onclick=async()=>{
  const r=await post('/api/save',{});
  if(!r.saved){ say(r.instruction,'err'); return; }
  const a=r.credential_audit||{};
  say('Saved.\\nprofile: '+r.profile_path+'\\nenv: '+r.env_path
    +'\\n\\nupdated in place: '+(r.env_written.updated_in_place.join(', ')||'none')
    +'\\nappended: '+(r.env_written.appended.join(', ')||'none')
    +(a.checked?('\\n\\ncredential lines in config/.env: '+a.protected_lines
      +'\\nall intact after write: '+(a.protected_lines_intact?'YES':'NO - '+a.altered)):''),'bot');
  $('#env').innerHTML=(a.diff&&a.diff.length?a.diff:r.env_preview)
    .map(l=>{const e=l.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
      return l.startsWith('+')?`<span class=add>${e}</span>`
           : l.startsWith('-')?`<span class=del>${e}</span>` : e;}).join('\\n');
};
$('#reset').onclick=async()=>{await post('/api/reset',{});chat.innerHTML='';started=false;
  field=null;$('#save').disabled=true;boot();};
function boot(){
  say("Hi - I set up the pricing and capacity config for your business by asking you "
    +"about it.\\n\\nTo begin, tell me the business name and what kind of business it is, "
    +"like: Rosewater Bakehouse, bakery",'bot');
  refresh();
}
boot();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--live",
        action="store_true",
        help="write config/business_profile.json and config/.env for real",
    )
    args = parser.parse_args()

    STATE["live"] = args.live
    if not args.live:
        use_sandbox()

    import uvicorn

    target = "config/ (LIVE)" if args.live else str(SANDBOX)
    print(f"intake playground -> http://127.0.0.1:{args.port}   writes to: {target}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
