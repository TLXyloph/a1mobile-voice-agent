"""Generate a dashboard fitted to one task, by shelling out to headless Claude.

`claude -p "<prompt>"` is a code generator we do not control the output of. It
is fast, it writes better CSS than a template does, and it will occasionally
return prose, a truncated file, or a page that pulls Tailwind off a CDN and
renders blank on conference wifi.

So the subprocess is treated as untrusted input:

    timeout      it cannot hang the request
    size cap     it cannot fill the disk
    validate     structural and security checks before anything is written
    fall back    `builtin_dashboard()` ships instead of something broken

That last one is the actual design decision. **A generator that emits a broken
dashboard is worse than one that emits a plain one** - a blank page in front of
a judge reads as a system that does not work, while a plain page that renders
the verdicts correctly reads as a system that works.

The contract handed to Claude is narrow on purpose. One self-contained file, no
network of any kind, and it must render the claim -> evidence -> verdict model
from `src/verify/receipts.py` with booked-versus-proven as the headline. Booked
is what the agent says happened. Proven is what an independent channel
confirmed. Showing only the first is the fabrication the rules disqualify for,
so the two numbers appear side by side or the dashboard is not fit for purpose.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.generator.questions import QuestionSet
from src.generator.spec import TaskProfile, slugify

logger = logging.getLogger("generator.dashboard")

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "config" / "generated"

#: Wall-clock ceiling on one generation. Long enough for a full page, short
#: enough that a hung CLI does not eat the demo slot.
TIMEOUT_S = 240

#: Anything bigger than this is not a dashboard, it is a runaway.
MAX_BYTES = 600_000

#: URL prefixes that are namespace identifiers, not fetches. An inline `<svg
#: xmlns="http://www.w3.org/2000/svg">` is self-contained; rejecting it would
#: throw away every correct page that happens to draw an icon.
_NAMESPACE_URLS = (
    "http://www.w3.org/2000/svg",
    "http://www.w3.org/1999/xlink",
    "http://www.w3.org/1999/xhtml",
    "http://www.w3.org/XML/1998/namespace",
)

_URL = re.compile(r"https?://[^\s\"'<>)]*")
_PROTOCOL_RELATIVE = re.compile(r"""(?:src|href)\s*=\s*["']//""", re.I)
_SCRIPT_SRC = re.compile(r"<script[^>]*\ssrc\s*=", re.I)
_EXTERNAL_STYLESHEET = re.compile(r"<link[^>]*stylesheet", re.I)
_CSS_IMPORT = re.compile(r"@import\b", re.I)
_BODY = re.compile(r"<body[^>]*>(.*?)</body>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_FENCE = re.compile(r"```(?:html)?\s*(.*?)```", re.S)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@dataclass
class HtmlCheck:
    """`problems` means do not save it. `warnings` means save it and say so."""

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_html(html: str) -> HtmlCheck:
    """Everything wrong with a candidate dashboard.

    Split deliberately. A page that loads a font from Google is *broken* - it
    renders wrong offline and leaks the demo. A page that renders verdicts but
    words the headline differently is *imperfect*, and imperfect beats the
    fallback.
    """
    check = HtmlCheck()
    text = html or ""

    if len(text.strip()) < 400:
        check.problems.append("output is too short to be a page")
        return check
    if len(text.encode("utf-8", "ignore")) > MAX_BYTES:
        check.problems.append(f"output exceeds {MAX_BYTES} bytes")
        return check

    lowered = text.lower()
    if "<html" not in lowered and "<!doctype" not in lowered:
        check.problems.append("not an HTML document")

    external = [
        u for u in _URL.findall(text) if not u.startswith(_NAMESPACE_URLS)
    ]
    if external:
        check.problems.append(
            "external URLs (must be self-contained): "
            + ", ".join(sorted(set(external))[:4])
        )
    if _PROTOCOL_RELATIVE.search(text):
        check.problems.append("protocol-relative URL in a src/href")
    if _SCRIPT_SRC.search(text):
        check.problems.append("<script src=...> - scripts must be inline")
    if _EXTERNAL_STYLESHEET.search(text):
        check.problems.append("<link rel=stylesheet> - CSS must be inline")
    if _CSS_IMPORT.search(text):
        check.problems.append("@import - CSS must be inline")

    body = _BODY.search(text)
    if body is None:
        check.problems.append("no <body>")
    else:
        visible = _TAG.sub(" ", body.group(1))
        visible = re.sub(r"<!--.*?-->", " ", visible, flags=re.S)
        if len(visible.strip()) < 40:
            check.problems.append("empty body")

    verdicts = sum(
        1 for word in ("verified", "unverified", "contradicted") if word in lowered
    )
    if verdicts < 2 or "evidence" not in lowered:
        check.problems.append(
            "does not render the claim/evidence/verdict model from "
            "src/verify/receipts.py"
        )

    if not ("booked" in lowered and "proven" in lowered):
        check.warnings.append("headline does not say booked vs proven")
    if "independent" not in lowered:
        check.warnings.append("does not distinguish independent evidence")
    return check


def extract_html(raw: str) -> str:
    """Pull the document out of whatever the CLI actually printed."""
    text = (raw or "").strip()
    if not text:
        return ""
    if m := _FENCE.search(text):
        text = m.group(1).strip()
    lowered = text.lower()
    for marker in ("<!doctype", "<html"):
        idx = lowered.find(marker)
        if idx != -1:
            end = lowered.rfind("</html>")
            return text[idx : end + 7] if end > idx else text[idx:]
    return text


# ---------------------------------------------------------------------------
# the subprocess
# ---------------------------------------------------------------------------

Runner = Callable[[str], str]


def claude_available() -> bool:
    return shutil.which("claude") is not None


def run_claude(prompt: str, timeout: int = TIMEOUT_S) -> str:
    """Headless Claude, one shot. Returns "" on any failure at all.

    No shell, argv only, and the return code is not trusted to mean anything -
    only the bytes it printed, which are then validated like any other input.
    """
    if not claude_available():
        logger.warning("claude CLI is not on PATH")
        return ""
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        logger.warning("claude -p timed out after %ss", timeout)
        return ""
    except OSError as exc:
        logger.warning("claude -p could not run: %s", exc)
        return ""
    if proc.returncode != 0:
        logger.warning("claude -p exited %s: %s", proc.returncode, (proc.stderr or "")[:300])
    return proc.stdout or ""


CONTRACT = """
Write ONE self-contained HTML file: an operations dashboard for a single
autonomous phone errand. Output the file and nothing else - no explanation, no
markdown fence, no commentary before or after. Start at <!doctype html>.

## Hard constraints - a violation makes the file unusable

* NO network of any kind. No CDN, no external stylesheet, no external font, no
  remote image, no fetch to another host. Every byte inline. It must render
  identically opened from a local file with the wifi off.
* No <script src=...> and no <link rel="stylesheet">. Inline <style> and inline
  <script> only. Images, if any, are inline SVG.
* A non-empty <body>. A page that renders blank is worse than no page.

## What it must show - this is the point of the whole product

The data model lives in src/verify/receipts.py and it is not negotiable:

    Claim     something the agent asserts happened. Born UNVERIFIED.
    Evidence  an artifact from a channel, each marked independent or not.
    Verdict   VERIFIED / UNVERIFIED / CONTRADICTED, derived from the evidence.

An agent's own words are recorded as evidence on the AGENT_ASSERTION channel
and can never promote a claim. Only independent channels - inbound SMS, inbound
email, a provider API, a web check, DTMF, an independent transcript, a human
callback - can.

THE HEADLINE OF THE PAGE IS BOOKED VERSUS PROVEN. Two numbers, side by side, at
the top, in the largest type on the page: how many claims the agent BOOKED, and
how many are PROVEN by independent evidence. Use those two words. When the
numbers differ, the page must make the gap obvious rather than pleasant - the
gap is the honest part of the product. Below it, one row per claim showing the
verdict, the expected side effect, and each piece of evidence with its channel
and whether that channel is independent.

## Design

Considered and quiet. No purple gradients, no glassmorphism, no emoji. One
restrained accent colour. System font stack. Readable at arm's length from a
laptop on a table - a judge will be reading this over someone's shoulder.
Respect prefers-color-scheme for dark mode. Semantic colour for verdicts:
green for verified, amber for unverified, red for contradicted, and never
colour alone - each verdict also carries its word.

Include a small inline JSON array of 3 plausible example claims for this
specific task so the page renders with content, assigned to a
`window.__CLAIMS__` global that a later process can overwrite, and render the
page from it with inline JS. At least one example must be UNVERIFIED, because a
dashboard that has only ever shown success has never been tested.
""".strip()


def build_prompt(profile: TaskProfile, questions: QuestionSet | None = None) -> str:
    """The contract plus this specific task. Task first, so it is not skimmed."""
    fields = ""
    if questions is not None:
        fields = "\n".join(
            f"  - {q.field}: {q.ask}" for q in questions.questions
        )

    economics = (
        "This task HAS unit economics - price per unit and discount authority "
        "are real and belong on the dashboard."
        if profile.unit_economics_apply
        else "This task has NO unit economics. Do NOT put price, margin, cost "
        "or discount anywhere on the page - there is no money changing hands "
        "per unit and inventing those panels makes the dashboard read as a "
        "generic template."
    )

    return f"""{CONTRACT}

## The task this dashboard is for

Goal: {profile.goal}
Kind of exchange: {profile.exchange.value}
Who gets called: {profile.callee or "not yet specified"}
What is being exchanged: {profile.subject or profile.goal}
Definition of done: {profile.done_when or "not yet specified"}
Counted in: {profile.units}
Involves a quantity of physical items: {"yes" if profile.physical_goods else "no"}
Closes on: {profile.closes_on.value}

{economics}

The intake questions this task asks, which tell you what the operator knows and
therefore what the dashboard can display:
{fields or "  (not yet generated)"}

Title the page after this task, not "Dashboard".
""".strip()


# ---------------------------------------------------------------------------
# the fallback
# ---------------------------------------------------------------------------


def builtin_dashboard(
    profile: TaskProfile, questions: QuestionSet | None = None
) -> str:
    """The template that ships when generation fails. Plain, and correct.

    Deliberately holds itself to the same contract it asks Claude for, and
    `tests/test_generator.py` runs `validate_html` over its output - a fallback
    that fails its own validator is not a fallback.
    """
    economics_row = ""
    if profile.unit_economics_apply:
        economics_row = (
            '<div class="fact"><dt>Unit economics</dt>'
            f"<dd>priced per {_esc(profile.units)}</dd></div>"
        )

    asked = ""
    if questions is not None and questions.questions:
        asked = "".join(
            f"<li><code>{_esc(q.field)}</code><span>{_esc(q.ask)}</span></li>"
            for q in questions.questions
        )
        asked = f'<section class="card"><h2>Intake</h2><ol class="asked">{asked}</ol></section>'

    # `</script>` inside a JSON string literal still ends the script element -
    # the HTML tokenizer does not know it is inside a string. The goal text is
    # whatever the user typed, so escape the sequence rather than hope.
    claims = _json_for_script(
        [
            {
                "description": f"Contacted {profile.callee or 'the other party'} about: {profile.goal[:90]}",
                "expected": profile.done_when
                or "a written confirmation arrives on a channel we control",
                "verdict": "VERIFIED",
                "evidence": [
                    {"channel": "agent_assertion", "independent": False,
                     "summary": "Agent reported the request was accepted."},
                    {"channel": "inbound_sms", "independent": True,
                     "summary": "Reply SMS received confirming the details."},
                ],
            },
            {
                "description": "Details read back and agreed on the call",
                "expected": "the confirmation restates the same details",
                "verdict": "UNVERIFIED",
                "evidence": [
                    {"channel": "agent_assertion", "independent": False,
                     "summary": "Agent reported the details were read back."},
                ],
            },
            {
                "description": "Requested slot held",
                "expected": "the provider record shows the hold",
                "verdict": "CONTRADICTED",
                "evidence": [
                    {"channel": "agent_assertion", "independent": False,
                     "summary": "Agent reported the slot was held."},
                    {"channel": "provider_api", "independent": True,
                     "supports": False,
                     "summary": "No matching record found on the provider side."},
                ],
            },
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(profile.goal[:70] or "Errand")} - evidence</title>
<style>
  :root {{
    --paper: #faf8f4; --ink: #1a1815; --muted: #6d675e; --line: #e2ddd3;
    --card: #ffffff; --accent: #8a5a2b;
    --ok: #2f6b3f; --warn: #8a6a1f; --bad: #9b2c2c;
    --ok-bg: #e8f1e9; --warn-bg: #f6efdc; --bad-bg: #f7e6e4;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --paper: #14130f; --ink: #ece7dd; --muted: #9c968b; --line: #2e2b25;
      --card: #1c1a16; --accent: #d59b5f;
      --ok-bg: #16281a; --warn-bg: #2b2410; --bad-bg: #2c1614;
      --ok: #7fbc8c; --warn: #d8b45e; --bad: #e08b81;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 24px 64px; background: var(--paper); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); font-size: 13px; margin: 0 0 28px; }}
  .scoreboard {{
    display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;
    gap: 8px; background: var(--card); border: 1px solid var(--line);
    border-radius: 12px; padding: 26px 20px; margin-bottom: 8px;
  }}
  .score {{ text-align: center; }}
  .score b {{ display: block; font-size: 56px; line-height: 1; font-weight: 620;
              font-variant-numeric: tabular-nums; letter-spacing: -0.03em; }}
  .score span {{ display: block; margin-top: 6px; font-size: 11px; letter-spacing: .14em;
                 text-transform: uppercase; color: var(--muted); }}
  .vs {{ color: var(--muted); font-size: 12px; letter-spacing: .12em; }}
  .gap {{ text-align: center; font-size: 13px; color: var(--accent); margin: 0 0 28px;
          padding: 10px; border: 1px dashed var(--line); border-radius: 8px; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px;
           padding: 18px 20px; margin-bottom: 14px; }}
  h2 {{ font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
        color: var(--muted); margin: 0 0 12px; font-weight: 600; }}
  .claim {{ border-top: 1px solid var(--line); padding: 14px 0; }}
  .claim:first-of-type {{ border-top: 0; padding-top: 0; }}
  .claim h3 {{ font-size: 15px; margin: 0 0 4px; font-weight: 560; }}
  .expected {{ color: var(--muted); font-size: 13px; margin: 0 0 10px; }}
  .chip {{ display: inline-block; font-size: 10px; letter-spacing: .1em; font-weight: 700;
           padding: 3px 8px; border-radius: 999px; margin-bottom: 6px; }}
  .VERIFIED {{ background: var(--ok-bg); color: var(--ok); }}
  .UNVERIFIED {{ background: var(--warn-bg); color: var(--warn); }}
  .CONTRADICTED {{ background: var(--bad-bg); color: var(--bad); }}
  ul.ev {{ list-style: none; margin: 0; padding: 0; }}
  ul.ev li {{ display: flex; gap: 10px; font-size: 13px; padding: 4px 0;
              border-left: 2px solid var(--line); padding-left: 12px; }}
  ul.ev li.ind {{ border-left-color: var(--accent); }}
  ul.ev code {{ font-size: 11px; color: var(--muted); white-space: nowrap; }}
  .tag {{ font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
          color: var(--muted); }}
  dl.facts {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
              gap: 12px; margin: 0; }}
  .fact dt {{ font-size: 10px; letter-spacing: .1em; text-transform: uppercase;
              color: var(--muted); }}
  .fact dd {{ margin: 2px 0 0; font-size: 14px; }}
  ol.asked {{ margin: 0; padding-left: 20px; }}
  ol.asked li {{ padding: 5px 0; font-size: 13px; }}
  ol.asked code {{ display: block; font-size: 11px; color: var(--accent); }}
  footer {{ color: var(--muted); font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(profile.goal[:110] or "Errand")}</h1>
  <p class="sub">{_esc(profile.exchange.value)} &middot; closes on
     {_esc(profile.closes_on.value.replace("_", " "))} &middot;
     calling {_esc(profile.callee or "not yet specified")}</p>

  <div class="scoreboard">
    <div class="score"><b id="booked">0</b><span>Booked</span></div>
    <div class="vs">VS</div>
    <div class="score"><b id="proven">0</b><span>Proven</span></div>
  </div>
  <p class="gap" id="gap">&nbsp;</p>

  <section class="card">
    <h2>Claims and evidence</h2>
    <div id="claims"></div>
  </section>

  <section class="card">
    <h2>Task</h2>
    <dl class="facts">
      <div class="fact"><dt>Counted in</dt><dd>{_esc(profile.units)}</dd></div>
      <div class="fact"><dt>Physical items</dt>
        <dd>{"yes" if profile.physical_goods else "no"}</dd></div>
      <div class="fact"><dt>Done when</dt>
        <dd>{_esc(profile.done_when or "not yet specified")}</dd></div>
      {economics_row}
    </dl>
  </section>

  {asked}

  <footer>Booked is what the agent said. Proven is what an independent channel
  confirmed. Agent assertions are recorded and never count toward proven.</footer>
</div>
<script>
window.__CLAIMS__ = window.__CLAIMS__ || {claims};
(function () {{
  // The goal text is whatever the user typed. It reaches innerHTML, so it
  // gets escaped on the way in - a dashboard is not a place to be relaxed
  // about that just because the file is local.
  function esc(s) {{
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {{
      return {{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}}[c];
    }});
  }}
  var claims = window.__CLAIMS__ || [];
  var proven = claims.filter(function (c) {{ return c.verdict === "VERIFIED"; }}).length;
  document.getElementById("booked").textContent = claims.length;
  document.getElementById("proven").textContent = proven;
  var gap = claims.length - proven;
  document.getElementById("gap").textContent = gap === 0
    ? "Every booked claim is proven by independent evidence."
    : gap + " of " + claims.length + " booked claim(s) are not proven. Reported as unconfirmed.";

  document.getElementById("claims").innerHTML = claims.map(function (c) {{
    var ev = (c.evidence || []).map(function (e) {{
      var supports = e.supports === false ? "contradicts" : "supports";
      return '<li class="' + (e.independent ? "ind" : "") + '">' +
             '<code>' + esc(e.channel) + '</code>' +
             '<span>' + esc(e.summary) + ' <span class="tag">' +
             (e.independent ? "independent" : "agent only") + ' &middot; ' + supports +
             '</span></span></li>';
    }}).join("");
    var verdict = ["VERIFIED", "UNVERIFIED", "CONTRADICTED"].indexOf(c.verdict) < 0
      ? "UNVERIFIED" : c.verdict;
    return '<div class="claim"><div class="chip ' + verdict + '">' + verdict +
           '</div><h3>' + esc(c.description) + '</h3>' +
           '<p class="expected">Expected side effect: ' + esc(c.expected) + '</p>' +
           '<ul class="ev">' + ev + '</ul></div>';
  }}).join("");
}})();
</script>
</body>
</html>
"""


def _json_for_script(value: object) -> str:
    """JSON safe to embed in an inline `<script>`."""
    return (
        json.dumps(value)
        .replace("</", "<\\/")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


@dataclass
class DashboardResult:
    path: Path
    html: str
    source: str
    """"claude" or "builtin". Shown in the UI - nobody should have to guess."""

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "source": self.source,
            "problems": self.problems,
            "warnings": self.warnings,
            "seconds": round(self.seconds, 1),
            "bytes": len(self.html),
        }


def target_dir(profile: TaskProfile, out_root: Path | None = None) -> Path:
    """`config/generated/<slug>/`, guaranteed to be under `out_root`.

    `slugify` cannot emit a separator or a dot segment, and the containment
    check is asserted anyway - path handling is not a place to rely on one
    layer being correct.
    """
    root = (out_root or OUT_ROOT).resolve()
    slug = slugify(profile.goal, fallback=profile.exchange.value)
    path = (root / slug).resolve()
    if root not in path.parents and path != root:
        raise ValueError(f"refusing to write outside {root}")
    return path


def generate_dashboard(
    profile: TaskProfile,
    questions: QuestionSet | None = None,
    *,
    runner: Runner | None = None,
    out_root: Path | None = None,
    timeout: int = TIMEOUT_S,
    write: bool = True,
) -> DashboardResult:
    """Generate, validate, and save - or save the fallback. Never raises.

    `runner` is the seam: tests pass a function, so the suite never shells out.
    """
    started = time.monotonic()
    prompt = build_prompt(profile, questions)
    run = runner if runner is not None else (lambda p: run_claude(p, timeout))

    html, source, problems, warnings = "", "builtin", [], []
    try:
        raw = run(prompt)
    except Exception as exc:  # noqa: BLE001 - the subprocess is untrusted
        logger.warning("dashboard runner raised %s", exc)
        raw = ""
        problems = [f"generator raised {type(exc).__name__}"]

    candidate = extract_html(raw)
    if candidate:
        check = validate_html(candidate)
        if check.ok:
            html, source, warnings = candidate, "claude", check.warnings
        else:
            problems = check.problems
            logger.warning("rejected generated dashboard: %s", "; ".join(check.problems))
    elif not problems:
        problems = ["generator produced no HTML"]

    if not html:
        html = builtin_dashboard(profile, questions)

    path = target_dir(profile, out_root) / "dashboard.html"
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        payload = {
            "profile": profile.to_dict(),
            "questions": questions.to_dict() if questions else None,
            "dashboard_source": source,
            "rejected_because": problems,
        }
        (path.parent / "profile.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    return DashboardResult(
        path=path,
        html=html,
        source=source,
        problems=problems,
        warnings=warnings,
        seconds=time.monotonic() - started,
    )
