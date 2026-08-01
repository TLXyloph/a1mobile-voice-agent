"""Demo front pages built from what the agent learned on the call.

One vertical cold-calls small businesses that have no website and offers to
build one. The pitch is only credible if the proof arrives immediately, so the
agent ends the call, turns the transcript into `BusinessFacts`, renders a real
page, and emails a live URL. A judge clicking that URL is the verifiable side
effect - not something the agent can assert its way into.

**Nothing is invented.** `extract_facts` omits any field the owner did not say.
A page that guesses opening hours and mails them to the owner is worse than a
page with no hours section: the owner reads it as us making things up about
their business.

**The page must survive partial data.** Every section is conditional and there
is no placeholder copy anywhere in this file - if a fact is missing, its markup
is never emitted. A page with a name and nothing else still has to read as a
deliberate design rather than an unfinished template.

Output is one self-contained HTML file: inline CSS, system fonts, no scripts,
no remote assets. It renders with the wifi off, which matters in a room with
bad conference wifi. The colour system and stylesheet live in `theme.py`; they
know nothing about business facts and are re-exported here.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.deliver.theme import Palette, accent_color, palette_for, stylesheet

__all__ = [
    "BusinessFacts", "Service", "accent_color", "deploy", "extract_facts",
    "extract_facts_sync", "facts_from_payload", "palette_for", "render_site",
    "serve_sites", "slugify",
]

ROOT = Path(__file__).resolve().parents[2]
SITES_DIR = ROOT / "evidence" / "sites"
DEFAULT_PORT = 8090


@dataclass
class Service:
    """One thing the business sells. Only `name` is required."""

    name: str
    description: str = ""
    price: str = ""
    """Free text on purpose: owners say '$4.50', 'from $12', '45/hour'."""


@dataclass
class BusinessFacts:
    """What the agent managed to learn on one phone call.

    Everything except `name` is optional and frequently absent - a two-minute
    cold call does not yield a full brand brief. Empty fields are not rendered.
    """

    name: str
    tagline: str | None = None
    services: list[Service] = field(default_factory=list)
    hours: dict[str, str] | str | None = None
    """Either {'Mon-Fri': '7am-3pm'} or a single free-text line."""

    phone: str | None = None
    email: str | None = None
    address: str | None = None
    years_in_business: int | None = None
    specialties: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # The LLM path hands back plain dicts, callers hand back Service
        # objects. Accept both so neither side has to care.
        keep = ("name", "description", "price")
        services = []
        for s in self.services:
            if isinstance(s, Service):
                services.append(s)
                continue
            d = dict(s)
            if d.get("name"):
                services.append(Service(**{k: v for k, v in d.items() if k in keep and v}))
        self.services = services
        self.specialties = [s.strip() for s in self.specialties if s and s.strip()]

    @property
    def hour_rows(self) -> list[tuple[str, str]]:
        """Hours as (label, value) pairs. A bare string gets an empty label."""
        if not self.hours:
            return []
        if isinstance(self.hours, str):
            return [("", self.hours)]
        return [(k, v) for k, v in self.hours.items() if v]

    @property
    def has_contact(self) -> bool:
        return bool(self.phone or self.email or self.address)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tagline": self.tagline,
            "services": [
                {"name": s.name, "description": s.description, "price": s.price}
                for s in self.services
            ],
            "hours": self.hours,
            "phone": self.phone,
            "email": self.email,
            "address": self.address,
            "years_in_business": self.years_in_business,
            "specialties": self.specialties,
        }


EXTRACTION_PROMPT = """You extract facts about a small business from a
transcript of a sales call with its owner. The facts become a demo web page we
email to that owner, so a wrong fact is a lie printed on their own website.

The single rule: include a field ONLY if the owner said it on the call. Never
infer, never guess, never fill a field with a plausible default, never write
marketing copy from nothing. Omitting a field is free - that section simply
does not appear. Inventing one loses the customer.

Fields:
  name              - the business name as the owner says it. Required.
  tagline           - one short line, but ONLY as a compression of how the
                      owner themselves described the business. Else null.
  services          - [{"name","description","price"}]. description and price
                      may be "" if unstated. Prices exactly as spoken, e.g.
                      "$4.50", "from $12", "45/hr".
  hours             - object mapping a day or day-range to hours, e.g.
                      {"Mon-Fri": "7am-3pm", "Sat": "8am-2pm"}. null if
                      unstated. Do not complete a partial week.
  phone, email      - only if stated on the call.
  address           - street address as stated.
  years_in_business - integer, only if stated or directly computable from a
                      stated founding year.
  specialties       - short noun phrases the owner emphasised, e.g.
                      ["sourdough", "custom cakes"]. [] if none.

Return a JSON object with exactly those keys. Use null or [] for anything the
transcript does not support."""


async def extract_facts(transcript: str, *, model: str | None = None) -> BusinessFacts:
    """Pull business facts out of a call transcript.

    Async to match the rest of the codebase - the caller is the agent's
    post-call hook, already inside an event loop. `extract_facts_sync` wraps it.
    """
    from openai import AsyncOpenAI

    resp = await AsyncOpenAI().chat.completions.create(
        model=model or os.getenv("EXTRACT_MODEL", "gpt-4.1"),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": transcript[:20000]},
        ],
    )
    return facts_from_payload(json.loads(resp.choices[0].message.content or "{}"))


def extract_facts_sync(transcript: str, *, model: str | None = None) -> BusinessFacts:
    """Blocking wrapper for scripts."""
    import asyncio

    return asyncio.run(extract_facts(transcript, model=model))


def facts_from_payload(payload: dict[str, Any]) -> BusinessFacts:
    """Build `BusinessFacts` from raw model JSON, dropping anything unusable.

    Split out from the network call so the mapping is testable offline.
    """
    name = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("transcript yielded no business name; refusing to render")

    years = payload.get("years_in_business")
    try:
        years = int(years) if years is not None else None
    except (TypeError, ValueError):
        years = None

    hours = payload.get("hours")
    if isinstance(hours, dict):
        hours = {str(k): str(v) for k, v in hours.items() if v} or None

    return BusinessFacts(
        name=name,
        tagline=(payload.get("tagline") or "").strip() or None,
        services=payload.get("services") or [],
        hours=hours,
        phone=(payload.get("phone") or "").strip() or None,
        email=(payload.get("email") or "").strip() or None,
        address=(payload.get("address") or "").strip() or None,
        years_in_business=years,
        specialties=list(payload.get("specialties") or []),
    )


def _e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _monogram(name: str) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name) if w]
    if not words:
        return "&bull;"
    letters = "".join(w[0] for w in words[:2]) if len(words) > 1 else words[0][:2]
    return _e(letters.upper())


def _favicon(f: BusinessFacts, p: Palette) -> str:
    """Monogram favicon as a percent-encoded inline SVG.

    A `data:` URI makes no request, so the page stays offline-capable and a
    judge's console stays clean of a favicon 404.
    """
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
        f"<rect width='64' height='64' rx='14' fill='{p.accent_deep}'/>"
        "<text x='50%' y='50%' dy='.35em' text-anchor='middle' fill='#fff' "
        "font-family='Georgia,serif' font-size='30' font-weight='700'>"
        f"{_e(_monogram(f.name))}</text></svg>"
    )
    return quote(svg, safe="")


def _pretty_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone


def _hero(f: BusinessFacts, solo: bool) -> str:
    bits = [f'<div class="mark">{_monogram(f.name)}</div>']
    # The eyebrow slot takes the tenure if we have it, otherwise the whole
    # specialty list - one lone specialty above the name reads like a caption
    # for something missing, and one lone chip below looks like a bug.
    chips = list(f.specialties)
    if f.years_in_business:
        bits.append(f'<p class="kicker">{f.years_in_business} years in business</p>')
    elif chips:
        line = " &middot; ".join(_e(s) for s in chips)
        bits.append(f'<p class="kicker">{line}</p>')
        chips = []
    bits.append(f"<h1>{_e(f.name)}</h1>")

    foot = [f'<p class="lede">{_e(f.tagline)}</p>'] if f.tagline else []
    cta = []
    if f.phone:
        tel = _e(re.sub(r"[^\d+]", "", f.phone))
        cta.append(f'<a class="btn" href="tel:{tel}">Call {_e(_pretty_phone(f.phone))}</a>')
    if f.email:
        label = "Email us" if f.phone else f"Email {_e(f.email)}"
        cls = "btn ghost" if f.phone else "btn"
        cta.append(f'<a class="{cls}" href="mailto:{_e(f.email)}">{label}</a>')
    if cta:
        foot.append(f'<div class="cta">{"".join(cta)}</div>')

    if solo:
        bits.append('<div class="rule"></div>')
    if foot:
        bits.append(f'<div class="foot">{"".join(foot)}</div>')
    if chips:
        bits.append(f'<ul class="chips">{"".join(f"<li>{_e(s)}</li>" for s in chips)}</ul>')

    cls = "hero solo" if solo else "hero"
    return f'<header class="{cls}"><div class="wrap">{"".join(bits)}</div></header>'


def _services(f: BusinessFacts) -> str:
    if not f.services:
        return ""
    items = []
    for s in f.services:
        row = f"<h3>{_e(s.name)}</h3>"
        if s.price:
            row += f'<span class="dots"></span><span class="price">{_e(s.price)}</span>'
        body = f'<div class="row">{row}</div>'
        if s.description:
            body += f"<p>{_e(s.description)}</p>"
        items.append(f"<li>{body}</li>")
    return (
        '<section class="section" id="services"><div class="wrap">'
        '<div class="head"><p class="kicker">What we make</p><h2>Services</h2></div>'
        f'<ul class="menu">{"".join(items)}</ul></div></section>'
    )


def _visit(f: BusinessFacts) -> str:
    """Hours and contact share one band so either can stand alone gracefully."""
    cols = []
    rows = f.hour_rows
    if rows:
        if len(rows) == 1 and not rows[0][0]:
            body = f'<p class="lede">{_e(rows[0][1])}</p>'  # free-text hours
        else:
            body = '<dl class="hours">' + "".join(
                f"<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>" for k, v in rows
            ) + "</dl>"
        cols.append(
            '<div><p class="kicker">When</p><h2>Hours</h2>'
            f'<div class="pad">{body}</div></div>'
        )

    if f.has_contact:
        items = []
        if f.phone:
            tel = _e(re.sub(r"[^\d+]", "", f.phone))
            items.append(
                f'<li><span>Phone</span><a class="big" href="tel:{tel}">'
                f"{_e(_pretty_phone(f.phone))}</a></li>"
            )
        if f.email:
            items.append(
                f'<li><span>Email</span><a href="mailto:{_e(f.email)}">{_e(f.email)}</a></li>'
            )
        if f.address:
            lines = "<br>".join(
                _e(part.strip()) for part in f.address.split(",") if part.strip()
            )
            items.append(f"<li><span>Find us</span><address>{lines}</address></li>")
        cols.append(
            '<div><p class="kicker">Say hello</p><h2>Get in touch</h2>'
            f'<ul class="contact pad">{"".join(items)}</ul></div>'
        )

    if not cols:
        return ""
    return (
        '<section class="section alt" id="visit">'
        f'<div class="wrap cols">{"".join(cols)}</div></section>'
    )


def render_site(facts: BusinessFacts) -> str:
    """One self-contained HTML file. No external requests, ever."""
    p = palette_for(facts.name)
    body = [_services(facts), _visit(facts)]
    solo = not any(body)  # nothing but a name: centre it and mean it
    title = f"{facts.name} - {facts.tagline}" if facts.tagline else facts.name
    meta = (
        f'<meta name="description" content="{_e(facts.tagline)}">' if facts.tagline else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title>{meta}
<link rel="icon" href="data:image/svg+xml,{_favicon(facts, p)}">
<style>{stylesheet(p)}</style>
</head>
<body>
{_hero(facts, solo)}
{"".join(body)}
<footer><div class="wrap">
<div class="mark">{_monogram(facts.name)}</div>
<p>&copy; {datetime.now().year} {_e(facts.name)}</p>
</div></footer>
</body>
</html>
"""


def slugify(name: str) -> str:
    """Filesystem- and URL-safe slug. Traversal segments cannot survive this."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:60] or "site"


def deploy(html_text: str, slug: str, *, root: Path | None = None) -> str:
    """Write the page and return the URL a judge can click.

    `SITE_BASE_URL` swaps in an ngrok base so the link works from the
    recipient's phone; without it the link is local-only.
    """
    slug = slugify(slug)
    directory = (root or SITES_DIR) / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.html").write_text(html_text, encoding="utf-8")

    base = os.getenv("SITE_BASE_URL") or f"http://localhost:{DEFAULT_PORT}"
    return f"{base.rstrip('/')}/{slug}/"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # keep the demo console readable
        pass


_SERVERS: dict[int, ThreadingHTTPServer] = {}


def serve_sites(
    port: int = DEFAULT_PORT, *, root: Path | None = None, background: bool = True
) -> str:
    """Serve `evidence/sites` over HTTP and return the base URL.

    Pass `port=0` for an ephemeral port (tests). Calling twice on the same port
    reuses the running server rather than failing to bind.
    """
    directory = root or SITES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    if port and port in _SERVERS:
        return f"http://localhost:{port}"

    server = ThreadingHTTPServer(("0.0.0.0", port), partial(_QuietHandler, directory=str(directory)))
    bound = server.server_address[1]
    _SERVERS[bound] = server
    url = f"http://localhost:{bound}"

    if background:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return url

    print(f"serving {directory} at {url}")
    server.serve_forever()
    return url


if __name__ == "__main__":
    serve_sites(int(os.getenv("SITE_PORT", DEFAULT_PORT)), background=False)
