"""The look: one hue per business, and the stylesheet built from it.

Split out of `sitegen` because it is a self-contained design system - nothing
here knows what a business fact is. The two rules it enforces are that the
colour is derived, never chosen at random (so a page is stable across
regenerations), and that the result never lands in the purple-gradient register
that reads as machine-generated.
"""

from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass

#: Hue bands that read as considered brand colours. 240-330 is excluded on
#: purpose: the purple-to-indigo gradient is the tell of a generated page, and
#: this one has to pass as a page a designer was paid for.
_HUE_BANDS: tuple[tuple[int, int], ...] = (
    (8, 34),  # brick, terracotta
    (34, 52),  # amber, ochre
    (96, 152),  # olive through forest
    (156, 196),  # teal
    (200, 234),  # slate blue
)


def _hsl(h: float, s: float, lightness: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h / 360.0, lightness, s)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


@dataclass(frozen=True)
class Palette:
    """A whole page's colour, derived from one hue."""

    accent: str
    accent_deep: str
    tint: str
    ink: str
    muted: str
    paper: str
    surface: str
    line: str


def palette_for(name: str) -> Palette:
    """Deterministic palette from the business name.

    The same name always gives the same site; two businesses in one demo reel
    look unrelated, which is the point.
    """
    digest = hashlib.sha256(name.strip().lower().encode()).digest()
    lo, hi = _HUE_BANDS[digest[0] % len(_HUE_BANDS)]
    hue = lo + (digest[1] / 255.0) * (hi - lo)
    return Palette(
        accent=_hsl(hue, 0.55, 0.40),
        accent_deep=_hsl(hue, 0.62, 0.27),
        tint=_hsl(hue, 0.42, 0.93),
        ink=_hsl(hue, 0.22, 0.11),
        muted=_hsl(hue, 0.10, 0.40),
        paper=_hsl(hue, 0.30, 0.985),
        surface=_hsl(hue, 0.26, 0.965),
        line=_hsl(hue, 0.20, 0.86),
    )


def accent_color(name: str) -> str:
    """The one colour a viewer would name if asked about the site."""
    return palette_for(name).accent


def stylesheet(p: Palette) -> str:
    """The whole stylesheet. Two families, one hue, a lot of air."""
    return f"""
*,*::before,*::after{{box-sizing:border-box}}
:root{{--accent:{p.accent};--deep:{p.accent_deep};--tint:{p.tint};--ink:{p.ink};--muted:{p.muted};--paper:{p.paper};--surface:{p.surface};--line:{p.line};--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;--serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif}}
html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--paper);color:var(--ink);font:400 17px/1.65 var(--sans);-webkit-font-smoothing:antialiased}}
::selection{{background:var(--accent);color:#fff}}
a{{color:var(--deep);text-underline-offset:.22em;text-decoration-thickness:1px}}
p{{margin:0}}
h1,h2,h3{{margin:0;font-family:var(--serif);font-weight:600;letter-spacing:-.02em;line-height:1.08}}
.wrap{{width:min(1080px,100% - 3rem);margin-inline:auto}}
.kicker{{margin:0 0 .85rem;font-size:.72rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--accent)}}
.mark{{display:grid;place-items:center;width:54px;height:54px;border-radius:15px;background:var(--deep);color:#fff;font:700 1rem/1 var(--sans);letter-spacing:.06em}}
/* Hero padding-bottom is small on purpose: the next .section brings its own
   generous padding-top, and stacking both leaves a dead band. */
.hero{{border-top:6px solid var(--deep);padding:clamp(3rem,8vw,6rem) 0 clamp(1.5rem,3vw,2.5rem);background:linear-gradient(172deg,var(--tint) 0%,var(--paper) 58%)}}
.hero h1{{margin:1.6rem 0 0;font-size:clamp(2.6rem,7vw,4.4rem)}}
.hero .kicker{{margin:2.2rem 0 0}}
/* Lede and buttons share one baseline-aligned row so a wide hero does not
   leave half its width empty. Collapses to stacked on narrow screens. */
.foot{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:1.6rem 3rem;margin-top:1.9rem}}
.lede{{max-width:38ch;margin:0;font-size:clamp(1.12rem,2.2vw,1.4rem);line-height:1.5;color:var(--muted)}}
.cta{{display:flex;flex-wrap:wrap;gap:.75rem}}
.btn{{display:inline-flex;align-items:center;padding:.95rem 1.7rem;border-radius:999px;background:var(--deep);color:#fff;text-decoration:none;font-weight:600;font-size:.98rem;box-shadow:0 1px 2px rgba(0,0,0,.12);transition:transform .12s ease,box-shadow .12s ease}}
.btn:hover{{transform:translateY(-1px);box-shadow:0 6px 18px rgba(0,0,0,.14)}}
.btn.ghost{{background:transparent;color:var(--deep);box-shadow:inset 0 0 0 1.5px var(--line)}}
.btn.ghost:hover{{box-shadow:inset 0 0 0 1.5px var(--deep)}}
.chips{{display:flex;flex-wrap:wrap;gap:.5rem;margin:2.6rem 0 0;padding:0;list-style:none}}
.chips li{{padding:.35rem .9rem;border-radius:999px;background:rgba(255,255,255,.7);border:1px solid var(--line);font-size:.85rem;color:var(--deep)}}
.section{{padding:clamp(3.5rem,8vw,6rem) 0}}
.section.alt{{background:var(--surface);border-block:1px solid var(--line)}}
.section h2{{font-size:clamp(1.8rem,3.4vw,2.6rem)}}
.head{{max-width:38ch;margin-bottom:3rem}}
.menu{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));column-gap:4.5rem;margin:0;padding:0;list-style:none}}
.menu li{{padding:1.5rem 0;border-top:1px solid var(--line)}}
.menu p{{margin-top:.5rem;color:var(--muted);font-size:.96rem;max-width:44ch}}
.row{{display:flex;align-items:baseline;gap:1rem}}
.row h3{{font-family:var(--sans);font-size:1.06rem;letter-spacing:0;line-height:1.35}}
.dots{{flex:1;border-bottom:1px dotted var(--line);transform:translateY(-.28em)}}
.price{{font-weight:600;color:var(--accent);font-variant-numeric:tabular-nums;white-space:nowrap}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:clamp(2.5rem,6vw,5rem)}}
.pad{{margin-top:1.8rem}}
.hours{{margin:0}}
.hours div{{display:flex;justify-content:space-between;gap:2rem;padding:.72rem 0;border-bottom:1px solid var(--line)}}
.hours dt{{font-weight:600}}
.hours dd{{margin:0;color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}}
.contact{{margin:0;padding:0;list-style:none}}
.contact li{{padding:.9rem 0;border-bottom:1px solid var(--line)}}
.contact span{{display:block;margin-bottom:.35rem;font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}}
.contact a,.contact address{{font-size:1.12rem;font-style:normal;line-height:1.45}}
.big{{font-family:var(--serif);font-size:clamp(1.5rem,3vw,2rem);text-decoration:none;letter-spacing:-.01em}}
.big:hover{{text-decoration:underline}}
footer{{padding:2.5rem 0 3.5rem;border-top:1px solid var(--line);color:var(--muted);font-size:.88rem}}
footer .wrap{{display:flex;flex-wrap:wrap;align-items:center;gap:1rem 1.5rem}}
footer .mark{{width:38px;height:38px;border-radius:11px;font-size:.8rem}}
/* Name-only pages: centre everything and let the page be a calling card. */
.solo{{min-height:88vh;display:grid;align-content:center;text-align:center}}
.solo .mark{{margin-inline:auto}}
.solo .foot{{justify-content:center}}
.solo .lede,.solo .cta,.solo .chips{{justify-content:center;margin-inline:auto}}
.solo .rule{{width:64px;height:3px;margin:2.2rem auto 0;background:var(--deep);border-radius:2px}}
@media (max-width:640px){{
body{{font-size:16px}}
.wrap{{width:min(1080px,100% - 2.25rem)}}
.menu{{grid-template-columns:1fr;column-gap:0}}
}}
"""
