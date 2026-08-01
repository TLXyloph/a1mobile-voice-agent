"""Decide whether an errand can be done on the web, or needs a phone call.

This is the step that makes the voice call defensible. Calling a business that
has a working online booking form is worse than useless - it wastes a human's
time to do something a form would have done silently. But a great many small
businesses (a dojo, a municipal golf course, a family dentist) have a website
that is a brochure and a phone number, and nothing else. Those are precisely
the errands a voice agent should own.

Two failure modes to keep distinct:

  - "Has a website" is not "is bookable on the web." Most small-business sites
    are brochures. Checking only for a domain sends everything to WEB and the
    agent completes nothing.
  - "Has a booking widget" is not "is bookable *for this intent*." A golf course
    may take tee times online but not junior clinic signups. The probe is
    therefore intent-specific, not site-generic.

When the probe is uncertain we route to PHONE. A needless call costs a few
minutes of someone's time; a wrongly-skipped errand silently fails to happen,
and under this hackathon's rules an errand that never happened scores zero.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Awaitable, Callable

from src.tasks.errands import Channel, Errand

logger = logging.getLogger("tasks.triage")

#: Booking-provider domains. Matched as domains, never as bare substrings -
#: "tock" appears inside "Stockton" and "resy" inside plenty of words, and a
#: false WEB routing is the dangerous direction: it silently skips a call that
#: needed making.
BOOKING_DOMAINS = (
    "opentable.com", "resy.com", "sevenrooms.com", "exploretock.com",
    "calendly.com", "acuityscheduling.com", "squarespace.com/scheduling",
    "setmore.com", "mindbodyonline.com", "wellnessliving.com",
    "zenplanner.com", "pushpress.com", "booksy.com", "square.site",
    "vagaro.com", "schedulicity.com", "teesnap.com", "golfnow.com",
    "foreup.com", "chronogolf.com", "yelp.com/reservations",
)

#: Phrases indicating self-service booking. Matched on word boundaries.
BOOKING_PHRASES = (
    "book now", "book online", "book a table", "schedule online",
    "reserve online", "request appointment", "online booking",
    "book appointment", "reserve a tee time", "sign up online",
)

_DOMAIN_RE = re.compile(
    "|".join(re.escape(d) for d in BOOKING_DOMAINS), re.IGNORECASE
)
_PHRASE_RE = re.compile(
    "|".join(rf"\b{re.escape(p)}\b" for p in BOOKING_PHRASES), re.IGNORECASE
)


def find_booking_signals(html: str, text: str) -> list[str]:
    """Signals that a site offers real self-service booking.

    Domains are looked for in raw markup (they live in href attributes);
    phrases in stripped text (so we don't match inside CSS or script blobs).
    """
    found = {m.group(0).lower() for m in _DOMAIN_RE.finditer(html)}
    found |= {m.group(0).lower() for m in _PHRASE_RE.finditer(text)}
    return sorted(found)

PHONE_RE = re.compile(
    r"(?:\+1[\s.-]?)?\(?([2-9]\d{2})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})"
)


def extract_phone(text: str) -> str | None:
    """Pull the first plausible US number out of page text."""
    m = PHONE_RE.search(text)
    if not m:
        return None
    return f"+1{m.group(1)}{m.group(2)}{m.group(3)}"


async def _default_fetch(url: str) -> str:
    import httpx

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=15.0,
        headers={"User-Agent": "Mozilla/5.0 (compatible; errand-agent/1.0)"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def probe_web_booking(
    url: str,
    intent: str,
    *,
    fetch: Callable[[str], Awaitable[str]] | None = None,
    use_llm: bool = True,
) -> tuple[bool, str, str | None]:
    """Can `intent` be completed on `url` without talking to anyone?

    Returns (bookable, reason, phone_found).

    The cheap signal scan runs first because it is free and catches the common
    case. The LLM pass only runs when signals are present, to answer the harder
    question of whether the widget covers *this* intent.
    """
    fetch = fetch or _default_fetch
    try:
        html = await fetch(url)
    except Exception as exc:  # noqa: BLE001
        return False, f"could not fetch site ({type(exc).__name__}) - assume phone", None

    # Strip script/style first: their contents are not user-visible text and
    # routinely contain third-party domains that mean nothing about booking.
    stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", stripped)
    text = re.sub(r"\s+", " ", text)

    phone = extract_phone(text)
    hits = find_booking_signals(html, text)

    if not hits:
        return False, "no self-service booking signals found on site", phone

    if not use_llm or not os.getenv("OPENAI_API_KEY"):
        return True, f"booking signals present: {', '.join(hits[:3])}", phone

    # Signals exist - but do they cover this specific intent?
    try:
        from openai import AsyncOpenAI

        resp = await AsyncOpenAI().chat.completions.create(
            model=os.getenv("TRIAGE_MODEL", "gpt-4.1-mini"),
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Decide whether a specific task can be completed on this "
                        "website without contacting a human. A booking widget that "
                        "does not cover the stated task means NO. Uncertainty means "
                        "NO - a wasted phone call is cheaper than a task that never "
                        'gets done. Return {"bookable": bool, "reason": str}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"TASK: {intent}\n"
                        f"BOOKING SIGNALS DETECTED: {hits[:6]}\n"
                        f"PAGE TEXT (truncated):\n{text[:3000]}"
                    ),
                },
            ],
        )
        verdict = json.loads(resp.choices[0].message.content or "{}")
        return (
            bool(verdict.get("bookable", False)),
            str(verdict.get("reason", "no reason given")),
            phone,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM triage failed, defaulting to phone: %s", exc)
        return False, f"triage inconclusive ({type(exc).__name__}) - assume phone", phone


async def triage(
    errand: Errand,
    *,
    resolve: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    fetch: Callable[[str], Awaitable[str]] | None = None,
) -> Errand:
    """Set `channel`, `phone`, `url` on an errand.

    Args:
        resolve: business name -> {"url": ..., "phone": ...}. Inject the Exa or
            Playwright MCP path at runtime; without it we can only use a URL
            already present on the errand.
    """
    if errand.url is None and resolve is not None:
        try:
            info = await resolve(errand.target)
            errand.url = info.get("url")
            errand.phone = errand.phone or info.get("phone")
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not resolve %s: %s", errand.target, exc)

    if errand.url is None:
        # No website at all is the strongest possible signal for a voice agent.
        errand.channel = Channel.PHONE if errand.phone else Channel.SKIP
        errand.reason = (
            "no website found; phone number known"
            if errand.phone
            else "no website and no phone number - cannot contact"
        )
        return errand

    bookable, reason, phone = await probe_web_booking(
        errand.url, errand.intent, fetch=fetch
    )
    errand.phone = errand.phone or phone
    errand.reason = reason

    if bookable:
        errand.channel = Channel.WEB
        errand.booking_url = errand.url
    elif errand.phone:
        errand.channel = Channel.PHONE
    else:
        errand.channel = Channel.SKIP
        errand.reason = f"{reason}; and no phone number found"

    logger.info(
        "triaged %r -> %s (%s)", errand.intent, errand.channel.value, errand.reason
    )
    return errand


async def triage_all(
    errands: list[Errand],
    *,
    resolve: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    fetch: Callable[[str], Awaitable[str]] | None = None,
) -> list[Errand]:
    """Triage concurrently. Network-bound, so parallelism is nearly free."""
    import asyncio

    return list(
        await asyncio.gather(
            *(triage(e, resolve=resolve, fetch=fetch) for e in errands)
        )
    )
