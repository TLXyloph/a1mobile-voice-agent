"""Triage must route correctly, and must fail toward PHONE when unsure.

`use_llm=False` throughout so these are deterministic and run offline; the LLM
pass only ever makes triage *stricter*, so passing here is a lower bound.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.errands import Channel, Errand  # noqa: E402
from src.tasks.triage import extract_phone, probe_web_booking, triage  # noqa: E402

DOJO_BROCHURE = """
<html><body>
<h1>Golden Dragon Tae Kwon Do</h1>
<p>Family martial arts in the Sunset since 1994.</p>
<p>Interested in a demo class? Call us at (415) 555-0142.</p>
<p>Hours: Mon-Fri 4pm-8pm</p>
</body></html>
"""

GOLF_WITH_BOOKING = """
<html><body>
<h1>Harding Park Golf Course</h1>
<a href="https://foreup.com/booking/harding">Book Now - Reserve Tee Times Online</a>
<p>Pro shop: 415-555-0199</p>
</body></html>
"""


def _fetch(html: str):
    async def fetcher(url: str) -> str:
        return html

    return fetcher


def _fail_fetch():
    async def fetcher(url: str) -> str:
        raise ConnectionError("dns failure")

    return fetcher


@pytest.mark.asyncio
async def test_brochure_site_routes_to_phone():
    """The core case: a real website that cannot actually book anything."""
    e = Errand(intent="schedule a tae kwon do demo class",
               target="Golden Dragon Tae Kwon Do", source="todo:1",
               url="https://example.com")
    await triage(e, fetch=_fetch(DOJO_BROCHURE))
    assert e.channel is Channel.PHONE
    assert e.phone == "+14155550142"


@pytest.mark.asyncio
async def test_site_with_booking_widget_routes_to_web():
    bookable, reason, phone = await probe_web_booking(
        "https://example.com", "book a tee time", fetch=_fetch(GOLF_WITH_BOOKING),
        use_llm=False,
    )
    assert bookable is True
    assert "foreup" in reason or "book now" in reason.lower()


@pytest.mark.asyncio
async def test_no_website_but_phone_routes_to_phone():
    e = Errand(intent="book a haircut", target="Joe's Barbers", source="todo:2",
               phone="+14155550100")
    await triage(e)
    assert e.channel is Channel.PHONE


@pytest.mark.asyncio
async def test_no_website_no_phone_is_skipped_not_guessed():
    e = Errand(intent="book something", target="Unknown Place", source="todo:3")
    await triage(e)
    assert e.channel is Channel.SKIP
    assert not e.is_actionable


@pytest.mark.asyncio
async def test_unreachable_site_falls_back_to_phone():
    """Fail toward calling: a skipped errand silently never happens."""
    e = Errand(intent="book a class", target="Somewhere", source="todo:4",
               url="https://down.example", phone="+14155550111")
    await triage(e, fetch=_fail_fetch())
    assert e.channel is Channel.PHONE


def test_phone_extraction_formats():
    assert extract_phone("call (415) 555-0142 today") == "+14155550142"
    assert extract_phone("415.555.0142") == "+14155550142"
    assert extract_phone("+1 415-555-0142") == "+14155550142"
    assert extract_phone("no number here") is None
    # Must not match an invalid area code starting with 0 or 1.
    assert extract_phone("call 155-5550142") is None


# -- regression: substring false positives --------------------------------
# "tock" inside "Stockton", "resy" inside other words, and provider domains
# appearing in analytics blobs all previously produced a false WEB routing,
# which silently skips a call that needed making.

BROCHURE_WITH_TRAPS = """
<html><head><style>.resylver{color:red}</style>
<script>var x="https://cdn.tracker.com/stock-analytics.js";</script></head>
<body>
<h1>Stockton Family Dentistry</h1>
<p>We are not currently taking online appointments.</p>
<p>Please call (415) 555-0177 to schedule.</p>
</body></html>
"""


@pytest.mark.asyncio
async def test_substrings_do_not_trigger_false_web_routing():
    bookable, reason, phone = await probe_web_booking(
        "https://example.com", "schedule a cleaning",
        fetch=_fetch(BROCHURE_WITH_TRAPS), use_llm=False,
    )
    assert bookable is False, f"false positive from substring match: {reason}"
    assert phone == "+14155550177"


@pytest.mark.asyncio
async def test_real_provider_domain_still_detected():
    html = '<a href="https://www.exploretock.com/restaurant">Reserve</a>'
    bookable, _, _ = await probe_web_booking(
        "https://example.com", "book a table", fetch=_fetch(html), use_llm=False,
    )
    assert bookable is True
