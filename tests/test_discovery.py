"""Discovery must be biased toward excluding, and must say why.

The expensive error is not a missed lead. It is phoning a business to tell
them they need a website when they already have a working one - the call is
over, and so is the credibility of the next call from that number. So the
assertions below care as much about what is *not* flagged, and about the
exclusion reasons, as about the leads themselves.

Fetchers are stubbed throughout, so this runs offline and deterministically.
`probe_web_booking` is exercised for real underneath - only the network is
faked.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.business.discovery import (  # noqa: E402
    SHEET_COLUMNS,
    Disqualified,
    Lead,
    find_no_website_leads,
    qualify_business,
    screen_businesses,
    to_sheet,
)

# -- fixtures: real-shaped pages -------------------------------------------

BROCHURE = """
<html><body>
<h1>Golden Dragon Tae Kwon Do</h1>
<p>Family martial arts in the Sunset since 1994.</p>
<p>Classes for kids and adults. Call (415) 555-0142 to ask about a demo.</p>
<p>Hours: Mon-Fri 4pm-8pm</p>
</body></html>
"""

MULTI_PAGE_BROCHURE = """
<html><body>
<nav><a href="/about">About</a> <a href="/classes">Classes</a>
<a href="/gallery">Gallery</a></nav>
<h1>Sunset Family Dentistry</h1>
<p>Gentle dentistry for the whole family. Serving the neighbourhood since 2003.</p>
<p>We are not currently taking appointments over the internet.</p>
<p>Please telephone (415) 555-0177 during office hours.</p>
</body></html>
"""

RICH_RESTAURANT = """
<html><body>
<nav><a href="/menu">Menu</a> <a href="/about">About</a>
<a href="/catering">Catering</a> <a href="/hours">Hours</a></nav>
<h1>Blue Plate Kitchen</h1>
<a href="https://www.toasttab.com/blue-plate/v3">Order Online</a>
<p>Add to cart and pay online for pickup in 20 minutes.</p>
<form method="post" action="/subscribe">
  <input type="email" name="email"><input type="submit" value="Join">
</form>
<p>Seasonal Californian cooking, seven days a week. Call 415-555-0155.</p>
</body></html>
"""

SITE_WITH_BOOKING = """
<html><body>
<h1>Ocean Avenue Pilates</h1>
<a href="https://clients.mindbodyonline.com/classic/ws?studioid=1234">Schedule</a>
<p>Reformer and mat classes. Studio line: (415) 555-0188.</p>
</body></html>
"""

SUBSTANTIAL_STATIC = """
<html><body>
<nav>
<a href="/about">About</a><a href="/team">Team</a><a href="/services">Services</a>
<a href="/pricing">Pricing</a><a href="/careers">Careers</a>
<a href="/press">Press</a><a href="/faq">FAQ</a><a href="/locations">Locations</a>
</nav>
<h1>Harbour Point Veterinary Group</h1>
<p>Four locations, twelve veterinarians, and a fully equipped surgical suite.</p>
<p>Telephone (415) 555-0166.</p>
</body></html>
"""


def _fetch(pages: dict[str, str]):
    """Stub fetcher over a url -> markup map. Unknown urls fail like the net."""

    async def fetcher(url: str) -> str:
        if url not in pages:
            raise ConnectionError(f"no stub for {url}")
        return pages[url]

    return fetcher


# -- the dangerous direction: a good site must never become a lead ---------


@pytest.mark.asyncio
async def test_rich_functional_site_is_not_a_lead():
    result = await qualify_business(
        {"name": "Blue Plate Kitchen", "phone": "+14155550155",
         "website": "https://blueplate.example"},
        fetch=_fetch({"https://blueplate.example": RICH_RESTAURANT}),
    )
    assert isinstance(result, Disqualified), getattr(result, "qualification_reason", "")
    assert "working functionality" in result.reason


@pytest.mark.asyncio
async def test_site_with_a_booking_widget_is_not_a_lead():
    result = await qualify_business(
        {"name": "Ocean Avenue Pilates", "phone": "+14155550188",
         "website": "https://oceanpilates.example"},
        fetch=_fetch({"https://oceanpilates.example": SITE_WITH_BOOKING}),
    )
    assert isinstance(result, Disqualified)
    assert "mindbodyonline" in result.reason


@pytest.mark.asyncio
async def test_unfetchable_site_is_excluded_not_assumed_bad():
    """We could not look, so we cannot claim. A blocked user agent is not a
    missing website."""
    result = await qualify_business(
        {"name": "Somewhere", "phone": "+14155550111",
         "website": "https://down.example"},
        fetch=_fetch({}),
    )
    assert isinstance(result, Disqualified)
    assert "could not be fetched" in result.reason


@pytest.mark.asyncio
async def test_substantial_static_site_is_excluded_as_uncertain():
    result = await qualify_business(
        {"name": "Harbour Point Veterinary Group", "phone": "+14155550166",
         "website": "https://harbourpointvet.example"},
        fetch=_fetch({"https://harbourpointvet.example": SUBSTANTIAL_STATIC}),
    )
    assert isinstance(result, Disqualified)
    assert "too uncertain to pitch" in result.reason


@pytest.mark.asyncio
async def test_brochure_without_a_phone_is_not_a_lead():
    """A prospect we cannot call is not a prospect."""
    page = "<html><body><h1>Quiet Books</h1><p>Used books since 1988.</p></body></html>"
    result = await qualify_business(
        {"name": "Quiet Books", "website": "https://quietbooks.example"},
        fetch=_fetch({"https://quietbooks.example": page}),
    )
    assert isinstance(result, Disqualified)
    assert "no phone" in result.reason


# -- the qualifying cases, each with a citable reason ----------------------


@pytest.mark.asyncio
async def test_no_website_is_flagged_with_a_reason():
    result = await qualify_business(
        {"name": "Joe's Barbers", "phone": "+14155550100", "website": None}
    )
    assert isinstance(result, Lead)
    assert "no website" in result.qualification_reason
    assert result.website is None
    assert result.score >= 0.9


@pytest.mark.asyncio
async def test_no_website_and_no_phone_is_not_a_lead():
    result = await qualify_business({"name": "Unknown Place"})
    assert isinstance(result, Disqualified)
    assert "unreachable" in result.reason


@pytest.mark.asyncio
async def test_brochure_site_is_flagged_with_a_specific_reason():
    result = await qualify_business(
        {"name": "Golden Dragon Tae Kwon Do", "website": "https://goldendragon.example"},
        fetch=_fetch({"https://goldendragon.example": BROCHURE}),
    )
    assert isinstance(result, Lead)
    reason = result.qualification_reason
    assert "single-page site" in reason
    assert "no online booking" in reason
    assert "no contact form" in reason
    # The phone came off the page, via triage's extractor - so the lead is callable.
    assert result.phone == "+14155550142"
    assert 0.0 <= result.score <= 1.0
    assert any("words of visible text" in s for s in result.signals)


@pytest.mark.asyncio
async def test_multi_page_brochure_is_flagged_and_scored_lower():
    result = await qualify_business(
        {"name": "Sunset Family Dentistry", "website": "https://sunsetdental.example"},
        fetch=_fetch({"https://sunsetdental.example": MULTI_PAGE_BROCHURE}),
    )
    assert isinstance(result, Lead)
    assert "brochure site (4 pages" in result.qualification_reason
    assert result.score == 0.65


@pytest.mark.asyncio
async def test_social_page_only_is_flagged_but_lookalike_domain_is_not():
    lead = await qualify_business(
        {"name": "Rosa's Tamales", "phone": "+14155550123",
         "website": "https://www.facebook.com/rosastamales"}
    )
    assert isinstance(lead, Lead)
    assert "facebook.com" in lead.qualification_reason

    # A real domain that merely contains a social host's name is a normal site.
    page = RICH_RESTAURANT
    other = await qualify_business(
        {"name": "Facebook Marketing Co", "phone": "+14155550124",
         "website": "https://facebookmarketing.example"},
        fetch=_fetch({"https://facebookmarketing.example": page}),
    )
    assert isinstance(other, Disqualified)


# -- the list-level API ----------------------------------------------------


BUSINESSES = [
    {"name": "Joe's Barbers", "phone": "+14155550100", "website": None},
    {"name": "Golden Dragon Tae Kwon Do", "website": "https://goldendragon.example"},
    {"name": "Sunset Family Dentistry", "website": "https://sunsetdental.example"},
    {"name": "Blue Plate Kitchen", "phone": "+14155550155",
     "website": "https://blueplate.example"},
    {"name": "Ocean Avenue Pilates", "phone": "+14155550188",
     "website": "https://oceanpilates.example"},
    {"name": "Unknown Place"},
]

PAGES = {
    "https://goldendragon.example": BROCHURE,
    "https://sunsetdental.example": MULTI_PAGE_BROCHURE,
    "https://blueplate.example": RICH_RESTAURANT,
    "https://oceanpilates.example": SITE_WITH_BOOKING,
}


@pytest.mark.asyncio
async def test_screening_keeps_both_halves_of_the_funnel():
    leads, dropped = await screen_businesses(BUSINESSES, fetch=_fetch(PAGES))
    assert [lead.business_name for lead in leads] == [
        "Joe's Barbers",                # 0.95, no website
        "Golden Dragon Tae Kwon Do",    # 0.85, single-page brochure
        "Sunset Family Dentistry",      # 0.65, four-page brochure
    ]
    assert len(dropped) == 3
    assert all(d.reason for d in dropped)
    assert len(leads) + len(dropped) == len(BUSINESSES)


@pytest.mark.asyncio
async def test_leads_come_back_best_first_and_can_be_thresholded():
    leads = await find_no_website_leads(BUSINESSES, fetch=_fetch(PAGES))
    assert [lead.score for lead in leads] == sorted(
        (lead.score for lead in leads), reverse=True
    )
    strong = await find_no_website_leads(BUSINESSES, fetch=_fetch(PAGES), min_score=0.8)
    assert len(strong) == 2


# -- the demo table --------------------------------------------------------


@pytest.mark.asyncio
async def test_to_sheet_produces_parseable_csv_with_one_row_per_lead():
    leads = await find_no_website_leads(BUSINESSES, fetch=_fetch(PAGES))
    sheet = to_sheet(leads)

    rows = list(csv.DictReader(io.StringIO(sheet)))
    assert len(rows) == len(leads) == 3
    assert tuple(rows[0].keys()) == SHEET_COLUMNS

    # Reasons contain commas by construction; they must survive the round trip
    # rather than shearing into extra columns.
    assert rows[1]["qualification_reason"] == leads[1].qualification_reason
    assert "," in rows[1]["qualification_reason"]
    assert rows[0]["business_name"] == "Joe's Barbers"
    assert rows[0]["website"] == ""          # None renders as blank, not "None"
    assert float(rows[0]["score"]) == leads[0].score


def test_to_sheet_of_nothing_is_a_header_only_sheet():
    rows = list(csv.DictReader(io.StringIO(to_sheet([]))))
    assert rows == []
    assert to_sheet([]).strip().split(",")[0] == "business_name"
