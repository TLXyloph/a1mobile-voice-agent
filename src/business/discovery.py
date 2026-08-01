"""Lead discovery for the NO_WEBSITE strategy.

The freelance-webdev campaign's prospects are defined by an absence: no
website, a social page standing in for one, or a brochure that looks like a
site but cannot do anything. That last group is exactly what
`src.tasks.triage` already detects. Triage asks "can this errand be completed
on the web?" and reads a NO as "we must phone them"; discovery asks it of the
whole site and reads a NO as "this business is losing bookings to their own
website". Same probe, opposite side of the trade - so this module calls
`probe_web_booking` and `find_booking_signals` rather than fetching or parsing
anything itself, and adds only the second half: is there *any* other working
functionality here?

**The bias runs toward exclusion.** A false positive is not a wasted API call,
it is an agent phoning a business to say they need a website when they already
have a good one. The call is over, and so is the credibility of every later
call from that number. So an unreachable site is not a lead (we could not
look, so we do not claim), a rich static site is not a lead (uncertain), and
any working functionality disqualifies. Exclusions keep their reason, so the
funnel stays auditable.

One inversion worth naming: `use_llm=False` is the *conservative* setting
here, unlike in triage. The LLM pass only ever turns "bookable" into "not
bookable for this intent" - one extra phone call in triage, but here it would
manufacture leads out of businesses that do have a working booking widget.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from src.tasks.triage import (
    _default_fetch,  # the one fetching implementation; do not write another
    find_booking_signals,
    probe_web_booking,
)

logger = logging.getLogger("business.discovery")

#: Hosts that are a social or directory presence, not a website. A business
#: whose only web address is one of these has no domain they control.
SOCIAL_ONLY_HOSTS = (
    "facebook.com", "instagram.com", "twitter.com", "x.com", "linktr.ee",
    "yelp.com", "tripadvisor.com", "nextdoor.com", "linkedin.com",
    "business.site", "sites.google.com", "wordpress.com", "about.me",
)

#: Third-party functionality a working site hands off to, matched in markup
#: where they live in hrefs and iframe sources.
COMMERCE_DOMAINS = (
    "toasttab.com", "doordash.com", "ubereats.com", "grubhub.com",
    "chownow.com", "slicelife.com", "clover.com", "shopify.com",
    "myshopify.com", "squareup.com", "ecwid.com", "bigcartel.com",
    "gumroad.com", "stripe.com", "paypal.com", "woocommerce",
    "mailchimp.com", "hubspot.com", "typeform.com", "jotform.com",
)

#: Phrases that only appear on a site that does something. Matched on word
#: boundaries in visible text, never inside script or style blocks.
COMMERCE_PHRASES = (
    "add to cart", "order online", "shopping cart", "checkout", "buy now",
    "order now", "place your order", "my account", "sign in", "log in",
    "create account", "start free trial", "subscribe now", "pay online",
    "get a quote", "request a quote", "contact form", "join waitlist",
)

_COMMERCE_DOMAIN_RE = re.compile(
    "|".join(re.escape(d) for d in COMMERCE_DOMAINS), re.IGNORECASE
)
_COMMERCE_PHRASE_RE = re.compile(
    "|".join(rf"\b{re.escape(p)}\b" for p in COMMERCE_PHRASES), re.IGNORECASE
)
_FORM_RE = re.compile(r"<form\b[^>]*>", re.IGNORECASE)
_INPUT_RE = re.compile(r"<input\b[^>]*type=[\"']?(email|tel|submit|password)", re.I)
_APP_RE = re.compile(r"__NEXT_DATA__|ng-app|data-reactroot|__NUXT__|wp-json", re.I)
_HREF_RE = re.compile(r"""href=["']([^"'>]+)""", re.IGNORECASE)
_ASSET_SUFFIXES = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".pdf", ".xml",
)

#: A site under this many visible words is a business card, not a website.
THIN_WORDS = 220
#: Above this many internal pages we stop calling it a brochure even with no
#: functionality found - better to miss the lead than be wrong on the call.
BROCHURE_PAGES = 5

DEFAULT_INTENT = "book an appointment, place an order, or contact the business online"


@dataclass
class Lead:
    """A business that qualifies for the no-website campaign.

    `qualification_reason` is the sentence the agent has to defend on a live
    call, so it cites what was observed - "single-page site, no online
    booking" - never a judgement like "weak web presence".
    """

    business_name: str
    phone: str | None
    website: str | None
    qualification_reason: str
    score: float
    signals: tuple[str, ...] = field(default_factory=tuple)
    """The raw observations behind the reason, for the audit trail."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "business_name": self.business_name,
            "phone": self.phone,
            "website": self.website,
            "qualification_reason": self.qualification_reason,
            "score": self.score,
            "signals": list(self.signals),
        }


@dataclass
class Disqualified:
    """A business we looked at and chose not to call, and why.

    Kept because the exclusions are the credible half of the demo: they show
    the list was screened rather than scraped.
    """

    business_name: str
    website: str | None
    reason: str


class _OnceFetcher:
    """Fetch each URL at most once and keep the markup.

    `probe_web_booking` fetches internally and does not hand the body back,
    but qualification also needs to see how thin the page is. Wrapping the
    fetch keeps that function as the single fetching path, at a cost of
    exactly one request per site.
    """

    def __init__(self, fetch: Callable[[str], Awaitable[str]] | None = None) -> None:
        self._fetch = fetch or _default_fetch
        self.body: str | None = None
        self.error: str | None = None

    async def __call__(self, url: str) -> str:
        if self.body is not None:
            return self.body
        try:
            self.body = await self._fetch(url)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        return self.body


def _visible_text(html: str) -> str:
    """Markup to user-visible text, script and style contents dropped."""
    stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", stripped)).strip()


def _internal_pages(html: str, base_url: str) -> set[str]:
    """Distinct internal pages linked from this one, excluding assets."""
    host = (urlparse(base_url).netloc or "").lower().removeprefix("www.")
    pages: set[str] = set()
    for m in _HREF_RE.finditer(html):
        href = m.group(1).strip()
        low = href.lower()
        if not href or low.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        parsed = urlparse(href)
        if parsed.netloc:
            other = parsed.netloc.lower().removeprefix("www.")
            if other != host:
                continue  # external link, says nothing about their site size
        path = parsed.path.split("?")[0].rstrip("/").lower()
        if not path or path in ("/index.html", "/index.htm", "/index.php"):
            continue
        if path.endswith(_ASSET_SUFFIXES):
            continue
        pages.add(path if path.startswith("/") else f"/{path}")
    return pages


def _functional_signals(html: str, text: str) -> list[str]:
    """Everything on this page that indicates the site actually does something.

    Booking detection is delegated wholesale to triage. Anything found here
    disqualifies the business as a lead, so the list is deliberately broad -
    over-detecting costs us a prospect, under-detecting costs us the call.
    """
    found: list[str] = [f"booking: {s}" for s in find_booking_signals(html, text)]
    found += sorted(
        {f"commerce: {m.group(0).lower()}" for m in _COMMERCE_DOMAIN_RE.finditer(html)}
    )
    found += sorted(
        {f"phrase: {m.group(0).lower()}" for m in _COMMERCE_PHRASE_RE.finditer(text)}
    )
    if _FORM_RE.search(html) and _INPUT_RE.search(html):
        found.append("form: submittable form with typed inputs")
    if _APP_RE.search(html):
        found.append("app: dynamic web application markers")
    return found


def _normalise_url(website: str | None) -> str | None:
    if not website or not website.strip():
        return None
    url = website.strip()
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


def _social_host(url: str) -> str | None:
    """The social host this URL *is*, or None.

    Host equality or a dotted suffix only, never a substring: telling someone
    who owns `facebookmarketing.co` that they only have a Facebook page is the
    same category of error as the false lead.
    """
    host = (urlparse(url).netloc or "").lower().split(":")[0]
    for social in SOCIAL_ONLY_HOSTS:
        if host == social or host.endswith(f".{social}"):
            return social
    return None


async def qualify_business(
    business: dict[str, Any],
    *,
    intent: str = DEFAULT_INTENT,
    fetch: Callable[[str], Awaitable[str]] | None = None,
) -> Lead | Disqualified:
    """Screen one business. Returns a Lead, or a Disqualified saying why not.

    Args:
        business: {"name": str, "phone": str | None, "website": str | None}.
        intent: what a customer would try to do on the site, passed to
            `probe_web_booking`.
    """
    name = str(business.get("name") or business.get("business_name") or "").strip()
    if not name:
        return Disqualified("", None, "record has no business name")

    listed_phone = (business.get("phone") or "").strip() or None
    url = _normalise_url(business.get("website"))

    # 1. No website at all. The strongest signal there is, and no fetch needed.
    if url is None:
        if not listed_phone:
            return Disqualified(name, None, "no website and no phone - unreachable")
        return Lead(
            business_name=name,
            phone=listed_phone,
            website=None,
            qualification_reason="no website found in listing; reachable by phone only",
            score=0.95,
            signals=("no website in source listing",),
        )

    # 2. A social page standing in for a website. Also decidable without a fetch.
    social = _social_host(url)
    if social:
        if not listed_phone:
            return Disqualified(name, url, f"{social} page only, and no phone to call")
        return Lead(
            business_name=name,
            phone=listed_phone,
            website=url,
            qualification_reason=(
                f"only web presence is a {social} page - no site and no domain "
                f"they control"
            ),
            score=0.85,
            signals=(f"listed website is {social}",),
        )

    # 3. A real domain. Now the site has to be looked at.
    once = _OnceFetcher(fetch)
    bookable, probe_reason, page_phone = await probe_web_booking(
        url, intent, fetch=once, use_llm=False
    )
    phone = listed_phone or page_phone

    if once.body is None:
        # We could not see the site, so we know nothing about it. Saying "you
        # need a website" to someone whose site merely blocked our user agent
        # is the exact failure this module exists to avoid.
        return Disqualified(name, url, f"site could not be fetched ({once.error}) - not assessed")

    html = once.body
    text = _visible_text(html)
    functional = _functional_signals(html, text)

    if bookable or functional:
        top = ", ".join(functional[:3]) or probe_reason
        return Disqualified(name, url, f"site has working functionality ({top})")

    if not phone:
        return Disqualified(name, url, "brochure site, but no phone number to call")

    words = len(text.split())
    pages = _internal_pages(html, url)
    page_count = len(pages) + 1  # the page we fetched
    observed = (
        f"{page_count} page(s) linked", f"{words} words of visible text",
        "no booking, ordering, checkout or contact-form signals found",
    )

    if page_count <= 1 and words < THIN_WORDS:
        shape = f"single-page site ({words} words)"
        score = 0.85 if words < 100 else 0.80
    elif page_count <= BROCHURE_PAGES and words < THIN_WORDS * 3:
        shape = f"brochure site ({page_count} pages, {words} words)"
        score = 0.65
    else:
        # Substantial content, no functionality we could detect. Possibly a
        # real prospect, but not one we can open a call with a defensible
        # claim about, so it does not get called.
        return Disqualified(
            name, url,
            f"substantial site ({page_count} pages, {words} words) with no "
            f"detected functionality - too uncertain to pitch",
        )

    return Lead(
        business_name=name,
        phone=phone,
        website=url,
        qualification_reason=(
            f"{shape}, no online booking, no online ordering, no contact form"
        ),
        score=score,
        signals=observed,
    )


async def screen_businesses(
    businesses: list[dict[str, Any]],
    *,
    intent: str = DEFAULT_INTENT,
    fetch: Callable[[str], Awaitable[str]] | None = None,
) -> tuple[list[Lead], list[Disqualified]]:
    """Screen a list, keeping both halves of the funnel."""
    results = await asyncio.gather(
        *(qualify_business(b, intent=intent, fetch=fetch) for b in businesses)
    )
    leads = [r for r in results if isinstance(r, Lead)]
    dropped = [r for r in results if isinstance(r, Disqualified)]
    leads.sort(key=lambda lead: (-lead.score, lead.business_name))
    logger.info("screened %d businesses -> %d leads, %d excluded",
                len(businesses), len(leads), len(dropped))
    return leads, dropped


async def find_no_website_leads(
    businesses: list[dict[str, Any]],
    *,
    intent: str = DEFAULT_INTENT,
    fetch: Callable[[str], Awaitable[str]] | None = None,
    min_score: float = 0.0,
) -> list[Lead]:
    """Qualified leads only, best first. The NO_WEBSITE discovery strategy."""
    leads, _ = await screen_businesses(businesses, intent=intent, fetch=fetch)
    return [lead for lead in leads if lead.score >= min_score]


SHEET_COLUMNS = ("business_name", "phone", "website", "score",
                 "qualification_reason", "signals")


def to_sheet(leads: list[Lead]) -> str:
    """Qualified leads as CSV, one row per lead plus a header.

    Written with the csv module rather than joins because the qualification
    reasons contain commas by construction, and a demo table that shears at
    the first comma is worse than no table.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(SHEET_COLUMNS)
    for lead in leads:
        writer.writerow(
            [
                lead.business_name, lead.phone or "", lead.website or "",
                f"{lead.score:.2f}", lead.qualification_reason,
                "; ".join(lead.signals),
            ]
        )
    return buf.getvalue()
