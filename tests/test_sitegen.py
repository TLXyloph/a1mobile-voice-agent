"""The demo page is the verifiable side effect, so it has to hold up unattended.

Two failure modes matter and both are pinned here: a page that leaks a
placeholder ("Hours: TBD") tells the owner we made their site up, and a page
with a single external reference goes blank on bad conference wifi in front of
a judge. Everything else is layout taste, which tests cannot check.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deliver.sitegen import (  # noqa: E402
    BusinessFacts,
    Service,
    accent_color,
    deploy,
    facts_from_payload,
    render_site,
    serve_sites,
    slugify,
)

FULL = BusinessFacts(
    name="Rosewater Bakehouse",
    tagline="Slow sourdough and pastry, baked before dawn on Irving Street.",
    services=[
        {"name": "Country sourdough", "description": "48-hour cold ferment.", "price": "$9"},
        {"name": "Morning bun", "description": "Cardamom sugar.", "price": "$4.50"},
        {"name": "Custom cakes", "description": "Two weeks' notice.", "price": "from $65"},
    ],
    hours={"Tue-Fri": "7am-3pm", "Sat-Sun": "8am-2pm", "Mon": "Closed"},
    phone="+14155550142",
    email="hello@rosewaterbakehouse.com",
    address="1412 Irving Street, San Francisco, CA 94122",
    years_in_business=12,
    specialties=["sourdough", "laminated pastry", "custom cakes"],
)

PLACEHOLDERS = ("TBD", "TODO", "N/A", "undefined", "null", "lorem ipsum", "example.com")


def visible_text(page: str) -> str:
    """Everything a viewer actually reads: no CSS, no tags, no attributes."""
    body = re.sub(r"<style.*?</style>", " ", page, flags=re.DOTALL)
    body = re.sub(r"<head.*?</head>", " ", body, flags=re.DOTALL)
    return re.sub(r"<[^>]+>", " ", body)


def test_full_facts_render_every_section():
    page = render_site(FULL)

    assert page.startswith("<!doctype html>")
    assert page.count("<html") == 1 and page.count("</html>") == 1
    assert "<title>Rosewater Bakehouse" in page

    text = visible_text(page)
    assert "Rosewater Bakehouse" in text
    assert "Slow sourdough and pastry" in text          # hero
    assert "Country sourdough" in text and "$4.50" in text  # services + pricing
    assert "Tue-Fri" in text and "7am-3pm" in text      # hours
    assert "(415) 555-0142" in text                     # contact
    assert "hello@rosewaterbakehouse.com" in text
    assert "1412 Irving Street" in text
    assert "12 years in business" in text

    # Clickable in the way a small-business page has to be.
    assert 'href="tel:+14155550142"' in page
    assert 'href="mailto:hello@rosewaterbakehouse.com"' in page


def test_name_only_page_has_no_empty_sections_or_placeholders():
    page = render_site(BusinessFacts(name="Golden Dragon Tae Kwon Do"))
    text = visible_text(page)

    assert "Golden Dragon Tae Kwon Do" in text
    for section in ("Services", "Hours", "Get in touch"):
        assert section not in text, f"empty {section} section was rendered"
    assert "<ul" not in page and "<dl" not in page  # no empty lists left behind
    assert "tel:" not in page and "mailto:" not in page

    lowered = text.lower()
    for bad in PLACEHOLDERS:
        assert bad.lower() not in lowered, f"placeholder {bad!r} leaked into the page"
    assert not re.search(r"\bnone\b", lowered)
    assert "{" not in text and "}" not in text  # unrendered template braces


def test_partial_facts_only_render_what_is_known():
    """The realistic case: a service list and a phone number, nothing else."""
    facts = BusinessFacts(
        name="Sunset Shoe Repair",
        services=[Service(name="Resole", price="$55")],
        phone="4155550188",
    )
    text = visible_text(render_site(facts))

    assert "Resole" in text and "$55" in text
    assert "Hours" not in text          # never asked, never shown
    assert "Get in touch" in text       # phone alone still earns the section
    assert "(415) 555-0188" in text
    for bad in PLACEHOLDERS:
        assert bad.lower() not in text.lower()


def test_page_makes_zero_external_requests():
    page = render_site(FULL)

    # Inline data: URIs are the intended way to stay self-contained, and they
    # make no network request. Strip them before scanning for remote refs -
    # an SVG data URI legitimately contains the xmlns string
    # "http://www.w3.org/2000/svg", which is a namespace identifier a browser
    # never fetches.
    scannable = re.sub(r'"data:[^"]*"', '""', page)

    assert "http://" not in scannable and "https://" not in scannable
    assert "//" not in re.sub(r"tel:|mailto:", "", scannable).replace("<!doctype", "")
    for attr in re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', page):
        assert attr.startswith(
            ("tel:", "mailto:", "#", "data:")
        ), f"offsite reference: {attr}"
    # No fetch surface at all: no scripts, no fonts, no images.
    assert "<script" not in page and "<img" not in page
    assert "@import" not in page and "url(" not in page


def test_accent_colour_is_deterministic_and_business_specific():
    a = accent_color("Rosewater Bakehouse")
    b = accent_color("Golden Dragon Tae Kwon Do")

    assert a != b
    assert accent_color("Rosewater Bakehouse") == a       # stable across runs
    assert accent_color("  rosewater bakehouse ") == a    # and across casing
    assert re.fullmatch(r"#[0-9a-f]{6}", a)

    assert a in render_site(BusinessFacts(name="Rosewater Bakehouse"))
    assert a not in render_site(BusinessFacts(name="Golden Dragon Tae Kwon Do"))


def test_deploy_writes_a_file_that_is_actually_reachable(tmp_path, monkeypatch):
    base = serve_sites(port=0, root=tmp_path)
    monkeypatch.setenv("SITE_BASE_URL", base)

    page = render_site(FULL)
    url = deploy(page, "Rosewater Bakehouse!", root=tmp_path)

    assert (tmp_path / "rosewater-bakehouse" / "index.html").read_text() == page
    assert url == f"{base}/rosewater-bakehouse/"

    with urllib.request.urlopen(url, timeout=5) as resp:
        assert resp.status == 200
        assert "Rosewater Bakehouse" in resp.read().decode()


def test_slug_cannot_escape_the_sites_directory(tmp_path):
    deploy("<p>x</p>", "../../etc/passwd", root=tmp_path)

    written = list(tmp_path.rglob("index.html"))
    assert len(written) == 1
    assert written[0].parent.parent == tmp_path  # exactly one level deep


def test_extraction_payload_never_invents_missing_fields():
    facts = facts_from_payload(
        {
            "name": "Rosewater Bakehouse",
            "tagline": None,
            "services": [{"name": "Sourdough", "price": "$9"}, {"description": "orphan"}],
            "hours": {},
            "phone": "",
            "years_in_business": "not a number",
            "specialties": ["sourdough", "  "],
        }
    )

    assert facts.tagline is None and facts.hours is None and facts.phone is None
    assert facts.years_in_business is None
    assert [s.name for s in facts.services] == ["Sourdough"]  # nameless one dropped
    assert facts.specialties == ["sourdough"]
    assert slugify(facts.name) == "rosewater-bakehouse"
