"""Read a price sheet without inventing the numbers that are not on it.

Owners already have the answers written down somewhere - a menu, a costing
spreadsheet, a note in a text file. Reading it is worth a lot of interview time,
and it is also the single easiest place to fabricate a business profile.

Two rules, both of which are the same rule as `src/verify/receipts.py`:

**A field this file cannot find is reported missing, never defaulted.** A
missing materials cost defaulted to zero does not fail loudly - it succeeds
quietly, every quote clears every margin, and the floor is gone. `missing` is
part of the return value, not an error case.

**A price is not a cost.** A menu says what customers pay; the cost model wants
what the owner pays. Menu rows are returned under `menu_items` with an explicit
warning and are never mapped onto a cost field, because a $3.50 muffin read as
a $3.50 cost inverts the entire margin calculation.

Extraction is keyword-and-number over rows: deliberately dumb, and it says what
line each value came from so the owner can be read back the evidence before it
is accepted.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

#: Extensions we will open. Anything else is refused rather than guessed at.
SUPPORTED_SUFFIXES = frozenset({".csv", ".tsv", ".xlsx", ".xlsm", ".txt", ".md"})

#: 5 MB. A menu is kilobytes; anything larger is not a price sheet.
MAX_BYTES = 5 * 1024 * 1024

#: Field -> the phrases that identify it. Longest phrase wins, so "minimum
#: margin" is never shadowed by "margin". Phrases must be specific: a bare
#: "margin" is ambiguous between the floor and the target and is reported as
#: such instead of being assigned to one of them.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "materials_per_unit": (
        "materials cost", "material cost", "ingredient cost", "ingredients cost",
        "cost of goods", "food cost", "cogs", "materials", "ingredients",
    ),
    "labor_per_unit": (
        "labour cost", "labor cost", "labour", "labor", "wage cost", "staff cost",
    ),
    "transport_cost": (
        "transport cost", "delivery cost", "shipping cost", "courier cost",
        "freight", "transport", "delivery fee", "fuel cost",
    ),
    "min_margin_pct": (
        "minimum margin", "min margin", "margin floor", "floor margin",
        "lowest margin",
    ),
    "target_margin_pct": (
        "target margin", "desired margin", "goal margin", "wanted margin",
    ),
    "max_discount_pct": (
        "maximum discount", "max discount", "discount cap", "discount limit",
        "discount ceiling",
    ),
    "capacity_total": (
        "weekly capacity", "monthly capacity", "capacity per week",
        "capacity per month", "max output", "capacity",
    ),
    "items_per_person": (
        "per person", "per head", "per guest", "items per person",
    ),
}

#: Phrases that look like a field but are not specific enough to assign.
AMBIGUOUS: dict[str, str] = {
    "margin": "says 'margin' without saying whether it is the floor or the target",
    "cost": "says 'cost' without saying materials, labour or delivery",
    "discount": "says 'discount' without saying it is the agent's ceiling",
}

_NUMBER = re.compile(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?\s*%?")
_PRICE_HEADER = re.compile(r"\b(price|rrp|retail|charge|sell)\b", re.IGNORECASE)


@dataclass
class ParsedDocument:
    """What one document yielded, and - as loudly - what it did not.

    `found` values are raw strings on purpose. They go back through the same
    coercers a spoken answer does, so a spreadsheet cannot smuggle in a value
    an owner would have been refused for saying out loud.
    """

    path: str
    found: dict[str, dict[str, str]] = dc_field(default_factory=dict)
    missing: list[str] = dc_field(default_factory=list)
    menu_items: list[dict[str, str]] = dc_field(default_factory=list)
    notes: list[str] = dc_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "found": self.found,
            "missing": self.missing,
            "menu_items": self.menu_items[:50],
            "notes": self.notes,
        }


class DocumentError(Exception):
    """The document cannot be read at all. Distinct from 'read it, found little'."""


def safe_path(raw: str, *, root: Path | None = None) -> Path:
    """Resolve a user-supplied path, refusing the shapes that are not documents.

    Input validation at the boundary, per the project rules: null bytes and
    unexpanded traversal are rejected before anything is opened, and only the
    handful of suffixes we can actually parse get through.
    """
    text = str(raw or "").strip()
    if not text:
        raise DocumentError("No path given. Which file should I read?")
    if "\x00" in text:
        raise DocumentError("That path contains a null byte and will not be opened.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (root or Path.cwd()) / path
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise DocumentError(f"No file at {path}. Check the path and say it again.")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DocumentError(
            f"I cannot read {path.suffix + ' files' if path.suffix else 'a file with no extension'}. "
            "Send me a CSV, XLSX, TXT or Markdown price sheet."
        )
    if path.stat().st_size > MAX_BYTES:
        raise DocumentError(f"{path.name} is larger than 5 MB - that is not a price sheet.")
    return path


def _rows(path: Path) -> list[list[str]]:
    """Every document flattened to rows of cells. One shape to scan."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".xlsx", ".xlsm"):
        try:
            import pandas as pd
        except ImportError:  # host launched us on an interpreter without the extras
            raise DocumentError(
                f"I cannot open spreadsheets on this Python - pandas is not installed "
                f"for {sys.executable}. Either point the MCP host at the project's "
                ".venv/bin/python, or paste the numbers to me and I will take them "
                "by voice."
            ) from None

        if suffix in (".xlsx", ".xlsm"):
            frame = pd.read_excel(path, header=None, dtype=str)
        else:
            frame = pd.read_csv(
                path, header=None, dtype=str, sep="\t" if suffix == ".tsv" else ",",
                engine="python", on_bad_lines="skip",
            )
        return [
            ["" if cell is None or str(cell) == "nan" else str(cell).strip() for cell in row]
            for row in frame.itertuples(index=False, name=None)
        ]
    text = path.read_text(encoding="utf-8", errors="replace")
    # Markdown tables are pipe-delimited; plain lines become one-cell rows.
    return [
        [c.strip() for c in line.split("|") if c.strip(" -:")] if "|" in line else [line.strip()]
        for line in text.splitlines()
        if line.strip()
    ]


def _number_in(text: str, *, after: int = 0) -> str | None:
    """First numeric token at or after `after`, normalised of $ and thousands commas."""
    match = _NUMBER.search(text, after)
    if match is None:
        return None
    token = match.group(0).replace("$", "").replace(",", "").replace(" ", "")
    return token or None


def _match_field(text: str) -> tuple[str, str] | None:
    """(field, matched phrase) for the longest keyword present, or None."""
    lowered = text.lower()
    best: tuple[str, str] | None = None
    for name, phrases in KEYWORDS.items():
        for phrase in phrases:
            if phrase in lowered and (best is None or len(phrase) > len(best[1])):
                best = (name, phrase)
    return best


def _looks_like_menu(rows: list[list[str]]) -> int | None:
    """Index of a header row that declares a price column, if there is one."""
    for index, row in enumerate(rows[:10]):
        if any(_PRICE_HEADER.search(cell) for cell in row) and len(row) >= 2:
            return index
    return None


def parse_document(raw_path: str, *, root: Path | None = None) -> ParsedDocument:
    """Pull profile fields out of a menu or price sheet.

    Returns everything found *and* everything not found. The caller is expected
    to ask the owner for the second list rather than assume it away.
    """
    path = safe_path(raw_path, root=root)
    try:
        rows = _rows(path)
    except DocumentError:
        raise
    except Exception as exc:  # unreadable spreadsheet, bad encoding, corrupt file
        raise DocumentError(f"Could not read {path.name}: {exc}") from exc

    doc = ParsedDocument(path=str(path))
    if not rows:
        doc.notes.append(f"{path.name} is empty - there is nothing in it to read.")

    menu_header = _looks_like_menu(rows)

    for index, row in enumerate(rows):
        text = " ".join(cell for cell in row if cell).strip()
        if not text:
            continue

        hit = _match_field(text)
        if hit is not None:
            name, phrase = hit
            if name in doc.found:
                continue  # first mention wins; a later one is not evidence of a change
            number = _number_in(text, after=text.lower().find(phrase) + len(phrase))
            number = number or _number_in(text)
            if number is None:
                doc.notes.append(
                    f"line {index + 1} mentions {phrase!r} but carries no number, so "
                    f"{name} is still unknown."
                )
                continue
            doc.found[name] = {"value": number, "evidence": text[:160]}
            continue

        lowered = text.lower()
        for word, why in AMBIGUOUS.items():
            if word in lowered and _NUMBER.search(text):
                doc.notes.append(f"line {index + 1} {why} - not used: {text[:80]!r}")
                break

        # A row under a price header is a menu line: a price, which is not a cost.
        if menu_header is not None and index > menu_header and len(row) >= 2:
            price = _number_in(" ".join(row[1:]))
            if price is not None and row[0]:
                doc.menu_items.append({"item": row[0][:60], "price": price})

    if doc.menu_items:
        doc.notes.append(
            f"{len(doc.menu_items)} menu rows read as PRICES, not costs. What a "
            "customer pays is not what the item costs you, so none of these were "
            "used as a cost - I still need your costs."
        )

    doc.missing = [name for name in KEYWORDS if name not in doc.found]
    if doc.missing:
        doc.notes.append(
            "Nothing in this file was assumed. Missing values stay missing until "
            "you give them to me."
        )
    return doc
