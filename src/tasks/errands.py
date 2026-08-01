"""Errands: actionable tasks pulled out of a person's inbox and to-do list.

The agent is not handed a goal in plain English. It reads the same sources a
person actually keeps their obligations in - email threads, a to-do file - and
works out for itself which items require contacting somebody.

The extraction is deliberately conservative. A false positive here means the
system phones a real business about something the user never asked for, which is
worse than missing an item. When in doubt, the item is dropped and the reason is
recorded.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Channel(str, Enum):
    """How an errand should be executed."""

    WEB = "web"
    """A self-service booking path exists. Cheaper, faster, more reliable."""

    PHONE = "phone"
    """No usable web path. This is where a voice agent earns its keep."""

    UNKNOWN = "unknown"
    """Not yet triaged."""

    SKIP = "skip"
    """Deliberately not actionable. `reason` explains why."""


@dataclass
class Errand:
    """One thing that needs doing on someone's behalf."""

    intent: str
    """The action, as an imperative: 'schedule a tae kwon do demo class'."""

    target: str
    """Who to contact: business name, as specific as the source allows."""

    source: str
    """Provenance, e.g. 'gmail:18f2a...' or 'todo:line-4'. Always keep this -
    a judge asking 'why did it call them?' needs a traceable answer."""

    constraints: str = ""
    """What the agent may decide alone. Everything else escalates."""

    channel: Channel = Channel.UNKNOWN
    reason: str = ""
    phone: str | None = None
    url: str | None = None
    booking_url: str | None = None
    confidence: float = 0.0
    id: str = field(default_factory=lambda: f"errand_{uuid.uuid4().hex[:8]}")

    @property
    def is_actionable(self) -> bool:
        return self.channel in (Channel.WEB, Channel.PHONE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "intent": self.intent,
            "target": self.target,
            "source": self.source,
            "constraints": self.constraints,
            "channel": self.channel.value,
            "reason": self.reason,
            "phone": self.phone,
            "url": self.url,
            "booking_url": self.booking_url,
            "confidence": self.confidence,
        }


EXTRACTION_PROMPT = """\
You read someone's messages and to-do items and identify tasks that require \
contacting an outside business or person to complete.

Extract ONLY items where all three hold:
  1. An outside party must be contacted (a business, office, or provider).
  2. There is a concrete action, not a vague intention. "Look into golf lessons"
     is not actionable; "book a tee time at Harding Park" is.
  3. The user clearly wants it done - not something they were merely told about,
     invited to consider, or are complaining about.

Do NOT extract: newsletters, marketing, notifications, things already completed,
anything requiring a decision the user has not already made, or anything
involving money above a trivial amount unless an explicit budget is stated.

For each item return:
  intent      - imperative phrase, e.g. "schedule a tae kwon do demo class"
  target      - the business or person to contact, as specifically as stated
  constraints - what is already decided (times, dates, party size, budget).
                Leave empty if nothing is specified. Do not invent constraints.
  confidence  - 0.0 to 1.0, how sure you are this is a real actionable request

Return a JSON object: {"errands": [...]}. Return {"errands": []} if none qualify.
Being wrong in the direction of extracting too much is the costly error here: it
causes real phone calls about things nobody asked for.
"""


async def extract_errands(
    documents: list[dict[str, str]], *, model: str | None = None
) -> list[Errand]:
    """Pull actionable errands out of raw text.

    Args:
        documents: [{"source": "gmail:abc", "text": "..."}]. Source strings flow
            into the Errand so every call is traceable to its origin.
        model: Override the extraction model.
    """
    if not documents:
        return []

    from openai import AsyncOpenAI

    blob = "\n\n---\n\n".join(
        f"[source: {d['source']}]\n{d['text'][:4000]}" for d in documents
    )

    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=model or os.getenv("EXTRACT_MODEL", "gpt-4.1"),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": blob},
        ],
    )

    payload = json.loads(resp.choices[0].message.content or "{}")
    out: list[Errand] = []
    known_sources = {d["source"] for d in documents}

    for item in payload.get("errands", []):
        if not item.get("intent") or not item.get("target"):
            continue
        source = item.get("source", "")
        if source not in known_sources:
            # The model invented or mangled a provenance string. Keep the errand
            # but mark the source unknown rather than assert a false origin.
            source = f"unattributed:{source or 'none'}"
        out.append(
            Errand(
                intent=item["intent"],
                target=item["target"],
                source=source,
                constraints=item.get("constraints", ""),
                confidence=float(item.get("confidence", 0.0)),
            )
        )
    return out


def filter_confident(errands: list[Errand], threshold: float = 0.6) -> list[Errand]:
    """Drop low-confidence extractions before anything places a call."""
    return [e for e in errands if e.confidence >= threshold]
