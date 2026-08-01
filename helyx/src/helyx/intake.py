"""The intake agent: turns operator conversation into a validated Mandate.

The negotiation is only as good as its parameters, so intake's job is to get
every field that bounds the negotiation -- item, quantity, target, walk-away
ceiling, date, constraints -- before a call may be placed.

The same discipline as the rest of Helyx applies here. The model extracts
fields, but it does **not** decide whether intake is finished. Completeness is
``mandate.missing_fields()``, computed from the accumulated slots. A model that
says "great, I have everything I need" while a slot is empty changes nothing:
``ready`` stays False and the call stays blocked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient, LLMError
from .mandate import (
    FIELD_PROMPTS,
    REQUIRED_FIELDS,
    Mandate,
    MandateError,
    build_mandate,
    missing_fields,
)

logger = logging.getLogger("helyx.intake")

SYSTEM_PROMPT = """You are the intake desk for Helyx, which negotiates wholesale and \
catering orders with restaurants and bakehouses on behalf of an operator.

Your only job is to collect the parameters that bound the negotiation. Call the \
`record_parameters` tool with every field you can extract from what the operator said. \
Extract only what they actually stated - never invent a price, a quantity, or a date. \
Omit a field entirely rather than guessing it.

Money fields are in dollars per unit (e.g. 4.25). `needed_by` is YYYY-MM-DD.
`ceiling_unit_price` is the walk-away point: the most the operator would ever pay per \
unit. It must be greater than or equal to `target_unit_price`.

After the tool call, reply with one short sentence: acknowledge what you captured and \
ask for the single most important missing item. Do not claim intake is complete."""

RECORD_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "record_parameters",
        "description": "Record negotiation parameters stated by the operator.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "e.g. 'sourdough loaves'"},
                "quantity": {"type": "integer"},
                "target_unit_price": {"type": "number", "description": "dollars per unit"},
                "ceiling_unit_price": {
                    "type": "number",
                    "description": "walk-away max dollars per unit",
                },
                "opening_unit_price": {"type": "number"},
                "needed_by": {"type": "string", "description": "YYYY-MM-DD"},
                "counterparty_name": {"type": "string"},
                "counterparty_phone": {"type": "string", "description": "E.164, e.g. +1..."},
                "max_rounds": {"type": "integer"},
                "constraints": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
}

#: Tool field name -> internal slot name.
_ALIASES = {
    "target_unit_price": "target_unit_price_cents",
    "ceiling_unit_price": "ceiling_unit_price_cents",
    "opening_unit_price": "opening_unit_price_cents",
}


@dataclass
class IntakeSession:
    """Accumulates slots across operator turns. Completeness is computed."""

    slots: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, str]] = field(default_factory=list)
    last_reply: str = ""
    served_model: str = ""

    # -- derived, not asserted --------------------------------------------
    @property
    def missing(self) -> list[str]:
        return missing_fields(self.slots)

    @property
    def ready(self) -> bool:
        """True only when every required slot is filled AND they validate."""
        if self.missing:
            return False
        try:
            build_mandate(self.slots)
        except MandateError:
            return False
        return True

    @property
    def next_question(self) -> str:
        for f in REQUIRED_FIELDS:
            if f in self.missing:
                return FIELD_PROMPTS.get(f, f"Please provide {f}.")
        return ""

    def validation_error(self) -> str:
        """Why a fully-populated set of slots still will not build."""
        if self.missing:
            return ""
        try:
            build_mandate(self.slots)
        except MandateError as exc:
            return str(exc)
        return ""

    def mandate(self) -> Mandate:
        return build_mandate(self.slots)

    def apply(self, extracted: dict[str, Any]) -> list[str]:
        """Merge extracted fields into slots. Returns the names actually set."""
        applied: list[str] = []
        for key, value in (extracted or {}).items():
            if value in (None, "", [], {}):
                continue
            slot = _ALIASES.get(key, key)
            self.slots[slot] = value
            applied.append(slot)
        return applied

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": dict(self.slots),
            "missing": self.missing,
            "ready": self.ready,
            "next_question": self.next_question,
            "validation_error": self.validation_error(),
            "last_reply": self.last_reply,
            "served_model": self.served_model,
            "required_fields": list(REQUIRED_FIELDS),
            "transcript": list(self.transcript),
        }


class IntakeAgent:
    """Conversational front end over an IntakeSession."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def turn(self, session: IntakeSession, operator_text: str) -> IntakeSession:
        """Process one operator utterance. Never raises on LLM failure."""
        text = (operator_text or "").strip()
        if not text:
            return session
        session.transcript.append({"role": "operator", "text": text})

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if session.slots:
            messages.append(
                {
                    "role": "system",
                    "content": f"Already captured: {session.slots}. Missing: {session.missing}.",
                }
            )
        for turn in session.transcript[-12:]:
            role = "user" if turn["role"] == "operator" else "assistant"
            messages.append({"role": role, "content": turn["text"]})

        try:
            completion = self.client.complete(messages, tools=[RECORD_TOOL])
        except LLMError as exc:
            logger.warning("intake LLM unavailable: %s", exc)
            session.last_reply = (
                "I could not reach the language model, so I did not update anything. "
                + (session.next_question or "Please try again.")
            )
            session.transcript.append({"role": "helyx", "text": session.last_reply})
            return session

        session.served_model = completion.served_model
        extracted = completion.first_tool_args("record_parameters") or {}
        session.apply(extracted)

        # The reply is cosmetic. What gates the call is `ready`, below.
        reply = completion.text or ""
        if session.missing:
            reply = f"{reply}\n\n{session.next_question}".strip()
        elif err := session.validation_error():
            reply = f"{reply}\n\nThat does not add up: {err}".strip()
        else:
            m = session.mandate()
            reply = (
                f"{reply}\n\nMandate locked: {m.quantity} x {m.item} for "
                f"{m.counterparty_name}, target ${m.target_unit_price_cents / 100:.2f}/unit, "
                f"walk away above ${m.ceiling_unit_price_cents / 100:.2f}/unit."
            ).strip()

        session.last_reply = reply
        session.transcript.append({"role": "helyx", "text": reply})
        return session
