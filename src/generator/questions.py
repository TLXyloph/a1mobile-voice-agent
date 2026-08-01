"""Generate the intake questions this particular task needs.

A generated form is only worth having if it is *narrower* than the union of
every form we could have written by hand. So the model proposes and this module
disposes, in three passes:

    propose   one LLM call returns a TaskProfile plus a draft question set
    harden    canonical questions injected, nonsense dropped, duplicates merged
    validate  `QuestionSet.problems()` - empty list or the set does not ship

The hardening pass is what makes the output trustworthy rather than plausible.
Two rules in it are not negotiable:

* **No pricing questions when `unit_economics_apply` is False.** A dentist
  booking has no margin. Asking for one is how a bespoke product starts sounding
  like a template that was pointed at the wrong industry.

* **Physical items always get the units-vs-headcount question.** A live call
  turned "thirty" into thirty muffins when it meant thirty people and lost $311.
  That ambiguity is invisible the moment it becomes an integer, so it has to be
  *asked*. `src/agents/flow.py` blocks a quote until `units_confirmed`; this is
  the same guard moved forward to intake, where it costs one sentence instead of
  a live-call recovery.

The model is behind `Planner`, a one-method seam, mirroring `Responder` in
`src/webapp/intake.py`. `ScriptedPlanner` drives the whole pipeline in tests
with no network, and `heuristic_profile` + `canonical_set` give a complete
offline path when the model is unreachable or returns junk.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from src.generator.spec import (
    Exchange,
    TaskProfile,
    heuristic_profile,
)

logger = logging.getLogger("generator.questions")

MODEL = os.getenv("GENERATOR_MODEL", "openai/gpt-5.4-mini")

#: Answer shapes a question can ask for. Anything else is coerced to "text".
KINDS: tuple[str, ...] = (
    "text", "integer", "number", "money", "date", "choice", "boolean", "phone",
)

#: The field that carries the $311 lesson. Named once so nothing can drift.
UNITS_FIELD = "units_basis"

#: Field-name tokens that mean a question is about unit economics.
PRICING_TOKENS: frozenset[str] = frozenset(
    {
        "price", "pricing", "margin", "markup", "cost", "cogs", "discount",
        "profit", "rate", "economics", "wholesale", "msrp",
    }
)

#: Spending limits are not unit economics. "The most I will pay in total" is a
#: sane thing to ask a person ordering a cake; "your gross margin" is not.
SPEND_FIELDS: frozenset[str] = frozenset(
    {"spend_ceiling", "budget", "budget_ceiling", "max_spend", "total_budget"}
)

#: Phrases in the question text that give away a unit-economics question even
#: when the field name is innocent.
_PRICING_PHRASES = (
    "per unit", "per-unit", "gross margin", "profit margin", "cost to make",
    "cost per", "your margin", "unit cost", "how much do you charge",
    "discount", "markup",
)

#: Model field names that mean the same thing as a canonical field.
#:
#: Hardening injects a canonical question for any required field the draft did
#: not cover, and "did not cover" is decided by field id. Without this, a model
#: that asks "where should they send written confirmation?" under the name
#: `written_confirmation_destination` gets the identical canonical question
#: bolted on beside it, and the form the user reads asks them the same thing
#: twice. The prompt asks for canonical ids; this is the backstop for when it
#: does not comply, and it is kept to exact matches so nothing is merged that
#: is merely adjacent.
FIELD_ALIASES: dict[str, str] = {
    "confirmation_channel": "confirm_to",
    "confirmation_contact": "confirm_to",
    "confirmation_destination": "confirm_to",
    "confirmation_method": "confirm_to",
    "written_confirmation": "confirm_to",
    "written_confirmation_destination": "confirm_to",
    "where_to_send_confirmation": "confirm_to",
    "business_to_call": "callee",
    "who_to_call": "callee",
    "target_business": "callee",
    "business_name": "callee",
    "vendor": "callee",
    "how_many": "quantity",
    "quantity_needed": "quantity",
    "total_quantity": "quantity",
    "budget": "spend_ceiling",
    "total_budget": "spend_ceiling",
    "max_spend": "spend_ceiling",
    "budget_ceiling": "spend_ceiling",
    "needed_by": "deadline",
    "due_date": "deadline",
    "deadline_date": "deadline",
    "definition_of_done": "done_definition",
    "success_criteria": "done_definition",
    "units_or_people": UNITS_FIELD,
    "items_or_people": UNITS_FIELD,
    "unit_basis": UNITS_FIELD,
    "headcount_or_items": UNITS_FIELD,
    "preferred_times": "preferred_windows",
    "preferred_dates": "preferred_windows",
    "availability": "preferred_windows",
    "patient_name": "on_whose_behalf",
    "booking_name": "on_whose_behalf",
    "insurance": "insurance_or_membership",
    "insurance_details": "insurance_or_membership",
    "reference_number": "account_reference",
    "order_reference": "account_reference",
}

_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_E164 = re.compile(r"^\+[1-9]\d{6,15}$")


# ---------------------------------------------------------------------------
# a question
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A validation rule that actually runs.

    A "validation rule" that is only a sentence in a UI is decoration. `check`
    is called server-side on every answer, so the rule is the same object the
    form renders and the API enforces.
    """

    kind: str = "text"
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()
    pattern: str = ""
    help: str = ""
    """Plain-English statement of the rule, shown under the field."""

    def problems(self) -> list[str]:
        out: list[str] = []
        if self.kind not in KINDS:
            out.append(f"unknown answer kind {self.kind!r}")
        if self.kind == "choice" and not self.choices:
            out.append("choice questions need choices")
        if self.min is not None and self.max is not None and self.min > self.max:
            out.append(f"min {self.min} exceeds max {self.max}")
        if self.pattern:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                out.append(f"bad pattern: {exc}")
        return out

    def check(self, raw: Any) -> tuple[bool, str]:
        """(ok, message). An empty answer is always ok here - requiredness is
        the `Question`'s business, not the rule's."""
        text = "" if raw is None else str(raw).strip()
        if not text:
            return True, ""

        if self.kind in ("integer", "number", "money"):
            try:
                value = float(text.replace(",", "").lstrip("$"))
            except ValueError:
                return False, "needs to be a number"
            if self.kind == "integer" and value != int(value):
                return False, "needs to be a whole number"
            if self.min is not None and value < self.min:
                return False, f"needs to be at least {self.min:g}"
            if self.max is not None and value > self.max:
                return False, f"needs to be at most {self.max:g}"
            return True, ""

        if self.kind == "date":
            try:
                date.fromisoformat(text[:10])
            except ValueError:
                return False, "needs to be a date like 2026-08-14"
            return True, ""

        if self.kind == "choice":
            lowered = {c.lower() for c in self.choices}
            if text.lower() not in lowered:
                return False, "pick one of: " + ", ".join(self.choices)
            return True, ""

        if self.kind == "boolean":
            if text.lower() not in ("yes", "no", "true", "false", "y", "n"):
                return False, "answer yes or no"
            return True, ""

        if self.kind == "phone":
            digits = re.sub(r"[^\d+]", "", text)
            if not _E164.match(digits):
                return False, "needs a full number with country code, e.g. +14155551234"
            return True, ""

        if self.min is not None and len(text) < self.min:
            return False, f"needs at least {int(self.min)} characters"
        if self.pattern and not re.search(self.pattern, text):
            return False, self.help or "does not match the expected format"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "min": self.min,
            "max": self.max,
            "choices": list(self.choices),
            "pattern": self.pattern,
            "help": self.help,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Rule:
        data = data or {}
        kind = str(data.get("kind") or "text").strip().lower()
        if kind not in KINDS:
            kind = "text"

        def _num(key: str) -> float | None:
            try:
                raw = data.get(key)
                return float(raw) if raw not in (None, "") else None
            except (TypeError, ValueError):
                return None

        choices = data.get("choices") or ()
        if isinstance(choices, str):
            choices = [c.strip() for c in choices.split("|")]
        pattern = str(data.get("pattern") or "")
        try:
            re.compile(pattern)
        except re.error:
            pattern = ""
        return cls(
            kind=kind,
            min=_num("min"),
            max=_num("max"),
            choices=tuple(str(c).strip() for c in choices if str(c).strip()),
            pattern=pattern,
            help=str(data.get("help") or "").strip(),
        )


@dataclass(frozen=True)
class Question:
    """One thing to ask, and what a good answer looks like."""

    field: str
    ask: str
    rule: Rule = field(default_factory=Rule)
    required: bool = True
    why: str = ""
    """One line on why this task needs this. Shown to the user, because a form
    that explains itself gets better answers than one that does not."""

    def problems(self) -> list[str]:
        out: list[str] = []
        if not _FIELD_RE.match(self.field or ""):
            out.append(f"field {self.field!r} is not a snake_case identifier")
        if len(self.ask.strip()) < 8:
            out.append(f"{self.field}: question text is too short to answer")
        out += [f"{self.field}: {p}" for p in self.rule.problems()]
        return out

    @property
    def is_pricing(self) -> bool:
        """Is this a unit-economics question?

        Field name first, question text second. A spend ceiling is explicitly
        not pricing: it bounds what the user pays, not what anybody earns.
        """
        if self.field in SPEND_FIELDS:
            return False
        tokens = set(self.field.split("_"))
        if tokens & PRICING_TOKENS:
            return True
        text = self.ask.lower()
        return any(p in text for p in _PRICING_PHRASES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "ask": self.ask,
            "required": self.required,
            "why": self.why,
            "rule": self.rule.to_dict(),
            "pricing": self.is_pricing,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question | None:
        """None for anything unusable. Dropping a malformed question is always
        better than shipping a form field nobody can answer."""
        raw_field = str(data.get("field") or data.get("id") or "").strip().lower()
        field_id = re.sub(r"[^a-z0-9_]+", "_", raw_field).strip("_")[:48]
        field_id = FIELD_ALIASES.get(field_id, field_id)
        ask = str(data.get("ask") or data.get("question") or "").strip()
        if not _FIELD_RE.match(field_id) or len(ask) < 8:
            return None
        required = data.get("required")
        return cls(
            field=field_id,
            ask=ask,
            rule=Rule.from_dict(data.get("rule") or data.get("validation")),
            required=True if required is None else bool(required),
            why=str(data.get("why") or "").strip(),
        )


# ---------------------------------------------------------------------------
# the canonical bank
# ---------------------------------------------------------------------------

_UNITS_ASK = (
    "How many, and is that a count of ITEMS or a count of PEOPLE? If you are "
    "telling me people, say how many items each person gets."
)

CANONICAL: dict[str, Question] = {
    "callee": Question(
        field="callee",
        ask="Who should we call? Give the business or person's name, and the number if you have it.",
        rule=Rule(kind="text", min=3, help="A name we could look up, not 'the dentist'."),
        why="Nothing happens without somebody to dial.",
    ),
    "done_definition": Question(
        field="done_definition",
        ask="What has to be true afterwards for this to count as done? Describe something you could check yourself.",
        rule=Rule(kind="text", min=10, help="A checkable sentence, not 'it went well'."),
        why="This becomes the claim on the receipt. If it is not checkable, nothing can verify it.",
    ),
    "confirm_to": Question(
        field="confirm_to",
        ask="Where should they send written confirmation - a mobile number for a text, or an email address?",
        rule=Rule(kind="text", min=5, help="A number or an email we control."),
        why="A reply we receive is the only thing that turns 'they said yes' into proof.",
    ),
    "deadline": Question(
        field="deadline",
        ask="What is the last date this is still useful? (YYYY-MM-DD)",
        rule=Rule(kind="date", help="A calendar date."),
        why="After this, calling is worse than not calling.",
    ),
    "quantity": Question(
        field="quantity",
        ask="How many are you after, as a plain number?",
        rule=Rule(kind="integer", min=1, help="A whole number, at least 1."),
        why="Sizes the order and the capacity hold.",
    ),
    UNITS_FIELD: Question(
        field=UNITS_FIELD,
        ask=_UNITS_ASK,
        rule=Rule(
            kind="text",
            min=4,
            help="Say 'items' or 'people'. If people, give the items-per-person number.",
        ),
        why=(
            "A headcount is not an item count. Reading 'thirty' as thirty items when "
            "it meant thirty people has already mispriced a real order by $311."
        ),
    ),
    "unit_price_floor": Question(
        field="unit_price_floor",
        ask="What is the lowest price per {unit} you will accept? The agent will never go below it.",
        rule=Rule(kind="money", min=0, help="A number in your currency."),
        why="This is a hard floor in code, not advice in a prompt.",
    ),
    "max_discount_pct": Question(
        field="max_discount_pct",
        ask="How much can the agent discount without asking you, as a percentage?",
        rule=Rule(kind="number", min=0, max=100, help="0 to 100. Zero is a fine answer."),
        why="Anything deeper escalates instead of being improvised under pressure.",
    ),
    "spend_ceiling": Question(
        field="spend_ceiling",
        ask="What is the most you are willing to spend in total?",
        rule=Rule(kind="money", min=0, help="A total, not a per-item price."),
        why="The agent walks away above this rather than guessing what you'd tolerate.",
    ),
    "preferred_windows": Question(
        field="preferred_windows",
        ask="Which days and times work? Give two or three options, in case the first is taken.",
        rule=Rule(kind="text", min=6, help="Real alternatives, e.g. 'Tue or Thu after 3pm'."),
        why="One option means the call fails the moment that slot is gone.",
    ),
    "on_whose_behalf": Question(
        field="on_whose_behalf",
        ask="Whose name is this under, and what is their date of birth or account number if they'll ask?",
        rule=Rule(kind="text", min=2, help="The name the other side will look up."),
        why="Most booking desks cannot proceed without it.",
    ),
    "urgency": Question(
        field="urgency",
        ask="How urgent is this - today, this week, this month, or whenever?",
        rule=Rule(
            kind="choice",
            choices=("today", "this week", "this month", "whenever"),
            help="Pick one.",
        ),
        why="Decides whether the agent takes a worse slot or holds out for a better one.",
    ),
    "insurance_or_membership": Question(
        field="insurance_or_membership",
        ask="Any insurance, membership or account details they will ask for?",
        rule=Rule(kind="text", help="Leave blank if none."),
        required=False,
        why="Being asked mid-call for something we do not have ends the errand.",
    ),
    "questions_to_ask": Question(
        field="questions_to_ask",
        ask="What exactly do you need to find out? List the questions you want answered.",
        rule=Rule(kind="text", min=8, help="One per line is fine."),
        why="An information call with no list comes back with an anecdote.",
    ),
    "account_reference": Question(
        field="account_reference",
        ask="What order, booking or account reference will they ask for?",
        rule=Rule(kind="text", min=2, help="The reference as it appears on your confirmation."),
        why="Nobody changes a record they cannot find.",
    ),
    "fallback_if_refused": Question(
        field="fallback_if_refused",
        ask="If they refuse, what should the agent do instead - accept it, push once, or ask for a manager?",
        rule=Rule(kind="text", min=4, help="A plain instruction."),
        why="Otherwise the agent decides that for you, live, unsupervised.",
    ),
    "never_do": Question(
        field="never_do",
        ask="Anything the agent must never agree to on your behalf?",
        rule=Rule(kind="text", help="Leave blank if nothing comes to mind."),
        required=False,
        why="These become absolute limits - there is nobody to ask mid-call.",
    ),
}


def canonical_for(field_id: str, profile: TaskProfile) -> Question | None:
    """The canonical question, with `{unit}` resolved against the profile."""
    base = CANONICAL.get(field_id)
    if base is None:
        return None
    if "{unit}" in base.ask:
        unit = profile.unit_label or profile.units.rstrip("s") or "unit"
        return Question(
            field=base.field,
            ask=base.ask.replace("{unit}", unit),
            rule=base.rule,
            required=base.required,
            why=base.why,
        )
    return base


# ---------------------------------------------------------------------------
# the set
# ---------------------------------------------------------------------------


@dataclass
class QuestionSet:
    """A form, plus the profile that justifies each field in it."""

    profile: TaskProfile
    questions: list[Question] = field(default_factory=list)
    source: str = "heuristic"
    """"model", "model+repaired" or "heuristic". Surfaced in the UI so nobody
    has to guess whether the LLM was actually reachable."""

    repairs: list[str] = field(default_factory=list)

    @property
    def fields(self) -> list[str]:
        return [q.field for q in self.questions]

    @property
    def required(self) -> list[Question]:
        return [q for q in self.questions if q.required]

    def get(self, field_id: str) -> Question | None:
        return next((q for q in self.questions if q.field == field_id), None)

    def problems(self) -> list[str]:
        """Empty means shippable. Everything here is a reason not to render."""
        out: list[str] = []

        seen: set[str] = set()
        for q in self.questions:
            if q.field in seen:
                out.append(f"duplicate field: {q.field}")
            seen.add(q.field)
            out += q.problems()

        for needed in self.profile.required_fields():
            if needed not in seen:
                out.append(f"required field {needed} has no question")

        if not self.profile.unit_economics_apply:
            for q in self.questions:
                if q.is_pricing:
                    out.append(
                        f"pricing question {q.field!r} on a task with no unit "
                        "economics"
                    )

        if self.profile.physical_goods and UNITS_FIELD not in seen:
            out.append(
                "physical items with no units-vs-headcount question - this is "
                "the ambiguity that cost $311"
            )

        if not self.questions:
            out.append("empty question set")
        return out

    @property
    def is_valid(self) -> bool:
        return not self.problems()

    def check_answers(self, answers: dict[str, Any]) -> dict[str, str]:
        """{field: error} for whatever is wrong. Empty dict means launchable."""
        errors: dict[str, str] = {}
        for q in self.questions:
            raw = answers.get(q.field)
            text = "" if raw is None else str(raw).strip()
            if q.required and not text:
                errors[q.field] = "required"
                continue
            ok, message = q.rule.check(text)
            if not ok:
                errors[q.field] = message
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "questions": [q.to_dict() for q in self.questions],
            "source": self.source,
            "repairs": self.repairs,
            "problems": self.problems(),
            "valid": self.is_valid,
        }


def canonical_set(profile: TaskProfile) -> QuestionSet:
    """The offline form: every required field, plus the always-useful extras."""
    ordered = list(profile.required_fields())
    if profile.exchange is Exchange.BOOKING:
        ordered.append("insurance_or_membership")
    ordered.append("never_do")

    questions = [q for f in ordered if (q := canonical_for(f, profile))]
    return QuestionSet(profile=profile, questions=questions, source="heuristic")


# ---------------------------------------------------------------------------
# hardening
# ---------------------------------------------------------------------------


def _canonicalise(q: Question) -> Question:
    """Rename an aliased field, keeping the model's wording.

    The wording is usually better than the canonical one - it mentions the
    actual bakery - so only the id is replaced.
    """
    target = FIELD_ALIASES.get(q.field)
    if target is None or target == q.field:
        return q
    return Question(
        field=target, ask=q.ask, rule=q.rule, required=q.required, why=q.why
    )


def harden(profile: TaskProfile, drafted: list[Question]) -> QuestionSet:
    """Turn a draft into something that passes `problems()`, or explain why not.

    Order matters: drop first, then inject. Injecting first would let a dropped
    pricing question take its canonical replacement with it.
    """
    repairs: list[str] = []
    kept: list[Question] = []
    seen: set[str] = set()

    drafted = [_canonicalise(q) for q in drafted]
    for q in drafted:
        if q.field in seen:
            repairs.append(f"dropped duplicate {q.field}")
            continue
        if q.problems():
            repairs.append(f"dropped malformed {q.field}")
            continue
        if not profile.unit_economics_apply and q.is_pricing:
            repairs.append(
                f"dropped pricing question {q.field} - this task has no unit economics"
            )
            continue
        seen.add(q.field)
        kept.append(q)

    # The units question is non-negotiable and the model's phrasing of it is
    # not trusted: this exact sentence is the one that survived a live call.
    if profile.physical_goods:
        canonical_units = canonical_for(UNITS_FIELD, profile)
        if UNITS_FIELD in seen:
            kept = [canonical_units if q.field == UNITS_FIELD else q for q in kept]
        else:
            kept.append(canonical_units)
            seen.add(UNITS_FIELD)
            repairs.append("injected the units-vs-headcount question")

    for needed in profile.required_fields():
        if needed in seen:
            continue
        if (q := canonical_for(needed, profile)) is not None:
            kept.append(q)
            seen.add(needed)
            repairs.append(f"injected required field {needed}")

    return QuestionSet(profile=profile, questions=kept, repairs=repairs)


# ---------------------------------------------------------------------------
# the model seam
# ---------------------------------------------------------------------------


class Planner(Protocol):
    async def plan(self, system: str, user: str) -> str: ...


class LiveKitPlanner:
    """The real model, over LiveKit inference - same rails as the voice agent.

    A fresh client inside `http_context.open()`: the shared aiohttp session is
    a context variable, and a client built outside one belongs to a session
    that may already be closed.
    """

    def __init__(self, model: str = MODEL) -> None:
        self.model = model

    async def plan(self, system: str, user: str) -> str:
        from livekit.agents import llm as lkllm
        from livekit.agents.inference import LLM
        from livekit.agents.utils import http_context

        async with http_context.open():
            client = LLM(model=self.model)
            ctx = lkllm.ChatContext.empty()
            ctx.add_message(role="system", content=system)
            ctx.add_message(role="user", content=user)
            parts: list[str] = []
            async with client.chat(chat_ctx=ctx) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        parts.append(chunk.delta.content)
            return "".join(parts)


@dataclass
class ScriptedPlanner:
    """A fixed reply. What the tests use, and what offline rehearsal uses."""

    replies: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    default: str = ""
    raises: Exception | None = None

    async def plan(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.raises is not None:
            raise self.raises
        return self.replies.pop(0) if self.replies else self.default


SYSTEM = """
You design the intake form for a service that makes real outbound phone calls on
someone's behalf. You are given one sentence describing a goal. You return the
smallest set of questions that, once answered, lets an agent make that call
unsupervised and come back with checkable proof.

## Judge every question by one test

Would a competent human, holding only these answers, be able to run this errand
without calling the user back? If yes, the set is complete. If a question does
not change what the agent says or does, delete it - a form padded with generic
fields is worse than a short one, because it makes the product feel like it was
built for somebody else.

## Not every task is a sale

Classify the exchange as exactly one of:
  sale        - we are selling for the operator. Margin, floors and discount
                authority are real.
  purchase    - we are buying or ordering for the user. A total spend ceiling
                is real; margin is not.
  booking     - a slot in someone else's calendar. Dates, names, eligibility.
  information - we need an answer and nothing else changes.
  admin       - changing a record that exists: cancel, reschedule, dispute.

If the exchange is anything other than "sale", DO NOT ask about price per unit,
cost to make, gross margin, markup or discount authority. Those fields do not
exist for a dentist appointment, and asking for them is the single clearest
sign that a form was generated without thinking.

## The one question you may never omit

If completing the task involves a quantity of physical items - food, print,
flowers, parts, anything countable - you MUST include a question with field
`units_basis` that asks explicitly whether the number is a count of ITEMS or a
count of PEOPLE, and if people, how many items each. A live call once read
"thirty" as thirty muffins when it meant thirty people, and lost $311. The
ambiguity is invisible once it becomes an integer, so it has to be asked.

## Output

One JSON object, nothing else:

{
  "profile": {
    "exchange": "...", "callee": "...", "subject": "...", "done_when": "...",
    "physical_goods": true|false, "unit_label": "...", "vertical": "..."
  },
  "questions": [
    {"field": "snake_case_id",
     "ask": "plain English a non-technical person can answer",
     "required": true|false,
     "why": "one short line on why this task needs it",
     "rule": {"kind": "text|integer|number|money|date|choice|boolean|phone",
              "min": null, "max": null, "choices": [], "help": "the rule stated plainly"}}
  ]
}

## Use these exact field ids where a question means one of them

Anything you leave out that the task needs gets added back automatically, and
it gets added under these names - so a question of yours that means the same
thing under a different name shows the user the same question twice.

  callee              who to call
  done_definition     what must be true afterwards
  confirm_to          where written confirmation should be sent
  deadline            the last date this is useful
  quantity            how many
  units_basis         items or people, and items per person
  spend_ceiling       the most the user will spend in total
  preferred_windows   which days and times work
  on_whose_behalf     whose name it is under
  urgency             how soon
  questions_to_ask    what to find out
  account_reference   the order or booking reference
  fallback_if_refused what to do if they say no
  never_do            what the agent must never agree to

Invent your own snake_case id for anything genuinely specific to this task -
that is where a generated form earns its keep.

Rules on the questions themselves: 4 to 10 of them, ordered the way you would
ask them out loud. No duplicated fields. No question whose answer you could
infer from the goal. `ask` is one sentence, no jargon, no field names showing
through. Always include a question for where written confirmation should be
sent - a text or an email we receive is the only thing that can prove the
errand happened.
""".strip()


def _parse(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    if m := _FENCE.search(text):
        text = m.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


async def generate(goal: str, planner: Planner | None = None) -> QuestionSet:
    """Free-text goal in, validated question set out. Never raises.

    A model failure degrades to `canonical_set`, which is a real form - shorter
    and blunter than a good generation, but complete and correct. The UI shows
    which one it got.
    """
    goal = (goal or "").strip()
    if not goal:
        return canonical_set(heuristic_profile(""))

    planner = planner or LiveKitPlanner()
    try:
        raw = await planner.plan(SYSTEM, goal)
    except Exception as exc:  # noqa: BLE001 - every model failure is one failure
        logger.warning("planner failed (%s); using the canonical set", exc)
        out = canonical_set(heuristic_profile(goal))
        out.repairs.append(f"model unreachable ({type(exc).__name__})")
        return out

    parsed = _parse(raw)
    if parsed is None:
        logger.warning("planner returned unparseable output; using the canonical set")
        out = canonical_set(heuristic_profile(goal))
        out.repairs.append("model returned no usable JSON")
        return out

    profile_data = dict(parsed.get("profile") or {})
    profile_data.setdefault("goal", goal)
    if not str(profile_data.get("goal") or "").strip():
        profile_data["goal"] = goal
    profile = TaskProfile.from_dict(profile_data)

    drafted: list[Question] = []
    for item in parsed.get("questions") or []:
        if isinstance(item, dict) and (q := Question.from_dict(item)):
            drafted.append(q)

    if not drafted:
        out = canonical_set(profile)
        out.repairs.append("model proposed no usable questions")
        return out

    out = harden(profile, drafted)
    out.source = "model+repaired" if out.repairs else "model"
    return out
