"""How an owner's spoken answer is judged, one value at a time.

Split out from `intake_profile.py` so the *rules* sit apart from the *script*:
this module knows what a valid margin is, that module knows what order to ask
things in. Nothing here knows the business exists.

The one shape that matters: a coercer that will not accept a value raises
`Refusal`, carrying the sentence to say back to the owner. That is the same
contract as the Gate in `src/agents/flow.py` - a refusal is the next instruction,
not a failure - and `intake_server.py` is the only place it is caught.

Refusals are written to be read aloud. "must be in [0, 100)" is correct and
useless on a phone call; "say it as a whole number like 30" is what gets the
right answer on the second try.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping


class Refusal(Exception):
    """A value intake will not take, carrying the sentence to say back.

    Raised only by coercers, caught only at the tool boundary. Callers of the
    MCP tools never see an exception - they see the instruction.
    """


def today() -> date:
    """Indirection so tests can pin 'today' without freezing the clock globally."""
    return date.today()  # noqa: DTZ011 - a delivery date is local, not UTC


# ---------------------------------------------------------------------------
# Coercers. Each takes what a human said and returns a typed value, or refuses
# with the exact words to say next.
# ---------------------------------------------------------------------------


def text(value: Any, *, what: str, max_len: int = 60) -> str:
    said = str(value or "").strip()
    if not said:
        raise Refusal(f"I did not catch the {what}. Say it again in a word or two.")
    if len(said) > max_len:
        raise Refusal(f"That is a long {what}. Give me a short one, under {max_len} characters.")
    return said


def unit_name(value: Any) -> str:
    """The singular thing they sell. Rejecting a number here is not pedantry -
    a unit called '30' makes every downstream sentence unreadable."""
    said = text(value, what="name of what you sell", max_len=40).lower()
    if said.replace(".", "").isdigit():
        raise Refusal(
            "I need the name of the thing, not a number - 'muffin', 'crew-hour', "
            "'website'. What do you call one of them?"
        )
    return said


def positive_int(value: Any, *, what: str) -> int:
    try:
        number = int(str(value).strip().replace(",", "").split(".")[0])
    except (TypeError, ValueError):
        raise Refusal(f"I need a whole number for {what}. About how many?") from None
    if number < 1:
        raise Refusal(
            f"{what} has to be at least 1 - {number} means there is nothing to sell. "
            "What is the real number?"
        )
    return number


def money(value: Any, *, what: str) -> Decimal:
    """Costs. Zero is allowed (a consultancy has no materials); negative is not."""
    raw = str(value).strip().lower()
    for junk in ("$", ",", "usd", "dollars", "each", "per unit", "per delivery"):
        raw = raw.replace(junk, "")
    raw = raw.strip()
    try:
        amount = Decimal(raw)
    except (InvalidOperation, ValueError):
        raise Refusal(
            f"I could not read {value!r} as money. Say {what} as a plain amount, "
            "like 0.80 or 1.25."
        ) from None
    if amount < 0:
        raise Refusal(f"{what} cannot be negative. What does it actually cost you?")
    if amount > Decimal("100000"):
        raise Refusal(
            f"{amount} is a very large per-unit cost. Did you mean the cost of one, "
            "or the cost of a whole batch? Give me the cost of one."
        )
    return amount


def percent(value: Any, *, what: str) -> Decimal:
    """0-100. A bare value in (0, 1) is ambiguous and gets asked about, not guessed."""
    raw = str(value).strip().lower().replace("percent", "%")
    explicit_pct = "%" in raw
    raw = raw.replace("%", "").strip()
    try:
        pct = Decimal(raw)
    except (InvalidOperation, ValueError):
        raise Refusal(
            f"I could not read {value!r} as a percentage. Say {what} as a number like 30."
        ) from None
    if not 0 <= pct <= 100:
        raise Refusal(
            f"{what} has to be between 0 and 100. {pct} is outside that - what did "
            "you mean?"
        )
    if pct >= 100:
        raise Refusal(f"A 100% {what} has no finite price. Give me a number under 100.")
    if 0 < pct < 1 and not explicit_pct:
        # 0.3 is either thirty percent or a third of one percent. Guessing wrong
        # sells the whole book at cost, so this is a question, not a default.
        raise Refusal(
            f"Did you mean {pct * 100:.0f} percent? Say it as a whole number like "
            f"{pct * 100:.0f}. If you really meant {pct} of a percent, say '{pct}%'."
        )
    return pct


def choice(value: Any, options: Mapping[str, tuple[str, ...]], *, what: str) -> str:
    """Map loose speech onto one of a few canonical values."""
    said = str(value).strip().lower()
    for canonical, synonyms in options.items():
        if said == canonical or any(word in said for word in synonyms):
            return canonical
    raise Refusal(
        f"I did not follow that for {what}. Answer with one of: " + ", ".join(options) + "."
    )


def date_or_days(value: Any, *, what: str) -> date:
    """Either an actual date or 'how many days from now' - owners say both."""
    said = str(value).strip().lower()
    if said in ("today", "now", "same day"):
        return today()
    if said == "tomorrow":
        return today() + timedelta(days=1)
    try:
        return date.fromisoformat(said)
    except ValueError:
        pass
    digits = "".join(c for c in said if c.isdigit())
    if digits:
        return today() + timedelta(days=int(digits))
    raise Refusal(
        f"I could not read {value!r} as a date for {what}. Give me a number of days "
        "from today, like 3, or a date like 2026-08-10."
    )


#: Canonical weekday names, and what people actually say for them.
WEEKDAYS: dict[str, tuple[str, ...]] = {
    "monday": ("mon",),
    "tuesday": ("tue", "tues"),
    "wednesday": ("wed", "weds"),
    "thursday": ("thu", "thur", "thurs"),
    "friday": ("fri",),
    "saturday": ("sat",),
    "sunday": ("sun",),
}


def weekdays(value: Any) -> tuple[str, ...]:
    """Days the business never delivers. 'none' is a real answer, not a blank."""
    said = str(value).strip().lower()
    if said in ("", "none", "no", "nope", "never", "open every day", "n/a"):
        return ()
    tokens = [t.strip(" .,;/") for t in said.replace("and", ",").replace(" ", ",").split(",")]
    out: list[str] = []
    for token in (t for t in tokens if t):
        stem = token.rstrip("s") if token not in WEEKDAYS else token
        match = next(
            (day for day, alts in WEEKDAYS.items() if stem == day or stem in alts), None
        )
        if match is None:
            raise Refusal(
                f"I did not recognise {token!r} as a day. Name the days you are "
                "closed - 'Sunday and Monday' - or say 'none'."
            )
        if match not in out:
            out.append(match)
    if len(out) == 7:
        raise Refusal(
            "That is every day of the week - the business would never deliver. "
            "Which days are you actually closed?"
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# One question
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One thing to ask, and everything needed to judge the answer.

    `prompt` may be a callable so a question can read back what is already
    known ("how many muffins can you make in a week?"). `asked_when` makes a
    question conditional; `check` catches the errors that only exist between
    two answers, like a target margin under the floor.
    """

    field: str
    prompt: str | Callable[[Mapping[str, Any]], str]
    why: str
    coerce: Callable[[Any], Any]
    asked_when: Callable[[Mapping[str, Any]], bool] | None = None
    check: Callable[[Any, Mapping[str, Any]], str | None] | None = None

    def applies(self, answers: Mapping[str, Any]) -> bool:
        return self.asked_when is None or self.asked_when(answers)

    def ask(self, answers: Mapping[str, Any]) -> str:
        return self.prompt(answers) if callable(self.prompt) else self.prompt

    def parse(self, value: Any, answers: Mapping[str, Any]) -> Any:
        """Coerce and cross-check. Raises `Refusal` with what to say next."""
        parsed = self.coerce(value)
        if self.check is not None and (problem := self.check(parsed, answers)):
            raise Refusal(problem)
        return parsed


# ---------------------------------------------------------------------------
# Wording helpers, so prompts can name the thing the owner already told us
# ---------------------------------------------------------------------------


def unit_of(answers: Mapping[str, Any]) -> str:
    return str(answers.get("unit", "unit"))


def plural(word: str) -> str:
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and not word.endswith(("ay", "ey", "iy", "oy", "uy")):
        return word[:-1] + "ies"
    return word + "s"


def target_not_below_floor(value: Decimal, answers: Mapping[str, Any]) -> str | None:
    floor = answers.get("min_margin_pct")
    if floor is not None and value < floor:
        return (
            f"You said you would not go below {floor}%, so a target of {value}% is "
            f"under your own floor. What margin do you actually aim for - {floor}% "
            "or better?"
        )
    return None


def latest_after_earliest(value: date, answers: Mapping[str, Any]) -> str | None:
    earliest = answers.get("earliest_date")
    if earliest is not None and value < earliest:
        return (
            f"That is before the {earliest} you gave as your soonest date. How far "
            "ahead will you take bookings?"
        )
    return None
