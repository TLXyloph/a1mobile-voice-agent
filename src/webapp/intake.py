"""The intake interview: a conversation that has to end with a usable spec.

An interview is a loop with a language model inside it, which means two things
can go wrong that a normal form cannot:

* **It never finishes.** The model asks a fifteenth clarifying question, or
  loops on a field the operator already answered. `MAX_TURNS` and
  `Conversation.stalled` make termination a property of this module rather than
  a hope about the prompt.

* **It fills fields nobody answered.** A model asked for JSON will happily
  produce `min_margin_pct: 30` because thirty is a reasonable margin. That
  number then becomes a real floor on a real call. So the patch is applied by
  `TaskSpec.apply`, which drops anything it cannot coerce, and the system
  prompt says the rule twice.

The model is behind `Responder`, a two-method seam. `LiveKitResponder` is the
real one; `ScriptedResponder` lets the tests drive an entire interview with no
network, which is what makes "the interview terminates" a test rather than an
anecdote.

The deterministic path matters as much as the model path. `next_question()`
computes the next thing to ask from `missing_fields()` alone, and it is what
runs when the model is unreachable, returns junk, or returns valid JSON with no
question in it. Conference wifi is a real threat model; an intake that stops
working when the LLM 502s takes the whole demo with it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from src.webapp.spec import FIELDS_BY_ID, KINDS, TaskSpec

logger = logging.getLogger("webapp.intake")

#: Hard ceiling on interview length. Roughly two questions per required field
#: plus slack. Reaching it is a failure mode, not a normal ending, and it is
#: reported as `stalled` rather than quietly producing a half-spec.
MAX_TURNS = 40

MODEL = os.getenv("INTAKE_MODEL", "openai/gpt-5.4-mini")


# ---------------------------------------------------------------------------
# the model seam
# ---------------------------------------------------------------------------


class Responder(Protocol):
    """Anything that can turn a system prompt plus a transcript into text."""

    async def respond(self, system: str, messages: list[dict[str, str]]) -> str: ...


class LiveKitResponder:
    """The real model, over LiveKit inference.

    A fresh `LLM` per call, inside `http_context.open()`: the shared aiohttp
    session is a context variable, and a client built outside one belongs to a
    session that may already be closed by the time the next request arrives.
    """

    def __init__(self, model: str = MODEL) -> None:
        self.model = model

    async def respond(self, system: str, messages: list[dict[str, str]]) -> str:
        from livekit.agents import llm as lkllm
        from livekit.agents.inference import LLM
        from livekit.agents.utils import http_context

        async with http_context.open():
            client = LLM(model=self.model)
            ctx = lkllm.ChatContext.empty()
            ctx.add_message(role="system", content=system)
            for m in messages[-24:]:
                role = "assistant" if m.get("role") == "assistant" else "user"
                ctx.add_message(role=role, content=m.get("text", ""))

            parts: list[str] = []
            async with client.chat(chat_ctx=ctx) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        parts.append(chunk.delta.content)
            return "".join(parts)


@dataclass
class ScriptedResponder:
    """A fixed list of replies, for tests and for offline rehearsal."""

    replies: list[str] = field(default_factory=list)
    calls: list[tuple[str, list[dict[str, str]]]] = field(default_factory=list)
    default: str = ""

    async def respond(self, system: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((system, list(messages)))
        if self.replies:
            return self.replies.pop(0)
        return self.default


# ---------------------------------------------------------------------------
# the conversation
# ---------------------------------------------------------------------------


@dataclass
class Message:
    role: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "text": self.text}


@dataclass
class Conversation:
    """One operator, one intake, one spec being filled in."""

    id: str = field(default_factory=lambda: f"cv_{uuid.uuid4().hex[:10]}")
    spec: TaskSpec = field(default_factory=TaskSpec)
    messages: list[Message] = field(default_factory=list)
    turns: int = 0
    stalled: bool = False
    """True once `MAX_TURNS` was hit with the spec still incomplete. The UI
    shows the remaining gaps and stops the loop rather than asking forever."""

    launched_room: str | None = None

    def say(self, role: str, text: str) -> Message:
        m = Message(role=role, text=text.strip())
        self.messages.append(m)
        return m

    @property
    def finished(self) -> bool:
        return self.spec.is_complete or self.stalled

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "spec": self.spec.to_dict(),
            "turns": self.turns,
            "max_turns": MAX_TURNS,
            "stalled": self.stalled,
            "finished": self.finished,
            "launched_room": self.launched_room,
        }


OPENING = (
    "I run outbound phone calls for you. Tell me what you need done - who to "
    "call and what you want out of it - and I will ask for whatever is still "
    "missing. I place the call only once the whole brief is filled in on the "
    "right, and I stick to the limits you set here: there is nobody to ask "
    "mid-call."
)


def start(spec: TaskSpec | None = None) -> Conversation:
    c = Conversation(spec=spec or TaskSpec())
    c.say("assistant", OPENING)
    return c


# ---------------------------------------------------------------------------
# the deterministic spine
# ---------------------------------------------------------------------------


def next_question(spec: TaskSpec) -> tuple[str, str] | None:
    """(field id, question) for the next gap, or None when the spec is full.

    Field order in `FIELDS` is interview order: what the job is, who to call,
    what is on offer, how much, what it costs, what the limits are, what counts
    as done. Asking limits before knowing the unit produces a limit in the
    wrong currency of thought.
    """
    missing = spec.missing_fields()
    if not missing:
        return None
    fid = missing[0]
    return fid, FIELDS_BY_ID[fid].ask(spec.effective_kind)


def build_system_prompt(spec: TaskSpec) -> str:
    """The instructions, regenerated each turn against the live spec."""
    kind = spec.effective_kind
    missing = spec.missing_fields()
    gaps = "\n".join(
        f"  - {fid}: {FIELDS_BY_ID[fid].ask(kind)}" for fid in missing
    ) or "  (nothing - the brief is complete)"
    kind_options = " | ".join(f'"{k}"' for k in KINDS)
    known = json.dumps(
        {
            k: v
            for k, v in spec.to_dict().items()
            if v not in (None, "", [], False)
            and k not in ("missing", "required", "blockers")
        },
        indent=2,
    )

    return f"""
You are the intake desk for a service that makes real outbound phone calls on a
business owner's behalf. You are talking to the owner, not to the person who
will be called. Your one job is to fill in a brief. When it is full, someone
dials a real phone and works to the limits you wrote down, with nobody to ask.

## The two rules

NEVER invent a value. If the owner has not said a number, that number stays
empty and you ask for it. A plausible margin, a plausible price, a plausible
date is worse than an empty field: the empty field asks a question, the
plausible one silently becomes a floor on a live call. Only put something in
`spec` if the owner said it or it follows unambiguously from what they said.

Units are not headcount. "Thirty people" is not thirty items, and "three
dentists" is not three appointments. Whenever a quantity appears, establish
which one it is and how they convert, and record that in `units_basis`. Getting
this wrong has already mispriced a real job at a third of its value.

## How to talk

Plain language. One or two questions per message, never a numbered
questionnaire. Acknowledge what they just told you in a few words, then ask the
next thing. If they answer three fields at once, take all three and skip ahead.
If they say they do not know or it does not apply, say what you will do instead
and move on - do not ask twice.

If the task is an errand rather than a sale (booking an appointment, chasing a
quote, arranging a pickup) there are no unit economics. Say so plainly, set
`kind` to "errand" and write one line in `pricing_note` explaining that pricing
is skipped. Do not ask for materials cost for a dentist booking.

Never suggest that a human can be consulted during the call. There is no such
person. The limits collected here are absolute.

## Output

Reply with a single JSON object and nothing else:

{{"say": "<your message to the owner>", "spec": {{...}}, "ready": false}}

`spec` is a patch - only the fields you learned this turn. Omit it entirely if
you learned nothing. Set `ready` true only when nothing is left in the gap list.

Nothing is recorded except through `spec`. If your message says you have got a
number, that number must appear in this turn's `spec` patch - otherwise you have
told the owner you wrote something down and written nothing, and they will find
out when the brief refuses to launch.

Patch fields: kind ({kind_options}), objective, business_name,
vertical, offer, unit_label, units_basis, capacity_total (int), max_qty (int),
max_discount_pct (number), earliest_date / latest_date ("YYYY-MM-DD"),
close_condition ("written_confirmation" | "booked_meeting" |
"delivered_artifact"), done_definition, confirm_to, pricing_note,
targets (list of {{"name","phone","find_hint","notes"}} - phone in +E.164),
economics ({{"materials_per_unit","labor_per_unit","transport_per_unit",
"min_margin_pct","target_margin_pct"}}, per unit, numbers only).

Today is {date.today().isoformat()}. Resolve relative dates against it.

## The brief so far

{known}

## Still missing, in the order to ask

{gaps}
""".strip()


# ---------------------------------------------------------------------------
# one turn
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    reply: str
    changed: list[str]
    used_model: bool
    raw: str = ""


async def advance(
    convo: Conversation, user_text: str, responder: Responder
) -> TurnResult:
    """Take one message from the operator and produce one reply.

    Never raises on a model failure. An intake that 500s because the LLM
    hiccuped is worse than one that asks the next question from the field list,
    and the field list is right there.
    """
    if user_text.strip():
        convo.say("user", user_text)
    convo.turns += 1

    if convo.turns > MAX_TURNS:
        convo.stalled = True
        return _fallback(convo, stalled=True)

    system = build_system_prompt(convo.spec)
    transcript = [m.to_dict() for m in convo.messages]

    try:
        raw = await responder.respond(system, transcript)
    except Exception as exc:  # noqa: BLE001 - any model failure is the same failure
        logger.warning("intake model failed (%s); falling back to the field list", exc)
        return _fallback(convo)

    parsed = _parse(raw)
    if parsed is None:
        # Some models answer in prose when asked for JSON. Prose is still a
        # question, so keep it - but take no spec changes from it.
        text = raw.strip()
        if text and len(text) < 2000:
            convo.say("assistant", text)
            return TurnResult(text, [], True, raw)
        return _fallback(convo)

    changed = convo.spec.apply(parsed.get("spec") or {})
    if not changed and _looks_substantive(user_text) and convo.spec.missing_fields():
        # Observed live: the model replies "got it, I have the per-pastry costs"
        # and sends no patch. The owner is then told their numbers were recorded
        # when they were not, and finds out only when launch refuses. One narrow
        # second pass - extraction only, no conversation - closes that gap.
        changed = await _reextract(convo, user_text, responder)
    say = str(parsed.get("say") or "").strip()
    if not say:
        return _fallback(convo, changed=changed)

    convo.say("assistant", say)
    if convo.spec.is_complete:
        convo.stalled = False
    return TurnResult(say, changed, True, raw)


_SUBSTANTIVE = re.compile(r"\d|\b\w{4,}\b")


def _looks_substantive(text: str) -> bool:
    """Worth a second extraction pass? A digit or a real word, not "ok"."""
    return bool(text and len(text.strip()) > 6 and _SUBSTANTIVE.search(text))


EXTRACT_PROMPT = """
You extract structured fields from one sentence. You do not converse.

Return a single JSON object: only fields the message below actually states, and
nothing else. Return {{}} if it states none of them. Never infer a value from
what is typical - a number the message does not contain must not appear in your
answer.

Empty fields, with what each means:
{gaps}

Field formats: capacity_total / max_qty are ints; max_discount_pct is a number;
earliest_date / latest_date are "YYYY-MM-DD" (today is {today});
close_condition is one of written_confirmation, booked_meeting,
delivered_artifact; targets is a list of {{"name","phone","find_hint"}};
economics is {{"materials_per_unit","labor_per_unit","transport_per_unit",
"min_margin_pct","target_margin_pct"}} in currency per unit.

The message:
{message}
""".strip()


async def _reextract(
    convo: Conversation, user_text: str, responder: Responder
) -> list[str]:
    """One narrow extraction pass. Silent on failure - it is a backstop."""
    gaps = "\n".join(
        f"  - {fid}: {FIELDS_BY_ID[fid].ask(convo.spec.effective_kind)}"
        for fid in convo.spec.missing_fields()
    )
    prompt = EXTRACT_PROMPT.format(
        gaps=gaps, today=date.today().isoformat(), message=user_text
    )
    try:
        raw = await responder.respond(prompt, [{"role": "user", "text": user_text}])
    except Exception as exc:  # noqa: BLE001
        logger.warning("re-extraction failed: %s", exc)
        return []
    patch = _parse(raw)
    return convo.spec.apply(patch) if patch else []


def _fallback(
    convo: Conversation, *, stalled: bool = False, changed: list[str] | None = None
) -> TurnResult:
    """Ask the next missing field directly. No model involved."""
    nxt = next_question(convo.spec)
    if stalled:
        gaps = ", ".join(
            FIELDS_BY_ID[f].label.lower() for f in convo.spec.missing_fields()
        )
        text = (
            "I have asked as much as I usefully can. Still open: "
            f"{gaps or 'nothing'}. Fill those in and I can place the call."
        )
    elif nxt is None:
        text = (
            "That is the whole brief. Check the panel on the right, and if it "
            "reads correctly, launch the call."
        )
    else:
        text = nxt[1]
    convo.say("assistant", text)
    return TurnResult(text, changed or [], False)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse(raw: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply, fenced or not."""
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
