"""Drafting replies the model is not allowed to get wrong.

A prompt is advice. Under pressure - and a prospect texting "can you do 300?"
is pressure - advice becomes a rounding error. `src/business/pricing.py` says
this about discount instructions and `src/agents/flow.py` says it about
ordering; this module says it about text.

So nothing here trusts the draft. Every reply the model produces is parsed for
currency figures *before* it can be sent, and every figure found is pushed
through the same three checks the voice agent's `propose_price` runs:

    Gate.allow_quote(total)         phase + preconditions carried from the call
    thread.budget_floor             never under a number they already named
    CostModel.validate_quote(...)   never under the hard margin floor

A draft that fails any of them is not sent. It is fed back to the model as an
instruction - the same trick `flow.Gate` uses, where a refusal reads as "go
find out X" rather than as an error - and if the model still cannot produce a
clean draft after a few attempts, a number-free fallback goes out instead. The
model never gets to be the last word on a price.

Three rules are enforced structurally rather than prompted:

1. **No unvalidated number leaves.** Including on a thread with no cost model:
   if there is nothing to validate against, no figure may be stated at all.
2. **No escalation.** There is nobody to ask. "Let me check with the owner" is
   a lie with a delay on it, so the phrase itself is a refusal condition.
3. **No claiming the deal is closed.** The agent can ask for written
   confirmation - that is exactly how `Channel.INBOUND_SMS` evidence gets
   created - but it may never assert that confirmation exists.

Per-person figures are refused outright. A headcount is not a unit count; that
confusion is what produced the $74 quote on a $385 job, and it is invisible
once it has become an integer.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from src.business.pricing import QuoteVerdict
from src.messaging.thread import Thread

logger = logging.getLogger("messaging.closer")

#: LiveKit inference model id. Same key as the voice path, no new credential.
MODEL = os.getenv("SMS_LLM_MODEL", "openai/gpt-5.4-mini")

#: Hard ceiling on an outbound body. Long texts get segmented, arrive out of
#: order, and read like a wall - all three are bot tells.
MAX_LENGTH = 480

#: How many times the model may be handed its own refusal before we stop asking.
MAX_ATTEMPTS = 3


# -- the model seam ------------------------------------------------------


class Responder(Protocol):
    """Anything that turns a system prompt plus a transcript into text."""

    async def respond(self, system: str, messages: list[dict[str, str]]) -> str: ...


class LiveKitResponder:
    """The real model, over LiveKit inference.

    A fresh client inside `http_context.open()` per call: the shared aiohttp
    session is a context variable, and a client built outside one belongs to a
    session that may already be closed by the time the next text arrives.
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
            return "".join(parts).strip()


@dataclass
class ScriptedResponder:
    """A fixed list of drafts. For tests and offline rehearsal."""

    replies: list[str] = field(default_factory=list)
    default: str = "Thanks - let me know what works and I'll sort it out."
    calls: list[tuple[str, list[dict[str, str]]]] = field(default_factory=list)

    async def respond(self, system: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((system, list(messages)))
        if self.replies:
            return self.replies.pop(0)
        return self.default


# -- finding money in a draft --------------------------------------------

#: Times, blanked before the bare-decimal scan so "8:30" and "7.30pm" are not
#: read as prices. Blanked with spaces rather than removed so that character
#: offsets still line up with the original text.
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}[.:]\d{2}\s*(?:am|pm)\b|\b\d{1,2}\s*(?:am|pm)\b", re.I)

_MONEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\s?(?P<amt>\d[\d,]*(?:\.\d{1,2})?)"),
    re.compile(r"(?P<amt>\d[\d,]*(?:\.\d{1,2})?)\s*(?:dollars|usd|bucks)\b", re.I),
    re.compile(r"\b(?:usd|us\$)\s*(?P<amt>\d[\d,]*(?:\.\d{1,2})?)", re.I),
    # A bare figure written to the cent is a price. Two decimals is the tell -
    # nobody writes "we need 120.00 muffins".
    re.compile(r"(?<![\w.$])(?P<amt>\d[\d,]*\.\d{2})(?![\d%])"),
)

#: Only these mean "per item". Deliberately narrow: reading "per week" as
#: per-unit would multiply a total by the quantity and wave a below-floor
#: number straight through, which is the one direction that must never be wrong.
_PER_UNIT = re.compile(r"^\s*(?:each|apiece|a piece|per (?:unit|item|piece)|/\s*\w+)", re.I)
_PER_PERSON = re.compile(r"^\s*(?:per|a|/)\s*(?:person|head|guest|attendee|pax|people)", re.I)


class Basis(str, Enum):
    TOTAL = "total"
    PER_UNIT = "per_unit"
    PER_PERSON = "per_person"


@dataclass(frozen=True)
class Figure:
    """A currency amount found in a draft, and what it appears to mean."""

    raw: str
    amount: Decimal
    basis: Basis
    start: int

    def implied_total(self, qty: int) -> Decimal:
        """What this figure commits us to if it is said out loud.

        A per-unit figure implies qty times itself. Anything ambiguous is read
        as a total, because validating a per-week figure as a total can only
        cause a false refusal, while the reverse causes a real underquote.
        """
        if self.basis is Basis.PER_UNIT and qty >= 1:
            return (self.amount * qty).quantize(Decimal("0.01"))
        return self.amount

    def to_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "amount": str(self.amount), "basis": self.basis.value}


def extract_figures(text: str) -> list[Figure]:
    """Every currency figure in a draft, deduplicated by position."""
    if not text:
        return []
    cleaned = _TIME.sub(lambda m: " " * len(m.group()), text)

    found: dict[int, Figure] = {}
    for pattern in _MONEY_PATTERNS:
        for m in pattern.finditer(cleaned):
            start, end = m.span("amt")
            if any(s <= start < e for s, e in ((f.start, f.start + len(f.raw)) for f in found.values())):
                continue
            amount = Decimal(m.group("amt").replace(",", ""))
            tail = cleaned[end : end + 28]
            if _PER_PERSON.match(tail):
                basis = Basis.PER_PERSON
            elif _PER_UNIT.match(tail):
                basis = Basis.PER_UNIT
            else:
                basis = Basis.TOTAL
            found[start] = Figure(raw=m.group("amt"), amount=amount, basis=basis, start=start)
    return [found[k] for k in sorted(found)]


# -- language that is refused regardless of the numbers -------------------

_ESCALATION = re.compile(
    r"(check (?:with|in with) (?:the |my )?(?:owner|manager|boss|team|colleague)"
    r"|ask (?:the |my )?(?:owner|manager|boss|team)"
    r"|run (?:this|it|that) (?:by|past) (?:the |my )?\w+"
    r"|speak (?:to|with) (?:the |my )?(?:owner|manager|boss|team)"
    r"|get (?:approval|sign[- ]?off)"
    r"|have (?:someone|him|her|them) (?:call|text|reach out)"
    r"|(?:i'?ll|let me) (?:double[- ]?)?check (?:and|then)? ?(?:get back|come back|circle back)"
    r"|waiting (?:on|for) (?:the |my )?(?:owner|manager|boss|approval))",
    re.I,
)

_CLAIMS_CLOSED = re.compile(
    r"((?:you'?re|you are|we'?re|we are) (?:all set|confirmed|booked in|locked in)"
    r"|(?:your|the) (?:order|booking|reservation) is (?:now )?(?:confirmed|booked|locked in|in the system)"
    r"|i(?:'ve| have) (?:confirmed|booked you|locked (?:it|this) in|put you down|placed the order)"
    r"|consider it (?:done|booked)"
    r"|this is (?:now )?confirmed"
    r"|done deal"
    r"|it'?s (?:all )?confirmed)",
    re.I,
)


# -- the verdict on one draft ---------------------------------------------


@dataclass(frozen=True)
class DraftIssue:
    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class DraftCheck:
    """Whether a draft may be sent, and if not, what to do instead."""

    ok: bool
    figures: list[Figure] = field(default_factory=list)
    issues: list[DraftIssue] = field(default_factory=list)

    @property
    def instruction(self) -> str:
        """The refusal, written as the model's next instruction.

        Same convention as `flow.Gate`: being blocked should read as a task,
        not as a tool failure, because a model handed an error tends to
        apologise and try the same number again.
        """
        if self.ok:
            return ""
        return "Your draft was NOT sent. " + " ".join(i.detail for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "figures": [f.to_dict() for f in self.figures],
            "issues": [i.to_dict() for i in self.issues],
        }


def check_draft(thread: Thread, text: str) -> DraftCheck:
    """The whole guard. Pure, synchronous, and independently testable.

    Nothing in here consults the model, and nothing the model says can change
    an answer. That is the property worth having: the guard is the same code
    whether a language model, an operator or a test wrote the draft.
    """
    issues: list[DraftIssue] = []
    body = (text or "").strip()

    if not body:
        issues.append(DraftIssue("empty", "The draft was empty. Write one or two sentences."))
        return DraftCheck(ok=False, issues=issues)

    if len(body) > MAX_LENGTH:
        issues.append(
            DraftIssue(
                "too_long",
                f"It is {len(body)} characters; keep it under {MAX_LENGTH}. "
                "Cut it to two sentences.",
            )
        )

    if _ESCALATION.search(body):
        issues.append(
            DraftIssue(
                "escalation",
                "You offered to check with someone. There is nobody to ask - you "
                "are the only one on this thread. Decide inside the limits you "
                "were given, or say plainly what you cannot do.",
            )
        )

    if _CLAIMS_CLOSED.search(body):
        issues.append(
            DraftIssue(
                "claims_closed",
                "You stated the deal is confirmed. You may not - it is confirmed "
                "when they put it in writing, not when you say so. Ask them to "
                "reply with the quantity and total instead.",
            )
        )

    figures = extract_figures(body)
    currency = thread.costs.currency if thread.costs else "USD"

    for fig in figures:
        if fig.basis is Basis.PER_PERSON:
            issues.append(
                DraftIssue(
                    "per_person",
                    f"You priced {fig.raw} per person. A headcount is not an item "
                    "count - state the TOTAL for the order instead.",
                )
            )
            continue

        if not thread.can_quote_at_all:
            issues.append(
                DraftIssue(
                    "unvalidatable",
                    f"You stated {currency} {fig.raw}, but this thread has no "
                    "confirmed quantity or cost model, so no price can be "
                    "checked. Say no number at all until the quantity is settled.",
                )
            )
            continue

        total = fig.implied_total(thread.qty)

        # 1. Their own stated number. Checked before the Gate - the Gate refuses
        #    this too, but generically, and the specific message ("you are
        #    throwing away $311") is the one that actually changes the rewrite.
        floor_budget = thread.budget_floor
        if floor_budget and total < Decimal(str(floor_budget)):
            issues.append(
                DraftIssue(
                    "undercuts_budget",
                    f"You wrote {fig.raw}, but they already said they would pay "
                    f"{floor_budget:.2f}. Quoting under their own number throws "
                    f"away {(Decimal(str(floor_budget)) - total):.2f}. State "
                    f"{floor_budget:.2f} or more.",
                )
            )
            continue

        # 2. The Gate the call ran under: phase and preconditions. A thread
        #    whose call never reached QUALIFIED cannot start quoting by text.
        if refusal := thread.gate.allow_quote(float(total)):
            issues.append(DraftIssue("gate_blocked", f"{refusal} (you wrote {fig.raw})"))
            continue

        # 3. The margin floor, from the same CostModel the voice agent used.
        check = thread.costs.validate_quote(thread.qty, total)
        if check.verdict is QuoteVerdict.BELOW_FLOOR:
            issues.append(
                DraftIssue(
                    "below_floor",
                    f"{check.reason} Do not write {fig.raw}. The lowest total you "
                    f"may state is {currency} {check.floor}.",
                )
            )
            continue
        if check.verdict is QuoteVerdict.REQUIRES_APPROVAL:
            approved = thread.approved_total
            if approved is not None and total >= Decimal(str(approved)):
                continue  # cleared by the operator during the call
            issues.append(
                DraftIssue(
                    "requires_approval",
                    f"{check.reason} Nobody can approve it on a text thread, so "
                    f"you may not state it. The lowest total you may state is "
                    f"{currency} {check.target}.",
                )
            )
            continue

        # 4. Per-unit figures also have to clear the campaign's per-unit floor.
        if (
            fig.basis is Basis.PER_UNIT
            and thread.campaign
            and fig.amount < Decimal(str(thread.campaign.envelope.min_price))
        ):
            issues.append(
                DraftIssue(
                    "below_unit_floor",
                    f"{fig.raw} each is under the {currency} "
                    f"{thread.campaign.envelope.min_price:.2f} per-unit floor.",
                )
            )

    return DraftCheck(ok=not issues, figures=figures, issues=issues)


# -- generating a reply ---------------------------------------------------


class ReplyStatus(str, Enum):
    OK = "ok"
    """First draft passed the guard."""

    REGENERATED = "regenerated"
    """An earlier draft was refused; a later one passed."""

    FALLBACK = "fallback"
    """The model never produced a clean draft. A number-free line went instead."""

    BLOCKED = "blocked"
    """Nothing may be sent at all - they opted out."""


#: Sent when the model cannot produce something the guard accepts. Contains no
#: figures by construction, so it cannot misquote; it keeps the thread alive
#: and hands the next move back to the prospect.
FALLBACK_PRICE = (
    "I want to get this right rather than guess, so I won't throw a number at "
    "you I can't stand behind. I can't go lower than what we discussed, but I "
    "can look at the size of the order or the date. Which would help more?"
)
FALLBACK_GENERAL = (
    "Thanks for coming back to me. Can you tell me which part you'd like me to "
    "sort out first, and I'll come straight back with specifics?"
)


@dataclass
class Reply:
    """A draft that has already survived the guard, plus what it took."""

    text: str
    status: ReplyStatus
    attempts: int = 0
    check: DraftCheck | None = None
    rejected: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sendable(self) -> bool:
        return self.status is not ReplyStatus.BLOCKED and bool(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status.value,
            "attempts": self.attempts,
            "rejected": self.rejected,
            "check": self.check.to_dict() if self.check else None,
        }


def build_system_prompt(thread: Thread) -> str:
    """The prompt. Restates the limits the guard enforces anyway.

    Belt and braces: the guard is what makes a bad number impossible, but a
    model that knows the floor produces a usable draft on the first attempt
    instead of the third, and every extra attempt is latency the prospect
    reads as a bot.
    """
    business = (thread.campaign.name if thread.campaign else "the business")
    lines = [
        f"You are continuing a sales conversation by TEXT for {business}. You "
        "already spoke to this person on the phone. Pick up where that call left "
        "off - do not reintroduce yourself at length.",
        "",
        "## The limits you inherited from that call",
        thread.constraints_summary(),
        "",
        "## Hard rules",
        "Never state a total below the hard floor. Never state a total below a "
        "number they have already said they would pay. State TOTALS, not "
        "per-person prices - a headcount is not an item count.",
        "There is nobody to ask. No owner, no manager, no approvals. Never say "
        "you will check with anyone or get back to them after checking. Close "
        "inside the limits or politely stop.",
        "Never say the order is confirmed, booked or all set. It is confirmed "
        "when THEY put it in writing. Your job is to get that writing: ask them "
        "to reply with the quantity and the total in one message.",
        "",
        "## Voice",
        "Two sentences, maximum. Plain words, no exclamation marks, no emoji. "
        "One question at a time. This is a text, not an email.",
        "Reply with the message body only - no greeting block, no signature, no "
        "quotation marks around it.",
    ]
    if not thread.can_quote_at_all:
        lines.insert(
            len(lines) - 4,
            "You do not have a validated quantity for this thread, so you may "
            "not state ANY price. Ask what they need instead.",
        )
    return "\n".join(lines)


async def generate_reply(
    thread: Thread,
    responder: Responder,
    *,
    attempts: int = MAX_ATTEMPTS,
    extra_instruction: str = "",
) -> Reply:
    """Draft, check, and re-draft until the guard is satisfied.

    Returns a `Reply` whose `text` has *always* passed `check_draft`, including
    in the fallback case. There is no path out of this function that returns an
    unvalidated number.
    """
    if thread.opted_out:
        return Reply(
            text="",
            status=ReplyStatus.BLOCKED,
            check=DraftCheck(
                ok=False,
                issues=[DraftIssue("opted_out", "They asked us to stop texting.")],
            ),
        )

    system = build_system_prompt(thread)
    messages = thread.history_for_model()
    if extra_instruction:
        messages = messages + [{"role": "user", "text": f"[NOTE] {extra_instruction}"}]

    rejected: list[dict[str, Any]] = []
    used = 0

    for attempt in range(max(1, attempts)):
        used = attempt + 1
        try:
            draft = (await responder.respond(system, messages)).strip()
        except Exception as exc:  # noqa: BLE001
            # A model outage must not take the thread down with it. Fall back
            # to a line that says nothing we cannot stand behind.
            logger.warning("responder failed on attempt %d: %s", used, exc)
            rejected.append({"draft": None, "error": str(exc)})
            break

        draft = draft.strip().strip('"').strip()
        check = check_draft(thread, draft)
        if check.ok:
            return Reply(
                text=draft,
                status=ReplyStatus.OK if attempt == 0 else ReplyStatus.REGENERATED,
                attempts=used,
                check=check,
                rejected=rejected,
            )

        logger.info("draft refused (attempt %d): %s", used, check.instruction)
        rejected.append({"draft": draft, "issues": [i.to_dict() for i in check.issues]})
        messages = messages + [
            {"role": "assistant", "text": draft},
            {"role": "user", "text": f"[SYSTEM] {check.instruction} Rewrite the message."},
        ]

    priced = any(r.get("issues") and any(
        i["kind"] in {"below_floor", "undercuts_budget", "requires_approval",
                      "per_person", "unvalidatable", "below_unit_floor"}
        for i in r["issues"]
    ) for r in rejected if r.get("issues"))

    fallback = FALLBACK_PRICE if priced else FALLBACK_GENERAL
    check = check_draft(thread, fallback)
    if not check.ok:  # pragma: no cover - the fallbacks carry no figures
        fallback = FALLBACK_GENERAL
        check = check_draft(thread, fallback)
    return Reply(
        text=fallback,
        status=ReplyStatus.FALLBACK,
        attempts=used,
        check=check,
        rejected=rejected,
    )


# -- reading the prospect's side ------------------------------------------

_BUDGET_PHRASE = re.compile(
    r"(?:budget (?:is|of|around|about)?|can (?:do|pay|spend|go (?:to|up to))|"
    r"we(?:'re| are)? (?:paying|at)|willing to (?:pay|spend)|"
    r"(?:i|we) (?:pay|paid|spend)|happy (?:to pay|with)|works? (?:for|at))"
    r"[^\d$]{0,12}(\$?\s?\d[\d,]*(?:\.\d{1,2})?)",
    re.I,
)

_STOP = re.compile(r"^\s*(stop|unsubscribe|quit|cancel|end|opt\s?out)\b", re.I)


def read_inbound(thread: Thread, body: str) -> dict[str, Any]:
    """Pull the facts an inbound text establishes, without acting on them.

    Only two things are extracted, both conservative:

    - **A stated budget.** Recorded via `Thread.note_stated_budget`, which only
      ever raises the conversational floor. If they name a number, we are not
      allowed to quote under it afterwards.
    - **An opt-out.** STOP means stop, immediately and permanently.
    """
    out: dict[str, Any] = {"stated_budget": None, "opt_out": False}
    text = (body or "").strip()

    if _STOP.match(text):
        thread.opted_out = True
        out["opt_out"] = True
        return out

    m = _BUDGET_PHRASE.search(text)
    if m:
        raw = m.group(1).replace("$", "").replace(",", "").strip()
        try:
            amount = float(raw)
        except ValueError:
            amount = 0.0
        if amount > 0:
            thread.note_stated_budget(amount)
            out["stated_budget"] = amount
    return out
