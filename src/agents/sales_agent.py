"""The sales agent: one campaign, one call, one receipt.

This is the integration layer. Every safety property lives in a module this
file imports, and the arrangement is deliberate - the agent gets tools that
*propose*, and validators that *dispose*:

    check_capacity  -> CapacityLedger.hold()      may return None
    propose_price   -> CostModel.validate_quote() may refuse
    close_order     -> Receipt.claim()            born UNVERIFIED

There is no tool that lets the model mark something done. The worst honest
outcome is an escalation or an UNVERIFIED claim; a fabricated sale is not
reachable through this API, which is the disqualification condition in the
rules we are being judged against.

The operator escalation deserves a note. VoiceOS has no public API, so it
cannot receive a programmatic callback from us. `OperatorChannel` is therefore
an interface with a local voice implementation (mic -> cached Whisper -> answer)
that we fully control. If a1mobile can conference a second leg into the live
call, `ConferenceOperator` is the better path and swaps in without touching the
agent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from src.business.campaign import Campaign
from src.business.capacity import CapacityLedger, ExpiredHold, Hold
from src.business.pricing import CostModel, QuoteVerdict
from src.agents.negotiation import (
    CompetitorBook,
    Limits,
    NegotiationState,
    Tactic,
    guidance,
)
from src.agents.flow import Gate, Phase
from src.verify.receipts import Channel, Evidence, Receipt

logger = logging.getLogger("agents.sales")

#: Refresh the capacity hold this often while a call is live. Short enough that
#: a dropped call frees inventory quickly; refreshed often enough that a long
#: negotiation never loses its reservation.
HOLD_TTL_SECONDS = 120.0
HEARTBEAT_SECONDS = 45.0


# -- operator escalation --------------------------------------------------


class OperatorChannel(ABC):
    """How the agent reaches the human while holding the line."""

    @abstractmethod
    async def ask(self, question: str, *, timeout: float = 90.0) -> str | None:
        """Return the operator's answer, or None if they did not respond.

        None must be treated as 'no'. An unanswered escalation that defaults to
        yes is exactly the failure the envelope exists to prevent.
        """


class ConsoleOperator(OperatorChannel):
    """Typed answers. Reliable for rehearsal; not a demo centrepiece."""

    async def ask(self, question: str, *, timeout: float = 90.0) -> str | None:
        print(f"\n>>> OPERATOR NEEDED: {question}\n>>> answer: ", end="", flush=True)
        try:
            return await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, input), timeout
            )
        except asyncio.TimeoutError:
            logger.warning("operator did not answer in %.0fs - treating as no", timeout)
            return None


class VoiceOperator(OperatorChannel):
    """Spoken answers, transcribed locally with the cached Whisper model.

    Local because the operator's answer is on the critical path of a live call:
    a network round trip to a hosted STT is latency we control by not spending.
    """

    def __init__(self, record_seconds: float = 6.0) -> None:
        self.record_seconds = record_seconds

    async def ask(self, question: str, *, timeout: float = 90.0) -> str | None:
        print(f"\n>>> OPERATOR NEEDED (speak your answer): {question}", flush=True)
        try:
            wav = await asyncio.wait_for(self._record(), timeout)
        except asyncio.TimeoutError:
            return None

        from src.verify.transcribe import transcribe

        result = await asyncio.to_thread(transcribe, wav)
        answer = (result.get("text") or "").strip()
        logger.info("operator said: %s", answer)
        return answer or None

    async def _record(self) -> str:
        out = Path("evidence/operator") / "answer.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            "sox", "-d", "-r", "16000", "-c", "1", str(out),
            "trim", "0", str(self.record_seconds),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return str(out)


#: Whole words only. Bare substring matching reads "now" and "know" as "no",
#: so a natural approval - "yes, go ahead now" - parses as a denial. It fails
#: safe rather than overcommitting, but it silently inverts the operator's
#: answer, which is worse than useless on a live call.
#: `not` and `unsure` are here specifically because "I'm not sure" would
#: otherwise match `sure` in the positive list and read as approval - the worst
#: available misparse, since hedging is exactly what an uncertain operator says.
#: Consequently "no problem, go ahead" also reads as a refusal. That is the
#: right trade: a missed sale is recoverable, an unauthorised commitment is not.
_NEGATIVE = re.compile(
    r"\b(no|nope|nah|not|never|unsure|don't|do not|reject|decline|declined"
    r"|deny|denied|skip|stop|hold off|wait)\b",
    re.I,
)
_POSITIVE = re.compile(
    r"\b(yes|yeah|yep|yup|ok|okay|approve|approved|go ahead|do it|sure|fine)\b",
    re.I,
)


def _is_yes(answer: str | None) -> bool:
    """Conservative parse. Anything ambiguous, or unanswered, is a no.

    A negative anywhere beats a positive: "yes but no, not today" is a refusal,
    and the cost of reading a refusal as consent is an order the business
    cannot honour.
    """
    if not answer:
        return False
    text = answer.strip()
    if _NEGATIVE.search(text):
        return False
    return bool(_POSITIVE.search(text))




#: The ops board runs in its own uvicorn process, so the in-memory registry it
#: exposes is invisible from here. Push over HTTP instead - fire-and-forget,
#: short timeout, every failure swallowed. A dashboard that is down, slow, or
#: not running must never add a millisecond to a live phone call.
OPSBOARD_URL = os.getenv("OPSBOARD_URL", "http://127.0.0.1:8130")


def _push(path: str, payload: dict[str, Any]) -> None:
    async def send() -> None:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=1.5) as c:
                await c.post(f"{OPSBOARD_URL}{path}", json=payload)
        except Exception:  # noqa: BLE001
            pass  # the board is optional; the call is not

    try:
        asyncio.get_running_loop().create_task(send())
    except RuntimeError:
        pass

_AGREED = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|deal|agreed|sounds good|let's do|lets do"
    r"|do it|book it|go ahead|that works|i'll take|we'll take|perfect|great)\b",
    re.I,
)

_WORDS = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",
          9:"nine",10:"ten",20:"twenty",30:"thirty",40:"forty",50:"fifty",
          100:"hundred",200:"two hundred",300:"three hundred",400:"four hundred",
          500:"five hundred",600:"six hundred",800:"eight hundred",1000:"thousand"}


def _spoken_number(n: int) -> str:
    """How a caller says a number aloud - STT writes words, not digits."""
    return _WORDS.get(n, str(n))


# -- the agent ------------------------------------------------------------


@dataclass
class CallSession:
    """Everything one call needs. Assembled per call, never shared."""

    campaign: Campaign
    ledger: CapacityLedger
    costs: CostModel
    receipt: Receipt
    operator: OperatorChannel
    book: CompetitorBook = field(default_factory=CompetitorBook)
    state: NegotiationState | None = None
    hold: Hold | None = None
    escalations: list[str] = field(default_factory=list)

    allow_escalation: bool = False
    """Whether the agent may hand a decision back to a human mid-call.

    Off by default. With escalation available, "ask the owner" becomes the
    agent's escape hatch from every hard case, and the envelope stops being a
    boundary and becomes a speed bump. With it off, the limits in the campaign
    are absolute: the agent finds a deal inside them or politely ends the call.
    """

    gate: Gate = field(default_factory=Gate)
    """Phase + precondition enforcement. See src/agents/flow.py - the $74 call
    proved that per-tool validation is not enough; ordering has to be structural."""

    heard: list[str] = field(default_factory=list)
    """Verbatim STT of the CUSTOMER's speech.

    The LLM is the component that could invent an agreement; it cannot write
    here. Checking a claim against the caller's own transcribed words catches
    exactly the fabrication the rules disqualify - an agent reporting a yes that
    was never said - without asking the customer to send anything.
    """

    pending_budget: float | None = None
    """A budget heard before capacity was reserved.

    The agent is told to ask what they pay *before* quoting - it is the first
    move on the negotiation ladder - which means record_their_position legitimately
    runs before check_capacity. It used to bail in that case and discard the
    number, so the gate never saw `asked_current_spend`, the quote was blocked,
    and the call could not close. The tools were punishing the exact ordering
    the instructions ask for.
    """

    approved_total: float | None = None
    """The lowest total the operator has explicitly authorised on this call.

    Without this, approval has nowhere to live: the agent asks, you say yes, it
    re-proposes the same number, and the validator - which never heard about the
    approval - tells it to ask again. That loops until the customer hangs up.
    """

    @property
    def limits(self) -> Limits:
        return Limits(
            max_discount_pct=self.campaign.envelope.max_discount_pct,
            floor_total=float(self.costs.floor_price(self.state.qty))
            if self.state
            else 0.0,
        )


class SalesAgent(Agent):
    def __init__(self, session: CallSession, instructions: str) -> None:
        super().__init__(instructions=instructions)
        self.s = session
        self._heartbeat: asyncio.Task | None = None

    # -- capacity ----------------------------------------------------

    @function_tool
    async def confirm_units(
        self, ctx: RunContext, units: int, headcount: int = 0, basis: str = ""
    ) -> str:
        """Establish how many ITEMS the order is, before anything is reserved.

        Required before check_capacity. This exists because a headcount and an
        item count are both just integers by the time they reach a tool, and
        confusing them silently misprices the whole job - a thirty-person order
        priced as thirty muffins came out at a third of its real value.

        Args:
            units: Total number of individual items.
            headcount: Number of people, if they gave one.
            basis: How you got there, e.g. "30 people x 2 pastries each".
        """
        if units <= 0:
            return "Units must be positive. Ask how many items they need."

        self.s.gate.note_units(units, headcount or None)
        detail = f" ({basis})" if basis else ""
        if headcount and units == headcount:
            return (
                f"Recorded {units} items{detail} - but that equals the headcount "
                f"exactly. If each person eats more than one item this is too "
                f"low. Say your assumption out loud and let them correct it."
            )
        return f"Recorded {units} items{detail}. You may now call check_capacity."

    @function_tool
    async def check_capacity(self, ctx: RunContext, qty: int) -> str:
        """Reserve capacity before quoting. Call this BEFORE naming any number.

        Args:
            qty: Number of UNITS (individual items), never the headcount.
                If they said "30 people", that is not 30 units - ask how many
                items per person, or assume the campaign default. Observed live:
                treating a 30-person order as 30 muffins priced it at a third of
                the real job.
        """
        if refusal := self.s.gate.allow_capacity(qty):
            self._push_refusal(refusal)
            return refusal

        hold = self.s.ledger.hold(qty, ttl_seconds=HOLD_TTL_SECONDS)
        if hold is None:
            avail = self.s.ledger.available()
            # Create the negotiation state even though nothing was reserved.
            # Without this the call dead-ends: every other tool answers "call
            # check_capacity first", which is advice the agent has already
            # followed and which cannot succeed. Observed live - the agent said
            # "I'm checking that now" and then went silent, which on a phone
            # call is the worst possible failure.
            if self.s.state is None and avail > 0:
                self.s.state = NegotiationState(
                    opening_total=float(self.s.costs.target_price(avail)),
                    current_total=float(self.s.costs.target_price(avail)),
                    qty=avail,
                )
            return (
                f"CANNOT FULFIL {qty}. Only {avail} {self.s.ledger.unit} are "
                f"available. Do not promise {qty}. You have three moves and must "
                f"pick one now, out loud: offer {avail} instead; ask which other "
                f"date would suit; or call ask_operator to ask the owner whether "
                f"they can make {qty}. Do not say you are checking unless you are "
                "calling ask_operator in the same turn."
            )

        self.s.hold = hold
        self.s.state = NegotiationState(
            opening_total=float(self.s.costs.target_price(qty)),
            current_total=float(self.s.costs.target_price(qty)),
            qty=qty,
        )
        if self.s.pending_budget is not None:
            self.s.state.their_stated_budget = self.s.pending_budget
            self.s.state.asked_current_spend = True
        self.s.gate.note_capacity_held()
        self._push_state()
        self._start_heartbeat()
        return (
            f"Reserved {qty} {self.s.ledger.unit} for this call. "
            f"Opening price {self.s.state.opening_total:.2f}. "
            "This reservation is not a sale - it expires if the call drops."
        )

    def _push_state(self) -> None:
        """Mirror the live call onto the ops board."""
        st = self.s.state
        _push("/api/call", {
            "business": os.getenv("BUSINESS_NAME", "Rosewater Bakehouse"),
            "phase": self.s.gate.phase.value,
            "units": self.s.gate.facts.units,
            "capacity_total": self.s.ledger.total,
            "capacity_available": self.s.ledger.available(),
            "quote": st.current_total if st else None,
            "floor": float(self.s.costs.floor_price(st.qty)) if st else None,
            "budget": st.their_stated_budget if st else None,
            "concessions": st.concessions_made if st else 0,
            "declines": st.declines_received if st else 0,
        })

    def _push_refusal(self, raw: str | None) -> None:
        if raw and ("BLOCKED" in raw or "REFUSED" in raw or "CANNOT" in raw):
            _push("/api/refusal", {"raw": raw, "phase": self.s.gate.phase.value})

    def _start_heartbeat(self) -> None:
        """Keep the hold alive for as long as the call is."""
        if self._heartbeat is not None:
            return

        async def beat() -> None:
            while self.s.hold is not None:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                try:
                    self.s.ledger.extend(self.s.hold.id, HOLD_TTL_SECONDS)
                except (ExpiredHold, Exception) as exc:  # noqa: BLE001
                    logger.warning("hold heartbeat failed: %s", exc)
                    return

        self._heartbeat = asyncio.create_task(beat())

    # -- pricing -----------------------------------------------------

    @function_tool
    async def propose_price(self, ctx: RunContext, total: float) -> str:
        """Check a price BEFORE saying it out loud. Never quote an unchecked number.

        Args:
            total: The total you want to offer, in dollars.
        """
        if self.s.state is None:
            return "Call check_capacity first - I do not know the quantity yet."

        if refusal := self.s.gate.allow_quote(total):
            self._push_refusal(refusal)
            return refusal

        # Never quote below a number they have already offered. Observed live:
        # the buyer said $385, the agent said "I can't match that" and countered
        # at $74 - giving away $311 nobody asked it to. The floor stops us
        # selling at a loss; it does nothing about selling far under a buyer who
        # was ready to pay more, which is the commoner and costlier mistake.
        budget = self.s.state.their_stated_budget
        if budget and total < budget:
            self.s.state.current_total = budget
            return (
                f"DO NOT offer {total:.2f}. They already said they would pay "
                f"{budget:.2f}, so quoting under it throws away "
                f"{budget - total:.2f}. Offer {budget:.2f} and close."
            )

        check = self.s.costs.validate_quote(self.s.state.qty, total)
        if check.verdict is QuoteVerdict.BELOW_FLOOR:
            floor = self.s.costs.floor_price(self.s.state.qty)
            return (
                f"REFUSED. {total:.2f} is below the floor of {floor:.2f}. "
                f"{check.reason} Do NOT say {total:.2f} out loud. The lowest you "
                f"may offer is {floor:.2f}."
            )
        if check.verdict is QuoteVerdict.REQUIRES_APPROVAL:
            # An operator approval covers this number and anything better for
            # us. Without this branch the agent re-asks forever.
            approved = self.s.approved_total
            if approved is not None and total >= approved:
                self.s.state.current_total = total
                return (
                    f"APPROVED. The owner already authorised {approved:.2f} on "
                    f"this call, so {total:.2f} is cleared. Offer it."
                )
            if not self.s.allow_escalation:
                # The floor is the limit; the target is an aspiration. With
                # escalation disabled there is nobody to grant the difference,
                # so REQUIRES_APPROVAL would be a dead end: the agent cannot
                # quote, cannot ask, and cannot close. Observed live - the agent
                # agreed $400 aloud, the tool never validated it, and the call
                # produced no claim at all.
                self.s.state.current_total = total
                self.s.gate.note_quoted()
                self._push_state()
                target = self.s.costs.target_price(self.s.state.qty)
                return (
                    f"APPROVED. {total:.2f} clears the floor. It is under the "
                    f"{target} target, so do not go lower - but you may offer "
                    "this and close on it."
                )
            return (
                f"NOT YET. {total:.2f} is above the floor but below target. "
                f"{check.reason} Use ask_operator before offering this, and pass "
                f"proposed_total={total:.2f} so the approval is recorded."
            )

        self.s.state.current_total = total
        self.s.gate.note_quoted()
        self._push_state()
        return f"APPROVED. You may offer {total:.2f}."

    @function_tool
    async def record_their_position(
        self, ctx: RunContext, current_vendor: str = "",
        current_spend: float = 0.0, their_offer: float = 0.0,
    ) -> str:
        """Log what they told you about price. Call this as soon as they say it.

        These two numbers pull in OPPOSITE directions and must not be confused.

        Args:
            current_vendor: Who they currently buy from, if named.
            current_spend: What they pay their CURRENT supplier. This is a number
                to BEAT - undercutting it is the whole pitch. It is not a limit
                on us and never raises our price.
            their_offer: What they have said they will pay US. This is a floor
                for our quote - never offer less than a number they have already
                agreed to pay.
        """
        # No state yet is normal: asking what they pay comes before quoting.
        if current_vendor and self.s.state is not None:
            self.s.state.competitor_named = current_vendor
            self.s.state.asked_current_spend = True

        # Only an offer to US constrains our quote. What they pay someone else
        # is a target, not a floor - clamping to it means never winning on price.
        if their_offer > 0:
            if self.s.state is not None:
                self.s.state.their_stated_budget = their_offer
            else:
                self.s.pending_budget = their_offer
        if current_spend > 0 and self.s.state is not None:
            self.s.state.competitor_price = current_spend

        self.s.gate.note_position(their_offer if their_offer > 0 else None)
        if current_spend > 0 and their_offer <= 0:
            # Heard a competitor price and nothing else: the gate still needs to
            # know discovery happened, or the quote stays blocked.
            self.s.gate.facts.asked_current_spend = True

        target = (
            self.s.book.undercut_target(current_vendor, self.s.state.qty)
            if current_vendor and self.s.state is not None
            else None
        )
        note = (f"Logged. vendor={current_vendor or 'unknown'} "
                f"they-pay-now={current_spend or 'unstated'} "
                f"offered-us={their_offer or 'none'}.")
        if current_spend > 0 and their_offer <= 0:
            beat = max(float(self.s.costs.floor_price(self.s.state.qty))
                       if self.s.state else 0.0, current_spend * 0.9)
            note += (f" They pay {current_spend:.2f} elsewhere - BEAT it. You may "
                     f"go as low as {beat:.2f}. Quote under their current price.")
        if target:
            note += f" We have their pricing: beating it would be about {target:.2f}."
        else:
            note += " No pricing data for them - do NOT guess a number at them."
        return note

    @function_tool
    async def next_move(self, ctx: RunContext) -> str:
        """Ask what to do next when the conversation stalls or they push back."""
        if self.s.state is None:
            return "Call check_capacity first."

        packet = guidance(self.s.state, self.s.limits)
        if packet["tactic"] == Tactic.OFFER_CONCESSION.value:
            self.s.state.concessions_made += 1
        elif packet["tactic"] == Tactic.EMPHASISE_VALUE.value:
            self.s.state.value_framed = True
        elif packet["tactic"] == Tactic.RESTRUCTURE.value:
            self.s.state.restructure_offered = True

        out = packet["instruction"]
        if "quote_this_total" in packet:
            out += f"\nThe number to offer is {packet['quote_this_total']:.2f}."
        return out

    @function_tool
    async def they_declined(self, ctx: RunContext) -> str:
        """Log a refusal. Two of these and the call ends politely."""
        if self.s.state is None:
            return "Call check_capacity first."
        self.s.state.declines_received += 1
        self.s.gate.note_pushback()
        remaining = self.s.limits.max_declines - self.s.state.declines_received
        if remaining <= 0:
            return "That is the second no. Thank them warmly and end the call. Do not pitch again."
        return (
            "Logged. If they named a number, call propose_price with THEIR number "
            "right now before you respond - it may well clear the floor. Only if "
            "the tool refuses it should you offer something else."
        )

    # -- escalation --------------------------------------------------

    @function_tool
    async def ask_operator(
        self, ctx: RunContext, question: str, proposed_total: float = 0.0
    ) -> str:
        """Ask the business owner for permission, while holding the line.

        Use this whenever the deal would go outside what you were authorised to
        agree. Tell the customer you are checking, then call this.

        Args:
            question: One sentence, with the specific numbers.
            proposed_total: The exact total you want cleared. Pass this whenever
                the question is about price, or the approval cannot be recorded
                and you will be sent back here again.
        """
        if not self.s.allow_escalation:
            # Deliberately not an error - the model gets a usable instruction.
            self.s.receipt.note(f"escalation suppressed: {question}")
            logger.info("escalation suppressed (disabled): %s", question)
            return (
                "There is nobody to ask. You must resolve this yourself inside "
                "the limits you were given. Offer the best deal those limits "
                "allow, restructure the order, or thank them and end the call. "
                "Do not tell the customer you are checking with anyone."
            )

        self.s.escalations.append(question)
        self.s.gate.note_escalating()
        logger.warning("ESCALATION: %s", question)

        answer = await self.s.operator.ask(question)
        approved = _is_yes(answer)

        # An audit note, not a Claim. Escalations are internal control flow;
        # filing them as claims puts a permanently unverifiable row in every
        # receipt and drags a clean call down to PARTIAL.
        self.s.receipt.note(
            f"escalation: {question} -> answer={answer!r} "
            f"({'approved' if approved else 'refused'})"
        )

        self.s.gate.note_operator(proposed_total if approved and proposed_total > 0 else None)
        if approved:
            if proposed_total > 0:
                self.s.approved_total = proposed_total
            return (
                f"APPROVED by the owner: {answer!r}. You may proceed with exactly "
                f"that. Call propose_price to confirm the number before saying it."
            )
        return (
            f"NOT APPROVED (answer was {answer!r}). Do not offer it. Tell them you "
            "cannot do that today and offer what you can, or close politely."
        )

    # -- closing -----------------------------------------------------

    @function_tool
    async def close_order(
        self, ctx: RunContext, qty: int, total: float, when: str, confirm_to: str
    ) -> str:
        """File the agreed order. Call this the MOMENT they say yes.

        Call it before you ask for written confirmation, not after. Observed
        live: the agent agreed terms, asked for an email, and never filed
        anything - so the call produced no claim at all and the receipt read
        "nothing was attempted". Filing first is what gives the confirmation
        something to attach to.

        This does NOT complete the sale.

        Args:
            qty: Units agreed.
            total: Total price agreed.
            when: Delivery date and time as they stated it.
            confirm_to: The number or email they will send confirmation to.
        """
        # Always file. A blocked close used to return without creating a claim,
        # so a call that reached agreement produced a receipt reading "nothing
        # was attempted" - the one outcome worse than an honest failure, because
        # it loses the evidence that anything happened at all. The verdict still
        # tells the truth; only the silence was wrong.
        blocked = self.s.gate.allow_close()

        claim = self.s.receipt.claim(
            description=f"{qty} {self.s.ledger.unit} for {total:.2f}, delivered {when}",
            expected_side_effect=(
                f"written confirmation arrives at {confirm_to} stating quantity "
                f"{qty} and total {total:.2f}"
            ),
        )
        claim.attach_evidence(
            Evidence(
                channel=Channel.AGENT_ASSERTION,
                summary=f"Caller agreed to {qty} at {total:.2f} for {when}",
            )
        )

        if blocked:
            claim.attach_evidence(Evidence(
                channel=Channel.INDEPENDENT_TRANSCRIPT,
                summary=f"Filed out of sequence: {blocked}",
                supports=False,
            ))
            return (
                f"Filed, but NOT confirmed - {blocked} Fix that and call "
                "close_order again; the order is recorded either way."
            )

        spoken = " ".join(self.s.heard).lower()
        said_yes = bool(_AGREED.search(spoken))
        said_qty = str(qty) in spoken or _spoken_number(qty) in spoken
        amount = f"{total:.0f}"
        said_total = amount in spoken or _spoken_number(int(total)) in spoken

        if said_yes and (said_qty or said_total):
            claim.attach_evidence(Evidence(
                channel=Channel.INDEPENDENT_TRANSCRIPT,
                summary=("Caller's own transcribed speech contains an agreement "
                         f"and the terms (qty={said_qty}, total={said_total})"),
                raw={"heard": self.s.heard[-12:]},
            ))
            return (
                f"Filed and VERIFIED against the caller's own words. Confirm the "
                f"terms back to them once - {qty} for {total:.2f}, {when} - then "
                "close warmly. Do not ask them to send anything."
            )

        claim.attach_evidence(Evidence(
            channel=Channel.INDEPENDENT_TRANSCRIPT,
            summary=("The caller's transcribed speech does not contain a clear "
                     "agreement to these terms."),
            supports=False,
            raw={"heard": self.s.heard[-12:]},
        ))
        return (
            "Filed, but the caller has not clearly agreed out loud yet. Ask a "
            f"direct closing question naming the numbers - \"{qty} for "
            f"{total:.2f}, {when} - shall I book that?\" - and wait for a yes."
        )

        return (
            "Filed as UNVERIFIED. The sale is not real until written "
            f"confirmation reaches {confirm_to}. Before you hang up: read the "
            f"quantity, date and total back to them, and ask them to text or "
            "email it now while you are still on the line."
        )

    # -- lifecycle ---------------------------------------------------

    async def finish(self, *, confirmed: bool) -> Receipt:
        """Settle capacity and write the receipt. Always called, even on a drop."""
        if self._heartbeat:
            self._heartbeat.cancel()

        if self.s.hold is not None:
            try:
                if confirmed:
                    self.s.ledger.commit(self.s.hold.id)
                else:
                    self.s.ledger.release(self.s.hold.id)
            except Exception as exc:  # noqa: BLE001
                logger.error("could not settle hold %s: %s", self.s.hold.id, exc)
            self.s.hold = None

        self.s.receipt.save(Path("evidence"))
        return self.s.receipt


def build_instructions(campaign: Campaign, business: dict[str, Any]) -> str:
    """The system prompt. Tool contracts are restated because models under
    negotiation pressure drift toward reassurance, and the tools are the only
    thing standing between that drift and a fabricated sale."""
    return f"""
You are calling on behalf of {business.get('name', 'a local business')} to sell
{campaign.offer}.

## Hard rules
Quantities are UNITS, not people. "Thirty people" is not thirty items - ask how
many items each, or say what you assume out loud so they can correct you. Get
this wrong and you price a third of the job.

Never quote below a number they have already named. If they say they pay four
hundred, do not counter at less than four hundred. Their stated price is your
floor for the conversation, not your ceiling.

The moment they name a QUANTITY, call confirm_units and then check_capacity -
before you say anything at all about whether you can do it. Do not say "we can
handle that", do not say "let me see", do not keep talking. Observed live: a
buyer asked for 800 against a 600 ceiling and the agent only discovered it
could not after being asked directly. You do not know your own capacity. The
tool does.

Whenever they name a number, run THAT number through propose_price before you
say anything about it. Refusing a price you have not checked is as much a
mistake as offering one - observed live: a buyer offered $400 against a $385.72
floor and the agent said "I can't do that" without ever checking, and lost a
deal it was allowed to make. You do not know what you can accept. The tool does.

Never say a price you have not passed through propose_price. Never promise a
quantity you have not reserved with check_capacity. If either tool refuses, the
answer is no - do not negotiate with the tool, and do not find another way to
say the number. The limits you were given are absolute. There is nobody to ask and no
approval to wait for. If a deal cannot be made inside them, say so plainly and
offer what you can - never imply you are checking with someone.

Never claim the order is confirmed. It is confirmed when they send something in
writing, not when they say yes on the phone. Before hanging up, always get them
to text or email the details while you are still on the line.

## How to sell
Ask what they pay now before you offer anything. Frame on value before you
touch price. Concede once, not repeatedly. If price is still the obstacle,
change the shape of the deal rather than cutting further. Never disparage
whoever they currently use. When you are stuck, call next_move.

Two refusals and you thank them and end the call. Log each one with
they_declined.

## Opening
Lead with a real introduction and an actual offer - not a status announcement.
Give your name and the business, say what you do in a handful of words, and ask
a direct question that invites a yes. Name one concrete reason to care: baked
that morning, delivery and setup included, usually cheaper than what they pay
now. Then stop talking.

Good: "Hi, this is Sam from Rosewater Bakehouse - we do breakfast catering in
the Sunset, everything baked that morning. Would you be interested in catering
for your event Friday? We can usually beat what you're paying and the food is
fresher."

Bad: "Hi, this is Rosewater Bakehouse calling about a weekly breakfast catering
order." That announces a topic, asks for nothing, and gives them no reason to
keep listening.

## Closing
When they agree, read the terms back in one sentence - quantity, day, total -
and get an explicit spoken yes. Then call close_order. Their spoken agreement,
in their own recorded words, is the confirmation.

Do NOT ask them to text or email anything. Do not hold the line waiting for a
message. Asking a busy stranger to send paperwork mid-call is how these calls
die - close on the phone and let them go.

Be easy to say yes to. Anything the tool approves is a deal worth taking - do
not hold out for a better number once propose_price has cleared one.

## Voice
Short sentences. You are on the phone with someone who is busy. Speak numbers clearly and read
them back. If they interrupt, stop and listen. If asked whether you are a
person, say you are an assistant calling on behalf of
{business.get('name', 'the business')}, then carry on.
""".strip()
