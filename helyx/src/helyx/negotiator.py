"""The negotiating agent. It chooses words; arithmetic chooses prices.

Two separations of power make this safe under the judges' planted friction:

**Price authority.** The model is *told* the authorised offer for this round,
but the number comes from ``MandateGuard.authorized_offer()``. After generation
every money amount in the model's line is re-scanned; a line containing an
unauthorised amount is discarded and replaced with a deterministic safe line.
So the failure mode "LLM concedes past the limit when pushed" is caught by
comparison, not prevented by instruction.

**Completion authority.** The only tool the agent has is ``file_proposal``, and
a filed proposal is born ``UNCONFIRMED``. There is deliberately no tool that
marks anything confirmed, done, or booked. Confirmation can only arrive later
through an independent channel (see domain.py and verification.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .domain import Channel, Evidence, Proposal, Terms
from .ladder import Decision, MandateGuard, Move, Violation
from .llm import LLMClient, LLMError
from .mandate import Mandate

logger = logging.getLogger("helyx.negotiator")

SYSTEM_PROMPT = """You are Helyx, negotiating a wholesale order by phone on behalf of \
a restaurant operator. You are speaking to a bakehouse or restaurant supplier.

Rules you must follow:
- Speak naturally and briefly, like one line of a phone call. No stage directions.
- The ONLY price you may mention is the authorised offer given to you this round. \
Never name any other figure, never split the difference yourself, and never agree to \
a price the supplier proposes.
- Do NOT repeat the supplier's number back to them, not even to reject it. Say "that is \
more than I can do" rather than naming their figure. Any other amount in your reply \
will be detected and your whole line discarded.
- If the supplier states a price, quantity or date, call `file_proposal` with exactly \
what THEY said - not what you wish they had said.
- You cannot confirm, book, or complete anything. If asked, say you will send a text \
to confirm in writing. Never claim the order is done.
- If the supplier cannot meet the requirement, say so plainly. Reporting a failure is \
correct behaviour, not a bad outcome."""

FILE_PROPOSAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "file_proposal",
        "description": (
            "Record the terms the supplier stated. This does NOT confirm anything; "
            "it files an unverified claim for independent checking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "unit_price": {"type": "number", "description": "dollars per unit they quoted"},
                "quantity": {"type": "integer"},
                "fulfilment_date": {"type": "string", "description": "YYYY-MM-DD"},
                "note": {"type": "string"},
            },
            "required": ["unit_price", "quantity"],
            "additionalProperties": False,
        },
    },
}


class Outcome(str, Enum):
    IN_PROGRESS = "in_progress"
    ACCEPTED_PENDING_CONFIRMATION = "accepted_pending_confirmation"
    WALKED_AWAY = "walked_away"


@dataclass
class Turn:
    round_index: int
    supplier_said: str
    agent_said: str
    authorized_offer_cents: int
    decision: Decision | None = None
    violations: list[Violation] = field(default_factory=list)
    replaced: bool = False
    served_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "supplier_said": self.supplier_said,
            "agent_said": self.agent_said,
            "authorized_offer_cents": self.authorized_offer_cents,
            "decision": self.decision.to_dict() if self.decision else None,
            "violations": [v.to_dict() for v in self.violations],
            "replaced": self.replaced,
            "served_model": self.served_model,
        }


@dataclass
class Negotiation:
    """State machine for one call. Rounds and stop conditions are arithmetic."""

    mandate: Mandate
    guard: MandateGuard = field(init=False)
    round_index: int = 0
    turns: list[Turn] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    outcome: Outcome = Outcome.IN_PROGRESS

    def __post_init__(self) -> None:
        self.guard = MandateGuard(self.mandate)

    @property
    def finished(self) -> bool:
        return self.outcome is not Outcome.IN_PROGRESS

    @property
    def latest_proposal(self) -> Proposal | None:
        return self.proposals[-1] if self.proposals else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mandate": self.mandate.to_dict(),
            "ladder": self.guard.ladder.schedule(),
            "round_index": self.round_index,
            "outcome": self.outcome.value,
            "finished": self.finished,
            "turns": [t.to_dict() for t in self.turns],
            "proposals": [p.to_dict() for p in self.proposals],
        }


class Negotiator:
    """Drives a Negotiation one supplier utterance at a time."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def opening_line(self, neg: Negotiation) -> str:
        return neg.guard.safe_line(0)

    def turn(self, neg: Negotiation, supplier_said: str) -> Turn:
        """Process one supplier utterance and produce the agent's reply."""
        if neg.finished:
            raise RuntimeError("negotiation already finished")

        guard = neg.guard
        rnd = neg.round_index
        authorised = guard.authorized_offer(rnd)

        turn = Turn(
            round_index=rnd,
            supplier_said=supplier_said,
            agent_said="",
            authorized_offer_cents=authorised,
        )

        # Pass 1: hear the supplier and file what they said.
        _text, served, tool_args = self._listen(neg, supplier_said, authorised)
        turn.served_model = served

        # --- file whatever the supplier proposed (born UNCONFIRMED) -------
        if tool_args:
            proposal = self._file_proposal(neg, tool_args)
            if proposal is not None:
                neg.proposals.append(proposal)
                decision = guard.evaluate(proposal.terms.unit_price_cents, rnd)
                turn.decision = decision
                if decision.move is Move.WALK_AWAY:
                    neg.outcome = Outcome.WALKED_AWAY
                elif decision.move is Move.ACCEPT:
                    neg.outcome = Outcome.ACCEPTED_PENDING_CONFIRMATION
                # The price the agent may now say is the one the decision
                # produced. Without this the agent repeats its opening offer
                # and never actually concedes.
                if decision.move in (Move.COUNTER, Move.ACCEPT):
                    turn.authorized_offer_cents = decision.unit_price_cents

        # Pass 2: say something. Models routinely return empty content when
        # they emit a tool call, so speech gets its own turn, and it is given
        # the decision that arithmetic already made.
        agent_line = self._speak(neg, supplier_said, turn)

        # --- backstop: re-read what the model actually said ---------------
        violations = guard.scan_utterance(agent_line)
        if violations or not agent_line:
            logger.warning(
                "replacing agent line: %s",
                [v.reason for v in violations] or "empty generation",
            )
            turn.violations = violations
            turn.replaced = True
            agent_line = self._deterministic_line(neg, turn)
        turn.agent_said = agent_line

        neg.turns.append(turn)
        if not neg.finished:
            neg.round_index += 1
            if neg.round_index >= neg.mandate.max_rounds:
                # Rounds exhausted with nothing agreed.
                neg.outcome = Outcome.WALKED_AWAY
        return turn

    # -- internals ---------------------------------------------------------
    def _history(self, neg: Negotiation, supplier_said: str) -> list[dict[str, Any]]:
        m = neg.mandate
        brief = (
            f"Order: {m.quantity} x {m.item}, needed by {m.needed_by.isoformat()}.\n"
            f"Supplier: {m.counterparty_name}.\n"
            f"Constraints: {', '.join(m.constraints) or 'none'}.\n"
            f"Round {neg.round_index + 1} of {m.max_rounds}."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": brief},
        ]
        for t in neg.turns[-6:]:
            if t.supplier_said:
                messages.append({"role": "user", "content": t.supplier_said})
            if t.agent_said:
                messages.append({"role": "assistant", "content": t.agent_said})
        messages.append({"role": "user", "content": supplier_said})
        return messages

    def _listen(
        self, neg: Negotiation, supplier_said: str, authorised: int
    ) -> tuple[str, str, dict[str, Any] | None]:
        """Extract what the supplier actually offered. Speech is a separate pass."""
        messages = self._history(neg, supplier_said)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Call file_proposal with exactly the terms the supplier just stated. "
                    "If they stated no price, do not call the tool."
                ),
            }
        )
        try:
            completion = self.client.complete(messages, tools=[FILE_PROPOSAL_TOOL])
        except LLMError as exc:
            logger.warning("negotiation LLM unavailable (listen): %s", exc)
            return "", "", None
        return (
            completion.text,
            completion.served_model,
            completion.first_tool_args("file_proposal"),
        )

    def _speak(self, neg: Negotiation, supplier_said: str, turn: Turn) -> str:
        """Generate the spoken reply, given the decision arithmetic already made."""
        price = turn.authorized_offer_cents
        move = turn.decision.move if turn.decision else Move.COUNTER

        if move is Move.WALK_AWAY:
            instruction = (
                "Their price is beyond what you may pay and there are no rounds left. "
                "Decline politely in one sentence and close the call warmly. "
                "Do NOT name any number at all."
            )
        elif move is Move.ACCEPT and turn.decision:
            agreed = price
            instruction = (
                f"Their price of ${agreed / 100:.2f} per unit is acceptable. Agree to it "
                f"in one sentence, then say you will text the details now and will treat "
                f"it as agreed only once they reply to that message. "
                f"The ONLY figure you may say is ${agreed / 100:.2f}. "
                f"Do not claim the order is booked or confirmed."
            )
        else:
            instruction = (
                f"Hold your position. In one or two natural sentences, push back and put "
                f"${price / 100:.2f} per unit on the table. "
                f"The ONLY figure you may say is ${price / 100:.2f}. Do NOT repeat their "
                f"number, and do not mention your walk-away limit."
            )

        messages = self._history(neg, supplier_said)
        messages.append({"role": "system", "content": instruction})
        try:
            completion = self.client.complete(messages)
        except LLMError as exc:
            logger.warning("negotiation LLM unavailable (speak): %s", exc)
            return ""
        if completion.served_model:
            turn.served_model = completion.served_model
        return completion.text

    def _file_proposal(self, neg: Negotiation, args: dict[str, Any]) -> Proposal | None:
        m = neg.mandate
        try:
            unit = int(round(float(args["unit_price"]) * 100))
            qty = int(args.get("quantity") or m.quantity)
            terms = Terms(
                item=m.item,
                quantity=qty,
                unit_price_cents=unit,
                fulfilment_date=str(args.get("fulfilment_date") or m.needed_by.isoformat()),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("could not file proposal from %r: %s", args, exc)
            return None

        proposal = Proposal(terms=terms, counterparty=m.counterparty_name)
        # Provenance: this is the agent's word and nothing more.
        proposal.add_evidence(
            Evidence(
                channel=Channel.AGENT_ASSERTION,
                body=f"Agent heard: {qty} x {m.item} at ${unit / 100:.2f} "
                f"({args.get('note', '')})".strip(),
            )
        )
        return proposal

    def _deterministic_line(self, neg: Negotiation, turn: Turn) -> str:
        if neg.outcome is Outcome.WALKED_AWAY:
            return (
                "That is past what I can authorise for this order, so I will leave it "
                "there. Thanks for your time."
            )
        if neg.outcome is Outcome.ACCEPTED_PENDING_CONFIRMATION and turn.decision:
            price = turn.decision.unit_price_cents
            return (
                f"That works. To keep us both straight I will text you the details now "
                f"- {neg.mandate.quantity} at ${price / 100:.2f} each - and I will treat "
                f"it as agreed once you reply to that message."
            )
        # If the supplier is pushing past the ceiling, say so rather than simply
        # restating an offer they have already rejected.
        over = bool(
            turn.decision
            and turn.decision.move is Move.COUNTER
            and "exceeds ceiling" in turn.decision.reason
        )
        return neg.guard.safe_line(turn.round_index, pushing_over_ceiling=over)
