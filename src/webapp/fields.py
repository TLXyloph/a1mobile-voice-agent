"""The shape of a brief: every question the interview has to get an answer to.

Split out from `spec.py` because two very different things read this list - the
interview, which turns it into questions, and the panel in the browser, which
turns it into rows. Keeping it in one place is what stops the UI showing a
field the interview never asks about, or the interview asking for something the
operator can never see it recorded.

`kinds` is the versatility mechanism. A bakery selling catering needs unit
economics; "call three dentists and book the earliest cleaning" does not, and
demanding a materials cost for a dental appointment is how a generic tool
announces that it was really built for one vertical.
"""

from __future__ import annotations

from dataclasses import dataclass

SALE = "sale"
ERRAND = "errand"
KINDS = (SALE, ERRAND)


@dataclass(frozen=True)
class SpecField:
    """One thing the interview has to establish."""

    id: str
    label: str
    group: str
    question: str
    kinds: tuple[str, ...] = KINDS
    errand_question: str = ""

    def ask(self, kind: str) -> str:
        if kind == ERRAND and self.errand_question:
            return self.errand_question
        return self.question

    def applies_to(self, kind: str) -> bool:
        return kind in self.kinds


FIELDS: tuple[SpecField, ...] = (
    SpecField(
        "kind",
        "Task type",
        "job",
        "Is this selling something with a price attached, or an errand where "
        "you just need something booked or arranged?",
    ),
    SpecField(
        "objective",
        "Outcome wanted",
        "job",
        "In one sentence, what has to be true when this is done?",
    ),
    SpecField(
        "business_name",
        "Calling on behalf of",
        "job",
        "Whose name do I give when someone picks up?",
    ),
    SpecField(
        "targets",
        "Who to call",
        "who",
        "Who should I call? A name and a phone number each, or tell me how to "
        "find them.",
    ),
    SpecField(
        "offer",
        "The offer",
        "what",
        "What are you offering them, phrased the way they would hear it?",
        errand_question=(
            "What exactly am I asking them for, phrased the way they would hear it?"
        ),
    ),
    SpecField(
        "unit_label",
        "Unit",
        "what",
        "What is one unit of that? Muffins, crew-hours, site builds - your "
        "words, plural.",
        errand_question="What is the thing being booked - appointments, slots, seats?",
    ),
    SpecField(
        "units_basis",
        "Units vs headcount",
        "what",
        "Careful here, this one causes real mispricing: when you say a "
        "quantity, is that a number of PEOPLE or a number of ITEMS? If someone "
        "says 'thirty people', how many items is that - how many per person?",
        errand_question=(
            "Careful here: is a quantity a number of PEOPLE or a number of "
            "BOOKINGS? Three dentists is not three appointments - how many "
            "appointments do you actually need, and for how many people?"
        ),
    ),
    SpecField(
        "capacity_total",
        "Capacity",
        "capacity",
        "How many of those can you actually supply in the period we are "
        "selling into?",
        errand_question="How many do you need in total?",
    ),
    SpecField(
        "economics",
        "Unit economics",
        "economics",
        "Per unit: materials, labour, transport, your minimum margin and the "
        "margin you actually want. Give me the ones you know.",
        kinds=(SALE,),
    ),
    SpecField(
        "max_discount_pct",
        "Max discount",
        "limits",
        "What is the deepest discount off list I may agree to without you?",
        kinds=(SALE,),
    ),
    SpecField(
        "date_window",
        "Date window",
        "limits",
        "What is the earliest and latest date I may commit to?",
    ),
    SpecField(
        "max_qty",
        "Max quantity",
        "limits",
        "What is the largest quantity I may agree to on my own?",
    ),
    SpecField(
        "close_condition",
        "What counts as done",
        "done",
        "What counts as done - written confirmation by SMS or email, a booked "
        "slot on a calendar, or something delivered we can go and check?",
    ),
    SpecField(
        "confirm_to",
        "Confirmation lands at",
        "done",
        "Where should that confirmation arrive? A phone number or an email "
        "address I can read back to them.",
    ),
)

FIELDS_BY_ID: dict[str, SpecField] = {f.id: f for f in FIELDS}

GROUPS: tuple[tuple[str, str], ...] = (
    ("job", "The job"),
    ("who", "Who to call"),
    ("what", "What is on offer"),
    ("capacity", "Capacity"),
    ("economics", "Unit economics"),
    ("limits", "Hard limits"),
    ("done", "Definition of done"),
)

