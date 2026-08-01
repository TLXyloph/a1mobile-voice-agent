"""The meta-layer: any goal in plain language -> the right questions and a
dashboard fitted to it.

    .venv/bin/python -m uvicorn src.generator.app:app --port 8140

The fixed verticals in `src/business/campaign.py` each carry a hand-written
intake form. Three of them was already two too many: a bakery needs capacity and
cost per unit, a dentist booking needs dates, insurance and urgency, and asking
either one the other's questions is the noise that makes a product feel
generic.

    spec.py           TaskProfile - what kind of exchange is this, really
    questions.py      the intake set, generated then hardened then validated
    dashboard_gen.py  `claude -p` behind a timeout and an HTML validator
    app.py            the chat box, the editable form, the preview

Everything here emits *configuration* for machinery that already exists.
`TaskProfile.to_campaign()` hands off to the campaign engine; the generated
dashboard renders the claim/evidence/verdict model from `src/verify/receipts.py`
and nothing else. If this package ever grows a second way to decide whether
something happened, that is the bug.
"""
