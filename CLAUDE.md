# a1mobile Voice AI Hackathon — project context

12-hour build, Fri Jul 31, 9am–9pm SF. Outbound voice agents that complete real
errands against uncontrolled parties.

## The one rule that shapes everything

Judging scores **verifiable side effects, not agent claims. Fabricated success
is an automatic disqualification.** Judges run live calls with *planted
friction*: unavailable options, payment failures, deliberately ambiguous asks.

So the product is not "an agent that makes calls." It is **an agent that
produces evidence**. Treat any code path that lets the agent's own words mark
something as done as a bug.

## What we are building

An autonomous errand runner. It reads the user's email and to-do list, works out
which items require contacting an outside party, **checks whether each can be
done on the web first**, and phones only the ones that cannot. Every completed
errand ends in a receipt backed by independent evidence.

The web-first triage is the point, not a nicety. It is what makes the voice call
defensible: a dojo, a municipal golf course, a family dentist all have a phone
and a brochure website with no booking path. Those errands are unreachable by
any web agent, and they are exactly what this thing owns.

Pipeline: `extract` (email/todo -> Errand) -> `triage` (web or phone?) ->
`call` (voice agent) -> `verify` (independent evidence) -> `Receipt`.

## Architecture

```
src/tasks/errands.py     Errand model + conservative extraction from raw text
src/tasks/triage.py      web-vs-phone decision. The differentiator.
src/tools/telephony.py   swappable rails: livekit | a1mobile | twilio
src/agents/errand_agent.py   the calling agent (tools can only FILE claims)
src/verify/receipts.py   Claim -> Evidence -> Receipt. The invariant lives here.
src/verify/webhooks.py   inbound SMS listener — where UNVERIFIED becomes VERIFIED
src/verify/transcribe.py local Whisper second opinion on the recording
```

Two asymmetries are deliberate and should not be "fixed":

- **Extraction fails toward dropping items.** A false positive phones a real
  business about something nobody asked for.
- **Triage fails toward PHONE.** A needless call wastes a few minutes; a
  wrongly-skipped errand silently never happens and scores zero. Booking-provider
  matching is domain- and word-boundary-based for this reason — bare substring
  matching made "Stockton" look like Tock and routed a phone errand to WEB.

Data flow: agent files a `Claim` (born UNVERIFIED) → an *independent* channel
(inbound SMS, provider API, Gmail, web check, local transcript) attaches
`Evidence` → `Claim.verdict` derives VERIFIED / UNVERIFIED / CONTRADICTED →
`Receipt.headline` reports pessimistically.

`Verdict` is a derived property with no setter. `Channel.AGENT_ASSERTION` is
recorded but excluded from `INDEPENDENT_CHANNELS`, so no volume of agent
insistence can promote a claim. `tests/test_receipts.py` pins this — if those
tests go red, the DQ condition is reachable again.

## a1mobile

They have no public API, SDK, or docs — their kit only exists inside the event.
`A1MobileProvider` is a deliberate stub, not a guess.

At kickoff, **first check whether they expose a SIP trunk.** If they do, set
`LIVEKIT_SIP_TRUNK_ID` to it and keep `TELEPHONY_PROVIDER=livekit` — that reuses
the whole working path instead of writing a new backend. Only implement
`A1MobileProvider` if their rails are not SIP.

Their inbound SMS webhook shape matters more than their call API: it feeds
`_normalise()` in `src/verify/webhooks.py` and is what makes claims verifiable.
Wire that up before anything else.

Note the kickoff includes a baseline demo of their *stock* assistant on the same
scenario. You are being benchmarked against that default — differentiation has
to come from friction handling and verification, not from plumbing.

## Commands

```bash
.venv/bin/python scripts/preflight.py        # check every credential live
.venv/bin/python scripts/warm_models.py      # re-cache offline models
.venv/bin/python -m pytest tests/ -q         # the anti-fabrication invariant

uv run src/agents/errand_agent.py console    # talk to it via laptop mic, no phone
uv run src/agents/errand_agent.py dev        # connect to LiveKit

uv run uvicorn src.verify.webhooks:app --port 8080
ngrok http 8080                              # then point SMS webhook at it
```

Everything runs from `.venv` (Python 3.12 — chosen over the system 3.14 because
onnxruntime and mlx wheels lag new releases).

## Conventions

- `REALTIME=1` (default) uses OpenAI speech-to-speech — better interruption
  handling, matters for the "Most Human" category. `REALTIME=0` swaps to the
  Deepgram→LLM→ElevenLabs cascade if you need a stronger reasoning model.
- Secrets live in `config/.env` only. Gitignored. Never inline them.
- Every run must write a receipt, including crashed runs — a run with no receipt
  is indistinguishable from a run that lied.
