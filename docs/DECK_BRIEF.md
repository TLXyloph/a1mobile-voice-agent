# Deck brief — feed this to a design agent

Everything below is verified against the running system. Numbers are real; where
something is unproven it says so. **Do not let the deck claim more than this
brief does** — the product's entire argument is that it refuses to overclaim, and
a deck that oversells it undermines the thesis on the first slide.

---

## The one-sentence version

An outbound voice agent that negotiates real orders by phone and **cannot report
a sale it did not make**.

## The problem, framed for a judge

Every voice-agent demo ends with the agent saying it worked. That claim is worth
nothing — the agent is the least reliable witness to its own performance, and an
LLM under conversational pressure drifts toward reassurance because that is how
its training data ends phone calls. This hackathon scores *verifiable side
effects* and disqualifies fabricated success, which means the interesting
engineering problem is not "can it talk" but "can it be believed."

## The answer

Make fabrication **unreachable**, not discouraged.

- An agent can only ever file a `Claim`. Claims are born `UNVERIFIED`.
- `Verdict` is a derived property with **no setter**.
- `AGENT_ASSERTION` is recorded but excluded from `INDEPENDENT_CHANNELS`, so no
  volume of agent insistence promotes anything.
- Only a channel the agent cannot write to — the caller's own transcribed
  speech, an inbound message, a provider API — can verify.

The demonstrable version, which takes ten seconds and never fails:
**hand-edit a receipt to say `VERIFIED`, re-ingest it, watch it come back
`UNVERIFIED`.** Three independent locks enforce this — the write API takes no
verdict argument, a SQL trigger `RAISE(ABORT)`s on direct update, and re-ingest
deletes-and-rederives rather than assigning. The protection travels with the
database file, not the process. Hand a judge the DB and dare them.

## The second idea: limits are structural, not prompted

"Never quote below your floor" is a *request* to a model. Under negotiation
pressure, models concede. So the agent **proposes** and a validator **disposes**:

```
check_capacity  → CapacityLedger.hold()       may refuse
propose_price   → CostModel.validate_quote()  may refuse
close_order     → Receipt.claim()             born UNVERIFIED
```

A state graph gates the *ordering* too — units before capacity, capacity before
price, validated price before close. Illegal sequences are unreachable rather
than discouraged. There is **no escalation path**: with an escape hatch
available, "let me ask my manager" becomes the answer to every hard case and the
envelope stops being a boundary.

---

## The strongest narrative material: eight bugs, all found by live calls

This is the most honest and most persuasive part of the story. Every one was a
*combination* of individually-correct guards, and none would have been found by
testing. Each is now pinned by a regression test.

| # | What happened | Why it matters |
|---|---|---|
| 1 | A 30-person order was priced as 30 muffins | A headcount and an item count are identical once they're integers |
| 2 | Buyer offered $385; agent countered **$74** | The floor stops selling below cost; nothing was watching the other direction |
| 3 | Agent said "I'm checking that now" then went silent | A refused hold left no state, so every later tool said "call check_capacity first" |
| 4 | Agent refused $400 over a $385.72 floor without checking | It knew it should not *say* an unchecked price; nobody told it not to *refuse* one |
| 5 | Escalation removed → `REQUIRES_APPROVAL` became a dead end | Couldn't quote, couldn't ask, couldn't close |
| 6 | Buyer said "I pay Costco $500"; agent clamped at $500 | Fixing #2 removed the ability to win on price at all |
| 7 | Intake wrote a profile both call paths ignored | The whole configuration feature was cosmetic |
| 8 | Live call reached agreement, receipt said "nothing attempted" | Losing the claim loses the evidence anything happened |

**The line to land:** the guards never broke. Every failure was two correct rules
combining into a wrong outcome — which is exactly the failure mode you cannot
find without putting a real stranger on the phone.

---

## Verified numbers (safe to put on a slide)

- **595 tests** passing
- **$0.90** unit cost → **$257.15** floor / **$450.00** target for 200 units
- **0.84s** LLM time-to-first-token (`openai/gpt-5.4`); ~1.2s perceived turn latency
- Benchmarked 4 models to pick it — `gpt-5.5` was 4.85s, unusable on a call
- Telephony: a1mobile number over their SIP trunk (Telnyx) via LiveKit SIP
- Voice: Deepgram nova-3 → OpenAI gpt-5.4 → Cartesia sonic-2, Silero VAD on-device
- Three campaigns (restaurant catering, freelance web dev, B2B SaaS) differ
  **only in configuration** — the engine is vertical-agnostic

## What is NOT proven — do not claim these

- **No live call has produced a filed claim.** `close_order` runs, the receipt
  saves empty. The identical sequence files correctly in replay. Open bug.
- The `VERIFIED` receipt on the demo page came from the rehearsal pipeline
  (scripted transcript, real Claim/Evidence/verdict chain), **not live audio**.
  If asked, say so.
- a1mobile's inbound SMS webhook never fires — verified extensively. Their voice
  webhook works; messaging does not reach us.
- No speech-to-speech: a1mobile's LLM gateway is text-only (no `/realtime`, no
  `/audio`) and rejects every model string tried.

---

## Demo flow

**Link:** `https://c9d9ebd5b47dc6.lhr.life/demo` *(check `evidence/JUDGE_LINK.txt`
— tunnels die and the URL changes)*

The page shows the agent's persona, floor, capacity and limits **read live from
the config it actually runs under**, then takes the judge's number and calls
them. a1mobile only dials OTP-verified numbers — frame that as a guardrail
against agents cold-dialing strangers, not as friction.

Three scripted frictions, two of which are refusals:

1. *"200 croissants Thursday, I pay Costco $500"* → it undercuts to win the deal
2. *"I'll only pay $100"* → refuses, below the $257 floor
3. *"Can you do 800?"* → refuses, over capacity

**Lead with the refusals.** An agent visibly declining a deal it isn't allowed to
make is a stronger proof than one agreeing, and nobody else will show it.

---

## Suggested slide order

1. **The claim problem** — every voice demo ends with the agent saying it worked
2. **Make lying unreachable** — Claim → Evidence → Receipt, verdict has no setter
3. **Live proof** — forge a receipt on stage, watch it re-derive as UNVERIFIED
4. **Limits are structural** — the state graph; propose vs dispose
5. **Eight bugs from eight calls** — the table above; the guards never broke
6. **It generalises** — three verticals, config-only difference
7. **Try it** — the link, the three scripted frictions
8. **What's honest** — the open bug, stated plainly

Slide 8 is not a weakness. In a competition that disqualifies fabricated
success, being the team that volunteers its own unproven edges is the argument.

## Tone

Plain, specific, unhurried. No superlatives, no "revolutionary." Let the numbers
and the bug table carry it. The product's voice is the deck's voice: it would
rather report an honest failure than an unearned success.
