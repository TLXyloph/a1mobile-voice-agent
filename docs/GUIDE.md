# Build guide — a1mobile Voice AI Hackathon

For whoever (human or agent) is writing code in this repo on Jul 31. Read
`CLAUDE.md` first for architecture; this is the working manual.

---

## 1. The scoring rule, restated as an engineering constraint

> Scored on verifiable side effects, not agent claims. Fabricated successes are
> automatically disqualified. Live calls include planted friction.

Three consequences that should drive every decision:

1. **A confident agent is a liability.** LLMs on phone calls drift toward
   reassurance — "great, you're all set!" — because that is how the training
   data ends calls. Left alone, your agent will report success it did not
   achieve, and that is not a bug that costs points, it is disqualification.
   The `Receipt` system exists to make that outcome unreachable rather than
   merely discouraged.

2. **The verification channel is worth more than the call.** A mediocre call
   that produces a confirmation SMS scores above a beautiful call that produces
   nothing checkable. If you are short on time, cut call quality, not evidence
   capture.

3. **Friction is the exam, not an edge case.** They will plant a sold-out slot
   and a declined card. Handling those *specific* cases well beats general
   eloquence.

## 2. What is already installed

Python 3.12 venv at `.venv` (system Python is 3.14; onnxruntime and mlx wheels
lag new releases, so 3.12 was chosen deliberately — don't "upgrade" it).

- `livekit-agents 1.6.7` + 16 plugins (openai, deepgram, elevenlabs, cartesia,
  silero, turn-detector, anthropic, assemblyai, rime, groq, noise-cancellation)
- `lk`, `ngrok`, `ffmpeg`, `sox` on PATH
- Playwright + Chromium; Playwright MCP registered in `.mcp.json`
- **Cached offline**: Silero VAD, multilingual turn detector, Whisper
  large-v3-turbo (MLX). Venue wifi cannot break these.

Verify anytime: `.venv/bin/python scripts/preflight.py`

## 3. Adding a new errand

The agent is generic; errands are configuration. To add one:

1. Set `ERRAND_TASK` and `ERRAND_CONSTRAINTS` in `config/.env`. Constraints are
   the decisions the agent may make *without asking* — be explicit and narrow.
   Everything outside them routes to `needs_decision`, which is correct
   behaviour, not failure.
2. Decide the verification channel **before** writing any prompt. Ask: "what
   will exist in the world, outside this call, if this worked?" If you cannot
   answer, the errand is unwinnable under these rules — pick a different one.
3. Wire that channel into `src/verify/`. SMS is already done; Gmail is
   available via the connected Gmail MCP; web checks via Playwright MCP.

## 4. Voice-specific pitfalls

These cost more time than architecture problems:

- **Numbers spoken aloud.** "Four" vs "for", "fifteen" vs "fifty" — this is the
  single most common cause of a wrong booking that still reports success.
  Always read numbers back and require them in the confirmation tokens.
- **Latency kills the illusion.** Anything over ~800ms end-to-end reads as a
  bot. `REALTIME=1` (speech-to-speech) is materially better here than the
  cascade. Only drop to `REALTIME=0` if you need a stronger reasoning model.
- **Interruption handling** is most of what "sounds human" means. Test it: talk
  over the agent in `console` mode and see if it stops.
- **IVR trees** need DTMF, not speech. `send_dtmf` is wired; test against a real
  phone tree early — every provider handles DTMF slightly differently.
- **Hold music** looks like silence to VAD. Long holds can trigger the agent to
  start talking to nobody. Watch for it.
- **Never let the agent claim to be human.** If asked directly it must say it is
  an assistant calling on someone's behalf. Beyond ethics, callers who feel
  deceived hang up, and a hangup is an unverifiable outcome.

## 5. Rehearsing planted friction

`livekit-agents` ships a simulation harness — `Scenario`, `ScenarioGroup`,
`SimulationRun`, `SimulationVerdict`, `mock_tools`. Use it to run your agent
against an adversarial counterpart *before* the live call.

Highest-value scenarios, mirroring what judges said they will plant:

| Scenario | What it should do | Failure to catch |
|---|---|---|
| Requested slot unavailable | offer alternatives, `needs_decision` | silently books a different time |
| Card declined | ask for hold-without-payment | invents a payment success |
| "Which Thursday?" | ask one clarifying question | guesses a date |
| Transfers you twice | stays on, re-states purpose | reports "spoke to manager" |
| Agrees but sends nothing | claim stays UNVERIFIED | reports success |

That last row is the important one: an agent that behaves perfectly and gets a
verbal yes must still end at UNVERIFIED if no text arrives. If your pipeline
reports SUCCESS there, you have a DQ-shaped bug.

## 6. Kickoff sequence (9:00am)

```
1.  scripts/preflight.py                 # before anything else
2.  Get a1mobile's kit. FIRST QUESTION: "do you expose a SIP trunk?"
      yes -> LIVEKIT_SIP_TRUNK_ID=<theirs>, TELEPHONY_PROVIDER=livekit. Done.
      no  -> implement A1MobileProvider (3 methods).
3.  SECOND QUESTION: "what does your inbound SMS webhook POST look like?"
      -> that shape goes in _normalise() in src/verify/webhooks.py
4.  uvicorn src.verify.webhooks:app --port 8080 && ngrok http 8080
    Point their SMS webhook at <ngrok>/sms/a1mobile
5.  Send yourself a test SMS. Confirm it lands in evidence/inbox.jsonl.
    Do not proceed until this works — it is the whole scoring mechanism.
6.  console-mode call to yourself. Then one real call to a low-stakes target.
7.  Only now start on the interesting part.
```

Watch the baseline demo of their stock assistant closely and write down exactly
where it fails. Those failure modes are almost certainly what gets planted in
judging.

## 7. Demo notes

- `Receipt.render()` is written to be read aloud. Show the UNVERIFIED and
  CONTRADICTED cases, not just the wins — demonstrating that your system
  *refuses* to overclaim is the strongest possible answer to the judging rule,
  and no one else will show a failure on purpose.
- Record the screen for the Video Bonus category while you have a working run;
  don't leave it to hour 11.
- Keep one deliberately-failing scenario in your demo. When a judge plants
  friction you did not anticipate, having already shown honest failure means an
  UNVERIFIED result reads as designed behaviour rather than a broken demo.
