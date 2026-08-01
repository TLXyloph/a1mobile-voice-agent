# Helyx

Outbound voice-agent product for one vertical: **restaurants and bakehouses**.
Helyx phones a supplier and negotiates a wholesale or catering order against a
mandate the operator sets.

Everything below runs from the existing venv at the repo root. Helyx reads
secrets from `config/.env` at runtime and never copies them.

```bash
V=.venv/bin/python

$V -m pytest helyx/tests -q                  # 80 tests
$V helyx/scripts/probe_models.py             # which model ids actually answer
$V helyx/scripts/demo_end_to_end.py          # headless run, real model, planted friction

HELYX_PORT=8123 PYTHONPATH=helyx/src \
  $V -m uvicorn helyx.dashboard:app --port 8123    # dashboard at http://127.0.0.1:8123
```

Ports 8080 and 8095 belong to other processes and are left alone. Helyx uses
8123 (`HELYX_PORT` to change).

## The property everything else serves

Judging scores verifiable side effects, and fabricated success is
disqualifying. So Helyx is built so that **no code path lets the agent's own
words mark anything as done**:

- `Proposal.status` is a derived property with **no setter**. Nothing can assign
  `CONFIRMED`.
- `Channel.AGENT_ASSERTION` is recorded for provenance but excluded from
  `INDEPENDENT_CHANNELS`. A thousand agent assertions still derive
  `UNCONFIRMED` (pinned by test).
- Agreement is stricter than delivery. `PROVIDER_API` (an a1mobile message id)
  proves we sent something, never that anyone agreed, so it is excluded from
  `AGREEMENT_CHANNELS`.
- Confirmation requires the independent message to **restate the numbers**.
  A bare "confirmed!" does not confirm; neither does a message with the wrong
  price. Numeric matching is boundary-aware, so `1200` does not satisfy `120`.

The agent's only tool is `file_proposal`. There is deliberately no tool that
marks anything booked, confirmed, or complete.

## Negotiation limits are arithmetic, not instructions

An LLM told "never pay more than $22" will pay more than $22 under pressure.
So the model never picks a number:

- `ConcessionLadder.offer_for_round(r)` is a pure function of the mandate,
  conceding in shrinking increments and ending at the ceiling.
- `MandateGuard.evaluate()` decides accept / counter / walk-away by integer
  comparison. Exhaustive tests sweep the space and assert no decision ever
  authorises above the ceiling, and that a counter is never *above* the
  supplier's own ask (Helyx does not bid against itself).
- `MandateGuard.scan_utterance()` re-reads what the model actually said and
  flags any money amount outside the authorised envelope. A violating line is
  discarded and replaced with a deterministic mandate-safe line.

Intake follows the same rule: the model fills slots, but `ready` is computed
from `missing_fields()` plus mandate validation. A model announcing "I have
everything I need" does not unblock the call.

## Pipeline

```
intake (operator params)  ->  Mandate (validated at the boundary)
   -> negotiation (LLM words, guard prices)  -> Proposal (born UNCONFIRMED)
   -> independent channel (inbound SMS / email / human review) -> Evidence
   -> status derives CONFIRMED | UNCONFIRMED | CONTRADICTED
   -> email loop: check inbound, report to operator, write receipt
```

Every run writes a receipt to `helyx/var/`, including failed runs: a run with
no receipt is indistinguishable from a run that lied.

## Layout

| path | role |
|---|---|
| `src/helyx/domain.py` | evidence, channels, derived status. The invariant. |
| `src/helyx/mandate.py` | operator parameters, validated at the boundary |
| `src/helyx/ladder.py` | concession arithmetic + utterance backstop |
| `src/helyx/intake.py` | intake agent; completeness is computed |
| `src/helyx/negotiator.py` | two-pass call agent (listen, then speak) |
| `src/helyx/llm.py` | LiveKit inference gateway + model fallback |
| `src/helyx/sms.py` | a1mobile rails; inbound normalisation |
| `src/helyx/email_loop.py` | inbound check, report composition, recipient lock |
| `src/helyx/store.py` | event log + derived view + SSE pub/sub |
| `src/helyx/dashboard.py` | FastAPI app, webhook, live dashboard |

## Configuration

| env | default | meaning |
|---|---|---|
| `HELYX_MODEL` | `openai/gpt-5.6` | primary model |
| `HELYX_FALLBACK_MODEL` | `openai/gpt-5.4` | used only if the primary fails |
| `HELYX_GATEWAY_URL` | LiveKit agent gateway | OpenAI-compatible endpoint |
| `HELYX_PORT` | `8123` | dashboard port |
| `HELYX_SMTP_HOST/USER/PASSWORD` | unset | enables real email delivery |
| `HELYX_IMAP_HOST/USER/PASSWORD` | unset | enables real inbox reading |

## Known limits

- **Email is composed, not delivered.** No SMTP credential exists in this repo,
  so `run_email_loop` writes the exact RFC-822 message to `helyx/var/outbox/`
  and reports `delivered: false`. Supplying `HELYX_SMTP_*` switches on the real
  send path, which has never been exercised against a live server.
- Inbound email uses `helyx/var/inbox/*.json` unless `HELYX_IMAP_*` is set.
- Outbound SMS is real but a1mobile restricts destinations to judge numbers or
  a pre-verified number.
- The call is text-driven (dashboard / script). No audio rail is wired.
