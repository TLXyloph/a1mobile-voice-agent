# Inbound-call smoke test via Linphone

**Date:** 2026-07-31
**Status:** approved, executing

## Problem

The LiveKit side of inbound calling is fully configured, but has never taken a
real call. Configuration that looks right and configuration that works are
different things, and we have no evidence for the second.

Known-good, verified live via `lk`:

| Object | ID | Binding |
|---|---|---|
| Inbound trunk | `ST_532VsM4vbCwM` | `+19378608348`, auth `<SIP_USERNAME_REDACTED>` |
| Dispatch rule | `SDR_nbwd3YzeGkVy` | → agent `sales-agent`, room prefix `inbound` |
| Outbound trunk | `ST_7KjTLQ9Z7UiB` | `+19378608348`, auth `<SIP_USERNAME_REDACTED>` |
| Outbound trunk | `ST_t3GSDr7VkAyb` | `+19379750189`, auth `<SIP_USERNAME_REDACTED>` |

`sales-agent` is registered at `src/agents/run_call.py:92`, matching the
dispatch rule.

The untested hop is **Telnyx → LiveKit**. Nothing in the LiveKit control plane
can tell us whether Telnyx will actually route a PSTN call on that number to
our trunk.

## Why a softphone rather than a cell phone

Dialling from a cell answers *whether* it works. A SIP client answers *where it
broke*, because we see the response code on the wire. With a twelve-hour build
budget, the difference between "inbound is broken" and "Telnyx returns 404 for
this number" is most of an afternoon.

Linphone over Zoiper: one `brew` command instead of a manual `.dmg`, no nag
screens, and a readable SIP trace. Zoiper's free tier would also have worked —
it supports G.711, which is what Telnyx uses.

## Design

### Registration choice

Linphone registers as `<SIP_USERNAME_REDACTED>@sip.telnyx.com` (UDP) — the credential
bound to `+19379750189`.

It must **not** register as `<SIP_USERNAME_REDACTED>`. That credential owns the number
under test; registering it would pull inbound calls for `+19378608348` to the
laptop, and the test would be measuring its own softphone. Using the second
credential forces the call out to Telnyx and back through the real inbound
route.

### Procedure

1. `brew install --cask linphone`
2. Register the `<SIP_USERNAME_REDACTED>` SIP account.
3. Start the worker: `uv run src/agents/run_call.py dev`. Without it the trunk
   answers and no agent joins, which presents as failure for the wrong reason.
4. Dial `+19378608348` from Linphone.
5. Observe three surfaces together: Linphone's SIP trace, worker stdout, and
   `lk room list`.

### Interpreting the result

The response code localises the fault. This table is the deliverable:

| Signal | Failing hop |
|---|---|
| `401`/`403` on REGISTER | Softphone credential wrong — test never started |
| `404` / `484` | Telnyx has no inbound route for the number → Telnyx portal |
| `503` / `480` | Telnyx routes, cannot reach the LiveKit trunk |
| `200 OK`, no room in `lk room list` | Trunk accepted, dispatch rule not matching |
| Room appears, no agent joins | Dispatch fine; `agent_name` mismatch or worker down |
| Agent speaks | Inbound works |

## Risks

**Registration side effect.** While Linphone is registered, inbound calls to
`+19379750189` land on the laptop. Harmless during testing; quit Linphone
before demoing.

**Outbound trunk interference.** Outbound INVITE auth is challenge/response and
independent of registration state, so `ST_t3GSDr7VkAyb` should be unaffected.
To be confirmed by test rather than assumed.

## Out of scope

`config/sip-outbound-trunk.json` contains a plaintext SIP password and is not
gitignored. The directory is not currently a git repository, so nothing has
leaked, but `git init` would expose it. Flagged, not fixed.
