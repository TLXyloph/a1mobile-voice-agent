"""SMS continuation of a voice call, under the call's own limits.

A call ends. The prospect said "text me the numbers", or hung up mid-thought,
or asked for something the agent was not allowed to give. The deal is still
live - but only by text now. This package continues it.

The single rule that shapes every module here:

    **A deal the agent could not make by voice must not become makeable by
    text.**

So a `Thread` is not a chat log. It carries the same campaign envelope, the
same capacity quantity, the same `CostModel` floor and the same
`src.agents.flow.Gate` facts the call ran under, and every outbound draft is
re-validated against them before a single character is sent. There is no
operator escalation on this channel - there is nobody watching an SMS thread -
so the limits are absolute rather than a starting position.

And the anti-fabrication invariant carries over unchanged: an outbound text is
`Channel.AGENT_ASSERTION` and can never satisfy anything. Only a text *from*
the prospect is `Channel.INBOUND_SMS`, and only it can promote a Claim.

    thread.py    the conversation + its inherited constraints, sqlite-persisted
    closer.py    reply generation, with the price guard that cannot be prompted away
    evidence.py  outbound -> AGENT_ASSERTION, inbound -> INBOUND_SMS
    send.py      outbound over a1mobile POST /api/sms, dry-run by default
    routes.py    the APIRouter a dashboard and an inbound hook talk to

INBOUND REALITY, stated once here and repeated where it matters: **a1mobile's
inbound SMS webhook has never fired for us.** Their voice webhook works - a
real Telnyx IP hit our TeXML endpoint - but texting the number produces zero
requests and there is no API to read received messages. Outbound sending works
and goes out from a shared pool number, so replies land somewhere we cannot
read. Everything here therefore treats *our own* HTTP endpoint as the inbound
source of truth. If a1mobile ever starts delivering, it works unchanged.
"""

from __future__ import annotations

__all__ = ["thread", "closer", "evidence", "send", "routes"]
