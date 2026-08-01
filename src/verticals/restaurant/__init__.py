"""Restaurant vertical: a queryable data layer over calls, orders and receipts.

`src/verify/receipts.py` writes one JSON receipt per call. That is the right
primitive for a judge - self-contained, hashable, hard to fake - and the wrong
primitive for an owner, who wants to ask "what did I commit to this week" and
"how much of what we booked is actually proven".

This package answers those questions without ever becoming a second source of
truth about whether something happened. Receipts remain authoritative; the
store is a projection of them, and `Claim.verdict` is recomputed from evidence
on the way in rather than copied from the file.
"""

from __future__ import annotations

__all__ = ["config", "export", "ingest", "query", "seed", "store"]
