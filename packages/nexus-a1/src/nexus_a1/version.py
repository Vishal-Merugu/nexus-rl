"""The A1 policy schema version.

A1 is an **interface between two independently-deployed tiers**, so the schema is a real contract with a
version, not a shape that happens to match today. The rApp (Non-RT tier) and the xApp (Near-RT tier) are
separate services on separate release cycles; either can be older than the other at any moment.

Version negotiation is therefore explicit: a policy carries the schema version it was written against, and
the consumer rejects one it cannot interpret rather than guessing at the difference. A silently
misinterpreted policy is worse than a refused one — the operator is told their intent is in force while
the network does something else.

Bump this when a field changes meaning, becomes required, or is removed. Adding an optional field with a
safe default is backward compatible and does not need a bump.

History:

- **1** — Initial schema. Three policy types (QoS objective, traffic steering, class floor), a scope
  restricted to traffic classes and links, and mandatory bounded TTL on every policy.
"""

from __future__ import annotations

A1_SCHEMA_VERSION: int = 1

MIN_SUPPORTED_SCHEMA_VERSION: int = 1
"""Oldest schema this consumer still understands.

Kept separate from the current version so that supporting an older rApp is a deliberate, visible decision
rather than an accident of whichever version happens to be deployed.
"""
