"""The registry schema version.

The ledger is written by the training/eval tooling and read by the controller — two independently
released things, exactly like A1. A record written by a newer tool is refused rather than partially
interpreted: a version whose provenance is misread is worse than one that cannot be read at all, because
the controller would serve it while reporting the wrong origin.

Bump when a field changes meaning, becomes required, or is removed. Adding an optional field with a safe
default is backward compatible and does not need a bump.

History:

- **1** — Initial schema. Provenance (run, git SHA, MDP version, algorithm, topology, seed), a list of
  evaluation scores tied to a named suite, and a four-value deployment stage.
"""

from __future__ import annotations

REGISTRY_SCHEMA_VERSION: int = 1

MIN_SUPPORTED_SCHEMA_VERSION: int = 1
"""Oldest ledger schema this code still reads."""
