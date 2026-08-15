"""The MDP version.

Bump this whenever the observation encoding, action space, reward definition, or episode length changes. A
checkpoint trained under version N is meaningless under version N+1 — evaluation refuses to load a mismatched
one rather than silently producing wrong numbers.

When you bump it:

1. List every run under ``experiments/runs/*/config.yaml`` carrying the old version
2. Check whether any of them is cited in ``docs/evaluation/results.md``
3. Say both in the PR description
4. Spawn the ``mdp-reviewer`` agent

History:

- **1** — Initial formulation. Discrete allocation modes (8), per-link + per-flow observation, four named
  reward components.
"""

from __future__ import annotations

MDP_VERSION: int = 1
