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
- **2** — Network slicing (Stage J). Adds a fifth reward component, ``sla_attainment_reward``:
  priority-weighted per-slice SLA headroom against ``DEFAULT_SLICES``. The observation encoding and the
  action space are **unchanged**; only the reward is.

  *Why it is still a bump.* A policy trained under version 1 optimised aggregate satisfaction with no
  notion of a differentiated promise. It is not merely scored differently here — the objective it learned
  is a different objective, and comparing its numbers against a version 2 policy's would be comparing two
  answers to two different questions.

  *Invalidated:* every run recorded under version 1 — at the time of the bump, the ``smoke-001`` and
  ``smoke-002`` smoke runs. Neither is cited in ``docs/evaluation/results.md``, which contains baseline
  numbers only, so no published figure changes.
"""

from __future__ import annotations

MDP_VERSION: int = 2
