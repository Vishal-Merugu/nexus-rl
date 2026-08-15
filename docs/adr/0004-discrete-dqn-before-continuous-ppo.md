# 0004 — Discrete action space with DQN first, continuous with PPO later

Date: 2026-08-14
Status: Accepted

## Context

Resource allocation is naturally continuous — "give this flow 37.5% of the link" is the real problem.
PPO handles continuous action spaces and is what most current network-RL literature uses. DQN handles only
discrete spaces and is simpler.

The temptation is to start where the research ends: continuous actions, PPO, done properly from the
beginning.

The argument against is about **what fails first**. A new RL project has many places to be wrong: the
observation encoding, the reward, action decoding, the training loop, the environment's determinism, the
service plumbing. Starting with a continuous action space adds "is the policy failing because the action
space is hard to explore?" to that list — and it is the hardest of them to distinguish from the others,
because every failure mode looks like a flat reward curve.

## Decision

**v1: discrete action space with DQN. Stage G: continuous with PPO.**

The discrete space is not one value per flow — that grows combinatorially and is not learnable. It is a set
of eight named **allocation modes**: open, moderate throttle, aggressive throttle, prioritise
latency-sensitive, prioritise bulk, shed background, reroute to least-loaded, and prioritise-plus-reroute.

Both spaces are implemented in `nexus-env` from the start, selected by `action_mode`. Stage G is a config
change plus a retrain, not a rewrite.

## Consequences

- **The whole pipeline gets proven before the hard part.** Simulator, MDP, services, dashboard, agent layer
  all work end to end against a policy that is easy to train and easy to sanity-check.
- **Failures stay attributable.** With eight discrete actions, a flat reward is an environment or reward
  problem, not an exploration problem. That narrowing is worth a great deal early on.
- **Modes are legible, which the agent layer needs.** "The controller shed background traffic" is an
  explanation an operator understands. A vector of 8 floats is not, and Use Case 2's whole point is
  explanation.
- **Reachability is genuinely limited, and this is the real cost.** A mode set cannot express "60% here, 40%
  there". Some optimal allocations are outside the space, so a discrete policy will plateau below what a
  continuous one could reach. **That plateau is expected behaviour, not a training bug** — it is written into
  `action_space.py` and into `/diagnose-reward` so it is not rediscovered as a mystery at 1am.
- **The v1 → Stage G comparison becomes a result in itself.** "Discrete DQN vs. continuous PPO on the same
  environment, same seeds, same baselines" is exactly the benchmark the papers in this area run, and having
  both makes the writeup stronger than either alone would.
- **Mode design is now a load-bearing decision.** Eight modes chosen badly bound the policy's ceiling as
  surely as a bad reward does. They are ordered and append-only: reordering them silently invalidates every
  checkpoint, so `AllocationMode` documents that constraint.
- **`RunConfig` rejects DQN with `action_mode: continuous`** rather than failing deep inside
  Stable-Baselines3 with an opaque error.

## Alternatives considered

- **Continuous PPO from the start.** Rejected: conflates plumbing bugs with exploration difficulty at
  exactly the point where the plumbing is least trustworthy. Deferred, not abandoned.
- **Per-flow discrete allocation levels** (each flow picks from {0, 25%, 50%, 100%}). Rejected: the space is
  4^n in the flow count — 65,536 actions for 8 flows, unlearnable within any sane sample budget, and
  illegible to the agent layer.
- **Parameterised/hybrid action space** (pick a mode, then a continuous parameter). Rejected for v1: more
  expressive, but supported awkwardly by Stable-Baselines3 and it reintroduces the exploration difficulty
  this decision exists to defer.
- **Skip DQN entirely and go straight to PPO on the discrete space.** Reasonable, and PPO does handle
  discrete spaces. Rejected mainly because benchmarking DQN against PPO is itself a documented pattern in
  this literature, and having both is worth more in the writeup than saving one training run.
