---
name: mdp-formulation
description: Use when defining or changing the observation encoding, action space, or reward function — designing the MDP for a new capability, adding a reward term, changing action granularity, or reviewing whether the current formulation is well-posed. Covers Markov sufficiency, normalisation, reward scale, degenerate optima, and the version discipline that keeps recorded results meaningful.
---

# MDP Formulation

The observation/action/reward contract is the single most consequential design artifact in this project. It
determines what the policy can learn, what the numbers mean, and whether any recorded result survives the next
change. It lives in exactly one place: `packages/nexus-env`.

Getting this right before coding is the highest-leverage hour available. Getting it wrong is discovered forty
minutes into a training run, or — worse — never.

## Observation

**Markov sufficiency.** Ask directly: can the optimal action be chosen from this vector alone? If the right
action depends on something that happened three steps ago and is not encoded, the problem is not Markov and no
amount of training fixes it. Either encode the history (a short window, a moving average) or accept that the
policy is solving a harder problem than you think.

**Normalisation is not optional.** Every element roughly [0,1] or [-1,1]. Raw Mb/s alongside utilization ratios
means the network effectively cannot see the ratios. Normalise by capacity, by max demand, by episode length —
and document the divisor, because it becomes part of the contract.

**Encode what an operator would look at:** per-link utilization, per-link queue backlog, per-flow satisfaction
ratio, per-class aggregate demand. Include an incident indicator if incidents are in scope — a policy that
cannot see a failure cannot route around it deliberately, only accidentally.

**Fixed shape.** The observation vector's length must not depend on runtime state. A variable number of flows
needs padding and a mask, not a variable-length array.

**Ordering is part of the contract.** Element 7 must mean the same thing in every episode and every run. Derive
ordering from a sorted, stable key — never from dict iteration.

## Action

**Start discrete.** DQN over a small discrete space gets the full pipeline working end to end. Continuous
allocation with PPO is more realistic and more expressive, but debugging a continuous action space while the
plumbing is unproven conflates two problems. Discrete first, continuous once discrete works — this is ADR-0004.

**Reachability is the question that matters.** Can the action space express the strategy the reward rewards? A
4-way path choice cannot express a 60/40 split. If the reward rewards proportional fairness and the actions can
only pick one path, the policy will plateau and the plateau will look like a training bug.

**Decoding must be total.** Every action in the space maps to a valid, capacity-respecting allocation. Never
rely on the policy learning to avoid invalid actions — mask them instead.

**Watch the size.** Discrete action spaces grow combinatorially with flows and paths. A space that grows faster
than the sample budget is not learnable; reduce granularity before reducing the problem.

## Reward

**Named components, always.** Never a bare scalar:

```
reward = w_t·throughput_reward − w_l·latency_penalty − w_d·drop_penalty + w_f·fairness_bonus
```

Every term logged individually in `info["reward_components"]`. A flat total with a rising throughput term and a
falling fairness term tells you exactly what happened; a flat scalar tells you nothing.

**Comparable magnitudes.** Compute the expected range of each term under typical conditions before choosing
weights. A penalty two orders of magnitude larger than every bonus makes the other terms invisible, and the
policy will optimise the one term it can see.

**Dense, not sparse.** Flow-level allocation gives signal every step. Use it. Sparse end-of-episode reward makes
credit assignment hard for no benefit here.

**Hunt for the degenerate optimum — the single highest-value check.** Actively try to construct an action that
scores well while achieving nothing useful. Classic instances in this domain:

- Dropping traffic to reduce measured latency on what remains
- Perfect fairness by allocating nothing to anyone
- Maximising throughput on one link while everything queues elsewhere
- Idling to avoid a transition penalty

RL will find whichever of these you left open, reliably, and the result looks like success until someone reads
what the policy actually does.

**Clip, do not let terms explode.** An unbounded penalty produces gradient spikes that destabilise training in
ways that then get misdiagnosed as a hyperparameter problem.

## Version discipline

Any change to observation shape/meaning/normalisation, action space/decoding, reward terms/weights/clipping, or
episode length **is an MDP change**:

1. Bump `MDP_VERSION` in `packages/nexus-env/src/nexus_env/version.py`
2. List every checkpoint it invalidates — search `experiments/runs/*/config.yaml`
3. Check whether any invalidated run is cited in `docs/evaluation/results.md`. A silently-wrong published number is worse than a broken build
4. Say all of this in the PR description
5. Spawn `mdp-reviewer`

## Contract tests

Every MDP change updates these:

- [ ] Observation shape, dtype (`float32`), and bounds
- [ ] Action space bounds; every action decodes to a valid allocation
- [ ] Every reward component present and named in `info`
- [ ] **Sign test per term** — an obviously-good action scores above an obviously-bad one
- [ ] Determinism: same seed + same action sequence ⇒ identical trajectory
- [ ] No NaN or inf under adversarial inputs (zero capacity, zero demand, total oversubscription)

A new reward term without a sign test is not finished.
