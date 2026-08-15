# MDP Specification

The observation / action / reward contract. Defined in one place — `packages/nexus-mdp` — and nowhere else.

**Current version: `MDP_VERSION = 1`**

Everything downstream is relative to this contract: every checkpoint, every recorded metric, every number in
a writeup. A change here invalidates all of them, which is why the version exists and why bumping it is a
deliberate, announced act.

---

## The problem, as an MDP

At each timestep the orchestrator observes the state of a simulated network — how loaded each link is, how
much traffic is backed up, how well each flow is being served — and chooses how to allocate capacity among
competing flows. It is rewarded for carrying traffic quickly, fairly, and without drops.

- **Agent** — the RL controller
- **Environment** — the flow-level simulator (`nexus-sim`), plus injected incidents
- **Timestep** — one simulation tick; the unit of the inner control loop
- **Episode** — 512 timesteps by default. A training construct: the deployed simulator runs continuously

The problem is genuinely sequential rather than a disguised bandit: today's allocation determines tomorrow's
queue backlog, and a policy that over-subscribes a link pays for it several timesteps later.

---

## Observation

**Shape:** `links × 3 + flows × 7 + 3`, fixed for a given topology and flow count.
**Type:** `float32`, every element in `[0, 1]`.
**Space:** `Box(low=0, high=1)`.

### Per link (3 features)

| Feature           | Encoding                              | Why |
| ----------------- | ------------------------------------- | --- |
| Utilization       | already `[0, 1]`                      | The primary congestion signal |
| Normalised queue  | `queue_mb / MAX_QUEUE_MB`             | Carries history — backlog is an integral of past over-subscription |
| Capacity health   | `effective / nominal`                 | Makes incidents *visible*. A policy that cannot see a failure can only route around it by accident |

### Per flow (7 features)

| Feature            | Encoding                                    | Why |
| ------------------ | ------------------------------------------- | --- |
| Satisfaction ratio | `allocated / demand`, clipped               | How well this flow is being served right now |
| Normalised demand  | `demand / 400 Mb/s`                         | Divisor sits above flash-crowd peaks so the term does not saturate exactly when it matters |
| Normalised latency | `latency_ms / MAX_LINK_LATENCY_MS`          | What the latency penalty is computed from |
| Drop ratio         | `dropped / demand`                          | Distinguishes "throttled" from "losing traffic" |
| Class one-hot (3)  | LATENCY_SENSITIVE / BULK / BACKGROUND       | Lets the policy treat classes differently, which QoS requires |

### Global (3 features)

Mean utilization, network-wide satisfaction, network-wide drop ratio. Cheap summary statistics that let the
policy react to overall pressure without re-deriving it from per-link values every step.

### Contract properties

- **Ordering is fixed** — links and flows in simulator array order, which derives from a sorted stable key.
  Element 7 means the same thing in every episode and every run. Reordering silently invalidates every
  checkpoint.
- **Shape never varies at runtime.** It depends only on topology and flow count, both fixed at construction.
- **Everything is normalised.** Raw Mb/s next to utilization ratios would make the ratios effectively
  invisible to the network. Each divisor is part of the contract.

### Markov sufficiency — the honest caveat

The observation is sufficient for the control decision under the flow-level model: queue backlog carries the
relevant history, so the optimal allocation depends on the present state rather than on the trajectory that
produced it.

It is **not** sufficient for anticipating demand. A policy cannot see that a flash crowd is about to start;
it can only react once demand rises. That is a deliberate limit — demand forecasting is a different problem,
and conflating it with allocation would make both harder to evaluate.

---

## Action

Two modes, both implemented, selected by config.

### Discrete (v1, DQN) — `Discrete(8)`

Eight named allocation modes rather than a value per flow. Per-flow discrete levels would be `4^n` — 65,536
actions at 8 flows, unlearnable and illegible.

| Index | Mode | Effect |
| ----- | ---- | ------ |
| 0 | `OPEN` | Every flow at full demand |
| 1 | `THROTTLE_MODERATE` | Everything to 60% |
| 2 | `THROTTLE_AGGRESSIVE` | Everything to 30% — sheds load to drain backlog |
| 3 | `PRIORITISE_LATENCY_SENSITIVE` | Interactive at 100%, rest at 40% |
| 4 | `PRIORITISE_BULK` | Bulk at 100%, rest at 40% |
| 5 | `SHED_BACKGROUND` | Background to 10%, rest untouched |
| 6 | `REROUTE_LEAST_LOADED` | Full rates, every flow onto its least-utilized candidate path |
| 7 | `PRIORITISE_LATENCY_AND_REROUTE` | Modes 3 and 6 together |

**Index order is part of the contract. Append only; never reorder.**

**Reachability, stated plainly:** this space cannot express "60% to f1, 40% to f2". Some optimal allocations
are outside it, so a discrete policy will plateau below what a continuous one could reach. **That plateau is
expected, not a bug** — see ADR-0004 and `/diagnose-reward`.

### Continuous (Stage G, PPO) — `Box(flow_count,)`

One share in `[0, 1]` per flow. Fully expressive. Actions outside the box are clipped rather than rejected —
a policy exploring outside is normal early in training.

### Decoding is total

Every action in either space decodes to a valid, capacity-respecting allocation. Out-of-range discrete
indices clamp; short continuous vectors default the remaining flows to full demand; unknown flow ids are
ignored. Nothing relies on the policy learning to avoid invalid actions, and no action can crash a run.

Operator constraints restrict **routing** at decode time (`forbidden_link_ids`), never rates. If every
candidate path for a flow is forbidden, the flow is still routed — a constraint must never black-hole
traffic.

---

## Reward

Four named components, each logged individually in `info["reward_components"]`. Never a bare scalar.

```
reward = 1.0 · throughput_reward
       − 1.0 · latency_penalty
       − 2.0 · drop_penalty
       + 0.5 · fairness_bonus
```

| Component | Definition | Range |
| --------- | ---------- | ----- |
| `throughput_reward` | `total_allocated / total_demand` | `[0, 1]` |
| `latency_penalty` | class-weighted mean normalised flow latency | `[−1, 0]` |
| `drop_penalty` | `total_dropped / total_demand` | `[−2, 0]` |
| `fairness_bonus` | Jain's index × network satisfaction | `[0, 0.5]` |

**Class weighting for latency:** LATENCY_SENSITIVE 3.0, BULK 1.0, BACKGROUND 0.25, normalised by the maximum
so the term stays in `[0, 1]` while preserving the ordering between classes. (Normalising by the *sum* of
participating weights would cancel the weighting entirely — a weighted mean over one class is that class's
value regardless of weight. This was a real bug, caught by the class sign test.)

**Drops are weighted highest** so that shedding traffic to make the survivors look fast is never profitable.

**An idle network scores exactly zero.** Rewarding an empty network would teach the policy to suppress
demand.

### Degenerate optima, and what closes each

| Exploit | Closed by |
| ------- | --------- |
| Perfect fairness by allocating nothing (Jain's index = 1.0 when all flows get zero) | `fairness_bonus` scaled by network satisfaction |
| Cutting measured latency by dropping the congested flows | `drop_penalty` weighted above the achievable latency gain |
| Aggregate throughput while starving interactive traffic | Per-class latency weighting |
| Suppressing demand to avoid penalties | Idle network scores 0, not positive |

Each has a test in `packages/nexus-mdp/tests/test_reward.py`. This is the highest-value part of the test
suite: RL finds whichever exploit is left open, reliably, and the resulting policy looks successful.

---

## Episode

- **Length:** 512 timesteps, then `truncated`
- **Never `terminated`.** A network under load has no goal state; an early-termination signal would teach the
  policy to seek one
- **Reset** with a seed rebuilds the simulator including the flow set, making an episode exactly reproducible

---

## Changing any of this

Any change to observation shape/meaning/normalisation, action space/decoding, reward
terms/weights/clipping, or episode length is an **MDP change**:

1. Bump `MDP_VERSION` in `packages/nexus-mdp/src/nexus_mdp/version.py`
2. List invalidated runs — `find_runs_with_mdp_version(old)`
3. Check whether any is cited in `docs/evaluation/results.md`
4. Say all of it in the PR
5. Spawn `mdp-reviewer`

Evaluation refuses to load a checkpoint whose recorded version does not match, rather than producing numbers
that quietly mean nothing.

---

## Version history

| Version | Change | Invalidated |
| ------- | ------ | ----------- |
| 1 | Initial formulation | — |
