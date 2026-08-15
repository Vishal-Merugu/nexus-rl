# 0001 — Flow-level simulation, not packet-level

Date: 2026-08-14
Status: Accepted

## Context

The RL controller needs an environment to train against. The fidelity of that environment is the single
biggest determinant of how long everything else takes.

Two options:

1. **Packet-level.** Model individual packets, queues, retransmissions, protocol behaviour. Realistic.
   Available off the shelf via ns-3.
2. **Flow-level.** Model aggregate rates per flow per link, queues as scalar backlogs, latency derived
   analytically from utilization and backlog.

The deciding constraint is that RL training does **millions of environment steps**. Step cost is not a
performance detail here — it sets the iteration loop. A 20-minute training run and an overnight one are
different projects: one supports "try it, look at the reward components, change the reward, try again", and
the other does not.

## Decision

**Flow-level simulation**, with a custom graph model rather than ns-3 or networkx.

- Per-flow aggregate rates in Mb/s, resolved against per-link capacity by proportional sharing.
- Queues as a single scalar backlog per link, in Mb, with a finite limit; overflow is dropped.
- Latency derived analytically: an M/M/1-flavoured `1/(1−u)` factor on propagation delay, plus backlog
  drain time. Both clamped.
- Simulation state as numpy arrays with a flow × link incidence matrix, so a timestep is a few matrix
  operations rather than a loop over nodes.

The custom graph model is a deliberate sub-decision: networkx is excellent for graph algorithms, but the
only one needed here is k-shortest-paths, computed once at construction. Everything in the hot path is numpy
indexing. A dependency that touches the inner loop is not worth carrying for one setup-time algorithm.

Budget: **`step()` stays under ~1 ms for a 10-node topology.** Profile before adding fidelity.

## Consequences

- **Training runs in minutes.** Which means reward shaping is iterative rather than a once-a-day ritual, and
  that is the difference between debugging the MDP and guessing at it.
- **The behaviour that matters for control is preserved.** Congestion, queue buildup, drops under overload,
  delay rising sharply near saturation, multi-path routing choice. The policy learns to manage all of these.
- **Several real phenomena are simply absent**, and every writeup must say so: no packet-level jitter, no
  TCP congestion-control interaction, no retransmission dynamics, no head-of-line blocking, no protocol
  overhead. Latency is *derived*, never measured.
- **Results describe a model, not a network.** Nothing here has been validated against hardware. This is
  stated alongside the numbers in `docs/evaluation/results.md`, not in a footnote — the honest framing is
  also the more defensible one under questioning.
- **The analytical latency model needs clamping everywhere.** `1/(1−u)` diverges at full utilization, and a
  single infinity becomes a NaN reward that destroys a training run silently. `MAX_UTILIZATION_FOR_DELAY`
  and `MAX_LINK_LATENCY_MS` exist for this.
- **Packet-level remains reachable.** The env boundary is `nexus-env`; the simulator sits behind it. A future
  ns-3-backed environment implementing the same observation/action contract would be a swap, not a rewrite —
  though it would be an `MDP_VERSION` bump and would invalidate every existing checkpoint.

## Alternatives considered

- **ns-3.** Rejected for v1: high fidelity, but the Python bindings are awkward, step cost is orders of
  magnitude higher, and the engineering effort lands on plumbing rather than on the two-tier control design
  that is the point of the project. Listed as v2 scope in the specification, and this decision keeps that
  door open.
- **networkx for the graph model.** Rejected: pulls a dependency into the hot path for one construction-time
  algorithm. A ~150-line `topology.py` covers what is needed and keeps `step()` pure numpy.
- **Fluid model with no queues at all.** Rejected: without backlog there is no cost to over-subscription
  beyond instantaneous loss, drops stop being a meaningful consequence, and the policy has nothing to learn
  about congestion — the reward's drop and latency terms would be nearly vacuous.
- **Discrete-event simulation.** Rejected: variable time steps complicate the MDP's notion of a timestep for
  no gain at this fidelity.
