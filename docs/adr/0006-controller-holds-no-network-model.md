# 0006 — The controller decodes purely from the state it received

Date: 2026-08-15
Status: Accepted

## Context

The controller has to turn a policy's action index into an `AllocationAction`. Decoding needs more than the
observation vector: a discrete mode resolves to per-flow shares using each flow's `traffic_class`, and a
rerouting mode picks a path out of each flow's `candidate_paths`. Neither survives observation encoding —
the observation is a normalised float vector with a class one-hot, not a routing table.

The first implementation solved this by giving the controller its own copy of the world. It read
`NEXUS_SEED` and `NEXUS_TOPOLOGY` from its environment, rebuilt the flow set through
`NetworkSimulator.from_config`, and decoded against that.

This is wrong in a way that produces no error anywhere. Flow ids are **positional** — `f0`, `f1`, … `fn`
in construction order. If the controller's seed or topology differs from the simulator's by any amount, it
still builds a flow set whose ids line up perfectly with the ids in the incoming state. Every lookup
succeeds. Every share is applied to a real flow. But the `candidate_paths` behind `f3` belong to a
different `f3`, so the controller reroutes traffic onto links that are not on that flow's path at all.

The failure surface is: wrong routes, plausible-looking allocations, no exception, no log line, and a
policy that appears to have learned something incoherent. In deployment the two services are separate
containers with independently-set environment variables, so the configurations drifting is not a remote
possibility — it is the default outcome of editing one manifest.

## Decision

**The controller holds no model of the network.** It decodes only from the `NetworkState` it was sent.

`FlowState` carries `candidate_paths` as part of the wire contract, and
`PolicyHost._routing_from(state)` builds the decoder's `FlowRoutingOptions` from that field on every
request. The controller does not import `NetworkSimulator`, does not read a topology name, and does not
know what seed the simulator is running.

The same rule binds the evaluator: `nexus_rl.policy_loader.routing_from(state)` is the identical function
against the identical field, so a policy is measured through the same decoding path it is served with.

## Consequences

- **Configuration drift becomes impossible rather than merely unlikely.** There is no second copy of the
  flow set to disagree with the first. The question "are the controller and simulator configured the same?"
  stops existing.
- **The state message got bigger.** `candidate_paths` is a tuple of paths per flow, sent every timestep.
  At mesh-6 with 8 flows this is tens of bytes; the dashboard already receives downsampled state, so the
  cost lands only on the control path, which is local.
- **The controller is stateless with respect to the network.** It can be restarted, scaled, or pointed at a
  different simulator with no reconfiguration, because everything it needs arrives in the request.
- **The simulator owns the action space.** If the set of candidate paths changes, the controller follows
  automatically. That is the correct ownership — the simulator is the single writer of network state.
- **A malformed or truncated state now fails loudly** at decode time rather than being silently patched
  from local configuration, which is the outcome we want: an error naming the real cause beats a plausible
  wrong answer.

## Alternatives considered

- **Keep the config-derived model, add a consistency check** (compare a hash of the flow set on each
  request). Rejected: it detects the divergence instead of preventing it, adds a failure mode with no
  recovery, and still requires both services to be configured — the check is one more thing that can be
  wrong.
- **Send only the current path, not the candidates.** Rejected: the policy's rerouting modes are defined
  over the candidate set. Without it the controller can express "keep the current path" and nothing else,
  which removes two of the eight discrete actions.
- **Have the controller query the simulator for topology at startup.** Rejected: it reintroduces a cached
  copy that goes stale on any simulator reset, and it makes controller startup depend on simulator
  availability for information that every subsequent request already carries.

## Related

- ADR-0002 — constrained agent authority; the constraint path likewise carries everything it needs.
- `tasks/lessons.md` — "Two services must never derive shared state independently from config".
