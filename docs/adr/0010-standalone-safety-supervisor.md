# 0010 — The safety supervisor is a separate service in the fast path

Date: 2026-08-17
Status: Accepted

## Context

Before this, the system had no answer to "what stops the policy from doing something harmful?". The
degradation path was worse than absent — it was *misleading*. The simulator's control loop asked the
controller for an action; if the response failed to parse, it caught the error, logged a line, and stepped
with **no action at all**, meaning every flow ran at full demand. A diverged policy therefore produced an
unmanaged network that looked, from the outside, like a managed one.

The v2 specification's Track D asks for hard action bounds enforced outside the model, automatic fallback
to a deterministic heuristic, and drift detection. Its Decision 9 also answers where that lives:
*a separate service in the fast path — if it lives inside the controller, a bug in the policy-serving code
can bypass it.*

There is a second, subtler problem the design had to solve. `AllocationAction` validates itself: shares
outside [0, 1] and non-finite values fail its own validator. So **anything holding an `AllocationAction`
is holding an action that was already proven fine**, and a guardrail placed there would inspect only
actions that could not be wrong. The interesting failures had already become a parse error somewhere else,
with the diagnosis thrown away.

## Decision

A standalone `safety-supervisor` service that **speaks the controller's own `/act` contract** and sits
between the simulator and the controller. Deployment is one line of configuration — point the simulator's
`NEXUS_CONTROLLER_URL` at the supervisor. The controller does not know it exists and cannot be made to
bypass it.

It parses the controller's response **leniently**, into a `ProposedAction` that enforces nothing, and does
its own checking. It emits a validated `AllocationAction`. That asymmetry — inspect raw, emit trusted — is
the whole point of the boundary.

Its decisions, in order:

- **Correct where correction is unambiguous.** A share of 1.9 clamps to 1.0, preserving the policy's
  evident intent. An omitted flow is held at its current rate, because absent is not a decision to starve.
- **Fall back where it is not.** A non-finite share has no honest repair: 0.0 starves the flow, 1.0 floods
  it, and both are inventions. The deterministic allocator takes the timestep instead.
- **Fall back on sustained outcome breaches, with hysteresis.** Action bounds constrain what the policy
  may ask for; QoS thresholds constrain what the network actually does, and a policy can emit perfect
  shares while driving the network into the ground. Engagement needs consecutive breaches, and release
  needs a *longer* run of healthy steps — because the fallback fixes the metric it engaged on, so instant
  release produces an oscillator rather than a safety mechanism.
- **Report drift, never act on it.** Drift is a leading indicator and evidence for a human deciding to
  retrain. By the time QoS is breaching, the warning has come too late to be useful.

Shadow mode (Stage L) lives here too, for the same reason: the supervisor is already in the path and
already holds both the bounds and the comparison logic.

## Consequences

- **The degraded path is now deliberate.** There is no execution in which the simulator receives no
  action. "Unmanaged at full demand" has been replaced by a named, deterministic allocator and a
  `fallback_engaged` flag on `/health`.
- **A fallback action is never labelled as the policy's.** It carries `supervisor:fallback`. Reusing the
  controller's label would make the dashboard, the decision log and the agent layer's explanations all
  attribute the baseline's behaviour to the policy.
- **Policy-level observability exists**, on `/safety`: violation counts by kind, correction rate, fallback
  activations, starved flows, drift. Deliberately separate from `/health` — a liveness probe asks "is this
  process up", and that is a different question from "is the policy behaving".
- **One more hop in the fast loop, and one more service that can fail.** The honest cost. The hop is a
  local call against a simulator ticking at hundreds of milliseconds, and the failure mode is designed
  for: if the supervisor cannot reach the controller it runs the deterministic allocator rather than
  passing the problem on.
- **The guardrail holds no network model**, for the same reason the controller does not (ADR-0006). Its
  fallback allocator is derived from the state the simulator sent, once.
- **It is not a proof of anything.** The bounds are hard limits and the fallback is real, but neither
  constitutes a safety *guarantee*. Domain randomization (also Stage K) improves robustness without
  guaranteeing it, and the robustness table says so in its own output rather than in a footnote.

## Alternatives considered

- **A module inside the controller.** Rejected on the specification's own reasoning, which is correct: a
  guardrail sharing a process with the policy shares its failure modes, and a bug in policy-serving code
  can bypass an in-process check. Independence *is* the value.
- **Validation inside the simulator, where the action is applied.** Rejected: the simulator is the single
  writer of authoritative state and should not also be the arbiter of whether the controller is
  trustworthy. It also puts the guardrail downstream of the parse that was destroying the diagnosis.
- **Keep parsing into `AllocationAction` and let its validator reject bad actions.** Rejected — this is
  the trap described above. It is a *type* boundary, and a good one; it is not a behavioural guardrail,
  and treating it as one leaves "malformed response" as the only diagnosis a diverged policy ever gets.
- **Fall back on a single breached timestep.** Rejected: one bad step under a traffic burst is not a
  failing policy, and reverting on it produces flapping between allocators.

## Related

- ADR-0006 — the controller holds no network model; the supervisor holds none either, for the same reason.
- ADR-0009 — the registry's `shadow` stage is what a shadow-mode candidate is promoted from.
- `docs/operations/safety-supervisor.md` — thresholds, endpoints, and the deployment change.
