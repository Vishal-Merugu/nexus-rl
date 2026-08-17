# Safety Supervisor

An independent guardrail between the simulator and the controller. See
[ADR-0010](../adr/0010-standalone-safety-supervisor.md) for why it is a separate service.

## Putting it in the path

It speaks the controller's own `/act` contract, so deployment is one line of configuration:

```sh
# Before: the simulator asks the controller directly.
NEXUS_CONTROLLER_URL=http://controller:8001

# After: the simulator asks the supervisor, which asks the controller.
NEXUS_CONTROLLER_URL=http://safety-supervisor:8003
```

The controller does not know the supervisor exists and cannot be made to bypass it. That is the point —
a guardrail inside the thing it guards shares its failure modes.

```
  simulator ──/act──► supervisor ──/act──► controller (xApp)
                          │                     ▲
                          │ checks, corrects    │
                          │ or replaces         │
                          ▼                     │
                   deterministic allocator ─────┘  (only when the policy cannot be trusted)
```

## What it enforces

**Hard action bounds** — checked on the controller's raw output, before any type validation:

| Bound | What happens |
| ----- | ------------ |
| Share outside [0, 1] | Clamped. Preserves the policy's evident intent |
| Non-finite share (NaN/inf) | **Fallback.** No repair is honest — 0.0 starves, 1.0 floods, both are inventions |
| Flow not in the network | Dropped and named |
| Flow omitted from the action | Held at its current rate. Absent is not a decision to starve |
| Route not among the flow's candidate paths | Dropped; the flow keeps its default route |
| Flow near zero for `starvation_window` steps | Reported once. The degenerate optimum a policy finds |

**QoS thresholds** — checked on what the network actually *did*. A policy can emit perfect shares and
still drive the network into the ground; only this catches that.

| Setting | Default | Meaning |
| ------- | ------- | ------- |
| `NEXUS_SAFETY_MAX_P95_LATENCY_MS` | 250 | |
| `NEXUS_SAFETY_MAX_DROP_RATE` | 0.35 | Fraction of offered demand dropped |
| `NEXUS_SAFETY_MIN_FAIRNESS` | 0.25 | Jain's index |
| `NEXUS_SAFETY_BREACHES_TO_ENGAGE` | 10 | Consecutive breaches before falling back |
| `NEXUS_SAFETY_HEALTHY_TO_RELEASE` | 60 | Consecutive healthy steps before the policy is trusted again |

**Release is deliberately slower than engagement.** The fallback fixes the metric it engaged on, so
instant release hands the policy back the network it just broke — an oscillator, not a safety mechanism.

## Endpoints

| Endpoint | Answers |
| -------- | ------- |
| `POST /act` | Interchangeable with the controller's. Always returns an action |
| `GET /health` | Is the process up, and **is the policy or the fallback driving** |
| `GET /safety` | Is the *policy* behaving — violations by kind, correction rate, activations, drift |
| `GET /shadow` | Is the candidate ready for promotion (Use Case 10) |

`/safety` is separate from `/health` on purpose. Uptime and error rate say nothing about whether a
stochastic policy is behaving; these are the numbers that do.

## Shadow mode (Use Case 10)

```sh
NEXUS_SHADOW_CONTROLLER_URL=http://controller-candidate:8001
```

A second controller serving the `shadow`-staged registry version. The supervisor queries both, applies the
incumbent's action, and records the candidate's. **There is no code path from a shadow proposal to the
applied action** — the guarantee is structural, not a flag, and a test asserts it with a share value
nothing else would produce.

Off by default: it doubles the calls in the fast loop, and a feature that silently costs a control-loop
round trip is one nobody can budget for.

Verdicts are about **readiness, never superiority**:

| Verdict | Meaning |
| ------- | ------- |
| `insufficient_data` | Too few timesteps. The default, and the honest one |
| `unsafe` | Would have violated a hard bound. Disqualifying, checked before the sample gate |
| `unresponsive` | Failed to answer too often. Not deployable whatever its quality |
| `indistinguishable` | Behaves like the incumbent — promotion is risk with no expected gain |
| `ready` | Safe, responsive, meaningfully different. **Not "better"** |

Shadow mode cannot tell you which policy is better. The candidate's actions were never applied, so the
network never responded to them and there is no counterfactual latency to compare. Superiority is the
offline [evaluation suite](model-registry.md)'s verdict, where both policies really are applied.

## Robustness evaluation (Use Case 9)

Separate from the service — it runs offline, in `nexus_rl.robustness`:

```python
from nexus_rl.robustness import evaluate_robustness, render_robustness
```

Scores an allocator on the **nominal** network and on **held-out** configurations deliberately outside the
training randomization ranges, then reports the degradation. Both allocators meet identical networks, and
the baseline is built from the *randomized* topology so the policy gets no advantage.

Domain randomization is opt-in per episode via `NetworkSimulator.from_config(..., domain=...)`, and
determinism is preserved: `(seed, domain)` still fully determines the trajectory.

**Domain randomization improves robustness and provides no explicit safety guarantee.** Overly aggressive
ranges destabilise training rather than helping. `render_robustness` prints that caveat with the table
rather than leaving it to a footnote.
