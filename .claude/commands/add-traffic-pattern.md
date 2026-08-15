---
description: 'Add a seeded traffic pattern and wire it into evaluation. Usage: /add-traffic-pattern <name> <description>'
---

Add a traffic pattern: `$ARGUMENTS`

Traffic patterns are the axis along which this project proves a policy generalises. Adding one is cheap and
high-value — a policy that only wins on `steady` has not been shown to work.

### Step 1: Define the demand process

In `packages/nexus-sim/src/nexus_sim/traffic.py`, implement the `TrafficPattern` protocol:

```
demand_mbps(timestep, flows, rng) -> NDArray[np.float32]
```

Rules:

- **Every draw uses the injected `rng`.** No global RNG, no wall-clock. Same seed ⇒ same demand sequence, always.
- Demand is per-flow, in Mb/s, non-negative.
- The pattern is a pure function of `(timestep, flows, rng)`. No internal mutable state that survives a reset.
- Document what real-world condition it represents and what it is designed to stress.

Existing patterns and what each stresses:

| Pattern       | Stresses                                                      |
| ------------- | ------------------------------------------------------------- |
| `steady`      | Baseline competence. Everything should do well here            |
| `diurnal`     | Slow adaptation — can the policy track a smooth trend?        |
| `bursty`      | Reaction time and queue management                            |
| `flash_crowd` | Sudden concentrated overload — where fairness policies break  |

Pick a pattern that stresses something none of these do, or do not add it.

### Step 2: Register it

Add to the pattern registry so `--traffic-pattern <name>` resolves it. No new branch in calling code — the
registry is the extension point.

### Step 3: Test it

`packages/nexus-sim/tests/test_traffic.py`:

- [ ] Determinism: same seed ⇒ identical sequence
- [ ] Different seeds ⇒ different sequences (a pattern ignoring its rng is a real bug that hides for months)
- [ ] Non-negative, finite, no NaN
- [ ] The shape it claims to have: assert the actual characteristic (a burst pattern has a variance spike; a diurnal one has a period)

### Step 4: Wire into evaluation

Add to the evaluation matrix in `experiments/configs/evaluation.yaml`. Every policy is then evaluated against
it automatically.

### Step 5: Do not train on everything

Keep at least one pattern **out** of every training config, held back for off-distribution evaluation. A policy
trained on all patterns cannot demonstrate generalisation — and that demonstration is the point.

Then `/post-implement-check`.
