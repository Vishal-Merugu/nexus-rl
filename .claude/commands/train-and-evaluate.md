---
description: 'Launch a training run and produce the baseline comparison. Usage: /train-and-evaluate <config-name>'
---

Train and evaluate using config: `$ARGUMENTS` (a file under `experiments/configs/`).

### Step 1: Pre-flight

Before burning compute:

- [ ] Working tree clean (the git SHA gets recorded in the run; a dirty tree makes the run unreproducible)
- [ ] `make test PKG=nexus-env` passes — a broken env wastes the whole run
- [ ] Config exists and states `seed`, `total_timesteps`, `traffic_pattern`, algorithm and hyperparameters
- [ ] `MDP_VERSION` noted, so the resulting checkpoint is traceable

### Step 2: Smoke first

Never launch a long run untested. Run the same config with a tiny budget:

```bash
uv run nexus-train --config experiments/configs/$ARGUMENTS --total-timesteps 2000 --run-id smoke-$(date +%s)
```

Confirm: reward components all present and non-zero, no NaN, step rate reasonable, checkpoint writes.

### Step 3: Train

```bash
uv run nexus-train --config experiments/configs/$ARGUMENTS
```

Report the run-id. Run in the background if long; do not block on it.

### Step 4: Evaluate — three seeds minimum, two patterns minimum

```bash
uv run nexus-eval --run-id <run-id> \
  --traffic-pattern diurnal --traffic-pattern bursty \
  --baseline static --baseline round_robin --baseline greedy_throughput \
  --seeds 3
```

Off-distribution evaluation is mandatory. A policy evaluated only on its training pattern has not been shown to
work, and reporting it that way is the most common way this project could embarrass itself.

### Step 5: Hand to `evaluation-analyst`

Spawn `evaluation-analyst` with the results. It checks the four-part standard — run-id, named baseline, named
pattern, seed count — and drafts the defensible claim.

### Step 6: Report

| Metric              | static | round_robin | greedy | **policy** |
| ------------------- | ------ | ----------- | ------ | ---------- |
| p95 latency (ms)    |        |             |        |            |
| Total throughput    |        |             |        |            |
| Fairness (Jain)     |        |             |        |            |
| Drops               |        |             |        |            |

Report mean ± spread across seeds. Then state, in one sentence each:

- What this shows
- What it does **not** show (fidelity limits, patterns not tested, incident behaviour not covered)

If the numbers go into `docs/evaluation/results.md`, they carry the run-id, seed count, `MDP_VERSION` and git SHA
with them. A number without those is not a result.
