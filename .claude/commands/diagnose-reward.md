---
description: 'Triage a stuck, collapsed, or suspicious training run. Usage: /diagnose-reward [run-id]'
---

Diagnose the training run: `$ARGUMENTS` (if no run-id given, use the most recent under `experiments/runs/`).

Spawn `rl-training-debugger`. Work the checklist **in order** — do not jump to hyperparameters. Most "RL
problems" in this repo have been environment bugs.

### Step 1: Read the artifacts

```bash
ls experiments/runs/<run-id>/
uv run nexus-eval --run-id <run-id> --summary
```

Load `config.yaml` (seed, `MDP_VERSION`, git SHA) and `metrics.jsonl`. Never diagnose from a description of a
curve when the data is on disk.

### Step 2: Decompose the reward

Tabulate each named component over time, **separately**: `throughput_reward`, `latency_penalty`, `drop_penalty`,
`fairness_bonus`. Also plot episode length and the exploration parameter.

Classify what you see:

| Pattern                                     | Likely cause                                            |
| ------------------------------------------- | ------------------------------------------------------- |
| Total flat, components flat at exactly 0    | Action decoding emitting zeros — shape mismatch         |
| Total flat, components moving oppositely    | Terms cancelling; weights need rebalancing, not more training |
| One term dominates by orders of magnitude   | Scale mismatch — the others are invisible to the policy |
| Rises then collapses                        | Instability: LR too high, or a degenerate optimum found |
| Rises then plateaus above every baseline    | **Converged.** Not a bug. Stop and report that          |
| Noisy with no trend                         | Observation uninformative, or reward too sparse/delayed |

### Step 3: Rule out determinism leaks

```bash
uv run pytest packages/nexus-sim/tests/test_determinism.py packages/nexus-env/tests/test_determinism.py -q
```

If these fail, stop. Nothing else is diagnosable until runs reproduce.

### Step 4: Probe the environment, not the algorithm

Cheap probes, in order — none require retraining:

1. **Sign check.** Hand-write an obviously-good action and an obviously-bad one. Does the reward rank them correctly?
2. **Observation informativeness.** Can *you* tell good from bad state from the observation vector alone? If not, neither can the policy.
3. **Normalisation.** Are all observation elements roughly [0,1]/[-1,1]? Raw Mb/s next to ratios breaks learning.
4. **Reachability.** Can the action space express the strategy the reward rewards?
5. **Degenerate optimum.** Try to construct an action that scores well while achieving nothing. If you succeed, that is the bug.

### Step 5: Only now, hyperparameters

Exploration schedule (has ε already hit its floor at the plateau?), learning rate, buffer size, target-update
interval, batch size, discount vs. episode length. For PPO: entropy coefficient, advantage normalisation.

### Step 6: Compare against baselines before calling it a failure

```bash
uv run nexus-eval --run-id <run-id> --baseline static --baseline greedy_throughput
```

A policy plateauing above every baseline has converged. Report it as a result, not a problem.

## Output

```
Run: <run-id>  seed=<n>  MDP_VERSION=<v>
Pattern observed: <from the table above>
Hypothesis: <specific and falsifiable>
Probe run: <what you did> → <result>
Verdict: ENV BUG | REWARD DESIGN | ACTION SPACE | HYPERPARAMETER | CONVERGED
Fix: <change>
Confirm by: <the cheapest run that would prove it>
Ruled out: <what, and how>
```

Never conclude from one seed. If the verdict rests on learning behaviour, confirm across three.
