---
name: rl-experiment-hygiene
description: Use whenever running, recording, comparing, or writing up a training or evaluation run — before launching a run, when interpreting results, when a number is about to appear in a README, writeup, plot, or interview talking point. Covers seeds, baselines, traffic patterns, metric choice, and what makes a number a result rather than an anecdote.
---

# RL Experiment Hygiene

RL results are unusually easy to get wrong in ways that look right. Seed variance routinely exceeds the effect
being measured; a policy can beat a baseline on its training distribution and lose everywhere else; a reward
improvement can come entirely from a reward-weight change. This skill is the discipline that keeps the numbers
honest.

## The four-part standard

A number is a **result** only if it carries all four. Otherwise it is an anecdote.

1. **Run-id** — traceable to `experiments/runs/<id>/` with config, seed, git SHA, `MDP_VERSION`
2. **Named baseline** — `static`, `round_robin`, or `greedy_throughput`
3. **Named traffic pattern** — at least two, one of them not trained on
4. **Seed count ≥ 3** — with spread, not just a mean

**Not a result:** "PPO improved latency by 23%."
**A result:** "Run `a1b2c3` (PPO, MDP v4, 5 seeds) cut p95 latency 23% ± 4% vs. the `static` baseline on
`diurnal`, and 11% ± 7% on `flash_crowd`, which it was not trained on."

## Before launching a run

- [ ] Working tree clean — a dirty tree makes the recorded SHA a lie
- [ ] `make test PKG=nexus-env` passes
- [ ] Config file contains every hyperparameter. Nothing inline on the command line
- [ ] Seed set explicitly
- [ ] `MDP_VERSION` noted
- [ ] Smoke run first (2k steps) — confirm reward components non-zero, no NaN, checkpoint writes

Time spent here is trivial against a wasted overnight run.

## Metric choice

| Metric      | Report as               | Why                                                                  |
| ----------- | ----------------------- | -------------------------------------------------------------------- |
| Latency     | **p95**, plus mean      | QoS is about the tail. A mean hides exactly what the reward punishes |
| Throughput  | Total **and** per-class | An aggregate win that starves `LATENCY_SENSITIVE` is a loss          |
| Fairness    | Jain's index            | Always alongside throughput — they trade off                         |
| Drops       | Count and rate          | A "win" that drops traffic is not a win                              |
| Reward      | Diagnostics only        | It is your own objective. Weak as a headline                         |

Never lead with reward. It is the thing you designed to be maximised; that the policy maximises it is not news.
Lead with the QoS metrics an operator would care about.

## The failure modes

**Single-seed claims.** Seed variance often exceeds the effect. Three minimum, five if the claim matters.

**Training-pattern overfitting.** Always hold one pattern out. A policy evaluated only where it trained has
demonstrated memorisation, not competence.

**Baseline strawmanning.** Tune the baseline as honestly as the policy. A deliberately bad `static` split makes
the whole comparison worthless — and an interviewer will ask how the baseline was configured.

**Cherry-picked checkpoints.** Report the final checkpoint, or a selection rule declared *before* looking at
results.

**Silent MDP drift.** Two runs under different `MDP_VERSION` values are not comparable. Check before every
comparison; this is the easiest way to publish a wrong number without noticing.

**Reward-metric conflation.** If the reward weights changed, reward improvement means nothing about network
behaviour. Compare QoS metrics instead.

## Plots

- Axes labelled with units
- Baselines on the **same** axes as the policy, never a separate figure
- Per-seed traces or confidence bands, never a lone mean line
- Include the failure cases. An incident-recovery plot with a visible dip is more credible and more interesting than a smooth curve

## Honest framing

State the simulation boundary in the same paragraph as the numbers, not in a footnote:

> Flow-level simulation: aggregate rates and scalar queue backlogs, no packet traces, latency derived
> analytically. Not validated against physical hardware.

Say what the fidelity cannot show. The honest version of this project is strong; the overclaimed version fails
at the first informed question and takes the credible parts down with it.

## Writing it up

Each claim gets: the sentence, the evidence, and the limit.

```markdown
**Claim.** Run `a1b2c3` (PPO, MDP v4, 5 seeds) cut p95 latency 23% ± 4% vs. `static` on `diurnal`.

**Evidence.** experiments/runs/a1b2c3/, seeds 0–4, 512-step episodes, evaluated on `diurnal` and `flash_crowd`.

**Limit.** `flash_crowd` gain was 11% ± 7% — the spread overlaps zero, so generalisation to sudden overload is
not demonstrated.
```

That third line is what makes the first two believable.
