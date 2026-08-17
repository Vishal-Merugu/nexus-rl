# Model Registry & Evaluation Suite

How a trained policy becomes a deployed one, and how you find out what is running.

The rule this exists to enforce: **the controller loads a policy by version identifier, never by a path to
a file called `policy.zip`.** A path tells you a file exists. A version tells you which run produced it,
from which commit, against which MDP contract, and what it scored. See
[ADR-0009](../adr/0009-file-backed-model-registry.md).

## The lifecycle

```
  nexus-train ──► experiments/runs/<run-id>/     (config.yaml, metrics.jsonl, policy.zip — gitignored)
                            │
                            │  nexus-registry register <run-id>
                            ▼
                  experiments/registry/          (provenance ledger — COMMITTED)
                            │
                            │  nexus-registry score v1     ← the fixed evaluation suite
                            ▼
                      v1 + suite score
                            │
                            │  nexus-registry promote v1 --stage production
                            ▼
                  NEXUS_POLICY_VERSION=production
                            │
                            ▼
                       controller serves it, /health reports its provenance
```

Weights are gitignored; the ledger is committed. A fresh clone therefore knows the full history of every
policy and can serve none of them — which is the correct asymmetry, and why a missing checkpoint reads as
"weights absent", not "registry broken".

## Commands

```sh
# Register a completed run. Provenance is read from the run's own config.yaml — never defaulted.
uv run nexus-registry register dqn-20260817-101500 --notes "first full DQN run"

# Score it against the reproducible suite. Appends; never overwrites an earlier score.
uv run nexus-registry score v1

# Deploy. Refuses a stale MDP version, and an unscored or single-seed version unless forced.
uv run nexus-registry promote v1 --stage production

# What is running, and everything behind it.
uv run nexus-registry show production
uv run nexus-registry list
```

Rollback is promoting the previous version again — the incumbent is demoted to `archived`, not deleted,
precisely so it stays a rollback target.

## Configuring the controller

```sh
NEXUS_POLICY_VERSION=production   # an intent: serve whatever is promoted
NEXUS_POLICY_VERSION=v7           # an exact pin
```

The intent form is what makes promotion and rollback a registry operation rather than a redeploy with an
edited environment variable.

`NEXUS_POLICY_PATH` still works as an escape hatch for a checkpoint that was never registered. It is
ignored when `NEXUS_POLICY_VERSION` is set, and it carries **no provenance** — `/health` reports
`policy_provenance: null` rather than implying otherwise.

## What `/health` answers

```json
{
  "policy": "dqn:v7@mdp1",
  "trained_policy_loaded": true,
  "policy_provenance": {
    "version_id": "v7", "stage": "production",
    "run_id": "dqn-20260817-101500", "git_sha": "abc1234",
    "mdp_version": 1, "algorithm": "dqn", "trained_on": "diurnal",
    "suite": "standard@v1", "mean_p95_latency_ms": 9.5, "seeds": 5
  }
}
```

When a baseline is serving, `fallback_reason` distinguishes the two causes that must never read the same:
nothing was configured (supported, no action needed), or what was configured could not be served (an
incident).

## The evaluation suite

`experiments/configs/evaluation-suite.yaml` is the fixed problem every candidate faces. Two policies are
comparable only if they met the same scenarios, the same seeds and the same step count — so that set is a
versioned artifact rather than whatever flags were typed.

The loader refuses, at load time rather than after minutes of compute:

- fewer than 3 seeds — below that, seed variance routinely exceeds the effect being measured
- fewer than 2 traffic patterns — scoring only where a policy trained measures memorisation
- an unknown pattern or baseline name, or an unknown key in the file

Scores record `suite_version`. **Changing any scenario, seed count or step count makes previously recorded
scores incomparable — bump `suite_version`,** exactly as with `MDP_VERSION`. The per-pattern breakdown is
stored alongside the aggregate because an aggregate hides the policy that wins on its training
distribution and collapses off it.

## Refusals worth knowing about

| Situation | What happens |
| --------- | ------------ |
| Registering the same run twice | Refused — one checkpoint is one artifact |
| Run config missing `git_sha`, `seed`, `topology`… | Refused — a guessed field describes a policy that does not exist |
| Promoting an unscored version | Refused unless `--force` |
| Promoting on 1–2 seeds | Refused unless `--force` |
| Promoting a stale `MDP_VERSION` | Refused, always — its actions mean something different, not merely worse |
| Two versions claiming one stage | `resolve` refuses rather than picking one |
| Registered version, missing checkpoint | Controller degrades to baseline and says which version is missing |

## What this is not

There is no experiment-tracking UI — no reward-curve browser, no run comparison view. `metrics.jsonl` per
run plus this ledger answers "what did it score and where did it come from", not "show me forty runs on
one axis". At the scale where that matters, the answer is MLflow, and `PolicyRegistry` is small enough
that an MLflow-backed implementation would be a swap rather than a rewrite. ADR-0009 states this plainly
because it is the obvious question to ask about the choice.
