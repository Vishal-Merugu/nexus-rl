# 0009 — The model registry is a committed file ledger, not a tracking service

Date: 2026-08-17
Status: Accepted

## Context

Stage I of the v2 plan asks for experiment tracking, a model registry, versioned policy artifacts with
provenance, and a reproducible evaluation suite. The v2 specification names MLflow (self-hosted) or Weights
& Biases as the tooling, and the requirement it states is concrete: *"The controller service should load a
specific policy version by identifier, not a file called `final_model.zip`."*

Before this, a policy was a path. `NEXUS_POLICY_PATH=/models/policy.zip` told you a file existed and
nothing else — not which run produced it, not from which commit, not against which MDP contract, not what
it scored, not whether anybody had ever compared it to the baseline it was supposed to beat. Every one of
those is a question asked at exactly the moment nobody wants to go archaeology in a shell history.

Two constraints of this project shape the answer:

1. **There is no database.** `CLAUDE.md` states it as a rule with an ADR gate attached. MLflow self-hosted
   is a server plus a backing store plus an artifact store — three operational dependencies for a system
   whose current deployment story is `docker compose up` and whose selling point includes running entirely
   offline, with a local Ollama and no account anywhere (ADR-0005).
2. **The thing being stored is tiny.** A few hundred bytes of provenance per trained policy. This project
   will produce tens of runs, not tens of thousands.

## Decision

A **filesystem ledger**, in a new `nexus-registry` package: one JSON document per version under
`experiments/registry/versions/`, holding provenance, a list of evaluation scores, and a deployment stage.

**The ledger is committed to git; checkpoints stay gitignored.** This is the load-bearing half of the
decision. Weights are large, binary, and invalidated by the next `MDP_VERSION` bump — they do not belong in
history. Provenance is small, permanent, and is the actual answer to "what is deployed and where did it
come from"; gitignoring it alongside the weights would mean that answer dies with the machine. A fresh
clone therefore knows the full history of every policy and can serve none of them, which is the correct
asymmetry.

Consequences of the shape:

- **Stage lives inside each version record**, not in a separate alias table. An alias table is a second
  source of truth about what is deployed, and two components deriving that independently is the failure
  `tasks/lessons.md` records under "two services must never derive shared state independently". Resolving
  `production` scans for the version claiming it, and **refuses if two do** rather than picking one.
- **The controller is configured with an intent or a pin** — `NEXUS_POLICY_VERSION=production` or `=v7`.
  The intent form makes promotion and rollback a registry operation rather than a redeploy with an edited
  environment variable.
- **Promotion refuses** a stale MDP version outright, and an unscored or single-seed version unless
  explicitly forced. The project's own hygiene rule ("≥3 seeds before it is a result") is enforced by the
  thing that deploys, not by a checklist.
- **Scores are attached to a named, versioned suite** (`experiments/configs/evaluation-suite.yaml`,
  `suite_version`). A number detached from the scenario set that produced it is not comparable to any other
  number, and a registry full of incomparable numbers invites exactly the comparison it cannot support.
- **`nexus-registry` depends only on pydantic and `nexus-errors`.** That is what lets the controller
  resolve a version without pulling in the training harness — the same separation ADR-0007 made between the
  MDP contract and the gym wrapper.

## Consequences

- **Stage I's question is answerable in one command.** `nexus-registry show production` prints which
  policy is serving, the run and commit behind it, and its suite score. `/health` reports the same as
  structured `policy_provenance`.
- **`NEXUS_POLICY_PATH` survives as an escape hatch** for a checkpoint that was never registered. It is
  ignored when `NEXUS_POLICY_VERSION` is set, and it carries no provenance — `/health` says so rather than
  implying otherwise.
- **No experiment *tracking* UI.** This is the honest cost. There is no reward-curve browser, no run
  comparison view, no hyperparameter sweep dashboard. `metrics.jsonl` per run plus the registry covers
  "what did it score and where did it come from", not "show me forty runs on one axis".
- **Concurrency is single-writer.** The CLIs write, the controller reads. Two machines registering
  simultaneously against a shared checkout could collide on a version id. Acceptable at this scale, and
  the failure is visible (a git conflict) rather than silent.
- **This does not scale to a real team.** Stated plainly because it is the obvious interview question:
  at tens of thousands of runs, or with concurrent writers, or when a reward-curve UI matters, the right
  answer is MLflow. The interface (`PolicyRegistry`) is small enough that an MLflow-backed implementation
  is a swap rather than a rewrite, which is the property being preserved here.

## Alternatives considered

- **MLflow, self-hosted.** Rejected for now on operational cost, not capability. It is the right answer at
  a scale this project is not at, and adopting it would contradict the no-database rule for a few hundred
  bytes per policy. Reconsider when Stage M (distributed training) makes run volume real — many parallel
  rollout workers is exactly the situation a tracking server exists for.
- **Weights & Biases.** Rejected: it is a hosted account and a network dependency, in a project that
  deliberately runs with no external service (ADR-0005). Losing "clone it and everything works offline"
  costs more than the dashboard is worth here.
- **Keep `NEXUS_POLICY_PATH` and add a metadata sidecar next to each checkpoint.** Rejected: the metadata
  would live under `experiments/runs/`, which is gitignored, so provenance would still die with the
  machine. It also gives no way to ask "what is in production" without knowing which file to look at.
- **Store the ledger in the controller and let it be written over HTTP.** Rejected: that makes the
  controller stateful in a way its restart story does not support, and puts a write path to deployment
  metadata inside the fast tier.

## Related

- ADR-0007 — contract/framework split; the same reasoning keeps `nexus-registry` free of training deps.
- ADR-0005 — provider-agnostic, offline-capable by default; a hosted tracker would break that property.
- `.claude/skills/rl-experiment-hygiene/SKILL.md` — the seed/baseline rules the registry now enforces.
