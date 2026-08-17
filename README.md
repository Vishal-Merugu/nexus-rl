# NEXUS-RL

Autonomous network resource orchestration with reinforcement learning and an LLM agent layer.

A simulated network manages its own resources: an RL controller allocates bandwidth every timestep, and an
LLM agent layer above it turns operator intent ("prioritise low-latency traffic for the next five minutes")
into policy the controller optimises within — and explains, in natural language, what the controller
actually did.

The two tiers are structured according to **O-RAN**'s RIC architecture: the controller is an **xApp** on a
Near-RT RIC, the agent layer is an **rApp** on a Non-RT RIC, and guidance flows down an **A1**-shaped policy
interface that answers back with a decision — including a reasoned refusal.

> This runs against a **simulated** network, not production traffic. That limitation is deliberate and stated
> everywhere the results are — see [Risks & limitations](#risks--limitations).

## Project Structure

```path
├── apps/
│   ├── simulator/       # FastAPI — owns simulation state, advances timesteps
│   ├── controller/      # FastAPI — hosts the trained policy, owns active constraints
│   ├── agent-layer/     # MCP server + any OpenAI-compatible LLM (local Ollama by default)
│   ├── safety-supervisor/ # Independent guardrail in the fast path — bounds, fallback, drift
│   └── dashboard/       # React + WebSocket — live network view and operator chat
├── packages/
│   ├── nexus-sim/       # Flow-level network simulator (pure, seeded)
│   ├── nexus-mdp/       # the MDP contract — observation/action/reward/version, no gym
│   ├── nexus-env/       # gymnasium Env for training, built on nexus-mdp
│   ├── nexus-baselines/ # Static / round-robin / greedy allocators
│   ├── nexus-rl/        # Stable-Baselines3 training + evaluation harness
│   ├── nexus-a1/        # A1 policy contract — documents, decisions, validation, translation
│   ├── nexus-registry/  # Model registry — versioned policies, provenance, scores, deployment stage
│   ├── nexus-safety/    # Guardrails — action bounds, fallback activation, drift, shadow comparison
│   ├── nexus-mdp/       # …also the CTDE contract: agent scopes, local vs joint observations
│   ├── nexus-types/     # pydantic contracts crossing service boundaries
│   ├── nexus-errors/    # NexusError hierarchy
│   ├── nexus-config/    # pydantic-settings — the only reader of env vars
│   └── nexus-telemetry/ # Structured logging + run metrics
├── experiments/         # Training configs + evaluation suite + registry (versioned); runs (gitignored)
└── deploy/k8s/          # Kubernetes manifests
```

## Architecture

Two tiers, deliberately asymmetric in speed and authority:

```
  operator (natural language)
        │
        ▼
┌──────────────────┐  A1 policy ──► ┌──────────────────┐
│   Agent layer    │                │    Controller    │
│  rApp / Non-RT   │  MCP read      │  xApp / Near-RT  │
│  (LLM, slow)     │ ◄── decision   │  (RL, fast)      │
│  issues POLICY   │  ACCEPTED /    │  emits ACTIONS   │
└──────────────────┘  PARTIAL /     └────────┬─────────┘
                      REJECTED               │ action (E2-shaped)
                                             ▼
                                    ┌──────────────────┐
                                    │    Simulator     │──► Dashboard (WebSocket)
                                    │ (E2 nodes)       │
                                    └──────────────────┘
```

The agent layer can reweight objectives, restrict routing, and request per-class floors. It cannot allocate
bandwidth, and it cannot write a constraint directly — it offers a policy, and the controller decides what it
can honour. That boundary is the central design claim of the project — see
[ADR-0002](docs/adr/0002-constrained-agent-authority.md) and
[ADR-0008](docs/adr/0008-oran-aligned-a1-interface.md).

**Architectural alignment, not conformance.** This is built against O-RAN's published architecture and its
interface concepts; it is not a conformance-tested implementation, and the transport is plain HTTP rather
than the standard's own bindings.

Full walkthrough: [docs/architecture/00_overview.md](docs/architecture/00_overview.md).

## Quick Start

1. Install dependencies:

   ```sh
   make install
   ```

2. Set up environment variables:

   ```sh
   cp .env.example .env   # defaults work as-is — no account or API key needed
   ```

   The agent layer talks to any **OpenAI-compatible** endpoint and defaults to a local Ollama, so the whole
   system runs with no paid service anywhere:

   ```sh
   ollama serve && ollama pull qwen2.5:7b
   ```

   Switching to OpenRouter, Groq, OpenAI or Anthropic is two environment variables, not a code change —
   see [ADR-0005](docs/adr/0005-provider-agnostic-llm.md).

3. Run the whole stack:

   ```sh
   docker compose -f .docker/docker-compose.yml up -d
   ```

   Dashboard on http://localhost:5173, simulator API docs on http://localhost:8000/docs.

No trained policy is needed to start — the controller falls back to the `static` baseline and says so on
`/health`. See [AGENTS.md](AGENTS.md) for running services without Docker.

## Training a policy

```sh
# Smoke run — minutes, proves the pipeline
uv run nexus-train --config experiments/configs/dqn-smoke.yaml

# Full discrete DQN run
uv run nexus-train --config experiments/configs/dqn-discrete-baseline.yaml

# Evaluate against baselines across traffic patterns
uv run nexus-eval --run-id <run-id> --traffic-pattern diurnal --traffic-pattern bursty
```

Hyperparameters live in config files, never on the command line. Every run writes `config.yaml`,
`metrics.jsonl` and checkpoints under `experiments/runs/<run-id>/`; artifacts are gitignored.

## Deploying a policy

A policy is deployed by **version**, never by file path — so "what is running?" has a real answer.

```sh
uv run nexus-registry register <run-id>          # provenance read from the run's own config
uv run nexus-registry score v1                   # against the fixed evaluation suite
uv run nexus-registry promote v1 --stage production
uv run nexus-registry show production             # what is deployed, from where, and what it scored
```

Then `NEXUS_POLICY_VERSION=production` (or an exact pin, `v7`). Promotion refuses a stale `MDP_VERSION`
outright, and an unscored or single-seed version unless forced — the ≥3-seed rule is enforced by the thing
that deploys, not by a checklist. The ledger under `experiments/registry/` is committed; the weights are
not. Full workflow: [docs/operations/model-registry.md](docs/operations/model-registry.md).

## Build stages

The system is built so that it is demonstrable at the end of every stage, not only at the end.

| Stage | Deliverable                                          | Status |
| ----- | ---------------------------------------------------- | ------ |
| A     | Flow-level simulator + non-learned baselines         | ✅ built, tested, produces comparison tables |
| B     | gymnasium env + DQN harness (discrete actions)       | ✅ built and tested — **no policy trained yet** |
| C     | Simulator + controller as containerised services     | ✅ built, control loop verified end to end |
| D     | Live React dashboard over WebSocket                  | ✅ built — graph, metrics, flows; chat UI pending |
| E     | MCP agent layer — operator query + directed change   | ✅ tools + authority boundary built and tested |
| F     | Kubernetes deployment                                | ✅ manifests written — not yet applied to a cluster |
| G     | PPO + continuous actions, full evaluation writeup    | ⬜ config exists, needs compute |
| H     | O-RAN reframing — A1 policy interface between the tiers | ✅ contract, xApp consumption, rApp emission, docs |
| I     | MLOps — model registry, versioned policies, evaluation suite | ✅ registry + suite + controller loads by version |
| K     | Safety layer — supervisor, domain randomization, robustness eval | ✅ service in the fast path; Use Case 9 apparatus |
| L     | Shadow mode + promotion workflow                     | ✅ shadow mode (Use Case 10); CMDP/safety critic **not** built |
| J     | Network slicing + multi-agent decomposition + MAPPO  | ◐ built and **trains**; single-seed only — not yet a result |

Stages H, I, K, L and most of J are **v2** (Tracks A, B, D, E). What is *not* built, plainly:

- **The single-agent-vs-multi-agent comparison table** — Stage J's actual stated output. MAPPO trains and
  learns (see below), but on **one seed**, which this project's own rules say is not a result.
- **CMDP + safety critic** (Stage L, part 2).
- **Stage M** (RLlib on Kubernetes) — its deliverable is a wall-clock benchmark on a cluster.
- **Stage N** (packet-level ns-3) — needs a C++ simulator built from source. The spec places it last
  deliberately.

`MDP_VERSION` is now **2** — the slice-aware reward component invalidates every version-1 checkpoint. No
published figure changes; `docs/evaluation/results.md` holds baseline numbers only.

**What has not happened yet:** no policy has been trained. Stages B and G need `torch` installed
(`make install`) and real compute. Everything else runs today, with the controller on the `static` baseline.

## Results

**MAPPO trains.** A 60k-timestep run over three slices (`mesh-6`, 9 flows, `diurnal`, seed 0) moved mean
episode reward from 100.3 to 115.3 across 470 episodes, with every named component improving —
throughput 0.499 → 0.579, fairness 0.233 → 0.271, SLA attainment 0.062 → 0.077.

> **This is a pipeline proof, not a result.** One seed, no baseline comparison, no held-out pattern. Per
> `.claude/skills/rl-experiment-hygiene/SKILL.md` a stated claim needs a baseline, a traffic pattern and
> ≥3 seeds. Do not cite these numbers.

```sh
uv run nexus-train-marl --config experiments/configs/mappo-slicing.yaml
```

Baseline numbers are reproducible right now, before any training:

```sh
uv run nexus-eval --traffic-pattern diurnal --traffic-pattern flash_crowd --seeds 3
```

Policy results land in [docs/evaluation/results.md](docs/evaluation/results.md) via `make evaluate`. Every
figure there carries a run-id, a seed count, a traffic pattern, and a named baseline — `nexus-eval` labels
anything with fewer than three seeds "not a result" rather than printing it as one.

## Risks & limitations

- **Simulated, not real.** Flow-level fidelity: aggregate rates and scalar queue backlogs, no packet traces.
  Latency is derived analytically. Nothing here has been validated against physical hardware.
- **RL training is unstable by nature.** Plateaus and collapses are expected; `/diagnose-reward` exists for that.
- **No trained policy anywhere.** Every number in this repo is a baseline number. The registry can hold,
  score and deploy a policy; there is not yet one to deploy.
- **Multi-agent is a contract, not a result.** Slicing and the CTDE decomposition are built and tested;
  MAPPO is not trained, so no multi-agent claim is made.
- **Guardrails are not guarantees.** The safety supervisor enforces hard bounds and a deterministic
  fallback; domain randomization improves robustness. Neither is a proof of safety, and the robustness
  table says so in its own output.

## Development

- Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `chore:`, `test:`, `sim:`, `rl:`)
- Branch naming: `features/XX`, `fixes/XX`, `experiments/XX`
- Quality gate before every PR: `make pr`
- Agent workflow, hooks and skills: [.claude/WORKFLOW.md](.claude/WORKFLOW.md)

## Documentation

- [Architecture overview](docs/architecture/00_overview.md)
- [MDP specification](docs/mdp/state-action-reward.md) — the state/action/reward contract
- [Domain glossary](CONTEXT.md)
- [Model registry & evaluation suite](docs/operations/model-registry.md)
- [Safety supervisor](docs/operations/safety-supervisor.md) — guardrails, fallback, shadow mode
- [Architecture decisions](docs/adr/)
- [Lessons learned](tasks/lessons.md)
- [Security policy](SECURITY.md)

## Background

The two-tier split — an LLM outer tier for high-level sequencing and intent, an RL inner tier for low-level
control — mirrors a current research direction in autonomous 6G / O-RAN network orchestration. This is a
scaled-down, buildable version of that pattern, not a reimplementation of any specific paper.
