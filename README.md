# NEXUS-RL

Autonomous network resource orchestration with reinforcement learning and an LLM agent layer.

A simulated network manages its own resources: an RL controller allocates bandwidth every timestep, and an
LLM agent layer above it turns operator intent ("prioritise low-latency traffic for the next five minutes")
into constraints the controller optimises within — and explains, in natural language, what the controller
actually did.

> This runs against a **simulated** network, not production traffic. That limitation is deliberate and stated
> everywhere the results are — see [Risks & limitations](#risks--limitations).

## Project Structure

```path
├── apps/
│   ├── simulator/       # FastAPI — owns simulation state, advances timesteps
│   ├── controller/      # FastAPI — hosts the trained policy, owns active constraints
│   ├── agent-layer/     # MCP server + any OpenAI-compatible LLM (local Ollama by default)
│   └── dashboard/       # React + WebSocket — live network view and operator chat
├── packages/
│   ├── nexus-sim/       # Flow-level network simulator (pure, seeded)
│   ├── nexus-mdp/       # the MDP contract — observation/action/reward/version, no gym
│   ├── nexus-env/       # gymnasium Env for training, built on nexus-mdp
│   ├── nexus-baselines/ # Static / round-robin / greedy allocators
│   ├── nexus-rl/        # Stable-Baselines3 training + evaluation harness
│   ├── nexus-types/     # pydantic contracts crossing service boundaries
│   ├── nexus-errors/    # NexusError hierarchy
│   ├── nexus-config/    # pydantic-settings — the only reader of env vars
│   └── nexus-telemetry/ # Structured logging + run metrics
├── experiments/         # Training configs (versioned) + run artifacts (gitignored)
└── deploy/k8s/          # Kubernetes manifests
```

## Architecture

Two tiers, deliberately asymmetric in speed and authority:

```
  operator (natural language)
        │
        ▼
┌──────────────────┐   MCP tools    ┌──────────────────┐
│   Agent layer    │ ─────────────► │    Controller    │
│  (LLM, slow)     │  read state    │  (RL, fast)      │
│                  │  write         │                  │
│                  │  CONSTRAINTS   │  emits ACTIONS   │
└──────────────────┘   only         └────────┬─────────┘
                                             │ action
                                             ▼
                                    ┌──────────────────┐
                                    │    Simulator     │──► Dashboard (WebSocket)
                                    │ (state, reward)  │
                                    └──────────────────┘
```

The agent layer can reweight objectives and restrict actions. It cannot allocate bandwidth. That boundary is
the central design claim of the project — see [ADR-0002](docs/adr/0002-constrained-agent-authority.md).

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

**What has not happened yet:** no policy has been trained. Stages B and G need `torch` installed
(`uv sync --all-packages --extra train`) and real compute. Everything else runs today, with the controller
on the `static` baseline.

## Results

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
- **Single-agent, single-domain.** Multi-agent control and network slicing are v2 scope, deliberately excluded.

## Development

- Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `chore:`, `test:`, `sim:`, `rl:`)
- Branch naming: `features/XX`, `fixes/XX`, `experiments/XX`
- Quality gate before every PR: `make pr`
- Agent workflow, hooks and skills: [.claude/WORKFLOW.md](.claude/WORKFLOW.md)

## Documentation

- [Architecture overview](docs/architecture/00_overview.md)
- [MDP specification](docs/mdp/state-action-reward.md) — the state/action/reward contract
- [Domain glossary](CONTEXT.md)
- [Architecture decisions](docs/adr/)
- [Lessons learned](tasks/lessons.md)
- [Security policy](SECURITY.md)

## Background

The two-tier split — an LLM outer tier for high-level sequencing and intent, an RL inner tier for low-level
control — mirrors a current research direction in autonomous 6G / O-RAN network orchestration. This is a
scaled-down, buildable version of that pattern, not a reimplementation of any specific paper.
