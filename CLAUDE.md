ALWAYS FOLLOW: **Be extremely concise. Sacrifice grammar for the sake of concision**

## Common Development Commands

### Docker Dev Environment

```bash
# Start dev stack (simulator + controller + agent + dashboard)
docker compose -f .docker/docker-compose.yml up -d

# Attach to dev container
docker compose -f .docker/docker-compose.yml exec dev bash

# Override ports if defaults conflict
DASHBOARD_HOST_PORT=5180 docker compose -f .docker/docker-compose.yml up -d
```

See `.docker/README.md` for full documentation.

### Build and Development

```bash
# Install everything (uv workspace + pnpm for dashboard)
make install

# Type check (mypy --strict over all Python packages)
make types

# Lint + format (ruff)
make lint
make format

# Tests
make test

# Tests for one package
make test PKG=nexus-sim

# Single test file
uv run pytest packages/nexus-sim/tests/test_link.py -q

# Pre-commit gate (types + lint + test + dashboard build)
make pr
```

### Training & Evaluation

```bash
# Train a policy. Always via a config file — never inline hyperparameters.
uv run nexus-train --config experiments/configs/dqn-discrete-baseline.yaml

# Evaluate a run against the static + round-robin baselines
uv run nexus-eval --run-id <run-id> --traffic-pattern diurnal

# Regenerate the comparison table used in the writeup
make evaluate
```

Runs write to `experiments/runs/<run-id>/` (metrics.jsonl, config.yaml, checkpoints).
**Never commit checkpoints** — `experiments/runs/` is gitignored except `.gitkeep`.

## High-Level Architecture

### Monorepo Structure

Polyglot **uv workspace** (Python) + a single **pnpm** app (dashboard):

- **`apps/`** — deployable services, one container each
- **`packages/`** — importable libraries, no service code, no I/O at import time
- **`experiments/`** — training configs + run artifacts (artifacts gitignored)
- **`deploy/k8s/`** — Kubernetes manifests

### Services (`apps/`)

| Service        | Language        | Responsibility                                                        |
| -------------- | --------------- | --------------------------------------------------------------------- |
| `simulator`    | Python/FastAPI  | Owns simulation state. Advances timesteps, applies actions, emits state |
| `controller`   | Python/FastAPI  | Hosts the trained policy. State in → action out. Owns active constraints |
| `agent-layer`  | Python/MCP      | LLM agent. Reads state via MCP tools, adjusts controller constraints    |
| `dashboard`    | TypeScript/React | Live network view + operator chat. WebSocket client only               |

### Key Packages

- **`nexus-sim`** — flow-level network simulator. Pure, deterministic given a seed. No FastAPI, no LLM, no gym.
- **`nexus-mdp`** — the MDP contract: observation encoding, action decoding, reward, `MDP_VERSION`. Pure numpy — no gym, no services. This is what the controller imports.
- **`nexus-env`** — gymnasium `Env` wrapping `nexus-sim` against that contract. Training only; nothing in `apps/` imports it.
- **`nexus-baselines`** — non-learned allocators (static, round-robin, greedy-throughput). The comparison floor.
- **`nexus-rl`** — Stable-Baselines3 training/eval harness, config loading, run artifacts.
- **`nexus-types`** — pydantic contracts crossing service boundaries (`NetworkState`, `Action`, `ConstraintSet`).
- **`nexus-a1`** — the A1 interface contract: versioned policy documents the outer tier (rApp) issues to the inner tier (xApp), their validation against the live network, and translation into constraints. Pure pydantic — no simulator, no numpy. See ADR-0008.
- **`nexus-errors`** — `NexusError` hierarchy. Every raised error subclasses it.
- **`nexus-config`** — `pydantic-settings` app config. The only place env vars are read.
- **`nexus-telemetry`** — structured logging + run-metric emission.

### Domain Architecture

**NEXUS-RL** is a two-tier autonomous network orchestrator over a simulated network:

- **Inner tier (fast, numeric)** — the RL controller allocates bandwidth / picks routes every timestep.
- **Outer tier (slow, linguistic)** — the LLM agent layer translates operator intent into *constraints* the inner tier optimizes within, and explains inner-tier decisions after the fact.

The agent layer **never** emits raw allocations. See ADR-0002.

Domain vocabulary lives in `CONTEXT.md`. Use its terms; don't invent synonyms.

### Data Layer

There is no database. State lives in three places:

- **Live simulation state** — in-memory in the simulator service, authoritative, single writer.
- **Rolling history** — bounded ring buffer in the controller (default 512 timesteps) for agent explanations.
- **Run artifacts** — append-only `metrics.jsonl` per training/eval run on disk.

If you find yourself adding a database, stop and open an ADR first.

## Development Guidelines

### Code Organization

- Package layout is `packages/<dist-name>/src/<import_name>/` + `packages/<dist-name>/tests/`
- Dist names are kebab (`nexus-sim`), import names are snake (`nexus_sim`)
- One concept per module. Modules named after the noun they own (`link.py`, `flow.py`, `reward.py`)
- Service code lives in `apps/`, never in `packages/`

### Simulation Rules

- The simulator is **pure and seeded**. Same seed + same action sequence ⇒ identical trajectory, always.
- No wall-clock time, no `random` module, no global RNG. Take a `numpy.random.Generator` explicitly.
- Never mutate a `NetworkState` in place — `step()` returns a new one. Snapshots handed to the agent layer must stay valid.
- Physical invariants are asserted, not assumed: utilization ∈ [0, 1], queue ≥ 0, allocations sum ≤ capacity.

### MDP Rules

- The observation, action, and reward definitions live in exactly one place: `nexus-mdp`. Nothing else encodes them, and no service depends on gymnasium to use them.
- Reward terms are **named and individually logged** (`latency_penalty`, `fairness_bonus`, …). A scalar-only reward is undebuggable.
- Changing observation shape, action space, or reward weights **invalidates existing checkpoints**. Bump `MDP_VERSION` in `nexus_mdp/version.py` and say so in the PR.

### Agent Layer Rules

- **Provider-agnostic.** The LLM is any OpenAI-compatible Chat Completions endpoint, configured by `NEXUS_LLM_BASE_URL` / `NEXUS_LLM_MODEL` / `NEXUS_LLM_API_KEY`. Default is a local Ollama with no key. Never import a vendor SDK or hardcode a provider — ADR-0005.
- Tools are declared in `apps/agent-layer/src/nexus_agent/tools/` — one file per tool, with an explicit pydantic input schema.
- Read tools are unrestricted. **Write tools mutate constraints only** (objective weights, action masks, TTL), never allocations.
- Every write tool takes a mandatory TTL. No permanent operator overrides.
- Agent answers must cite retrieved numbers. If a tool returned no data, say so — never fill the gap with a plausible-sounding number.

### Testing & Documentation

- **pytest** for Python, **vitest** for the dashboard
- Tests live in `packages/<pkg>/tests/`, mirroring module names (`src/nexus_sim/link.py` → `tests/test_link.py`)
- Test observable behavior, never internals. Assert on emitted state and metrics, not on private attributes
- **Never train a model inside a test.** Test the env contract (shapes, bounds, determinism, reward sign) instead. Training belongs in `make evaluate`
- Use docstrings on public functions. Describe units — `latency_ms`, not `latency`

### Error Handling

- Every raised error subclasses `NexusError` from `nexus-errors`. Never raise bare `Exception`/`ValueError`.
- Never write a bare `except:` or a bare `except Exception:` without re-raising or logging with `telemetry.capture_exception`.
- Service boundaries (FastAPI handlers, MCP tools) map `NexusError` → structured response. Internal functions raise, they don't return error unions.
- Use `error_to_string(err)` for messages from `unknown`-shaped exceptions.

### Performance

- Simulator step must stay under ~1 ms for a 10-node graph — RL training does millions of steps. Profile before adding fidelity.
- Vectorize with numpy over per-node Python loops in the hot path (`step()`, reward, observation encoding).
- The dashboard receives **downsampled** state (default every 5th timestep). Never stream raw training-speed steps to a browser.

## Important Notes

### Planning

- When creating implementation plans, always write pseudo code. Do not use actual code.
- **Be extremely concise. Sacrifice grammar for the sake of concision when formulating the plan.**

## Code Quality & Common Patterns

- Never use `Any`. Use protocols, generics, `object` + narrowing, or a real type. `# type: ignore` needs an inline reason.
- Never use `cast()` to silence mypy. Fix the type.
- Hyperparameters and simulation constants come from config files or `nexus_sim/constants.py` — never inline magic numbers.
- Functions small, single-responsibility, side-effect free where possible. Simulation math must be pure functions of `(state, action)`.
- Before writing a new component, check whether one exists that a small parameter would generalize.
- Use TDD: write the failing test, then implement.

## Strict Prohibitions

- NEVER use `Any`, bare `except:`, or `cast()` to silence a type error
- NEVER use star imports (`from x import *`) or re-export barrels in `__init__.py`
- NEVER use the global `random` module or unseeded `np.random.*` — pass a `Generator`
- NEVER let the agent layer emit raw allocation actions — constraints only (ADR-0002)
- NEVER commit files under `experiments/runs/` or `*.zip` checkpoints
- NEVER train a model inside a unit test
- NEVER change observation/action/reward without bumping `MDP_VERSION`
- ALWAYS run `make lint`, `make types`, and relevant tests before declaring work done
- ALWAYS log reward components individually, not just the scalar total

## Self-Improvement Loop

After ANY correction from the user, update **both** knowledge layers:

1. **`tasks/lessons.md`** (versioned, shared) — Add the pattern, why, and when to apply it.
2. **Local memory** (personal, cross-session) — Save a feedback memory for future sessions.

At session start, read `tasks/lessons.md`. Do not repeat mistakes already documented there.

When a lesson conflicts with current code, trust the code — update the lesson.

## Automated Workflow (MUST FOLLOW)

Claude MUST automatically trigger the right skill/agent at the right moment. The user should never have to remember which skill to invoke.

### After Implementation (any non-trivial code change)

1. Run `/post-implement-check` automatically before saying "done"
2. If it fails, fix the issues and re-run until clean

### After Changing the MDP (observation / action / reward)

1. Spawn the `mdp-reviewer` agent on the diff
2. Confirm `MDP_VERSION` bumped and affected checkpoints flagged before the user reviews

### When a Training Run Plateaus or Collapses

1. If the user pastes a reward curve or says training is stuck, invoke `/diagnose-reward`
2. Do not ask "would you like me to investigate?" — just investigate

### Before Committing / Creating a PR

1. Spawn `pre-commit-auditor` agent as final gate
2. Only proceed with commit/PR if auditor returns READY

### On Context Exhaustion Warning

1. When context is running low, proactively run `/continue-session`
2. Output the handoff prompt so the user can start a new session cleanly

### Feature Development Lifecycle

1. `/create-plan` → design with agent teams
2. `/review-plan` → 3 parallel reviewer agents critique the plan (auto after plan)
3. Iterate on plan → fix CRITICAL/HIGH findings
4. Implement (TDD: tests first, then code)
5. `/post-implement-check` → quality gates (auto)
6. `pre-commit-auditor` → final audit (auto)
7. `/create-pr` → when user requests

## Agent skills

### Issue tracker

Issues live in GitHub Issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Experiment hygiene

Every training claim needs a seed, a baseline, and ≥3 seeds before it is stated as a result. See `.claude/skills/rl-experiment-hygiene/SKILL.md`.
