# AGENTS.md

Environment bootstrap and gotchas for coding agents running in this repo (Cursor Cloud, Claude Code, CI containers).

## Bootstrap

The repo is a **uv workspace** (Python 3.12) plus one **pnpm** app (the dashboard). Both installs are needed for a full check run.

```bash
# One-shot setup
make install
```

Which is equivalent to:

```bash
# Python workspace — creates .venv at the repo root, installs every packages/* and apps/* in editable mode
uv sync --all-packages --all-extras

# Dashboard only
pnpm --dir apps/dashboard install --frozen-lockfile
```

If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Do **not** create a venv by hand or `pip install` into the system Python. `uv` owns `.venv/`; anything installed outside it will be invisible to `make test`.

## Quick Start (services)

```bash
# 1. Copy the env template. The defaults work as-is: the agent layer points at a local Ollama and needs
#    no API key. Nothing here requires an account.
cp .env.example .env

# 2. Start the full stack
docker compose -f .docker/docker-compose.yml up -d

# 3. Wait for healthy containers (~15s)
until docker compose -f .docker/docker-compose.yml ps | grep -q "(healthy)"; do sleep 2; done

# 4. Dashboard at http://localhost:5173, simulator API docs at http://localhost:8000/docs
```

Without Docker, run the three Python services in separate shells:

```bash
uv run uvicorn nexus_simulator.main:app --port 8000 --reload
uv run uvicorn nexus_controller.main:app --port 8001 --reload
uv run nexus-agent-server                       # MCP server on stdio, or --http --port 8002
pnpm --dir apps/dashboard dev                   # Vite on 5173
```

## Running without a trained policy

The controller starts fine with no checkpoint — it falls back to the `static` baseline allocator and reports
`policy: "baseline:static"` on `/health`. Stages A, C, D and E of the build plan are all reachable before any
training has happened. If the dashboard shows flat, unremarkable allocations, check `/health` before assuming a bug.

To train the smallest useful policy (minutes, not hours):

```bash
uv run nexus-train --config experiments/configs/dqn-smoke.yaml
```

## Key gotchas

| Issue                                                    | Cause                                                                     | Fix                                                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `ModuleNotFoundError: nexus_sim` after editing a package | Ran `python` directly instead of through the workspace venv               | Prefix with `uv run`, or activate `.venv`                                        |
| `make test` passes locally, CI fails on types            | `mypy --strict` runs over all packages; local run was scoped to one       | Run `make types` (unscoped) before pushing                                       |
| Two runs with the same config give different rewards     | Something in the path used the global RNG                                 | Grep for `np.random.` and `import random`; every draw takes an injected `Generator` |
| Checkpoint loads but evaluation scores are nonsense      | `MDP_VERSION` changed since training                                      | Retrain, or check out the commit recorded in `experiments/runs/<id>/config.yaml` |
| Training is fast but reward is flat at exactly 0         | Action decoding produced an all-zero allocation — usually a shape mismatch | Assert on `info["reward_components"]`, not the scalar. See `/diagnose-reward`     |
| Agent layer answers with invented numbers                | An MCP tool errored and the model narrated around it                      | Tool errors must surface as tool errors, never as empty success payloads         |
| Agent replies but never calls a tool                     | The local model is too small for reliable tool calling                    | Use a tool-calling model (`qwen2.5:7b`, `llama3.1:8b`); 3B models are unreliable |
| `Could not reach the LLM endpoint`                       | Ollama not running, or wrong host from inside Docker                      | `ollama serve`; from a container use `host.docker.internal:11434`               |
| Dashboard WebSocket disconnects every few seconds        | Simulator restarted by `--reload` on every file save                      | Run the simulator without `--reload` when demoing                                |
| `docker compose up` fails on port 8000                   | Another local service holds it                                            | `SIMULATOR_HOST_PORT=8010 docker compose -f .docker/docker-compose.yml up -d`     |

## Service ports (local)

| Service        | Host Port | Container Port |
| -------------- | --------- | -------------- |
| Simulator API  | 8000      | 8000           |
| Controller API | 8001      | 8001           |
| Agent layer    | 8002      | 8002           |
| Dashboard      | 5173      | 5173           |

## Running checks

See `CLAUDE.md` for the full command reference. Summary:

- **Lint:** `make lint` (ruff)
- **Format:** `make format` (ruff format + prettier for the dashboard)
- **Types:** `make types` (mypy --strict)
- **Tests:** `make test` (pytest + vitest)
- **Single package:** `make test PKG=nexus-sim`
- **Pre-commit:** `make pr`

## Environment variables

All read in exactly one place — `packages/nexus-config/src/nexus_config/settings.py`. Adding a variable anywhere
else is a review blocker.

| Variable               | Required            | Purpose                                                                  |
| ---------------------- | ------------------- | ------------------------------------------------------------------------ |
| `NEXUS_LLM_BASE_URL`   | no                  | OpenAI-compatible endpoint. Defaults to local Ollama                     |
| `NEXUS_LLM_MODEL`      | no                  | Model id at that endpoint. Must support tool calling                     |
| `NEXUS_LLM_API_KEY`    | remote endpoints    | Empty is correct for Ollama / LM Studio / llama.cpp                      |
| `NEXUS_SIMULATOR_URL`  | controller, agent   | Defaults to `http://localhost:8000`                                      |
| `NEXUS_CONTROLLER_URL` | agent, dashboard    | Defaults to `http://localhost:8001`                                      |
| `NEXUS_POLICY_PATH`    | controller optional | Checkpoint to load; falls back to the `static` baseline                  |
| `NEXUS_SEED`           | no                  | Master seed for the simulator service (default `1337`)                   |
| `NEXUS_API_KEY`        | outside dev         | Guards mutating endpoints. Unrelated to `NEXUS_LLM_API_KEY`              |
| `NEXUS_LOG_LEVEL`      | no                  | `INFO` default                                                            |

### Running the agent layer against a local model

No account, no key, no cost:

```bash
ollama serve
ollama pull qwen2.5:7b          # tool-calling capable; llama3.1:8b also works
uv run nexus-agent-server
```

From inside Docker, Ollama on the host is `http://host.docker.internal:11434/v1`.

Switching to a hosted provider is two variables — no code change:

```bash
NEXUS_LLM_BASE_URL=https://api.groq.com/openai/v1
NEXUS_LLM_MODEL=llama-3.3-70b-versatile
NEXUS_LLM_API_KEY=<key>
```
