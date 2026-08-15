# Docker Development Environment

Four containers: simulator, controller, agent layer, dashboard.

## Start

```bash
docker compose -f .docker/docker-compose.yml up -d
```

Dashboard on http://localhost:5173, simulator API docs on http://localhost:8000/docs.

Nothing here needs an account or an API key. The agent layer points at an Ollama running on your host.

## Port conflicts

Every port is overridable — no need to edit the compose file:

```bash
SIMULATOR_HOST_PORT=8010 DASHBOARD_HOST_PORT=5180 \
  docker compose -f .docker/docker-compose.yml up -d
```

| Service | Default host port | Override |
| ------- | ----------------- | -------- |
| Simulator | 8000 | `SIMULATOR_HOST_PORT` |
| Controller | 8001 | `CONTROLLER_HOST_PORT` |
| Dashboard | 5173 | `DASHBOARD_HOST_PORT` |

## The agent layer and Ollama

The agent-layer container reaches Ollama on your **host** at `host.docker.internal:11434`. On Linux this
needs the `extra_hosts` mapping already in the compose file; on macOS and Windows it works out of the box.

```bash
ollama serve
ollama pull qwen2.5:7b
```

Switching to a hosted provider needs no rebuild:

```bash
NEXUS_LLM_BASE_URL=https://api.groq.com/openai/v1 \
NEXUS_LLM_MODEL=llama-3.3-70b-versatile \
NEXUS_LLM_API_KEY=<key> \
  docker compose -f .docker/docker-compose.yml up -d agent-layer
```

## Serving a trained policy

The controller runs on the `static` baseline by default and reports that on `/health`. To serve a
checkpoint:

1. Uncomment `EXTRAS: policy` under the controller build args (this adds torch — the image gets much
   larger).
2. Set `NEXUS_POLICY_PATH` to a path under `experiments/runs/`, which is mounted read-only at
   `/app/experiments/runs`.

```bash
NEXUS_POLICY_PATH=/app/experiments/runs/<run-id>/policy.zip \
  docker compose -f .docker/docker-compose.yml up -d --build controller
```

Confirm with `curl localhost:8001/health` — `trained_policy_loaded` should be `true`. If it is `false`,
`fallback_reason` says why; the usual cause is an `MDP_VERSION` mismatch.

## Images

`Dockerfile.python` is shared by all three Python services, parameterised by `SERVICE_PATH`. The dependency
layer is separate from the source layer, so editing code does not reinstall dependencies.

Torch is opt-in via `EXTRAS`. The simulator and agent layer never carry it, which is why those images are
an order of magnitude smaller than a single shared image would be.

## Useful commands

```bash
# Logs
docker compose -f .docker/docker-compose.yml logs -f simulator

# Health of everything
docker compose -f .docker/docker-compose.yml ps

# Inject an incident (Use Case 4)
curl -X POST localhost:8000/incidents -H 'Content-Type: application/json' -d '{
  "incident_id": "demo-1",
  "kind": "LINK_FAILED",
  "target_id": "n0->n1",
  "start_timestep": 0,
  "duration_timesteps": 200
}'

# Rebuild one service after a dependency change
docker compose -f .docker/docker-compose.yml up -d --build controller

# Tear down, including volumes
docker compose -f .docker/docker-compose.yml down -v
```

## Gotchas

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| Simulator healthy, dashboard empty | Wrong WebSocket URL after a port override | `VITE_SIMULATOR_WS` must match `SIMULATOR_HOST_PORT` |
| Agent layer cannot reach the LLM | Ollama not running on the host, or Linux without host-gateway | `ollama serve`; check `extra_hosts` is present |
| Controller reports `baseline:static` unexpectedly | No checkpoint, or it failed to load | `curl localhost:8001/health` and read `fallback_reason` |
| Dashboard reinstalls on every start | `dashboard_modules` volume was removed | Normal after `down -v`; the first start is slow |
| Simulator restarts constantly | Uvicorn `--reload` picking up file saves | The compose command does not use `--reload`; check for a local override |
