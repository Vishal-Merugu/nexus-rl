# NEXUS-RL

A research prototype for autonomous network resource orchestration in a **simulated network**. A reinforcement-learning controller allocates bandwidth each timestep; an LLM agent translates operator intent into bounded policy proposals that the controller can accept, partially apply, or reject.

The design is inspired by O-RAN’s Near-RT RIC / Non-RT RIC split and A1 policy concepts. It is not an O-RAN conformance implementation, and it has not been validated against physical network equipment.

## System overview

| Service | Role |
| --- | --- |
| Simulator | Seeded flow-level network simulation and control loop |
| Controller | FastAPI service that applies the active allocator and constraints |
| Safety supervisor | Checks proposed actions, enforces bounds, and falls back when needed |
| Agent layer | MCP tools and an OpenAI-compatible LLM endpoint for operator requests |
| Dashboard | React/Vite live view of network topology, flows, incidents, and QoS metrics |

The agent proposes policy; it cannot allocate bandwidth or write controller constraints directly. The controller remains authoritative. Shared Python packages define the MDP, baselines, policy contracts, safety rules, model registry, and telemetry.

## Current status

- The simulator, service APIs, dashboard, safety supervisor, training/evaluation harness, and versioned policy registry are in the repository.
- The controller uses the static baseline when no compatible policy checkpoint is available.
- Training checkpoints and run artifacts are ignored by Git; a trained policy is not shipped with a fresh clone.
- Kubernetes manifests are provided for local experimentation. They have not been validated as a production deployment.
- Results from a single training run are not treated as evidence of general performance. Use the evaluation suite and multiple seeds before making performance claims.

## Quick start

### Requirements

- Python 3.12+
- uv
- Docker Compose
- Node.js and pnpm for the dashboard
- Ollama on the host if you want local LLM responses (a hosted OpenAI-compatible endpoint can be configured instead)

Install the workspace and dashboard dependencies, then start the stack:

~~~bash
make install
cp .env.example .env
ollama serve
ollama pull qwen2.5:7b
make up
~~~

The dashboard is available at http://localhost:5173. The simulator API docs are at http://localhost:8000/docs. Docker Compose uses the host’s Ollama service for the agent layer; the simulator and controller can start on the static baseline without a trained model.

To stop the services, run make down.

## Train and evaluate

~~~bash
# Short pipeline smoke run
uv run nexus-train --config experiments/configs/dqn-smoke.yaml

# Evaluate the checked-in baseline suite
make evaluate

# Run the project quality checks
make types
make lint
make test
~~~

Training configurations are in experiments/configs. Run outputs and policy weights are written beneath experiments/runs and are not committed. See docs/operations/model-registry.md for registering, scoring, and promoting a checkpoint.

## Kubernetes

The manifests and local kind/minikube instructions are in deploy/k8s/README.md. They demonstrate service boundaries, health probes, resource limits, and replica choices; the presence of manifests does not mean the system has been deployed to a cluster.

## Repository map

- apps/simulator — FastAPI simulation service
- apps/controller — policy and constraint service
- apps/safety-supervisor — independent action guardrail
- apps/agent-layer — MCP server and LLM integration
- apps/dashboard — React/Vite dashboard
- packages/ — simulator, MDP, baselines, RL, safety, A1 contracts, registry, and shared types
- experiments/configs — training and evaluation configurations

## Limitations

- Simulation is flow-level: it models aggregate rates and queue backlogs, not packet traces.
- Latency is derived by the simulator and has not been calibrated to real equipment.
- O-RAN alignment is architectural; transport and interfaces are not conformance-tested.
- Safety checks reduce unsafe actions but do not prove system safety.
- No trained policy checkpoint is included in the repository.

## Further reading

- [Architecture overview](docs/architecture/00_overview.md)
- [State, action, and reward contract](docs/mdp/state-action-reward.md)
- [Model registry and evaluation suite](docs/operations/model-registry.md)
- [Safety supervisor](docs/operations/safety-supervisor.md)
- [Architecture decisions](docs/adr/0008-oran-aligned-a1-interface.md)
- [Security policy](SECURITY.md)
