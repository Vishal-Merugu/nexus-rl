# Kubernetes Deployment

Local-first: `kind` or `minikube`, no cloud account required. This is the infrastructure deliverable — the
claim being made is "I understand orchestration", not "I can adopt every CNCF project", so there is no Helm
chart, no operator, and no service mesh.

## Deploy locally

```bash
kind create cluster --name nexus

# Build and load images (kind does not read the host daemon's images)
docker build -t nexus-rl/simulator:0.1.0   --build-arg SERVICE_PATH=apps/simulator   -f .docker/Dockerfile.python .
docker build -t nexus-rl/controller:0.1.0  --build-arg SERVICE_PATH=apps/controller  -f .docker/Dockerfile.python .
docker build -t nexus-rl/agent-layer:0.1.0 --build-arg SERVICE_PATH=apps/agent-layer -f .docker/Dockerfile.python .
kind load docker-image --name nexus nexus-rl/simulator:0.1.0 nexus-rl/controller:0.1.0 nexus-rl/agent-layer:0.1.0

kubectl create secret generic nexus-secrets \
  --from-literal=NEXUS_API_KEY="$(openssl rand -hex 16)" \
  --from-literal=NEXUS_LLM_API_KEY=""

kubectl apply -f deploy/k8s/
kubectl get pods -w
```

Dashboard on `http://localhost:30080` once the NodePort is reachable.

## What scales, and what does not

This is the reasoning the manifests encode — the part worth explaining in an interview:

| Component | Replicas | Why |
| --------- | -------- | --- |
| Simulator | **1, `Recreate`** | Single writer of simulation state. A second replica is a second timeline, not more capacity. `RollingUpdate` would briefly run two |
| Controller | **1** | Inference is stateless and would scale, but active constraints are per-process state. Scaling it means moving constraints to shared storage — an ADR, not a replica count |
| Agent layer | **2+** | Genuinely stateless. Scales freely |
| Dashboard | **2+** | Static assets |

The simulator and controller having different scaling behaviour from the agent layer is the actual reason
this is four services rather than one process.

## Exposure

Only the dashboard is reachable from outside. Simulator, controller and agent layer are `ClusterIP`.
Mutating endpoints (`/incidents`, `/reset`, `/constraints`) require the `X-Nexus-Key` header because
`NEXUS_ENV` is `production` here.

## Probes

Readiness and liveness are deliberately different. A controller loading a checkpoint is **not ready**, not
dead — conflating them makes Kubernetes restart-loop it forever while the logs show a perfectly healthy
startup. The controller's readiness probe therefore allows a long failure threshold, and its liveness probe
starts late.

Health endpoints report what the service is actually doing: the simulator's includes the current timestep
and whether the controller is reachable; the controller's reports which allocator is live. A probe that
returns 200 unconditionally is a liability during a demo.

## Resources

Every container has requests and limits. The controller's memory limit matters most — with the `policy`
extra it carries torch, and an unbounded container gets OOM-killed on a kind cluster in a way that looks
mysterious rather than like a memory problem.

## The LLM in-cluster

`NEXUS_LLM_BASE_URL` defaults to `http://ollama:11434/v1`, expecting an Ollama deployment in the cluster.
For a demo it is usually simpler to point it at a hosted OpenAI-compatible endpoint and put the key in
`nexus-secrets` — two values in the ConfigMap and Secret, no image rebuild (ADR-0005).

## Teardown

```bash
kubectl delete -f deploy/k8s/
kind delete cluster --name nexus
```
