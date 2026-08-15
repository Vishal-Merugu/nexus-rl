# Context

NEXUS-RL is a two-tier autonomous network orchestrator over a simulated network. This file is the canonical glossary of domain terms used in code and conversation. Update inline when a term is sharpened.

## Glossary

### Orchestrator

The combined RL controller + LLM agent layer, taken as one system. The industry term for the component that decides how a network's shared resources are divided. When a doc says "the orchestrator decided", it means the system as a whole; name the tier explicitly when the distinction matters.

### Inner tier / Controller

The RL controller. Fast (every timestep), numeric, no language. Consumes an observation, emits an `Action`. This is the only component allowed to produce allocations.

_Avoid_: "the agent" for this component — that is ambiguous with the LLM agent. Say "the controller" or "the RL controller".

### Outer tier / Agent layer

The LLM agent. Slow (operator-paced), linguistic, tool-calling. Reads controller and simulator state via MCP tools; writes only `Constraint` objects. Never emits allocations.

_Avoid_: "the RL agent" for this component.

### Timestep

One simulation tick. The unit of the inner control loop: simulator computes state → controller returns action → simulator applies action, computes reward and next state. Wall-clock duration is a display concern only; the simulator has no notion of real time.

### Flow

A traffic demand between a source node and a destination node, with a `traffic_class` and a requested rate in Mb/s. Flows are the unit of allocation — the controller decides how much capacity each flow gets on each link of its path, not how individual packets move.

### Flow-level simulation

The chosen fidelity: aggregate rates per flow per link, queues as scalar backlogs. Explicitly **not** packet-level. Latency is derived analytically from utilization and backlog, not measured from packet traces. See ADR-0001.

### Link

A directed edge between two nodes with a fixed `capacity_mbps`. Carries a scalar `queue_mb` backlog. Its `utilization` is allocated rate ÷ capacity, always clamped to [0, 1].

### Utilization

Allocated rate on a link divided by its capacity. Always in [0, 1]. Demand above capacity does not show as utilization > 1 — the excess becomes queue growth and, past the queue limit, drops. Do not read utilization as a demand signal; read `offered_load_mbps` for that.

### Traffic class

The QoS category of a flow: `LATENCY_SENSITIVE`, `BULK`, `BACKGROUND`. Determines the weight a flow's latency and throughput carry in the reward, and which flows an operator constraint can target.

_Avoid_: "priority" as a noun for this — priority is the *outcome* the controller chooses, class is the *input* property of the flow.

### Reward components

The reward is a weighted sum of individually named terms, never an opaque scalar: `throughput_reward`, `latency_penalty`, `drop_penalty`, `fairness_bonus`. Every one is logged separately each step. A change to the weights is an MDP change (see MDP version).

### Fairness

Jain's fairness index over per-flow satisfaction ratios (allocated ÷ requested), in (0, 1]. 1.0 means every flow got the same fraction of what it asked for. This is a *proportional* notion — an equal-bytes split of an unequal demand set scores poorly, and that is intended.

### MDP version (`MDP_VERSION`)

Integer in `nexus_mdp/version.py`, bumped whenever the observation encoding, action space, or reward definition changes. A checkpoint trained under version N is meaningless under version N+1. Every run artifact records the version it was trained under; evaluation refuses to load a mismatched checkpoint.

### Constraint

The only thing the agent layer can write. A time-boxed modifier on the controller's optimization: an objective reweighting (`objective_weights`), an action restriction (`action_mask`), or a per-class floor. Carries a mandatory `ttl_timesteps` and an `issued_by` provenance string.

_Avoid_: "policy change" — the policy (the trained network) is not modified by operators; only the objective it is scored against, and the actions it is permitted, change.

### Constrained authority

The design rule that the outer tier may reweight or restrict but never directly allocate. Both a safety property and the central system-design claim of the project. See ADR-0002.

### Policy

The trained function mapping observation to action — the artifact produced by training. Frozen at inference time in v1. Distinct from "the controller", which is the *service* hosting a policy.

### Baseline

A non-learned allocator used as the comparison floor: `static` (fixed equal split), `round_robin` (rotating priority), `greedy_throughput` (maximize total allocated rate, ignore fairness and class). Every performance claim about a policy is stated relative to a named baseline on a named traffic pattern.

### Traffic pattern

A named, seeded demand generator: `steady`, `diurnal`, `bursty`, `flash_crowd`. Evaluation runs each policy across at least two patterns — a policy that only beats the baseline on its training pattern has not been shown to work.

### Incident

A deliberate degradation injected into the simulator: link capacity cut, link failure, or node failure, with a start timestep and duration. Used for Use Case 4. Incidents are part of the environment, not of the action space — the controller reacts to them, it cannot cause or clear them.

### Episode

A bounded sequence of timesteps used for training and evaluation (default 512). Episode boundaries are a training construct; the deployed simulator service runs continuously and has no episodes.

### Run

One training or evaluation execution, identified by a `run-id`, with its config, metrics, and checkpoints under `experiments/runs/<run-id>/`. A claim without a run-id is not a result.

## O-RAN vocabulary (v2, Track A)

The two-tier design of this system converged independently on the structure the O-RAN Alliance
standardized. From v2 onward the standard's terms are the primary ones — use these in code and
conversation, with the v1 term in parentheses where it aids the reader. See ADR-0008.

**Architectural alignment, not conformance.** Nothing here is conformance-tested, and the transport is
plain HTTP rather than the standard's own bindings. Say "structured according to O-RAN's architecture",
never "O-RAN compliant".

### xApp

An application on the Near-RT RIC performing near-real-time control. Here: the RL controller service —
the inner tier. Consumes A1 policy, retains sole authority to allocate.

### rApp

An application on the Non-RT RIC performing slower-timescale work — policy generation, SLA assurance,
analytics. Here: the LLM agent layer, the outer tier. Issues A1 policy; never a control action.

### Near-RT RIC / Non-RT RIC

RAN Intelligent Controller, near-real-time (10 ms – 1 s control loops, hosts xApps) and non-real-time
(> 1 s, hosted within SMO, hosts rApps and owns ML model lifecycle). The two-timescale split is the whole
point: the LLM is never in the fast path.

### A1

The interface carrying policy guidance from the Non-RT tier down to the Near-RT tier. In this repo,
`nexus-a1` plus the controller's `/a1/*` routes. Carries guidance, never an allocation — the standard's
own answer to the problem ADR-0002 solved independently.

_Avoid_: "the agent sets an allocation over A1". It cannot; there is no such policy type.

### A1 policy

A versioned document with a `policy_type`, a `scope`, a `statement`, provenance, and a **mandatory
bounded TTL** (a deliberate deviation — A1 itself requires no lifetime). Translated into a v1
`Constraint` on acceptance: A1 is the interface, the constraint is the enforcement.

### A1 policy decision

The xApp's answer to an offered policy: `ACCEPTED`, `PARTIALLY_ACCEPTED`, or `REJECTED`, always with an
operator-facing `reason`, and an `achievable_alternative` where one exists. Partial acceptance is a
distinct state on purpose — enforcing two of three requested links and reporting plain success is the
failure this type exists to prevent.

### E2

The interface between the Near-RT RIC and the network elements (E2 nodes) carrying telemetry up and
control down. Here: the existing `/act` call between simulator and controller. Not a real E2AP
implementation — the shape, not the protocol.

### SMO

Service Management and Orchestration — the framework hosting the Non-RT RIC. Here: nothing distinct yet;
the dashboard stands in for the SMO-side operator console.
