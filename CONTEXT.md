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

## Model lifecycle (Stage I)

### Policy version

A registered, addressable policy: an id (`v7`), the provenance of the run that produced it, every
evaluation score it has earned, and a deployment stage. The unit the controller is configured with —
`NEXUS_POLICY_VERSION=v7`, never a path to a checkpoint file.

_Avoid_: "the model", "the checkpoint" when you mean a version. A checkpoint is weights on disk; a version
is weights plus the record of where they came from and what they scored.

### Provenance

The origin of a policy version: run id, git SHA, `MDP_VERSION`, algorithm, action mode, topology, flow
count, training pattern, seed, timesteps. All required — a version missing any of them cannot be
reproduced, and "mostly traceable" is what lets an untraceable artifact into production.

### Stage

Where a version sits: `none`, `shadow`, `production`, `archived`. A **deployment claim**, and at most one
version holds `production` at a time. Demotion is to `archived` rather than `none` so a rollback target
stays findable.

### Evaluation suite

The fixed, versioned scenario set every candidate is scored against
(`experiments/configs/evaluation-suite.yaml`). Two policies are comparable only if they met the same
scenarios, seeds and step count. Scores record `suite_version`; changing the suite makes earlier scores
incomparable and requires a bump, exactly like `MDP_VERSION`.

### Promotion

Moving a version to a stage. Refuses a stale `MDP_VERSION` outright, and an unscored or fewer-than-three-
seed version unless explicitly forced.

_Avoid_: "deploying a model" for the act of copying a file. Promotion is a registry operation; the
controller follows it.

## Safety (Stages K–L)

### Safety supervisor

An independent service in the fast path that checks the controller's actions against hard bounds,
substitutes a deterministic allocator when the policy cannot be trusted, and reports policy-level
observability. Separate from the controller on purpose: a guardrail inside the thing it guards shares its
failure modes (ADR-0010).

### Action bounds

Limits enforced **outside the model**, on the action the policy produced. Share range, known flows, valid
routes, and starvation over a window. Distinct from QoS thresholds.

_Avoid_: calling `AllocationAction`'s own validator a guardrail. It is a type boundary — it rejects bad
values by refusing to exist, which destroys the diagnosis rather than producing one.

### QoS thresholds

Limits on what the network actually **did**, as opposed to what the policy asked for. A policy can emit
perfectly in-range shares and still drive the network into the ground.

### Fallback activation

The deterministic allocator taking over. Engages on an unrepairable action immediately, or on sustained
QoS breaches; releases only after a *longer* run of healthy timesteps, because the fallback fixes the
metric it engaged on and instant release oscillates. Its **rate is the clearest failure signal the system
has**.

### Drift

Standardised deviation of live state from the training-time reference. A **leading indicator**, reported
and never acted on — by the time QoS is breaching, the warning came too late to be useful.

### Domain randomization

Randomizing simulator parameters (capacity, demand, latency) per episode so the policy learns a
distribution rather than one network. Improves robustness; **provides no explicit safety guarantee**, and
over-wide ranges destabilise training.

_Avoid_: "the policy is safe because it was domain-randomized". Say "more robust across the range it saw".

### Held-out configuration

A simulator configuration deliberately **outside** the training randomization ranges. Evaluating there is
what makes a robustness claim evidence rather than a restatement of the training score.

### Shadow mode

Running a candidate policy alongside the incumbent, its decisions recorded and never applied. Answers
*readiness* — safe, responsive, meaningfully different — never *superiority*, because the candidate's
actions were never applied and no outcome is attributable to it.

_Avoid_: "the shadow policy performed better". It did not perform at all.
