# Architecture Overview

Four services, two control loops, two deliberately asymmetric tiers — structured according to the O-RAN
Alliance's RIC architecture. See [ADR-0008](../adr/0008-oran-aligned-a1-interface.md) for how the project
arrived at that structure independently before adopting its vocabulary.

## The shape

```
                        operator (natural language)
                                  │
                                  ▼
                        ┌───────────────────┐
                        │    Agent layer    │   Non-RT tier · rApp · > 1 s
                        │   (Non-RT RIC /   │   slow · linguistic · MCP tools
                        │        SMO)       │   reads widely, issues A1 POLICY only
                        └─────────┬─────────┘
                                  │ A1 policy ──────► decision
                                  │ (typed, versioned,  (ACCEPTED /
                                  ▼  bounded TTL)        PARTIAL / REJECTED)
   ┌──────────────┐  state   ┌───────────────────┐
   │  Simulator   │ ───────► │    Controller     │   Near-RT tier · xApp · every timestep
   │  (E2 nodes:  │   E2     │   (Near-RT RIC)   │   fast · numeric
   │  O-CU/O-DU   │ ◄─────── │                   │   emits ALLOCATIONS — sole authority
   │  stand-ins)  │  action  └───────────────────┘
   └──────┬───────┘
          │
          │ downsampled state (WebSocket) · O1-style management data
          ▼
   ┌──────────────┐
   │  Dashboard   │   SMO-side operator console · pure consumer
   └──────────────┘
```

The asymmetry is load-bearing. The Near-RT tier acts every timestep on numbers; the Non-RT tier acts at
operator pace on language. A change that puts the Non-RT tier synchronously in the fast loop, or gives it
Near-RT authority, breaks the design — see ADR-0002.

## The O-RAN mapping

| This project | O-RAN | Interface it speaks |
| ------------ | ----- | ------------------- |
| Controller service | **xApp** on a Near-RT RIC (10 ms – 1 s loop) | A1 (consumes policy), E2 (to network elements) |
| Agent layer | **rApp** on a Non-RT RIC within SMO (> 1 s loop) | A1 (issues policy), R1 |
| "Constrained authority" (ADR-0002) | A1 policy guidance | A1 |
| Simulator | **E2 nodes** — stand-ins for O-CU / O-DU | E2 |
| Dashboard | SMO-side operator console | O1-style management data |

**This is architectural alignment, not conformance.** Nothing is tested against an O-RAN conformance suite;
the transport is plain HTTP rather than the standard's own bindings, and "E2" here is the existing `POST
/act` call rather than an E2AP implementation. The honest claim is "structured according to O-RAN's
architecture", never "O-RAN compliant".

## Services

| Service | O-RAN role | Owns | Scales |
| ------- | ---------- | ---- | ------ |
| **Simulator** | E2 nodes | Live simulation state. The only writer | Singleton — stateful, one authoritative timeline |
| **Controller** | Near-RT RIC / xApp | The loaded policy version, accepted A1 policies, active constraints, decision history | Singleton; stateless inference could scale, accepted policy state could not |
| **Agent layer** | Non-RT RIC / rApp | MCP tool surface, LLM conversation | Stateless — scales freely |
| **Safety supervisor** | — (no O-RAN analogue) | Its own bounds, thresholds, fallback state and drift window | Singleton per control loop; stateful across timesteps |
| **Dashboard** | SMO console | Nothing. WebSocket in, pixels out | Static assets |

That different scaling profile is why these are four services rather than one process, and it is the actual
infrastructure deliverable — not "it runs in a container".

## The two control loops

Making these explicit is the whole point of the O-RAN framing.

### Fast loop (every timestep, no LLM anywhere in it)

1. **Simulator** computes demand for the timestep from the seeded traffic pattern.
2. **Simulator** asks for an action (`POST /act` — the E2-shaped call). In a deployment with the safety
   supervisor in the path, that call lands on the **supervisor**, which forwards it to the xApp, checks
   what comes back against hard bounds, and substitutes a deterministic allocator when the policy cannot
   be trusted (ADR-0010). The contract is identical either way.
3. **xApp** encodes the observation, runs the policy, decodes an `AllocationAction` — applying any active
   constraints as routing restrictions.
4. **Simulator** applies the action: resolves rates against link capacity, updates queues, computes drops,
   derives latency, and produces a new frozen `NetworkState`.
5. **Dashboard** receives every 5th state over WebSocket.

### Slow loop (operator pace, seconds to minutes)

6. **Operator** asks the rApp a question or gives an instruction.
7. **rApp** calls MCP read tools against simulator and xApp.
8. If the instruction implies a change, the rApp issues an **`A1Policy`** — never an allocation, and never a
   `Constraint` written directly.
9. **xApp** validates it against the network it last observed and answers with an `A1PolicyDecision`:
   ACCEPTED, PARTIALLY_ACCEPTED, or REJECTED, each carrying an operator-facing reason and, where one exists,
   an achievable alternative. Only what it accepted becomes a `Constraint`.
10. **rApp** relays that decision, citing the numbers it actually retrieved.

**The LLM is never in the fast path.** A non-deterministic component with multi-second latency cannot sit in
a loop that must run every timestep, and keeping it out is an architectural requirement rather than a
performance tuning choice.

**The simulator drives, not the controller.** The simulator is the single writer of state; if the controller
drove, two concurrent callers would interleave writes and produce a trajectory no seed can reproduce.

## A1 is the interface; `Constraint` is the enforcement

The two are deliberately not merged. `A1Policy` is the versioned wire contract between two independently
deployed tiers — it has a schema version, an advertised type set, and a negotiated response. `Constraint` is
the xApp's internal representation of guidance it has agreed to, and it carries the enforcement machinery
that was already tested: expiry, last-writer-wins objective precedence, accumulating link restrictions,
class floors.

Translation runs **after** validation and follows the *decision*, not the request. A partially-accepted
steering policy becomes a constraint covering only the links the xApp agreed to — translating the original
request would apply guidance the operator was explicitly told had been refused.

## Failure modes, designed first

| What fails | What happens | Why that is right |
| ---------- | ------------ | ----------------- |
| Agent layer (rApp) | xApp keeps running; accepted policies expire on schedule | Degrades to autonomous operation — the correct fallback for an autonomous orchestrator, and the reason the Non-RT tier is kept out of the fast loop |
| Controller (xApp) | Simulator steps with no action — every flow at full demand. `/health` reports `controller_reachable: false` | An unmanaged network is visibly unmanaged. A frozen one looks crashed |
| No checkpoint | Controller serves the `static` baseline and says so on `/health` | Stages A, C, D, E are reachable before any training exists |
| Policy emits an unsafe action | Supervisor substitutes the deterministic allocator, labels it `supervisor:fallback`, and reports it on `/safety` | A guardrail that silently repaired actions would hide a policy defect nobody ever learns about |
| Supervisor cannot reach the controller | Deterministic allocator drives; `/health` says `controller_unreachable` | Distinct from a *bad* action: nothing is wrong with the policy and the fix is elsewhere |
| Dashboard | Nothing stops | It is a pure consumer |
| Simulator | Everything stops | The only genuine SPOF. Acceptable in a simulation |

The degraded path is designed before the happy path throughout. A control loop that silently does nothing
while reporting healthy is the worst available failure for an orchestrator.

## State ownership

One writer per piece of state — the rule that prevents most of the bugs this architecture could have:

- **Live simulation state** → simulator, in memory, behind a lock
- **Accepted A1 policies and their decisions** → controller (xApp). An A1 policy is identified state at the
  *consumer*, not a message the issuer can retroactively edit — which is why a policy id already in force is
  refused rather than treated as an update, and why changing guidance means withdraw-then-reissue
- **Active constraints** → controller
- **Decision history** → controller, bounded ring buffer (512)
- **Run artifacts** → the training process, append-only files
- **Policy provenance and deployment stage** → the registry ledger under `experiments/registry/`, written
  by the CLIs and read by the controller. Committed to git, unlike the weights it describes — the answer
  to "what is deployed and where did it come from" must outlive the machine that produced it (ADR-0009)

There is no database. Adding persistence is an ADR, not a convenience.

## Package vs. app

`packages/` holds libraries: no FastAPI, no I/O at import, no service code. `apps/` holds services: thin,
parse-delegate-serialize. Simulation maths in a handler, or a web framework in a package, is a review
blocker — it is what makes the simulator testable in milliseconds without anything running.

## Further reading

- [MDP specification](../mdp/state-action-reward.md) — the observation/action/reward contract
- [ADR-0001](../adr/0001-flow-level-simulation.md) — flow-level fidelity
- [ADR-0002](../adr/0002-constrained-agent-authority.md) — the authority boundary
- [ADR-0003](../adr/0003-mcp-over-ad-hoc-rest.md) — MCP tool surface
- [ADR-0004](../adr/0004-discrete-dqn-before-continuous-ppo.md) — discrete first
- [ADR-0005](../adr/0005-provider-agnostic-llm.md) — provider-agnostic LLM
- [ADR-0006](../adr/0006-controller-holds-no-network-model.md) — the xApp holds no network model
- [ADR-0007](../adr/0007-mdp-contract-split-from-gym-wrapper.md) — MDP contract vs. gym wrapper
- [ADR-0008](../adr/0008-oran-aligned-a1-interface.md) — the A1-shaped policy interface
- [ADR-0009](../adr/0009-file-backed-model-registry.md) — the file-backed model registry
- [ADR-0010](../adr/0010-standalone-safety-supervisor.md) — the standalone safety supervisor
- [Model registry & evaluation suite](../operations/model-registry.md) — the deployment workflow

## Why the controller holds no model of the network

The controller decodes actions purely from the `NetworkState` it receives — including each flow's
`candidate_paths`. It builds no simulator, reads no topology, and does not know the seed.

It used to reconstruct the flow set from its own `NEXUS_SEED`/`NEXUS_TOPOLOGY` in order to learn the action
space. Two services deriving shared state independently from configuration diverge silently, and this case
was especially quiet: flow ids are positional (`f0`..`fn`), so a one-digit config difference produced ids
that lined up perfectly while the routes behind them belonged to entirely different flows. Wrong paths, no
error, anywhere.

Putting the action space in the state contract removes the failure class rather than guarding against it,
and drops `nexus-sim` from the controller's dependencies. A service should not need to agree with another
service about anything it can simply be told.

## Package split: contract vs. environment

`nexus-mdp` holds the MDP contract — observation encoding, action decoding, reward, `MDP_VERSION`. Pure
numpy. `nexus-env` holds only the gymnasium wrapper and depends on `nexus-mdp`.

The controller needs the *codec*, not the training environment. Before the split it imported `nexus-env`
and therefore pulled gymnasium into a service that never constructs an `Env`. The split also makes the
"MDP lives in exactly one place" rule structural: a service physically cannot depend on the training
environment to reach the contract.
