# Architecture Overview

Four services, one control loop, two deliberately asymmetric tiers.

## The shape

```
                        operator (natural language)
                                  │
                                  ▼
                        ┌───────────────────┐
                        │   Agent layer     │   slow · linguistic · MCP tools
                        │   (outer tier)    │   reads widely, writes CONSTRAINTS only
                        └─────────┬─────────┘
                                  │ Constraint (bounded TTL)
                                  ▼
   ┌──────────────┐  state   ┌───────────────────┐
   │  Simulator   │ ───────► │   Controller      │   fast · numeric · every timestep
   │ (single      │          │   (inner tier)    │   emits ALLOCATIONS
   │  writer)     │ ◄─────── │                   │
   └──────┬───────┘  action  └───────────────────┘
          │
          │ downsampled state (WebSocket)
          ▼
   ┌──────────────┐
   │  Dashboard   │   pure consumer
   └──────────────┘
```

The asymmetry is load-bearing. The inner tier acts every timestep on numbers; the outer tier acts at
operator pace on language. A change that makes the outer tier synchronous in the control loop, or gives it
inner-tier authority, breaks the design — see ADR-0002.

## Services

| Service | Owns | Scales |
| ------- | ---- | ------ |
| **Simulator** | Live simulation state. The only writer | Singleton — stateful, one authoritative timeline |
| **Controller** | The loaded policy, active constraints, decision history | Singleton in v1; stateless inference could scale, constraints could not |
| **Agent layer** | MCP tool surface, LLM conversation | Stateless — scales freely |
| **Dashboard** | Nothing. WebSocket in, pixels out | Static assets |

That different scaling profile is why these are four services rather than one process, and it is the actual
infrastructure deliverable — not "it runs in a container".

## The control loop

One tick, in order:

1. **Simulator** computes demand for the timestep from the seeded traffic pattern.
2. **Simulator** asks the **controller** for an action (`POST /act` with the current state).
3. **Controller** encodes the observation, runs the policy, decodes an `AllocationAction` — applying any
   active constraints as routing restrictions.
4. **Simulator** applies the action: resolves rates against link capacity, updates queues, computes drops,
   derives latency, and produces a new frozen `NetworkState`.
5. **Dashboard** receives every 5th state over WebSocket.

Asynchronously, at operator pace:

6. **Operator** asks the agent layer a question or gives an instruction.
7. **Agent layer** calls MCP read tools against simulator and controller.
8. If the instruction implies a change, the agent posts a **`Constraint`** to the controller — never an
   allocation.
9. **Agent layer** answers, citing the numbers it actually retrieved.

**The simulator drives, not the controller.** The simulator is the single writer of state; if the controller
drove, two concurrent callers would interleave writes and produce a trajectory no seed can reproduce.

## Failure modes, designed first

| What fails | What happens | Why that is right |
| ---------- | ------------ | ----------------- |
| Agent layer | Controller keeps running; constraints expire on schedule | Degrades to autonomous operation — the correct fallback for an autonomous orchestrator |
| Controller | Simulator steps with no action — every flow at full demand. `/health` reports `controller_reachable: false` | An unmanaged network is visibly unmanaged. A frozen one looks crashed |
| No checkpoint | Controller serves the `static` baseline and says so on `/health` | Stages A, C, D, E are reachable before any training exists |
| Dashboard | Nothing stops | It is a pure consumer |
| Simulator | Everything stops | The only genuine SPOF. Acceptable in a simulation |

The degraded path is designed before the happy path throughout. A control loop that silently does nothing
while reporting healthy is the worst available failure for an orchestrator.

## State ownership

One writer per piece of state — the rule that prevents most of the bugs this architecture could have:

- **Live simulation state** → simulator, in memory, behind a lock
- **Active constraints** → controller
- **Decision history** → controller, bounded ring buffer (512)
- **Run artifacts** → the training process, append-only files

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
